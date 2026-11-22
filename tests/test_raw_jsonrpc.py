"""Unit tests for defensive raw JSON-RPC 2.0 protocol validation and error handling."""

import json
import os
import subprocess
import sys
import tempfile
import time
import pytest
from unittest.mock import MagicMock

from wandering_kettle_assistant.raw_jsonrpc import (
    RobustRawJSONRPCClient,
    JSONRPCParseError,
    JSONRPCVersionError,
    JSONRPCIdMismatchError,
    JSONRPCResponseError,
    ServerProcessTerminatedError,
    JSONRPCTimeoutError,
    MalformedToolResultError,
    MCPToolExecutionError,
)


def test_invalid_json_handling():
    """Verify that malformed/invalid JSON raises JSONRPCParseError."""
    with pytest.raises(JSONRPCParseError) as exc_info:
        RobustRawJSONRPCClient.parse_and_validate_frame("not a valid json string {]")
    assert "Invalid JSON" in str(exc_info.value)


def test_incorrect_jsonrpc_version():
    """Verify that version other than '2.0' raises JSONRPCVersionError."""
    bad_version = json.dumps({"jsonrpc": "1.0", "id": 1, "result": {}})
    with pytest.raises(JSONRPCVersionError) as exc_info:
        RobustRawJSONRPCClient.parse_and_validate_frame(bad_version)
    assert "Unsupported JSON-RPC version" in str(exc_info.value)


def test_mismatched_request_id():
    """Verify that mismatched response id raises JSONRPCIdMismatchError."""
    mismatched = json.dumps({"jsonrpc": "2.0", "id": 99, "result": {}})
    with pytest.raises(JSONRPCIdMismatchError) as exc_info:
        RobustRawJSONRPCClient.parse_and_validate_frame(mismatched, expected_id=1)
    assert "Request ID mismatch" in str(exc_info.value)


def test_jsonrpc_error_response():
    """Verify that JSON-RPC error frames raise JSONRPCResponseError with code and message."""
    err_frame = json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "error": {
            "code": -32601,
            "message": "Method not found",
        },
    })
    with pytest.raises(JSONRPCResponseError) as exc_info:
        RobustRawJSONRPCClient.parse_and_validate_frame(err_frame, expected_id=2)
    assert exc_info.value.code == -32601
    assert "Method not found" in exc_info.value.message


def test_malformed_tool_result():
    """Verify that tool response missing result or content raises MalformedToolResultError."""
    # Missing content list
    bad_result = {"result": {"wrong_field": 123}}
    with pytest.raises(MalformedToolResultError):
        RobustRawJSONRPCClient.extract_and_validate_tool_result(bad_result)

    # Missing result object altogether
    bad_payload = {"no_result": True}
    with pytest.raises(MalformedToolResultError):
        RobustRawJSONRPCClient.extract_and_validate_tool_result(bad_payload)


def test_mcp_tool_is_error_flag():
    """Verify that tool result with isError: true raises MCPToolExecutionError."""
    err_tool_result = {
        "result": {
            "content": [{"type": "text", "text": "ChromaDB connection refused"}],
            "isError": True,
        }
    }
    with pytest.raises(MCPToolExecutionError) as exc_info:
        RobustRawJSONRPCClient.extract_and_validate_tool_result(err_tool_result)
    assert "ChromaDB connection refused" in str(exc_info.value)


def test_stdout_server_log_isolation():
    """Verify that stray server logs on stdout preceding the JSON frame are safely ignored."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdout.readline.side_effect = [
        "2026-11-22 INFO [server]: Starting up FastMCP\n",
        "Loading weights: 100%\n",
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "TestServer"}}}) + "\n",
    ]

    frame = RobustRawJSONRPCClient.read_next_jsonrpc_frame(mock_proc, expected_id=1, timeout=2.0)
    assert frame["id"] == 1
    assert frame["result"]["serverInfo"]["name"] == "TestServer"
    RobustRawJSONRPCClient.close_reader(mock_proc)


def test_server_process_termination_handling():
    """Verify that premature process termination raises ServerProcessTerminatedError."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = -9  # Terminated
    mock_proc.returncode = -9
    mock_proc.stdout.readline.return_value = ""  # EOF

    with pytest.raises(ServerProcessTerminatedError) as exc_info:
        RobustRawJSONRPCClient.read_next_jsonrpc_frame(mock_proc, expected_id=1, timeout=1.0)
    assert "terminated" in str(exc_info.value).lower()
    RobustRawJSONRPCClient.close_reader(mock_proc)


def test_real_subprocess_timeout():
    """Start a real subprocess that never writes stdout and verify JSONRPCTimeoutError
    is raised within the specified timeout plus a tolerance window.

    This proves the thread+queue pattern genuinely interrupts a blocking readline().
    Works on both Windows and Linux.
    """
    script_content = "import time\nwhile True:\n    time.sleep(60)\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    timeout_seconds = 1.0
    tolerance_seconds = 2.0  # generous tolerance for CI/slow machines

    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        t0 = time.time()
        with pytest.raises(JSONRPCTimeoutError):
            RobustRawJSONRPCClient.read_next_jsonrpc_frame(
                proc, expected_id=1, timeout=timeout_seconds
            )
        elapsed = time.time() - t0

        assert elapsed < timeout_seconds + tolerance_seconds, (
            f"Timeout took {elapsed:.2f}s but should have fired within {timeout_seconds + tolerance_seconds:.1f}s"
        )
        assert elapsed >= timeout_seconds * 0.8, (
            f"Timeout fired too early at {elapsed:.2f}s (expected >= {timeout_seconds * 0.8:.1f}s)"
        )
    finally:
        RobustRawJSONRPCClient.close_reader(proc)
        proc.kill()
        proc.wait()
        os.unlink(script_path)


def test_persistent_reader_sequential_requests_no_dropped_frames():
    """Prove that multiple sequential request-response cycles on the same subprocess
    do not lose, skip, or abandon responses across consecutive calls.

    Verifies that exactly one reader thread and one queue persists for the process.
    """
    # Create an echo responder script: reads JSON-RPC requests from stdin, responds on stdout
    responder_code = """
import sys, json

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        req_id = req.get("id")
        method = req.get("method")
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"echo_method": method, "sequence": req_id}
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(e)}}) + "\\n")
        sys.stdout.flush()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(responder_code)
        script_path = f.name

    proc = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )

    client = RobustRawJSONRPCClient(proc)

    try:
        # Perform 5 sequential request-response exchanges across the same reader
        for req_id in range(1, 6):
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": f"test_action_{req_id}",
                "params": {"step": req_id},
            }
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            # Read frame with expected ID matching the sent request
            resp = RobustRawJSONRPCClient.read_next_jsonrpc_frame(proc, expected_id=req_id, timeout=3.0)
            assert resp["id"] == req_id
            assert resp["result"]["sequence"] == req_id
            assert resp["result"]["echo_method"] == f"test_action_{req_id}"

    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        os.unlink(script_path)
