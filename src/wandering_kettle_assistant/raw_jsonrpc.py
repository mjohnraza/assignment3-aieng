"""Robust, production-grade Raw JSON-RPC 2.0 Client implementation for MCP.

Implements explicit failure handling and defensive protocol checks for:
- Invalid JSON responses (JSONRPCParseError)
- Incorrect JSON-RPC protocol versions (JSONRPCVersionError)
- Mismatched request IDs (JSONRPCIdMismatchError)
- Server-side JSON-RPC error frames (JSONRPCResponseError)
- Server process premature termination (ServerProcessTerminatedError)
- Request timeouts (JSONRPCTimeoutError)
- Malformed tool results missing content blocks (MalformedToolResultError)
- MCP tool execution errors with isError: true (MCPToolExecutionError)
- Isolation of JSON-RPC protocol frames from stray stdout server logs
"""

import json
import queue
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class JSONRPCError(Exception):
    """Base class for all JSON-RPC and MCP protocol errors."""
    pass


class JSONRPCParseError(JSONRPCError):
    """Raised when wire bytes/text cannot be parsed as valid JSON."""
    pass


class JSONRPCVersionError(JSONRPCError):
    """Raised when the JSON-RPC response version is not '2.0'."""
    pass


class JSONRPCIdMismatchError(JSONRPCError):
    """Raised when response id does not match the sent request id."""
    pass


class JSONRPCResponseError(JSONRPCError):
    """Raised when the server returns an error response object."""
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(f"JSON-RPC Error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class ServerProcessTerminatedError(JSONRPCError):
    """Raised when the server subprocess terminates unexpectedly or closes pipe."""
    pass


class JSONRPCTimeoutError(JSONRPCError):
    """Raised when a request exceeds the allotted timeout waiting for response."""
    pass


class MalformedToolResultError(JSONRPCError):
    """Raised when an MCP tool result is missing required content fields."""
    pass


class MCPToolExecutionError(JSONRPCError):
    """Raised when an MCP tool result returns isError: true."""
    pass


class SubprocessJSONRPCReader:
    """Persistent background reader thread and queue for a single subprocess stdout stream."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._sentinel = object()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self.proc.stdout is None:
                    break
                line = self.proc.stdout.readline()
                if not line:
                    # EOF reached
                    self.queue.put(self._sentinel)
                    break
                stripped = line.strip()
                # Discard non-JSON log lines so server logs never contaminate the protocol stream
                if stripped.startswith("{"):
                    self.queue.put(stripped)
        except Exception:
            self.queue.put(self._sentinel)

    def read_frame(self, expected_id: Optional[int] = None, timeout: float = 5.0) -> Dict[str, Any]:
        start_time = time.time()
        while True:
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                raise JSONRPCTimeoutError(f"Timed out after {timeout:.1f}s awaiting JSON-RPC response.")

            try:
                item = self.queue.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise ServerProcessTerminatedError(
                        f"Server process terminated with code {self.proc.returncode} while awaiting response."
                    )
                continue

            if item is self._sentinel:
                raise ServerProcessTerminatedError(
                    f"Server process terminated unexpectedly (exit code: {self.proc.returncode})."
                )

            return RobustRawJSONRPCClient.parse_and_validate_frame(item, expected_id=expected_id)

    def close(self) -> None:
        self._stop_event.set()


class RobustRawJSONRPCClient:
    """Defensive Raw JSON-RPC 2.0 Client communicating over pipes with persistent reader."""

    _readers: Dict[Any, SubprocessJSONRPCReader] = {}
    _lock = threading.Lock()

    def __init__(self, proc: Optional[subprocess.Popen] = None):
        self.proc = proc
        if proc is not None:
            self.reader = self.get_or_create_reader(proc)
        else:
            self.reader = None

    @classmethod
    def get_or_create_reader(cls, proc: subprocess.Popen) -> SubprocessJSONRPCReader:
        with cls._lock:
            key = id(proc)
            if key not in cls._readers:
                cls._readers[key] = SubprocessJSONRPCReader(proc)
            return cls._readers[key]

    @classmethod
    def close_reader(cls, proc: subprocess.Popen) -> None:
        with cls._lock:
            key = id(proc)
            reader = cls._readers.pop(key, None)
            if reader:
                reader.close()

    @classmethod
    def close_all_readers(cls) -> None:
        with cls._lock:
            for reader in cls._readers.values():
                reader.close()
            cls._readers.clear()

    def close(self) -> None:
        if self.proc is not None:
            self.close_reader(self.proc)

    @staticmethod
    def parse_and_validate_frame(
        raw_line: str,
        expected_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate an individual line against standard JSON-RPC 2.0 specifications."""
        line = raw_line.strip()
        if not line:
            raise JSONRPCParseError("Received empty wire frame.")

        try:
            payload = json.loads(line)
        except Exception as exc:
            raise JSONRPCParseError(f"Invalid JSON payload: {exc}. Raw line: {line!r}") from exc

        if not isinstance(payload, dict):
            raise JSONRPCParseError(f"Expected JSON object in frame, got {type(payload).__name__}.")

        # Validate JSON-RPC 2.0 version field
        version = payload.get("jsonrpc")
        if version != "2.0":
            raise JSONRPCVersionError(f"Unsupported JSON-RPC version {version!r}; expected '2.0'.")

        # Validate request ID alignment
        if expected_id is not None:
            resp_id = payload.get("id")
            if resp_id != expected_id:
                raise JSONRPCIdMismatchError(
                    f"Request ID mismatch: sent request {expected_id}, but server replied with id {resp_id}."
                )

        # Check for JSON-RPC error responses
        if "error" in payload:
            err = payload["error"]
            if isinstance(err, dict):
                code = err.get("code", -32000)
                msg = err.get("message", "Unknown JSON-RPC error")
                data = err.get("data")
            else:
                code = -32000
                msg = str(err)
                data = None
            raise JSONRPCResponseError(code=code, message=msg, data=data)

        return payload

    @staticmethod
    def extract_and_validate_tool_result(result_payload: Dict[str, Any]) -> str:
        """Inspect and validate an MCP tools/call result object."""
        result = result_payload.get("result")
        if not isinstance(result, dict):
            raise MalformedToolResultError(
                f"Expected 'result' object in tool call response, got {type(result).__name__}."
            )

        # Check MCP isError / is_error flag
        if result.get("isError") is True or result.get("is_error") is True:
            content_items = result.get("content", [])
            err_text = ""
            if isinstance(content_items, list) and content_items:
                err_text = content_items[0].get("text", str(content_items[0]))
            raise MCPToolExecutionError(
                f"MCP tool execution failed (isError: true). Details: {err_text}"
            )

        # Validate content list
        content = result.get("content")
        if not isinstance(content, list):
            raise MalformedToolResultError(
                f"Expected 'content' array in tool result, got {type(content).__name__}."
            )

        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts)

    @classmethod
    def read_next_jsonrpc_frame(
        cls,
        proc: subprocess.Popen,
        expected_id: Optional[int] = None,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Read the next line from process stdout with a genuine cross-platform timeout,
        using a persistent background reader thread and queue per subprocess.

        Guarantees:
        - Exactly one background reader thread per subprocess
        - Retains unread frames across sequential requests in arrival order
        - Bounded timeout via queue.get(timeout=...)
        - Safe EOF and process termination detection
        """
        reader = cls.get_or_create_reader(proc)
        return reader.read_frame(expected_id=expected_id, timeout=timeout)
