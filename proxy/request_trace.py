"""Optional bounded request/response capture, independent of loop detection."""
import json
import time
import uuid

from tool_tracker import preview


class RequestTrace:
    def __init__(self, settings, store, method, path, body):
        self.settings, self.store = settings, store
        config = settings.value.loopDetection
        self.enabled = config.traceEnabled and config.traceAllRequests
        self.generation = getattr(store, "generation", 0)
        self.request_generation = getattr(store, "request_generation", 0)
        self.content = config.traceContent
        self.limit = config.maxTraceContentKB * 1024
        self.started = time.monotonic()
        self.record = {"id": uuid.uuid4().hex, "timestamp": time.time(), "kind": "request",
                       "method": method, "path": path, "status": None, "complete": False}
        self.response = bytearray()
        self.request(body)

    def active(self):
        config = self.settings.value.loopDetection
        return (self.enabled and config.traceEnabled and config.traceAllRequests
                and self.generation == getattr(self.store, "generation", 0)
                and self.request_generation == getattr(self.store, "request_generation", 0))

    def request(self, body):
        self.record["request_bytes"] = len(body)
        if self.active() and self.content:
            self.record["request_content"] = self._redact(body[:self.limit])
            self.record["request_truncated"] = len(body) > self.limit

    def feed(self, chunk):
        if not self.active():
            self.enabled = False
            self.response.clear()
            return
        self.record["response_bytes"] = self.record.get("response_bytes", 0) + len(chunk)
        if self.content:
            self.response.extend(chunk[:max(0, self.limit - len(self.response))])

    def _redact(self, body):
        text = body.decode("utf-8", errors="replace")
        try:
            return json.loads(preview(json.loads(text), self.limit * 8))
        except (ValueError, RecursionError):
            # SSE: redact structured payloads as JSON, including sensitive keys.
            lines = []
            for line in text.splitlines():
                if line.startswith("data:"):
                    line = "data: " + preview(line[5:].lstrip(), self.limit * 8)
                else:
                    line = preview(line, self.limit * 8)
                lines.append(line)
            return "\n".join(lines)

    def finish(self, *, complete=False, error=None):
        if not self.active():
            self.response.clear()
            return
        config = self.settings.value.loopDetection
        self.record.update(complete=complete, duration_ms=round((time.monotonic() - self.started) * 1000))
        if error:
            self.record["error"] = preview(str(error))
        if self.content and config.traceContent:
            self.record["response_content"] = self._redact(bytes(self.response))
            self.record["response_truncated"] = self.record.get("response_bytes", 0) > len(self.response)
        else:
            self.record.pop("request_content", None)
        self.store.submit("request", self.record, config)
        self.response.clear()
        self.enabled = False
