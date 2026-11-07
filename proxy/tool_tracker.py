"""Bounded session histories, tool/result correlation and recovery coordination."""
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
import json
import logging
import re
import time
import uuid

from loop_detector import LoopDetector, ToolCall, canonical, digest
from recovery import recover

log = logging.getLogger(__name__)
SECRET = re.compile(r'(?i)(authorization|api[_-]?key|token|password|secret|credential|cookie)')
TEXT_SECRET = re.compile(r'(?i)(bearer\s+)[^\s"\']+|((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;"\']+')
ERROR = re.compile(r'(?i)\b(error|exception|traceback|failed|failure|TS\d{4})\b')


def preview(value, limit=512):
    """Best effort redaction; content recording remains opt-in."""
    def redact(obj):
        if isinstance(obj, dict):
            return {k: "[REDACTED]" if SECRET.search(k) else redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [redact(v) for v in obj]
        if isinstance(obj, str):
            return TEXT_SECRET.sub(lambda m: (m[1] or m[2] or "") + "[REDACTED]", obj)
        return obj
    try:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        return canonical(redact(value))[:limit]
    except (TypeError, ValueError, RecursionError):
        return "[unavailable]"


@dataclass
class Session:
    calls: OrderedDict = field(default_factory=OrderedDict)
    tail: deque = field(default_factory=lambda: deque(maxlen=80))
    budgets: OrderedDict = field(default_factory=OrderedDict)
    user_marker: str | None = None
    last_event: tuple | None = None
    touched: float = field(default_factory=time.monotonic)
    state: str = "NORMAL"


class ToolTracker:
    def __init__(self, settings, store):
        self.settings = settings
        self.store = store
        self.sessions = OrderedDict()
        self.detector = LoopDetector()
        self.failures = 0

    def identify(self, headers, payload):
        explicit = headers.get("x-proxy-session-id")
        if not explicit:
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                explicit = metadata.get("session_id")
        reliable = isinstance(explicit, str) and bool(explicit.strip()) and len(explicit) <= 512
        if not reliable and self.settings.value.loopDetection.singleClient:
            return digest("single-client"), "single-client", True
        external = explicit if reliable else uuid.uuid4().hex
        # Authorization scopes equal session IDs from clients with different credentials.
        return digest([headers.get("authorization", ""), external]), external, reliable

    def session(self, key):
        config = self.settings.value.loopDetection
        now = time.monotonic()
        for old, value in list(self.sessions.items()):
            if now - value.touched > config.sessionTTLSeconds:
                del self.sessions[old]
        if key not in self.sessions:
            self.sessions[key] = Session()
        state = self.sessions[key]
        state.touched = now
        self.sessions.move_to_end(key)
        while len(self.sessions) > config.maxSessions:
            self.sessions.popitem(last=False)
        return state

    def _record(self, key, call, new=False, content=None, content_field=None):
        config = self.settings.value.loopDetection
        if not config.traceEnabled:
            return
        record = asdict(call)
        record.update(session=key, new=new)
        if not config.traceContent:
            record.pop("arguments_preview")
            record.pop("result_preview")
        elif content_field:
            limit = config.maxTraceContentKB * 1024
            value = preview(content, limit + 1)
            record[content_field] = value[:limit]
            record[content_field + "_truncated"] = len(value) > limit
        self.store.submit("tool", record, config)

    def _call(self, key, state, tool, model, *, add_to_tail=True):
        function = tool.get("function", {})
        if not isinstance(function, dict) or not tool.get("id") or not function.get("name"):
            return
        call_id = digest(tool["id"])
        if call_id in state.calls:
            return
        call = ToolCall(call_id, str(function["name"])[:256], digest(function.get("arguments", "")),
                        model, time.time(), arguments_preview=preview(function.get("arguments", "")))
        state.calls[call_id] = call
        if add_to_tail:
            state.tail.append(call)
        while len(state.calls) > 256:
            state.calls.popitem(last=False)
        self._record(key, call, new=True, content=function.get("arguments", ""), content_field="arguments_content")

    def history(self, key, messages, model):
        state = self.session(key)
        users = [m for m in messages if m.get("role") == "user"]
        marker = digest(users)
        if state.user_marker is not None and marker != state.user_marker:
            state.tail.clear()
        state.user_marker = marker
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        # Skip replayed prefixes even if they exceed the bounded ID cache.
        anchor = -1
        for i, message in enumerate(messages):
            if message.get("role") == "assistant":
                if any(digest(t.get("id")) in state.calls for t in message.get("tool_calls", []) if t.get("id")):
                    anchor = i
        for i, message in enumerate(messages):
            if message.get("role") == "assistant":
                for tool in message.get("tool_calls", []):
                    if i >= anchor:
                        self._call(key, state, tool, model, add_to_tail=i > last_user)
            elif message.get("role") == "tool" and message.get("tool_call_id"):
                call = state.calls.get(digest(message["tool_call_id"]))
                if call is None:
                    continue
                value = message.get("content")
                if value is None:
                    continue
                result = digest(value)
                if call.result != result:
                    call.result = result
                    call.result_preview = preview(value)
                    call.error = message.get("is_error") is True or bool(ERROR.search(canonical(value)))
                    self._record(key, call, content=value, content_field="result_content")
        return state

    def prepare(self, key, payload, model, supports_thinking, reliable=True):
        config = self.settings.value
        if not config.loopDetection.enabled:
            if config.loopDetection.storeToolHistory and config.loopDetection.traceEnabled:
                if reliable:
                    self.history(key, payload["messages"], model)
                else:
                    self.session(key)
            return payload
        if not reliable:
            # Count newly generated output tools only. Replayed anonymous input
            # cannot be attributed safely and would inflate totals.
            self.session(key)
            return payload
        state = self.history(key, payload["messages"], model)
        loop = self.detector.detect(list(state.tail), config.loopDetection.repetitionThreshold,
                                    config.loopDetection.maxCycleLength)
        if not loop or state.last_event == (loop.signature, loop.last_call):
            return payload
        state.last_event = (loop.signature, loop.last_call)
        attempts = state.budgets.get(loop.signature, 0)
        event = {**asdict(loop), "id": uuid.uuid4().hex, "session": key, "model": model,
                 "timestamp": time.time(), "intervention": "observe", "attempt": attempts,
                 "state": "NORMAL", "limitation": None}
        result = payload
        if reliable and config.recovery.mode != "observe":
            if attempts >= config.recovery.maxAttempts:
                event.update(intervention="limit_reached", limitation="Limite atteinte : observation, sans bloquer les outils")
            else:
                state.state = "RECOVERY"
                try:
                    result, action, limitation = recover(payload, loop, config.recovery, supports_thinking)
                    attempts += 1
                    state.budgets[loop.signature] = attempts
                    state.budgets.move_to_end(loop.signature)
                    while len(state.budgets) > 64:
                        state.budgets.popitem(last=False)
                    event.update(intervention=action, limitation=limitation, attempt=attempts)
                finally:
                    state.state = "NORMAL"
        if not config.loopDetection.traceContent:
            event["results"] = ["sha256:" + c.result for c in list(state.tail)[-loop.cycle_length:]]
        if config.loopDetection.traceEnabled:
            self.store.submit("loop", event, config.loopDetection)
        return result

    def assistant(self, key, calls, model):
        config = self.settings.value.loopDetection
        if not (config.enabled or (config.traceEnabled and config.storeToolHistory)) or key not in self.sessions:
            return
        state = self.sessions[key]
        for call in calls:
            self._call(key, state, call, model)

    def safe_prepare(self, *args, **kwargs):
        try:
            return self.prepare(*args, **kwargs)
        except Exception:
            self.failures += 1
            log.warning("Observation des outils indisponible ; requête conservée")
            return args[1]


class ResponseObserver:
    """Tee bounded SSE events, assembling only tool deltas for choice zero."""
    MAX_EVENT = 65536
    MAX_TOOLS = 128
    MAX_ARGUMENTS = 262144

    def __init__(self, tracker, key, model, sse):
        self.tracker, self.key, self.model, self.sse = tracker, key, model, sse
        self.buffer = b""
        self.tools = {}
        self.event = []
        self.event_size = 0
        self.size = 0
        self.disabled = False

    def feed(self, chunk):
        if self.disabled:
            return
        try:
            if not self.sse:
                if len(self.buffer) + len(chunk) > 1048576:
                    self._disable()
                else:
                    self.buffer += chunk
                return
            self.buffer += chunk
            # Process line by line, supporting LF/CRLF split across chunks.
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if len(line) > self.MAX_EVENT:
                    raise ValueError("large event")
                line = line.rstrip(b"\r")
                if not line:
                    self._event()
                elif line.startswith(b"data:"):
                    value = line[5:].lstrip(b" ")
                    self.event_size += len(value)
                    if self.event_size > self.MAX_EVENT:
                        raise ValueError("large event")
                    self.event.append(value)
            if len(self.buffer) > self.MAX_EVENT:
                raise ValueError("large event")
        except Exception:
            self._disable()

    def _disable(self):
        self.disabled = True
        self.buffer = b""
        self.tools.clear()
        self.event.clear()
        self.tracker.failures += 1

    def _event(self):
        data = b"\n".join(self.event)
        self.event.clear()
        self.event_size = 0
        if data and data != b"[DONE]":
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                return
            # Resource-limit failures must reach feed() and disable observation.
            self._payload(payload)

    def _payload(self, data):
        for choice in data.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            if "message" in choice:
                self.tracker.assistant(self.key, choice["message"].get("tool_calls", []), self.model)
                continue
            for delta in choice.get("delta", {}).get("tool_calls", []):
                index = delta.get("index", 0)
                if index not in self.tools:
                    if len(self.tools) >= self.MAX_TOOLS:
                        raise ValueError("too many tools")
                    self.tools[index] = {"id": "", "function": {"name": "", "arguments": ""}}
                tool = self.tools[index]
                for k in ("id",):
                    part = delta.get(k, "")
                    tool[k] += part
                    self.size += len(part)
                for k in ("name", "arguments"):
                    part = delta.get("function", {}).get(k, "")
                    tool["function"][k] += part
                    self.size += len(part)
                if self.size > self.MAX_ARGUMENTS:
                    raise ValueError("large tools")
            if choice.get("finish_reason") in {"tool_calls", "stop"}:
                self.tracker.assistant(self.key, [self.tools[i] for i in sorted(self.tools)], self.model)
                self.tools.clear()

    def finish(self):
        if self.disabled or self.sse:
            return
        try:
            self._payload(json.loads(self.buffer))
        except Exception:
            pass
        self.buffer = b""
