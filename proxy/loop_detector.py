"""Deterministic suffix-cycle detection. Compare full digests, never lossy previews."""
from dataclasses import dataclass
import hashlib
import json
from typing import Sequence


def canonical(value) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    model: str
    timestamp: float
    result: str | None = None
    error: bool = False
    arguments_preview: str = ""
    result_preview: str = ""

    @property
    def signature(self):
        return (self.name, self.arguments, self.result, self.model)


@dataclass(frozen=True)
class Loop:
    kind: str
    repetitions: int
    cycle_length: int
    signature: str
    tools: tuple[str, ...]
    results: tuple[str, ...]
    last_call: str


class LoopDetector:
    def detect(self, calls: Sequence[ToolCall], threshold=3, max_cycle=4) -> Loop | None:
        # A pending tool breaks continuity; an absent result is never evidence.
        complete = []
        for call in calls:
            if call.result is None:
                complete.clear()
            else:
                complete.append(call)
        for size in range(1, max_cycle + 1):
            if len(complete) < size * threshold:
                continue
            cycle = complete[-size:]
            signatures = [c.signature for c in cycle]
            repetitions = 1
            while len(complete) >= (repetitions + 1) * size:
                end = len(complete) - repetitions * size
                if [c.signature for c in complete[end-size:end]] != signatures:
                    break
                repetitions += 1
            if repetitions >= threshold:
                kind = "repeated_failure" if any(c.error for c in cycle) else "simple" if size == 1 else "cycle"
                # Rotations of the same cycle share the intervention budget.
                rotations = [signatures[i:] + signatures[:i] for i in range(size)]
                signature = digest(min(rotations))
                return Loop(kind, repetitions, size, signature,
                            tuple(c.name for c in cycle), tuple(c.result_preview for c in cycle),
                            complete[-1].id)
        return None
