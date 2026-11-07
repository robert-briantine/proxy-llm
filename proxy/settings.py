"""Validated, atomic settings, kept separate from immutable model profiles."""
import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LoopDetection(StrictSettings):
    enabled: bool = True
    singleClient: bool = True
    traceEnabled: bool = True
    traceAllRequests: bool = False
    maxDiskMB: int = Field(default=256, ge=1, le=102400)
    maxTraceContentKB: int = Field(default=1024, ge=1, le=16384)
    repetitionThreshold: int = Field(default=3, ge=2, le=20)
    maxCycleLength: int = Field(default=4, ge=1, le=4)
    storeToolHistory: bool = True
    traceContent: bool = False
    retentionDays: int = Field(default=7, ge=1, le=365)
    maxTraceRows: int = Field(default=10000, ge=100, le=100000)
    maxSessions: int = Field(default=256, ge=1, le=4096)
    sessionTTLSeconds: int = Field(default=3600, ge=60, le=86400)


class Recovery(StrictSettings):
    mode: str = Field(default="observe", pattern=r"^(observe|prompt|thinking|prompt_and_thinking)$")
    prompt: str = Field(default="Une boucle d'utilisation des outils a été détectée. Analyse les dernières tentatives et leurs résultats. Ne reproduis pas les actions ayant échoué sans modification pertinente. Essaie une stratégie différente pour atteindre l'objectif demandé.", max_length=16000)
    temporaryThinking: bool = True
    maxAttempts: int = Field(default=2, ge=1, le=10)


class Settings(StrictSettings):
    loopDetection: LoopDetection = Field(default_factory=LoopDetection)
    recovery: Recovery = Field(default_factory=Recovery)


class SettingsStore:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.value = Settings.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.value = Settings()
        except (OSError, ValueError):
            logging.getLogger(__name__).warning("Paramètres de détection illisibles ; valeurs par défaut")
            self.value = Settings()

    def save(self, value: Settings):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(value.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self.value = value
