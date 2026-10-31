"""Permet de tester la logique sans daemon Docker ni téléchargement d'images."""
import importlib.util
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
os.environ["MODEL_CONFIG_PATH"] = str(ROOT / "proxy" / "models.json")
os.environ["MODEL_STATE_PATH"] = "/tmp/qwen-proxy-test-selection.json"

# Le paquet docker n'est pas présent dans tous les environnements CI.
if importlib.util.find_spec("docker") is None:
    docker = types.ModuleType("docker")
    errors = types.ModuleType("docker.errors")
    class DockerException(Exception):
        pass
    class NotFound(DockerException):
        pass
    class ImageNotFound(NotFound):
        pass
    errors.DockerException = DockerException
    errors.NotFound = NotFound
    errors.ImageNotFound = ImageNotFound
    typed = types.ModuleType("docker.types")
    class DeviceRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class Ulimit:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    typed.DeviceRequest = DeviceRequest
    typed.Ulimit = Ulimit
    docker.errors = errors
    docker.types = typed
    sys.modules.update({"docker": docker, "docker.errors": errors, "docker.types": typed})

sys.path.insert(0, str(ROOT / "proxy"))
