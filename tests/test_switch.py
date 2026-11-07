"""Tests unitaires de la sélection et de la sécurité du changement."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest

import app as proxy


@pytest.fixture(autouse=True)
def simulated_container_state(monkeypatch):
    async def state():
        return {"engine_profile": "qwen27", "status": "running"}
    monkeypatch.setattr(proxy, "container_state", state)
    monkeypatch.setattr(proxy, "log_engine_failure_sync", lambda model: None)


class FakeContainer:
    def __init__(self, profile, *, status="running"):
        self.attrs = {
            "Config": {"Image": proxy.MODELS[profile]["image"],
                       "Cmd": [proxy.MODELS[profile]["model"]],
                       "Labels": {"qwen.proxy.model_id": profile, "qwen.proxy.profile_hash": proxy.profile_digest(profile)}},
            "State": {"Status": status, "ExitCode": 0},
        }
        self.status = status
    def reload(self):
        pass


def test_profiles_keep_a_stable_public_model():
    assert proxy.PUBLIC_MODEL == "qwen3.8:27b-mtp"
    assert set(proxy.MODELS) == {"qwen27", "flash-next"}
    assert any("qwen3_coder" in arg for arg in proxy.MODELS["qwen27"]["args"])
    assert any("qwen3_xml" in arg for arg in proxy.MODELS["flash-next"]["args"])
    assert "b12x" in proxy.MODELS["flash-next"]["image"]
    assert proxy.MODELS["flash-next"]["env"]["VLLM_PLE_TABLE_MEMORY"] == "disk"
    assert any("262144" in arg for arg in proxy.MODELS["flash-next"]["args"])


def test_legacy_container_is_recognized():
    c = FakeContainer("qwen27")
    c.attrs["Config"]["Labels"] = {"com.docker.compose.service": "vllm"}
    assert proxy.runtime_profile_id(c) == "qwen27"


def test_generated_run_params_are_from_the_target_profile(monkeypatch):
    captures = []
    class Images:
        def get(self, image):
            pass
    class Containers:
        def create(self, image, **kwargs):
            captures.append((image, kwargs))
            return FakeContainer("flash-next")
    monkeypatch.setattr(proxy, "docker_api", lambda: SimpleNamespace(images=Images(), containers=Containers()))
    monkeypatch.setattr(proxy, "HOST_HOME", "/home/testuser")
    proxy.create_engine_sync("flash-next")
    image, kwargs = captures[0]
    assert image == "eugr/spark-vllm-b12x:nightly-20261015"
    assert kwargs["command"][0] == "nvidia/Qwen3.8-Flash-Next-FP4-Local"
    assert "--served-model-name=qwen3.8:27b-mtp" in kwargs["command"]
    assert kwargs["environment"]["VLLM_PLE_TABLE_MEMORY"] == "disk"
    assert kwargs["network"] == "qwen-on-demand_default"
    assert kwargs["labels"]["qwen.proxy.profile_hash"] == proxy.profile_digest("flash-next")
    assert kwargs["volumes"]["/home/testuser/.cache/huggingface"]["bind"] == "/root/.cache/huggingface"


@pytest.mark.asyncio
async def test_switch_waits_for_stream_completion(monkeypatch, tmp_path):
    # Simule un flux en cours, puis son achèvement. Le moteur ne doit
    # jamais être arrêté tant que le compteur de requêtes est non nul.
    proxy.selected_model = "qwen27"
    proxy.active_requests = 1
    proxy.no_active_requests.clear()
    proxy.switch_complete.clear()
    proxy.pending_model = "flash-next"
    proxy.switch_phase = "waiting"
    proxy.switch_error = None
    monkeypatch.setattr(proxy, "STATE_PATH", tmp_path / "selection.json")
    actions = []
    def fake_stop():
        assert proxy.active_requests == 0
        actions.append("stop")
    async def fake_start(client, model_id):
        assert proxy.active_requests == 0
        actions.append("start:" + model_id)
    monkeypatch.setattr(proxy, "stop_remove_sync", fake_stop)
    monkeypatch.setattr(proxy, "start_engine_locked", fake_start)
    task = asyncio.create_task(proxy.switch_worker(None, "flash-next"))
    await asyncio.sleep(0.03)
    assert actions == []
    assert not proxy.switch_complete.is_set()
    await proxy.release_slot()
    await asyncio.wait_for(task, 2)
    assert actions == ["stop", "start:flash-next"]
    assert proxy.selected_model == "flash-next"
    assert (tmp_path / "selection.json").exists()
    assert proxy.switch_complete.is_set()
    assert proxy.pending_model is None
    proxy.selected_model = "qwen27"
    proxy.switch_complete.set()


@pytest.mark.asyncio
async def test_failed_new_model_restores_previous(monkeypatch, tmp_path):
    proxy.selected_model = "qwen27"
    proxy.pending_model = "flash-next"
    proxy.active_requests = 0
    proxy.no_active_requests.set()
    proxy.switch_complete.clear()
    monkeypatch.setattr(proxy, "STATE_PATH", tmp_path / "selection.json")
    started = []
    monkeypatch.setattr(proxy, "stop_remove_sync", lambda: None)
    async def fake_start(client, name):
        started.append(name)
        if name == "flash-next":
            raise RuntimeError("Modele Flash indisponible")
    monkeypatch.setattr(proxy, "start_engine_locked", fake_start)
    await proxy.switch_worker(None, "flash-next")
    assert started == ["flash-next", "qwen27"]
    assert proxy.selected_model == "qwen27"
    assert "Modele Flash indisponible" in proxy.switch_error
    assert proxy.switch_complete.is_set()
    assert proxy.pending_model is None


def test_changed_profile_invalidates_existing_container(monkeypatch):
    c = FakeContainer("flash-next")
    assert proxy.runtime_profile_id(c) == "flash-next"
    c.attrs["Config"]["Labels"]["qwen.proxy.profile_hash"] = "stale-build"
    assert proxy.runtime_profile_id(c) is None


def test_startup_timeout_does_not_recreate_engine(monkeypatch):
    c = FakeContainer('flash-next')
    monkeypatch.setitem(proxy.MODELS['flash-next'], 'startup_timeout_seconds', 28800)
    assert proxy.runtime_profile_id(c) == 'flash-next'


@pytest.mark.asyncio
async def test_switch_reattaches_to_existing_target_after_proxy_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy, 'selected_model', 'qwen27')
    monkeypatch.setattr(proxy, 'STATE_PATH', tmp_path/'selection.json')
    proxy.active_requests = 0
    proxy.no_active_requests.set()
    proxy.switch_complete.clear()
    async def state():
        return {'engine_profile': 'flash-next', 'status': 'running'}
    async def start(client, model):
        assert model == 'flash-next'
    def stop():
        pytest.fail('The existing Flash-Next download must continue')
    monkeypatch.setattr(proxy, 'container_state', state)
    monkeypatch.setattr(proxy, 'start_engine_locked', start)
    monkeypatch.setattr(proxy, 'stop_remove_sync', stop)
    await proxy.switch_worker(None, 'flash-next')
    assert proxy.selected_model == 'flash-next' and proxy.switch_complete.is_set()


@pytest.mark.asyncio
async def test_flash_download_can_exceed_default_timeout_and_resets_idle_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(proxy, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(proxy, 'STARTUP_TIMEOUT', 1200)
    monkeypatch.setattr(proxy, 'last_activity', 0)
    monkeypatch.setattr(proxy, 'find_container', lambda: FakeContainer('flash-next'))
    async def health(client):
        return clock[0] >= 1800
    async def sleep(seconds):
        clock[0] += 600
    monkeypatch.setattr(proxy, 'health_ok', health)
    monkeypatch.setattr(proxy.asyncio, 'sleep', sleep)
    await proxy.start_engine_locked(None, 'flash-next')
    assert proxy.last_activity == 1800 and not proxy.starting


@pytest.mark.asyncio
async def test_new_requests_wait_while_switch_pending(monkeypatch):
    proxy.selected_model = "qwen27"
    proxy.active_requests = 0
    proxy.switch_complete.clear()
    proxy.no_active_requests.set()
    async def fake_state():
        return {"engine_profile": "qwen27", "status": "running"}
    async def fake_health(client):
        return True
    monkeypatch.setattr(proxy, "container_state", fake_state)
    monkeypatch.setattr(proxy, "health_ok", fake_health)
    future = asyncio.create_task(proxy.ensure_ready(None, reserve_slot=True))
    await asyncio.sleep(0.03)
    assert not future.done()
    assert proxy.active_requests == 0
    proxy.switch_complete.set()
    await asyncio.wait_for(future, 1)
    assert proxy.active_requests == 1
    await proxy.release_slot()
    assert proxy.active_requests == 0
