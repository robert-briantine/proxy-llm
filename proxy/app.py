"""Proxy OpenAI pour DGX Spark : un seul conteneur GPU / vLLM à la fois.

Le conteneur proxy reste allumé. Le conteneur d'inférence est recréé avec
l'image + les arguments du modèle sélectionné, APRES drainage des requêtes.
Sécurité : le socket Docker donne des privilèges élevés sur l'hôte.
Ne jamais exposer ce proxy sur Internet sans authentification.
"""
import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.types import DeviceRequest, Ulimit
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel

from settings import Settings, SettingsStore
from trace_store import TraceStore
from tool_tracker import ResponseObserver, ToolTracker
from request_trace import RequestTrace

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qwen_proxy")

CONTAINER_NAME = os.getenv("VLLM_CONTAINER_NAME", "qwen-on-demand-vllm")
UPSTREAM = os.getenv("VLLM_BASE_URL", f"http://{CONTAINER_NAME}:8000").rstrip("/")
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "qwen-on-demand_default")
HOST_HOME = os.getenv("HOST_HOME", "/home/ubuntu")
HF_TOKEN = os.getenv("HF_TOKEN", "")
STARTUP_TIMEOUT = max(10, int(os.getenv("STARTUP_TIMEOUT_SECONDS", "1200")))
IDLE_TIMEOUT = max(0, int(os.getenv("IDLE_TIMEOUT_SECONDS", "1200")))
CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", "/app/models.json"))
STATE_PATH = Path(os.getenv("MODEL_STATE_PATH", "/app/data/selection.json"))

loop_settings = SettingsStore(Path(os.getenv("LOOP_SETTINGS_PATH", str(STATE_PATH.parent / "loop-settings.json"))))
trace_store = TraceStore(Path(os.getenv("TOOL_HISTORY_PATH", str(STATE_PATH.parent / "tool-history.sqlite3"))))
tool_tracker = ToolTracker(loop_settings, trace_store)

# Configurations figées au démarrage du proxy ; modifier models.json + relancer proxy.
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
MODELS: dict = CONFIG["models"]
DEFAULT_MODEL = CONFIG["default_model"]
PUBLIC_MODEL = CONFIG["public_model_name"]
assert DEFAULT_MODEL in MODELS and PUBLIC_MODEL and MODELS
for _id, _profile in MODELS.items():
    assert _profile["image"] and _profile["model"] and isinstance(_profile["args"], list)


def load_selection() -> str:
    try:
        selected = json.loads(STATE_PATH.read_text(encoding="utf-8"))["model"]
        if selected in MODELS:
            return selected
    except (FileNotFoundError, ValueError, KeyError, OSError):
        pass
    return DEFAULT_MODEL


def save_selection(model: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"model": model}), encoding="utf-8")
    tmp.replace(STATE_PATH)


selected_model = load_selection()
pending_model: str | None = None
switch_phase = "idle"  # idle, waiting, replacing, rollback
switch_error: str | None = None
last_switch: dict | None = None
switch_task: asyncio.Task | None = None

manage_lock = asyncio.Lock()
switch_complete = asyncio.Event()
switch_complete.set()
no_active_requests = asyncio.Event()
no_active_requests.set()
active_requests = 0
last_activity = time.monotonic()
starting = False
stopping = False
previous_sample = None
docker_client = None

METRIC_RE = re.compile(r'^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([-+\d.eE]+)(?:\s|$)')


def parse_metrics(raw: str) -> dict[str, float]:
    values = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if match:
            try:
                number = float(match.group(2))
            except ValueError:
                continue
            values[match.group(1)] = values.get(match.group(1), 0.0) + number
    return values


def docker_api():
    global docker_client
    if docker_client is None:
        docker_client = docker.from_env()
    return docker_client


def find_container():
    try:
        return docker_api().containers.get(CONTAINER_NAME)
    except NotFound:
        return None


def profile_digest(model_id: str) -> str:
    config = {
        "model": {key: value for key, value in MODELS[model_id].items()
                  if key not in {"supports_thinking", "startup_timeout_seconds"}},
        "public_model_name": PUBLIC_MODEL,
        "host_home": HOST_HOME,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def runtime_profile_id(container) -> str | None:
    if container is None:
        return None
    container.reload()
    labels = (container.attrs.get("Config", {}).get("Labels") or {})
    managed = labels.get("qwen.proxy.model_id")
    if managed in MODELS:
        # Une modification du profil entraîne une recréation au prochain démarrage.
        return managed if labels.get("qwen.proxy.profile_hash") == profile_digest(managed) else None
    # Migration de la V3 (créée par docker compose).
    image = container.attrs.get("Config", {}).get("Image", "")
    cmd = " ".join(str(x) for x in (container.attrs.get("Config", {}).get("Cmd") or []))
    original = MODELS[DEFAULT_MODEL]
    if image == original["image"] and original["model"] in cmd:
        return DEFAULT_MODEL
    return None


def create_engine_sync(model_id: str):
    """Crée le conteneur d'inférence, mais ne le démarre pas."""
    profile = MODELS[model_id]
    client = docker_api()
    try:
        client.images.get(profile["image"])
    except ImageNotFound:
        log.info("Téléchargement de l'image vLLM : %s", profile["image"])
        client.images.pull(profile["image"])
    # Volumes sur le système HOTE (l'accès Docker n'est pas celui du proxy).
    mounts = {
        f"{HOST_HOME}/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"},
        "qwen-on-demand_vllm-cache": {"bind": "/root/.cache/vllm", "mode": "rw"},
        f"{HOST_HOME}/.cache/flashinfer": {"bind": "/root/.cache/flashinfer", "mode": "rw"},
        "qwen-on-demand_triton-cache": {"bind": "/root/.triton", "mode": "rw"},
        f"{HOST_HOME}/.cache/b12x": {"bind": "/root/.cache/b12x", "mode": "rw"},
    }
    env = {"HF_HOME": "/root/.cache/huggingface", "HF_TOKEN": HF_TOKEN}
    env.update(profile.get("env", {}))
    args = [profile["model"], f"--served-model-name={PUBLIC_MODEL}", "--host=0.0.0.0", "--port=8000"]
    args += profile["args"]
    return client.containers.create(
        profile["image"], command=args, entrypoint=["vllm", "serve"],
        name=CONTAINER_NAME, platform="linux/arm64", detach=True,
        network=DOCKER_NETWORK, environment=env, volumes=mounts,
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        ipc_mode="host", shm_size="16g", security_opt=profile.get("security_opt", []),
        ulimits=[Ulimit(name="memlock", soft=-1, hard=-1),
                 Ulimit(name="stack", soft=67108864, hard=67108864)],
        labels={"qwen.proxy.model_id": model_id, "qwen.proxy.managed": "true",
                "qwen.proxy.profile_hash": profile_digest(model_id)},
        restart_policy={"Name": "no"},
    )


def stop_remove_sync():
    c = find_container()
    if c is None:
        return
    c.reload()
    if c.status == "running":
        log.info("Arrêt de l'ancien moteur vLLM (%s)", CONTAINER_NAME)
        c.stop(timeout=120)
    c.remove()


def log_engine_failure_sync(model_id):
    """Keep bounded startup diagnostics before rollback removes the engine."""
    c = find_container()
    if c is not None:
        output = c.logs(tail=80)[-32768:].decode("utf-8", errors="replace")
        log.error("Derniers journaux du moteur %s avant restauration :\n%s", model_id, output)


def startup_timeout(model_id):
    return max(10, int(MODELS[model_id].get("startup_timeout_seconds", STARTUP_TIMEOUT)))


async def container_state() -> dict:
    def inspect():
        container = find_container()
        if container is None:
            return {"status": "absent", "health": None, "exit_code": None, "engine_profile": None}
        container.reload()
        state = container.attrs.get("State", {})
        return {
            "status": state.get("Status", "unknown"),
            "health": state.get("Health", {}).get("Status"),
            "exit_code": state.get("ExitCode"),
            "engine_profile": runtime_profile_id(container),
        }
    try:
        return await asyncio.to_thread(inspect)
    except DockerException as exc:
        raise HTTPException(503, f"Docker inaccessible : {exc}") from exc


async def health_ok(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get(f"{UPSTREAM}/health", timeout=2)
        return r.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


async def start_engine_locked(client: httpx.AsyncClient, model_id: str):
    """N'utiliser que sous manage_lock, avec les requêtes admises bloquées."""
    global starting, previous_sample, last_activity
    starting = True
    try:
        def create_or_start():
            c = find_container()
            if c is not None and runtime_profile_id(c) != model_id:
                log.warning("Moteur existant différent du profil %s : recréation", model_id)
                stop_remove_sync()
                c = None
            if c is None:
                c = create_engine_sync(model_id)
            c.reload()
            if c.status != "running":
                log.info("Démarrage de vLLM pour %s", model_id)
                c.start()
        await asyncio.to_thread(create_or_start)
        previous_sample = None
        timeout = startup_timeout(model_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await health_ok(client):
                last_activity = time.monotonic()
                return
            await asyncio.sleep(2)
            state = await container_state()
            if state["status"] not in ("running", "restarting", "created"):
                raise HTTPException(503, f"vLLM a quitté : {state['status']} (code {state['exit_code']})")
        raise HTTPException(504, f"vLLM non prêt après {timeout} s ({model_id}) ; premier téléchargement des poids ou compilation encore en cours : consulter les journaux du moteur")
    except DockerException as exc:
        raise HTTPException(503, f"Erreur Docker ({model_id}) : {exc}") from exc
    finally:
        starting = False


async def ensure_ready(client: httpx.AsyncClient, *, reserve_slot=False):
    global active_requests, last_activity
    while True:
        # Ne pas admettre de nouvelles requêtes après la demande de changement.
        await switch_complete.wait()
        async with manage_lock:
            if not switch_complete.is_set():
                continue
            state = await container_state()
            if state["engine_profile"] != selected_model or not await health_ok(client):
                if active_requests:
                    raise HTTPException(503, "Le moteur a quitté pendant une requête")
                await start_engine_locked(client, selected_model)
            if reserve_slot:
                active_requests += 1
                no_active_requests.clear()
            last_activity = time.monotonic()
            return


async def release_slot():
    global active_requests, last_activity
    async with manage_lock:
        active_requests = max(0, active_requests - 1)
        if active_requests == 0:
            no_active_requests.set()
        last_activity = time.monotonic()


async def stop_container(*, only_if_idle=False) -> bool:
    global stopping, previous_sample
    async with manage_lock:
        if not switch_complete.is_set():
            if only_if_idle:
                return False
            raise HTTPException(409, "Changement de modèle en cours")
        if active_requests:
            if only_if_idle:
                return False
            raise HTTPException(409, "Des requêtes sont encore en cours")
        state = await container_state()
        if state["status"] != "running":
            return False
        stopping = True
        try:
            await asyncio.to_thread(lambda: find_container().stop(timeout=120))
            previous_sample = None
            return True
        finally:
            stopping = False


async def switch_worker(client: httpx.AsyncClient, target: str):
    global selected_model, pending_model, switch_phase, switch_error, last_switch, previous_sample
    previous = selected_model
    began = time.monotonic()
    try:
        # L'événement est remis à zéro dès que le switch est demandé : les
        # NOUVELLES requêtes attendent ; celles en cours se terminent.
        await no_active_requests.wait()
        async with manage_lock:
            switch_phase = "replacing"
            log.info("Changement vLLM %s -> %s", previous, target)
            # A proxy restart can leave the requested engine still downloading.
            # Reattach to that matching profile instead of interrupting it again.
            state = await container_state()
            if state["engine_profile"] != target:
                await asyncio.to_thread(stop_remove_sync)
            try:
                await start_engine_locked(client, target)
            except Exception as exc:
                log.exception("Chargement de %s impossible, retour à %s", target, previous)
                switch_error = str(exc)
                switch_phase = "rollback"
                try:
                    await asyncio.to_thread(log_engine_failure_sync, target)
                except Exception:
                    log.warning("Journaux du moteur %s indisponibles", target)
                try:
                    await asyncio.to_thread(stop_remove_sync)
                    await start_engine_locked(client, previous)
                except Exception as rollback_error:
                    log.exception("Rollback de vLLM impossible")
                    switch_error += f" ; restauration impossible : {rollback_error}"
                return
            selected_model = target
            save_selection(target)
            last_switch = {
                "from": previous, "to": target, "seconds": round(time.monotonic() - began, 2)
            }
            previous_sample = None
            log.info("Modèle actif : %s", target)
    except Exception as exc:
        log.exception("Erreur de changement de modèle")
        switch_error = str(exc)
    finally:
        pending_model = None
        switch_phase = "idle"
        switch_complete.set()


async def auto_stop_loop():
    while True:
        await asyncio.sleep(15)
        if IDLE_TIMEOUT <= 0 or active_requests or starting or stopping or not switch_complete.is_set():
            continue
        if time.monotonic() - last_activity < IDLE_TIMEOUT:
            continue
        try:
            if await stop_container(only_if_idle=True):
                log.info("vLLM arrêté automatiquement après %s s", IDLE_TIMEOUT)
        except Exception:
            log.exception("Erreur arrêt automatique")


@asynccontextmanager
async def lifespan(app: FastAPI):
    trace_store.configure(loop_settings.value.loopDetection)
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=None, write=120, pool=15),
        limits=httpx.Limits(max_connections=32), follow_redirects=False,
    )
    housekeeping = asyncio.create_task(auto_stop_loop())
    try:
        yield
    finally:
        housekeeping.cancel()
        with suppress(asyncio.CancelledError):
            await housekeeping
        if switch_task and not switch_task.done():
            switch_task.cancel()
            with suppress(asyncio.CancelledError):
                await switch_task
        await trace_store.close()
        await app.state.http.aclose()
        if docker_client is not None:
            await asyncio.to_thread(docker_client.close)


app = FastAPI(title="Smart vLLM Proxy - DGX Spark", lifespan=lifespan)


class SwitchRequest(BaseModel):
    model: str


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8"))


@app.get("/api/models")
async def list_models():
    return {
        "active_model": selected_model,
        "pending_model": pending_model,
        "public_model_name": PUBLIC_MODEL,
        "models": [
            {"id": key, "supports_thinking": p.get("supports_thinking", False), "label": p["label"], "model": p["model"],
             "startup_timeout_seconds": startup_timeout(key),
             "image": p["image"], "max_context": p.get("max_context")}
            for key, p in MODELS.items()
        ],
    }


@app.get("/api/loop-settings")
async def get_loop_settings():
    return loop_settings.value.model_dump()


settings_lock = asyncio.Lock()


@app.put("/api/loop-settings")
async def put_loop_settings(payload: Settings):
    async with settings_lock:
        was_enabled = loop_settings.value.loopDetection.enabled
        try:
            await asyncio.to_thread(loop_settings.save, payload)
        except OSError as exc:
            raise HTTPException(503, "Impossible de sauvegarder les paramètres") from exc
        # Keep call IDs and recovery budgets when saving settings. A disabled
        # interval breaks continuity because tools may have been missed.
        if was_enabled != payload.loopDetection.enabled:
            for state in tool_tracker.sessions.values():
                state.tail.clear()
        trace_store.configure(payload.loopDetection)
    return loop_settings.value.model_dump()


@app.get("/api/traces")
async def get_traces(limit: int = 30, kind: str = "all"):
    if kind not in {"all", "tools", "loops", "requests"}:
        raise HTTPException(422, "Type de trace inconnu")
    result = await asyncio.to_thread(trace_store.read, max(1, min(limit, 100)), kind=kind)
    return {**result, "dropped": trace_store.dropped, "storage_errors": trace_store.errors,
            "disk_bytes": await asyncio.to_thread(trace_store.disk_bytes),
            "max_disk_bytes": loop_settings.value.loopDetection.maxDiskMB * 1024 * 1024,
            "trace_enabled": loop_settings.value.loopDetection.traceEnabled,
            "observation_errors": tool_tracker.failures, "sessions": len(tool_tracker.sessions)}


@app.get("/api/traces/{event_id}")
async def get_trace(event_id: str):
    result = await asyncio.to_thread(trace_store.read, event_id=event_id)
    if not result or "id" not in result:
        raise HTTPException(404, "Trace inconnue ou expirée")
    return result


@app.get("/api/status")
async def status(request: Request):
    global previous_sample
    state = await container_state()
    running = state["status"] == "running"
    ready = await health_ok(request.app.state.http) if running else False
    performance = None
    if ready:
        try:
            response = await request.app.state.http.get(f"{UPSTREAM}/metrics", timeout=3)
            response.raise_for_status()
            metrics = parse_metrics(response.text)
            count = metrics.get("vllm:generation_tokens_total")
            now = time.monotonic()
            tps = None
            if count is not None and previous_sample is not None:
                elapsed = now - previous_sample[0]
                delta = count - previous_sample[1]
                if elapsed > 0 and delta >= 0:
                    tps = round(delta / elapsed, 1)
            previous_sample = (now, count) if count is not None else None
            kv = metrics.get("vllm:kv_cache_usage_perc")
            performance = {
                "tokens_per_second": tps,
                "generated_tokens_total": count,
                "kv_cache_usage_percent": round(kv * 100, 1) if kv is not None else None,
                "vllm_requests_running": metrics.get("vllm:num_requests_running"),
                "vllm_requests_waiting": metrics.get("vllm:num_requests_waiting"),
                "prompt_tokens_total": metrics.get("vllm:prompt_tokens_total"),
            }
        except (httpx.HTTPError, ValueError):
            performance = None
    else:
        previous_sample = None
    memory = None
    if running:
        try:
            def get_memory():
                c = find_container()
                return c.stats(stream=False, one_shot=True).get("memory_stats", {}).get("usage") if c else None
            memory = await asyncio.to_thread(get_memory)
        except (DockerException, KeyError, TypeError):
            pass
    return {
        **state, "ready": ready,
        "starting": starting, "stopping": stopping,
        "model": PUBLIC_MODEL, "active_model": selected_model,
        "active_label": MODELS[selected_model]["label"],
        "pending_model": pending_model, "switch_phase": switch_phase,
        "startup_timeout_seconds": startup_timeout(pending_model or selected_model),
        "switch_error": switch_error, "last_switch": last_switch,
        "active_proxy_requests": active_requests,
        "idle_timeout_seconds": IDLE_TIMEOUT,
        "seconds_since_last_request": round(time.monotonic() - last_activity),
        "docker_memory_bytes": memory, "metrics": performance,
    }


@app.post("/api/start")
async def start(request: Request):
    await ensure_ready(request.app.state.http)
    return {"ok": True, "ready": True, "active_model": selected_model}


@app.post("/api/stop")
async def stop():
    return {"ok": True, "stopped": await stop_container()}


@app.post("/api/switch", status_code=202)
async def switch(request: Request, payload: SwitchRequest):
    global pending_model, switch_phase, switch_error, switch_task
    if payload.model not in MODELS:
        raise HTTPException(400, f"Modèle inconnu : {payload.model}")
    async with manage_lock:
        if not switch_complete.is_set():
            raise HTTPException(409, f"Changement déjà en cours vers {pending_model}")
        if payload.model == selected_model:
            return {"ok": True, "changed": False, "active_model": selected_model}
        pending_model = payload.model
        switch_phase = "waiting" if active_requests else "replacing"
        switch_error = None
        switch_complete.clear()
        switch_task = asyncio.create_task(switch_worker(request.app.state.http, payload.model))
        return {"ok": True, "queued": bool(active_requests), "target": payload.model}


HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


@app.api_route("/v1/{subpath:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def openai_proxy(subpath: str, request: Request):
    client: httpx.AsyncClient = request.app.state.http
    body = await request.body()
    data = None
    # Tous les profils exposent le même nom public pour éviter de modifier Qwen Code.
    if body and request.method in {"POST", "PUT", "PATCH"} and subpath in {
        "chat/completions", "completions", "responses", "embeddings"
    }:
        try:
            data = json.loads(body)
            if isinstance(data, dict) and "model" in data:
                data["model"] = PUBLIC_MODEL
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        except (ValueError, UnicodeDecodeError):
            pass

    trace = RequestTrace(loop_settings, trace_store, request.method, "/v1/" + subpath, body)
    try:
        await ensure_ready(client, reserve_slot=True)
    except BaseException as exc:
        trace.record["status"] = getattr(exc, "status_code", None)
        trace.finish(error=exc)
        raise
    trace.record["model"] = selected_model
    url = f"{UPSTREAM}/v1/{subpath}"
    if request.url.query:
        url += "?" + request.url.query
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in HOP_HEADERS and k.lower() != "accept-encoding"}
    headers["accept-encoding"] = "identity"
    response = None
    observer = None
    session_header = None
    try:
        if (subpath == "chat/completions" and request.method == "POST"
                and isinstance(data, dict) and isinstance(data.get("messages"), list)
                and all(isinstance(m, dict) for m in data["messages"])
                and (loop_settings.value.loopDetection.enabled or
                     (loop_settings.value.loopDetection.traceEnabled and loop_settings.value.loopDetection.storeToolHistory))):
            # Capture after admission: a pending switch may have changed selected_model.
            model_id = selected_model
            key, external, reliable = tool_tracker.identify(request.headers, data)
            if not reliable:
                session_header = external
            data = tool_tracker.safe_prepare(key, data, model_id,
                                            MODELS[model_id].get("supports_thinking", False), reliable)
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            observer = ResponseObserver(tool_tracker, key, model_id, bool(data.get("stream")))
        trace.request(body)
        upstream_req = client.build_request(request.method, url, content=body, headers=headers)
        response = await client.send(upstream_req, stream=True)
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        trace.record["status"] = 502 if isinstance(exc, httpx.HTTPError) else None
        trace.finish(error=exc)
        await release_slot()
        if isinstance(exc, asyncio.CancelledError):
            raise
        return JSONResponse({"error": f"vLLM inaccessible : {exc}"}, status_code=502)
    except BaseException as exc:
        trace.finish(error=exc)
        await release_slot()
        raise

    response_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_HEADERS}
    trace.record["status"] = response.status_code

    if session_header:
        response_headers["x-proxy-session-id"] = session_header
    if response.status_code >= 300:
        observer = None

    async def stream_bytes():
        async for chunk in response.aiter_raw():
            trace.feed(chunk)
            if observer is not None:
                observer.feed(chunk)
            yield chunk
        if observer is not None:
            observer.finish()
        trace.record["complete"] = True

    class TrackedStreamingResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            # Le slot doit être relâché même si le client coupe avant le
            # premier octet (un "finally" dans le générateur ne suffit pas).
            try:
                await super().__call__(scope, receive, send)
            finally:
                try:
                    await response.aclose()
                finally:
                    try:
                        trace.finish(complete=trace.record["complete"])
                    finally:
                        await release_slot()

    return TrackedStreamingResponse(stream_bytes(), status_code=response.status_code,
                                    headers=response_headers)
