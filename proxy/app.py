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
        "model": MODELS[model_id],
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
    global starting, previous_sample
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
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if await health_ok(client):
                return
            await asyncio.sleep(2)
            state = await container_state()
            if state["status"] not in ("running", "restarting", "created"):
                raise HTTPException(503, f"vLLM a quitté : {state['status']} (code {state['exit_code']})")
        raise HTTPException(504, f"vLLM non prêt après {STARTUP_TIMEOUT} s ({model_id})")
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
            await asyncio.to_thread(stop_remove_sync)
            try:
                await start_engine_locked(client, target)
            except Exception as exc:
                log.exception("Chargement de %s impossible, retour à %s", target, previous)
                switch_error = str(exc)
                switch_phase = "rollback"
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
            {"id": key, "label": p["label"], "model": p["model"],
             "image": p["image"], "max_context": p.get("max_context")}
            for key, p in MODELS.items()
        ],
    }


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

    await ensure_ready(client, reserve_slot=True)
    url = f"{UPSTREAM}/v1/{subpath}"
    if request.url.query:
        url += "?" + request.url.query
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in HOP_HEADERS and k.lower() != "accept-encoding"}
    headers["accept-encoding"] = "identity"
    response = None
    try:
        upstream_req = client.build_request(request.method, url, content=body, headers=headers)
        response = await client.send(upstream_req, stream=True)
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        await release_slot()
        if isinstance(exc, asyncio.CancelledError):
            raise
        return JSONResponse({"error": f"vLLM inaccessible : {exc}"}, status_code=502)

    response_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_HEADERS}

    async def stream_bytes():
        async for chunk in response.aiter_raw():
            yield chunk

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
                    await release_slot()

    return TrackedStreamingResponse(stream_bytes(), status_code=response.status_code,
                                    headers=response_headers)
