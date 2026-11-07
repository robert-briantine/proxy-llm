"""Single-client correlation, opt-in HTTP capture, manual stop and disk quota."""
import asyncio
import json
import sqlite3
import time

import httpx
import pytest

import app as proxy
from request_trace import RequestTrace
from settings import SettingsStore
from tool_tracker import ToolTracker
from trace_store import TraceStore
from test_loops import history, tracker, loop_records


def test_single_client_works_without_headers_and_deduplicates(tmp_path):
    t = tracker(tmp_path, 'prompt')
    key, _, reliable = t.identify({}, history())
    assert reliable
    assert key == t.identify({'authorization': 'changed'}, history())[0]
    assert t.prepare(key, history(), 'qwen27', True) != history()
    assert t.prepare(key, history(), 'qwen27', True) == history()
    assert len(loop_records(t)) == 1
    assert sum(k == 'tool' and r['new'] for k, r in t.store.records) == 3
    assert key != t.identify({'x-proxy-session-id': 'separate'}, history())[0]


def test_manual_stop_preserves_detection(tmp_path):
    t = tracker(tmp_path, 'prompt', traceEnabled=False)
    assert t.prepare('s', history(), 'qwen27', True) != history()
    assert not t.store.records


@pytest.mark.asyncio
@pytest.mark.parametrize('stream,status', [(False, 200), (True, 200), (False, 500)])
async def test_full_http_trace_and_tools(monkeypatch, tmp_path, stream, status):
    settings = SettingsStore(tmp_path/'settings.json')
    config = settings.value.loopDetection
    config.traceAllRequests = config.traceContent = True
    config.enabled = False  # Tracing also works without the loop detector.
    store = TraceStore(tmp_path/'traces.sqlite3')
    for name, value in {'loop_settings': settings, 'trace_store': store,
                        'tool_tracker': ToolTracker(settings, store), 'active_requests': 0}.items():
        monkeypatch.setattr(proxy, name, value)
    async def ready(client, *, reserve_slot=False):
        if reserve_slot:
            proxy.active_requests += 1
            proxy.no_active_requests.clear()
    monkeypatch.setattr(proxy, 'ensure_ready', ready)
    payload = {'choices': [{'message': {'content': 'bonjour', 'api_key': 'response-secret'}}]}
    raw = json.dumps(payload).encode()
    if stream:
        raw = b'data: ' + raw + b'\n\ndata: [DONE]\n\n'
    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(0, len(raw), 7):
                yield raw[i:i+7]
    def upstream(request):
        return httpx.Response(status, stream=Chunks(),
                              headers={'content-type': 'text/event-stream' if stream else 'application/json'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(proxy.app.state, 'http', upstream_client, raising=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={**history(), 'stream': stream,
                                         'api_key': 'request-secret'}, headers={'authorization': 'Bearer header-secret'})
            assert response.status_code == status and response.content == raw
            await asyncio.to_thread(store.queue.join)
            result = (await client.get('/api/traces?kind=requests')).json()
            assert result['requests'] == 1
            detail = (await client.get('/api/traces/' + result['recent'][0]['id'])).json()
            assert detail['status'] == status and detail['complete']
            encoded = json.dumps(detail)
            assert 'request-secret' not in encoded and 'response-secret' not in encoded and 'header-secret' not in encoded
            assert 'bonjour' in encoded and 'npm run build' in encoded
            tools = (await client.get('/api/traces?kind=tools')).json()['recent']
            assert len(tools) == 3
            tool = (await client.get('/api/traces/' + tools[0]['id'])).json()
            assert 'npm run build' in tool['arguments_content']
            assert 'unknown name' in tool['result_content']
            assert (await client.get('/api/traces?kind=invalid')).status_code == 422
    await store.close()
    assert proxy.active_requests == 0


@pytest.mark.asyncio
async def test_manual_disable_via_api_discards_inflight_and_can_resume(monkeypatch, tmp_path):
    settings = SettingsStore(tmp_path/'settings.json')
    config = settings.value.loopDetection
    config.traceAllRequests = config.traceContent = True
    store = TraceStore(tmp_path/'traces.sqlite3')
    monkeypatch.setattr(proxy, 'loop_settings', settings)
    monkeypatch.setattr(proxy, 'trace_store', store)
    monkeypatch.setattr(proxy, 'tool_tracker', ToolTracker(settings, store))
    pending = RequestTrace(settings, store, 'POST', '/v1/responses', b'{"input":"private"}')
    pending.feed(b'started')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
        value = settings.value.model_dump()
        value['loopDetection']['traceEnabled'] = False
        assert (await client.put('/api/loop-settings', json=value)).status_code == 200
        assert not SettingsStore(settings.path).value.loopDetection.traceEnabled
        proxy.tool_tracker.prepare('s', history(), 'qwen27', True)
        value['loopDetection']['traceEnabled'] = True
        assert (await client.put('/api/loop-settings', json=value)).status_code == 200
    pending.feed(b'finished')
    pending.finish(complete=True)
    await asyncio.to_thread(store.queue.join)
    assert store.read(kind='all')['recent'] == []
    resumed = RequestTrace(settings, store, 'GET', '/v1/models', b'')
    resumed.feed(b'{"data":[]}')
    resumed.finish(complete=True)
    await store.close()
    assert store.read(kind='requests')['requests'] == 1


@pytest.mark.asyncio
async def test_content_is_optional_and_capture_truncates_explicitly(tmp_path):
    settings = SettingsStore(tmp_path/'settings.json')
    config = settings.value.loopDetection
    config.traceAllRequests = True
    config.maxTraceContentKB = 1
    store = TraceStore(tmp_path/'traces.sqlite3')
    trace = RequestTrace(settings, store, 'POST', '/v1/embeddings', b'private-input')
    trace.feed(b'private-output')
    trace.finish(complete=True)
    config.traceContent = True
    trace = RequestTrace(settings, store, 'POST', '/v1/responses', b'x'*2000)
    trace.feed(b'y'*2000)
    trace.finish(complete=False)
    await store.close()
    records = [store.read(event_id=r['id']) for r in store.read(kind='requests')['recent']]
    assert records[0]['request_truncated'] and records[0]['response_truncated']
    assert not records[0]['complete']
    assert 'request_content' not in records[1] and 'response_content' not in records[1]


@pytest.mark.asyncio
async def test_disk_cap_evicts_oldest_and_shrinks_after_setting_change(tmp_path):
    settings = SettingsStore(tmp_path/'settings.json')
    config = settings.value.loopDetection
    config.maxDiskMB = 4
    store = TraceStore(tmp_path/'traces.sqlite3')
    for i in range(50):
        store.submit('request', {'id': str(i), 'timestamp': time.time() + i, 'content': 'x'*20000}, config)
    await asyncio.to_thread(store.queue.join)
    assert store.disk_bytes() > 1024*1024
    config.maxDiskMB = 1
    store.configure(config)
    await asyncio.to_thread(store.queue.join)
    assert store.disk_bytes() <= 1024*1024
    for i in range(50, 100):
        store.submit('request', {'id': str(i), 'timestamp': time.time() + i, 'content': 'x'*20000}, config)
    await store.close()
    assert store.disk_bytes() <= 1024*1024
    assert store.errors == 0 and store.dropped == 0
    assert store.read(event_id='0') is None and store.read(event_id='99')['id'] == '99'
    assert store.read(kind='requests')['requests'] == 100
    with sqlite3.connect(store.path) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


@pytest.mark.asyncio
async def test_oversized_record_does_not_remove_existing_traces(tmp_path):
    settings = SettingsStore(tmp_path/'settings.json')
    config = settings.value.loopDetection
    config.maxDiskMB = 1
    store = TraceStore(tmp_path/'traces.sqlite3')
    store.submit('request', {'id': 'kept', 'timestamp': time.time()}, config)
    store.submit('request', {'id': 'huge', 'timestamp': time.time(), 'content': 'x'*2000000}, config)
    await store.close()
    assert store.read(event_id='kept')['id'] == 'kept'
    assert store.dropped == 1 and store.errors == 0
    assert store.disk_bytes() <= 1024*1024


@pytest.mark.asyncio
async def test_existing_wal_database_migrates_and_releases_space(tmp_path):
    path = tmp_path/'traces.sqlite3'
    # Build an old database with retained free pages and no auto-vacuum.
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE legacy (value BLOB)')
        db.execute('INSERT INTO legacy VALUES (zeroblob(2000000))')
        db.commit()
        db.execute('DELETE FROM legacy')
    db.close()
    settings = SettingsStore(tmp_path/'settings.json')
    settings.value.loopDetection.maxDiskMB = 1
    store = TraceStore(path)
    store.configure(settings.value.loopDetection)
    await store.close()
    assert store.errors == 0 and store.disk_bytes() <= 1024*1024
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA auto_vacuum').fetchone()[0] == 1
        assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'


@pytest.mark.asyncio
@pytest.mark.parametrize('route,stream_failure', [('responses', False), ('embeddings', True)])
async def test_network_failures_are_traced_and_release_slots(monkeypatch, tmp_path, route, stream_failure):
    settings = SettingsStore(tmp_path/'settings.json')
    settings.value.loopDetection.traceAllRequests = True
    store = TraceStore(tmp_path/'traces.sqlite3')
    monkeypatch.setattr(proxy, 'loop_settings', settings)
    monkeypatch.setattr(proxy, 'trace_store', store)
    monkeypatch.setattr(proxy, 'active_requests', 0)
    async def ready(client, *, reserve_slot=False):
        proxy.active_requests += 1
        proxy.no_active_requests.clear()
    monkeypatch.setattr(proxy, 'ensure_ready', ready)
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'partial'
            raise httpx.ReadError('interrupted')
    def upstream(request):
        if stream_failure:
            return httpx.Response(200, stream=BrokenStream())
        raise httpx.ConnectError('unavailable', request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(proxy.app.state, 'http', upstream_client, raising=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
            if stream_failure:
                with pytest.raises(httpx.ReadError):
                    await client.post('/v1/' + route, json={'input': 'hello'})
            else:
                assert (await client.post('/v1/' + route, json={'input': 'hello'})).status_code == 502
    await store.close()
    event = store.read(kind='requests')['recent'][0]
    detail = store.read(event_id=event['id'])
    assert event['path'] == '/v1/' + route
    assert not detail['complete']
    assert detail['status'] == (200 if stream_failure else 502)
    assert proxy.active_requests == 0 and proxy.no_active_requests.is_set()
