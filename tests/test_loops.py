"""Recovery tests with simulated OpenAI history, HTTP and SSE; no GPU needed."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import time

import httpx
import pytest

import app as proxy
from loop_detector import LoopDetector, ToolCall, digest
from settings import Settings, SettingsStore
from tool_tracker import ResponseObserver, ToolTracker, preview
from trace_store import TraceStore


class MemoryStore:
    def __init__(self):
        self.records = []
        self.generation = 0
    def configure(self, config):
        if not config.traceEnabled or not config.traceAllRequests:
            self.generation += 1
    def submit(self, kind, record, config):
        self.records.append((kind, record))


def tracker(tmp_path, mode="observe", **detection):
    settings = SettingsStore(tmp_path / 'settings.json')
    settings.value = Settings(loopDetection=detection, recovery={"mode": mode})
    return ToolTracker(settings, MemoryStore())


def tool(call_id, name="build", arguments='{"cmd":"npm run build"}'):
    return {"id": str(call_id), "type": "function", "function": {"name": name, "arguments": arguments}}


def history(names=("build",)*3, results=None, start=0):
    messages = [{"role": "user", "content": "Fix the project"}]
    for index, name in enumerate(names):
        messages.extend([{"role": "assistant", "content": None, "tool_calls": [tool(index+start, name)]},
                         {"role": "tool", "tool_call_id": str(index+start),
                          "content": results[index] if results else "error TS2304: unknown name"}])
    return {"model": "public", "messages": messages, "temperature": .3, "max_tokens": 128,
            "tools": [{"type": "function", "function": {"name": "build", "parameters": {"type": "object"}}}],
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}


def loop_records(t):
    return [record for kind, record in t.store.records if kind == "loop"]


@pytest.mark.parametrize("names,threshold,kind,size", [
    (("A",)*3, 3, "simple", 1),
    (("A", "B")*2, 2, "cycle", 2),
    (("A", "B", "C")*3, 3, "cycle", 3),
    (("A", "B", "C", "D")*3, 3, "cycle", 4),
])
def test_cycles(tmp_path, names, threshold, kind, size):
    t = tracker(tmp_path, repetitionThreshold=threshold)
    t.prepare("s", history(names, ["unchanged"]*len(names)), "qwen27", True)
    event = loop_records(t)[0]
    assert (event['kind'], event['cycle_length'], event['repetitions']) == (kind, size, threshold)


def test_repeated_failure_and_full_result_progress(tmp_path):
    t = tracker(tmp_path)
    t.prepare("s", history(), "qwen27", True)
    assert loop_records(t)[0]['kind'] == 'repeated_failure'
    for results in (["1", "2", "3"], ["error TS2304 a", "error TS2304 b", "error TS2304 c"],
                    ["x"*2000+str(i) for i in range(3)]):
        t = tracker(tmp_path)
        t.prepare('s', history(results=results), 'qwen27', True)
        assert not loop_records(t)


def test_build_with_intervening_edits_is_not_simple_repetition(tmp_path):
    t = tracker(tmp_path)
    t.prepare('s', history(("build", "edit", "build", "edit", "build"),
                         ['error', 'changed file 1', 'error', 'changed file 2', 'error']), 'qwen27', True)
    assert not loop_records(t)


def test_normalized_arguments_and_pending_result():
    calls = [ToolCall(str(i), 'A', digest(arg), 'qwen27', 0, digest('ok'))
             for i, arg in enumerate(['{"b":2,"a":1}', '{"a":1, "b":2}', {"a": 1, "b": 2}])]
    assert LoopDetector().detect(calls)
    calls[-1].result = None
    assert not LoopDetector().detect(calls)


def test_history_deduplication_and_truncation(tmp_path):
    t = tracker(tmp_path)
    payload = history()
    for _ in range(3):
        t.prepare('s', payload, 'qwen27', True)
    assert len([r for k, r in t.store.records if k == 'tool' and r['new']]) == 3
    assert len(loop_records(t)) == 1
    # Keep an overlapping call, then add a new call after compaction.
    truncated = history(("build",)*2, start=2)
    t.prepare('s', truncated, 'qwen27', True)
    assert len([r for k, r in t.store.records if k == 'tool' and r['new']]) == 4


def test_long_replayed_prefix_is_skipped(tmp_path):
    t = tracker(tmp_path)
    payload = history(tuple('tool'+str(i) for i in range(300)))
    t.prepare('s', payload, 'qwen27', True)
    t.prepare('s', payload, 'qwen27', True)
    assert len(t.sessions['s'].calls) == 256
    assert sum(k == 'tool' and r['new'] for k, r in t.store.records) == 300


@pytest.mark.parametrize('mode,prompt,thinking', [('observe', False, False), ('prompt', True, False),
    ('thinking', False, True), ('prompt_and_thinking', True, True)])
def test_modes_are_temporary_and_preserve_generation(tmp_path, mode, prompt, thinking):
    t = tracker(tmp_path, mode)
    payload = history()
    original = deepcopy(payload)
    output = t.prepare('s', payload, 'qwen27', True)
    assert payload == original
    assert len(output['messages']) == len(payload['messages']) + int(prompt)
    assert output['chat_template_kwargs']['enable_thinking'] == thinking
    assert output['chat_template_kwargs']['preserve_thinking'] is False
    for key in ('tools', 'temperature', 'max_tokens'):
        assert output[key] == original[key]
    if prompt:
        assert 'repeated_failure' in output['messages'][0]['content']
        assert output['messages'][1:] == original['messages']
    assert t.sessions['s'].state == 'NORMAL'
    # Identical HTTP retries don't spend another recovery attempt.
    assert t.prepare('s', payload, 'qwen27', True) == payload


def test_existing_system_messages_and_tool_adjacency(tmp_path):
    t = tracker(tmp_path, 'prompt')
    payload = history()
    payload['messages'].insert(0, {'role': 'system', 'content': 'existing'})
    result = t.prepare('s', payload, 'qwen27', True)
    assert result['messages'][0] == payload['messages'][0]
    assert result['messages'][2:] == payload['messages'][1:]


def test_thinking_unsupported_and_disabled(tmp_path):
    t = tracker(tmp_path, 'prompt_and_thinking')
    result = t.prepare('s', history(), 'custom', False)
    assert result['chat_template_kwargs']['enable_thinking'] is False
    assert loop_records(t)[0]['limitation']
    assert loop_records(t)[0]['intervention'] == 'prompt'
    t = tracker(tmp_path, 'thinking')
    t.settings.value.recovery.temporaryThinking = False
    assert t.prepare('s', history(), 'qwen27', True) == history()


def test_recovery_budget_and_rotated_cycles(tmp_path):
    t = tracker(tmp_path, 'prompt', repetitionThreshold=2)
    for count in range(4, 10):
        t.prepare('s', history(tuple('AB'[i%2] for i in range(count))), 'qwen27', True)
    events = loop_records(t)
    assert [e['intervention'] for e in events[:3]] == ['prompt', 'prompt', 'limit_reached']
    assert len({e['signature'] for e in events}) == 1
    assert t.sessions['s'].state == 'NORMAL'


def test_detection_disabled(tmp_path):
    t = tracker(tmp_path, 'prompt_and_thinking', enabled=False)
    assert t.prepare('s', history(), 'qwen27', True) == history()
    assert not loop_records(t)
    assert sum(kind == 'tool' and record['new'] for kind, record in t.store.records) == 3


def test_sessions_and_credentials_are_isolated(tmp_path):
    t = tracker(tmp_path, 'prompt', singleClient=False)
    t.prepare('a', history(("build",)*2), 'qwen27', True)
    t.prepare('b', history(("build",), start=2), 'qwen27', True)
    assert not loop_records(t)
    a, _, _ = t.identify({'x-proxy-session-id': 'same', 'authorization': 'Bearer a'}, {})
    b, _, _ = t.identify({'x-proxy-session-id': 'same', 'authorization': 'Bearer b'}, {})
    assert a != b
    a, _, reliable = t.identify({}, history())
    b, _, _ = t.identify({}, history())
    assert a != b and not reliable
    assert t.prepare(a, history(), 'qwen27', True, reliable=False) == history()


def test_new_user_turn_breaks_continuity(tmp_path):
    t = tracker(tmp_path)
    payload = history(("build",)*2)
    t.prepare('s', payload, 'qwen27', True)
    payload['messages'].append({'role': 'user', 'content': 'Build once again please'})
    payload['messages'].extend(history(("build",), start=2)['messages'][1:])
    t.prepare('s', payload, 'qwen27', True)
    assert not loop_records(t)


def test_session_limits_and_ttl(tmp_path):
    t = tracker(tmp_path, maxSessions=2, sessionTTLSeconds=60)
    for key in 'abc':
        t.prepare(key, history(("build",)), 'qwen27', True)
    assert set(t.sessions) == {'b', 'c'}
    t.sessions['b'].touched -= 61
    t.session('c')
    assert set(t.sessions) == {'c'}


def test_model_change_breaks_cycle(tmp_path):
    t = tracker(tmp_path)
    t.prepare('s', history(("build",)*2), 'qwen27', True)
    t.prepare('s', history(), 'flash-next', True)
    assert not loop_records(t)


def test_redaction():
    value = preview({'api_key': 'sensitive', 'nested': {'password': 'private'}, 'command': 'TOKEN=hidden'})
    assert 'sensitive' not in value and 'private' not in value and 'hidden' not in value


def sse_payload():
    parts = [
        {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'call_a', 'type': 'function',
                     'function': {'name': 'build', 'arguments': '{"cmd":'}}]}, 'finish_reason': None}]},
        {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"écho"}'}}]}, 'finish_reason': None}]},
        {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]},
        {'choices': [], 'usage': {'completion_tokens': 20}},
    ]
    return b''.join(('data: '+json.dumps(p, ensure_ascii=False)+'\r\n\r\n').encode() for p in parts)+b'data: [DONE]\r\n\r\n'


def test_stream_parser_fragments_and_deduplicates_echoed_history(tmp_path):
    t = tracker(tmp_path)
    t.prepare('s', {'messages': []}, 'qwen27', True)
    observer = ResponseObserver(t, 's', 'qwen27', True)
    for byte in sse_payload():
        observer.feed(bytes([byte]))
    calls = list(t.sessions['s'].calls.values())
    assert len(calls) == 1
    assert calls[0].arguments == digest('{"cmd":"écho"}')
    payload = {'messages': [{'role': 'assistant', 'tool_calls': [tool('call_a', arguments='{"cmd":"écho"}')]},
                            {'role': 'tool', 'tool_call_id': 'call_a', 'content': 'done'}]}
    t.prepare('s', payload, 'qwen27', True)
    assert sum(k == 'tool' and r['new'] for k, r in t.store.records) == 1
    assert list(t.sessions['s'].calls.values())[0].result == digest('done')


def test_multiline_sse_and_oversized_observation(tmp_path):
    t = tracker(tmp_path)
    t.prepare('s', {'messages': []}, 'qwen27', True)
    observer = ResponseObserver(t, 's', 'qwen27', True)
    observer.feed(b'data: {"choices":\ndata: [{"message":{"tool_calls":'+json.dumps([tool('x')]).encode()+b'}}]}\n\n')
    assert len(t.sessions['s'].calls) == 1
    observer.feed(b'data: '+b'x'*70000)
    assert observer.disabled and not observer.buffer


@pytest.mark.asyncio
async def test_sqlite_persistence_privacy_and_replay(tmp_path):
    settings = SettingsStore(tmp_path/'settings.json')
    store = TraceStore(tmp_path/'traces.sqlite3')
    t = ToolTracker(settings, store)
    t.prepare('s', history(), 'qwen27', True)
    await store.close()
    result = store.read()
    assert (result['tool_calls'], result['loops']) == (3, 1)
    assert store.read(event_id=result['recent'][0]['id'])['kind'] == 'repeated_failure'
    with sqlite3.connect(store.path) as db:
        records = '\n'.join(row[0] for row in db.execute('SELECT record FROM tools'))
    assert 'npm run build' not in records and 'unknown name' not in records
    # Replay after a tracker restart: SQLite uniqueness retains the exact count.
    t = ToolTracker(settings, store)
    t.prepare('s', history(), 'qwen27', True)
    await store.close()
    assert store.read()['tool_calls'] == 3


@pytest.mark.asyncio
async def test_storage_disabled_and_unwritable_database(tmp_path):
    settings = SettingsStore(tmp_path/'settings.json')
    settings.value.loopDetection.storeToolHistory = False
    store = TraceStore(tmp_path/'traces.sqlite3')
    t = ToolTracker(settings, store)
    t.prepare('s', history(), 'qwen27', True)
    await store.close()
    assert store.read()['tool_calls'] == 3
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM tools').fetchone()[0] == 0
    store = TraceStore(tmp_path)  # a directory cannot be a SQLite database
    t = ToolTracker(settings, store)
    assert t.prepare('s', history(), 'qwen27', True) == history()
    await store.close()
    assert store.errors > 0


@pytest.mark.asyncio
async def test_retention(tmp_path):
    store = TraceStore(tmp_path/'traces.sqlite3')
    config = Settings().loopDetection
    store.submit('loop', {'id': 'old', 'timestamp': time.time()-86400*10}, config)
    await store.close()
    assert not store.read()['recent']


@pytest.mark.asyncio
async def test_settings_api_persistence_validation_and_trace_detail(monkeypatch, tmp_path):
    settings = SettingsStore(tmp_path/'settings.json')
    store = TraceStore(tmp_path/'traces.sqlite3')
    monkeypatch.setattr(proxy, 'loop_settings', settings)
    monkeypatch.setattr(proxy, 'trace_store', store)
    monkeypatch.setattr(proxy, 'tool_tracker', ToolTracker(settings, store))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
        value = (await client.get('/api/loop-settings')).json()
        value['recovery'].update(mode='prompt_and_thinking', prompt='Analyse {freely} and retry differently')
        assert (await client.put('/api/loop-settings', json=value)).status_code == 200
        assert SettingsStore(settings.path).value.model_dump() == value
        value['loopDetection']['maxCycleLength'] = 5
        assert (await client.put('/api/loop-settings', json=value)).status_code == 422
        assert (await client.get('/api/traces/missing')).status_code == 404
        proxy.tool_tracker.prepare('s', history(), 'qwen27', True)
        await store.close()
        traces = (await client.get('/api/traces')).json()
        assert traces['tool_calls'] == 3 and traces['loops'] == 1
        detail = await client.get('/api/traces/'+traces['recent'][0]['id'])
        assert detail.json()['intervention'] == 'prompt+thinking'
        dashboard = (await client.get('/')).text
        assert 'Détection des boucles' in dashboard and 'Traces d’exécution' in dashboard


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_http_recovery_preserves_tool_responses_and_slots(monkeypatch, tmp_path, stream):
    t = tracker(tmp_path, 'prompt_and_thinking')
    monkeypatch.setattr(proxy, 'loop_settings', t.settings)
    monkeypatch.setattr(proxy, 'tool_tracker', t)
    monkeypatch.setattr(proxy, 'selected_model', 'flash-next')
    monkeypatch.setattr(proxy, 'active_requests', 0)
    seen = []
    async def ready(client, *, reserve_slot=False):
        if reserve_slot:
            proxy.active_requests += 1
            proxy.no_active_requests.clear()
    monkeypatch.setattr(proxy, 'ensure_ready', ready)
    raw = sse_payload() if stream else json.dumps({'choices': [{'message': {'role': 'assistant', 'tool_calls': [tool('new')]}}]}).encode()
    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index in range(0, len(raw), 7):
                yield raw[index:index+7]
    def upstream(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, stream=Chunks(), headers={'content-type': 'text/event-stream' if stream else 'application/json'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(proxy.app.state, 'http', upstream_client, raising=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
            payload = history()
            payload['stream'] = stream
            response = await client.post('/v1/chat/completions', json=payload, headers={'X-Proxy-Session-ID': 'test-session'})
            assert response.content == raw
            assert response.status_code == 200
    assert seen[0]['model'] == proxy.PUBLIC_MODEL
    assert seen[0]['chat_template_kwargs']['enable_thinking'] is True
    assert len(seen[0]['messages']) == len(payload['messages'])+1
    assert proxy.active_requests == 0 and proxy.no_active_requests.is_set()
    assert any(r['model'] == 'flash-next' for k, r in t.store.records if k == 'tool')


@pytest.mark.asyncio
async def test_sqlite_row_cap(tmp_path):
    store = TraceStore(tmp_path/'traces.sqlite3')
    config = Settings().loopDetection
    config.maxTraceRows = 100
    for i in range(105):
        store.submit('loop', {'id': str(i), 'timestamp': time.time()}, config)
    await store.close()
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM loops').fetchone()[0] == 100
    assert store.read()['loops'] == 105


def test_anonymous_input_history_is_not_counted_again(tmp_path):
    t = tracker(tmp_path, 'prompt', singleClient=False)
    for _ in range(3):
        key, _, reliable = t.identify({}, history())
        assert t.prepare(key, history(), 'qwen27', True, reliable) == history()
    assert not t.store.records


@pytest.mark.asyncio
async def test_saving_settings_keeps_call_ids_and_recovery_budget(monkeypatch, tmp_path):
    t = tracker(tmp_path, 'prompt')
    monkeypatch.setattr(proxy, 'loop_settings', t.settings)
    monkeypatch.setattr(proxy, 'tool_tracker', t)
    t.prepare('s', history(), 'qwen27', True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
        assert (await client.put('/api/loop-settings', json=t.settings.value.model_dump())).status_code == 200
    t.prepare('s', history(), 'qwen27', True)
    assert len(loop_records(t)) == 1
    assert sum(k == 'tool' and r['new'] for k, r in t.store.records) == 3


def test_thinking_capability_does_not_invalidate_engine_profile(monkeypatch):
    previous = proxy.profile_digest('qwen27')
    monkeypatch.setitem(proxy.MODELS['qwen27'], 'supports_thinking', False)
    assert proxy.profile_digest('qwen27') == previous


@pytest.mark.asyncio
async def test_upstream_failure_releases_reserved_slot(monkeypatch, tmp_path):
    t = tracker(tmp_path)
    monkeypatch.setattr(proxy, 'loop_settings', t.settings)
    monkeypatch.setattr(proxy, 'tool_tracker', t)
    monkeypatch.setattr(proxy, 'active_requests', 0)
    async def ready(client, *, reserve_slot=False):
        proxy.active_requests += 1
        proxy.no_active_requests.clear()
    monkeypatch.setattr(proxy, 'ensure_ready', ready)
    def upstream(request):
        raise httpx.ConnectError('unavailable', request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(proxy.app.state, 'http', upstream_client, raising=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
            result = await client.post('/v1/chat/completions', json=history())
    assert result.status_code == 502
    assert proxy.active_requests == 0 and proxy.no_active_requests.is_set()


@pytest.mark.asyncio
async def test_stream_error_keeps_slot_until_closed(monkeypatch, tmp_path):
    t = tracker(tmp_path)
    monkeypatch.setattr(proxy, 'loop_settings', t.settings)
    monkeypatch.setattr(proxy, 'tool_tracker', t)
    monkeypatch.setattr(proxy, 'active_requests', 0)
    closed = []
    async def ready(client, *, reserve_slot=False):
        proxy.active_requests += 1
        proxy.no_active_requests.clear()
    monkeypatch.setattr(proxy, 'ensure_ready', ready)
    class FailingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            assert proxy.active_requests == 1
            yield b'data: {"choices":[]}\n\n'
            raise httpx.ReadError('stream interrupted')
        async def aclose(self):
            closed.append(True)
    def upstream(request):
        return httpx.Response(200, stream=FailingStream(), headers={'content-type': 'text/event-stream'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(proxy.app.state, 'http', upstream_client, raising=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
            with pytest.raises(httpx.ReadError):
                await client.post('/v1/chat/completions', json={**history(), 'stream': True})
    assert closed
    assert proxy.active_requests == 0 and proxy.no_active_requests.is_set()


def test_missing_result_is_not_a_completed_tool(tmp_path):
    t = tracker(tmp_path)
    payload = history()
    for message in payload['messages']:
        if message['role'] == 'tool':
            message.pop('content')
    t.prepare('s', payload, 'qwen27', True)
    assert not loop_records(t)


def test_stream_tool_accumulation_is_bounded(tmp_path):
    t = tracker(tmp_path)
    t.prepare('s', {'messages': []}, 'qwen27', True)
    observer = ResponseObserver(t, 's', 'qwen27', True)
    for _ in range(20):
        payload = {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'x',
                    'function': {'name': 'build', 'arguments': 'x'*50000}}]}}]}
        observer.feed(('data: '+json.dumps(payload)+'\n\n').encode())
    assert observer.disabled
    assert not observer.tools and not observer.event and not observer.buffer
    assert t.failures == 1


@pytest.mark.asyncio
async def test_integrated_model_switch_waits_for_observed_stream(monkeypatch, tmp_path):
    t = tracker(tmp_path, 'prompt_and_thinking')
    for name, value in {'loop_settings': t.settings, 'tool_tracker': t, 'selected_model': 'qwen27',
                        'pending_model': 'flash-next', 'active_requests': 0,
                        'no_active_requests': asyncio.Event(), 'switch_complete': asyncio.Event(),
                        'STATE_PATH': tmp_path/'selection.json'}.items():
        monkeypatch.setattr(proxy, name, value)
    proxy.no_active_requests.set()
    proxy.switch_complete.set()
    streaming = asyncio.Event()
    finish = asyncio.Event()
    actions = []
    async def ready(client, *, reserve_slot=False):
        await proxy.switch_complete.wait()
        if reserve_slot:
            proxy.active_requests += 1
            proxy.no_active_requests.clear()
    async def start(client, model):
        actions.append('start:'+model)
    async def state():
        return {'engine_profile': 'qwen27', 'status': 'running'}
    def stop():
        assert proxy.active_requests == 0
        actions.append('stop')
    monkeypatch.setattr(proxy, 'ensure_ready', ready)
    monkeypatch.setattr(proxy, 'container_state', state)
    monkeypatch.setattr(proxy, 'stop_remove_sync', stop)
    monkeypatch.setattr(proxy, 'start_engine_locked', start)
    raw = sse_payload()
    class DelayedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield raw[:100]
            streaming.set()
            await finish.wait()
            yield raw[100:]
    def upstream(request):
        return httpx.Response(200, stream=DelayedStream(), headers={'content-type': 'text/event-stream'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(proxy.app.state, 'http', upstream_client, raising=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
            response_task = asyncio.create_task(client.post('/v1/chat/completions',
                json={**history(), 'stream': True}, headers={'X-Proxy-Session-ID': 'switching'}))
            await asyncio.wait_for(streaming.wait(), 1)
            proxy.switch_complete.clear()
            switch_task = asyncio.create_task(proxy.switch_worker(upstream_client, 'flash-next'))
            await asyncio.sleep(0)
            assert actions == [] and proxy.active_requests == 1
            finish.set()
            response, _ = await asyncio.wait_for(asyncio.gather(response_task, switch_task), 2)
    assert response.content == raw
    assert actions == ['stop', 'start:flash-next']
    assert proxy.selected_model == 'flash-next'
    assert proxy.active_requests == 0
    assert all(r['model'] == 'qwen27' for kind, r in t.store.records if kind == 'tool')
