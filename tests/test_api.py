import asyncio
import json

import httpx
import pytest

import app as proxy


@pytest.mark.asyncio
async def test_public_model_is_rewritten_and_slot_released(monkeypatch):
    proxy.selected_model = 'qwen27'
    proxy.pending_model = None
    proxy.switch_complete.set()
    proxy.active_requests = 0
    proxy.no_active_requests.set()
    seen = []

    async def fake_ensure(client, *, reserve_slot=False):
        if reserve_slot:
            proxy.active_requests += 1
            proxy.no_active_requests.clear()

    class StreamingData(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'

    def upstream(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, headers={'content-type': 'text/event-stream'},
            stream=StreamingData())

    monkeypatch.setattr(proxy, 'ensure_ready', fake_ensure)
    proxy.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
        result = await client.post('/v1/chat/completions', json={
            'model': 'qwen-other', 'messages': [{'role': 'user', 'content': 'Bonjour'}], 'stream': True})
        assert result.status_code == 200
        assert b'data:' in result.content
    await proxy.app.state.http.aclose()
    assert seen[0]['model'] == proxy.PUBLIC_MODEL
    assert proxy.active_requests == 0
    assert proxy.no_active_requests.is_set()


@pytest.mark.asyncio
async def test_api_lists_models_and_rejects_unknown_switch():
    proxy.switch_complete.set()
    proxy.pending_model = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
        models = (await client.get('/api/models')).json()
        assert {x['id'] for x in models['models']} == {'qwen27', 'flash-next'}
        failure = await client.post('/api/switch', json={'model': 'non-existent'})
        assert failure.status_code == 400
