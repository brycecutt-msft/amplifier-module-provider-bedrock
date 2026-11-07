"""Real SDK serialization across httpx/httpx2, never a live AWS request."""
import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import pytest
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_bedrock import BedrockProvider

MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"


def http_module():
    return importlib.import_module("httpx2" if any(
        cls.__module__.startswith("httpx2")
        for cls in anthropic.DefaultAsyncHttpxClient.__mro__
    ) else "httpx")


def response(http, stop="end_turn"):
    return http.Response(200, json={
        "id": "fixture", "type": "message", "role": "assistant", "model": MODEL,
        "content": [{"type": "text", "text": "ok"}], "stop_reason": stop,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    })


def provider(handler, **config):
    http = http_module()
    hooks = SimpleNamespace(emit=AsyncMock())
    p = BedrockProvider({"aws_region": "us-east-1", "aws_access_key": "fixture-key",
        "aws_secret_key": "fixture-secret", "use_cross_region_inference": False,
        "model_filters": {"include": []}, "raw": True, **config}, SimpleNamespace(hooks=hooks))
    original = p.client
    p.client = original.with_options(http_client=http.AsyncClient(transport=http.MockTransport(handler)), max_retries=0)
    return p, original, hooks


@pytest.mark.asyncio
@pytest.mark.parametrize("config,request_temp,options,expected", [
    ({}, 0, {}, 0),
    ({}, .25, {}, .25),
    ({"temperature": .4}, None, {}, .4),
    ({}, None, {"request_options": {"temperature": .6}}, .6),
])
async def test_exact_temperature_reaches_wire_and_raw_event(config, request_temp, options, expected):
    http = http_module()
    seen = []
    def handler(req):
        seen.append(req)
        return response(http)
    p, original, hooks = provider(handler, **config)
    request = ChatRequest(model=MODEL, messages=[Message(role="user", content="fixture")],
                          max_output_tokens=17, temperature=request_temp)
    try:
        result = await p.complete(request, **options)
        assert result.content[0].text == "ok" and len(seen) == 1
        payload = json.loads(seen[0].content)
        assert payload["temperature"] == expected and payload["max_tokens"] == 17
        assert payload["anthropic_version"] == "bedrock-2023-05-31"
        assert "extra_body" not in payload and "model_filters" not in payload
        assert seen[0].url.path == "/model/"+MODEL+"/invoke"
        assert "Credential=fixture-key/" in seen[0].headers["authorization"]
        event = next(c.args[1] for c in hooks.emit.call_args_list if c.args[0] == "llm:request")
        assert event["raw"]["temperature"] == expected and "extra_body" not in event["raw"]
        assert request.temperature == request_temp
    finally:
        await p.client.close()
        await original.close()


@pytest.mark.asyncio
async def test_continuation_uses_same_temperature_model_and_account():
    http = http_module()
    seen = []
    def handler(req):
        seen.append(req)
        return response(http, "max_tokens" if len(seen) == 1 else "end_turn")
    p, original, _ = provider(handler)
    try:
        result = await p.complete(ChatRequest(model=MODEL, messages=[Message(role="user", content="fixture")],
                                             max_output_tokens=17, temperature=.25))
        assert len(seen) == 2 and result.finish_reason == "end_turn"
        for req in seen:
            payload = json.loads(req.content)
            assert payload["temperature"] == .25 and payload["max_tokens"] == 17
            assert req.url.path == "/model/"+MODEL+"/invoke"
            assert "Credential=fixture-key/" in req.headers["authorization"]
            assert req.extensions["timeout"] == {"connect": 5, "pool": 5, "read": None, "write": None}
        assert len(json.loads(seen[1].content)["messages"]) == 2
    finally:
        await p.client.close()
        await original.close()


@pytest.mark.asyncio
async def test_unsupported_temperature_rejection_is_not_dropped_or_retried():
    http = http_module()
    seen = []
    def handler(req):
        seen.append(req)
        return http.Response(400, json={"message": "fixture validation failure"})
    p, original, _ = provider(handler)
    try:
        with pytest.raises(anthropic.BadRequestError):
            await p.complete(ChatRequest(model=MODEL, messages=[Message(role="user", content="fixture")], temperature=.25))
        assert len(seen) == 1 and json.loads(seen[0].content)["temperature"] == .25
    finally:
        await p.client.close()
        await original.close()


@pytest.mark.asyncio
async def test_cancellation_with_serialized_temperature_propagates():
    entered, wait = asyncio.Event(), asyncio.Event()
    seen = []
    async def handler(req):
        seen.append(req)
        entered.set()
        await wait.wait()
        raise AssertionError("unreachable")
    p, original, _ = provider(handler)
    task = asyncio.create_task(p.complete(ChatRequest(model=MODEL, messages=[Message(role="user", content="fixture")], temperature=.25)))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(seen) == 1 and json.loads(seen[0].content)["temperature"] == .25
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await p.client.close()
        await original.close()
