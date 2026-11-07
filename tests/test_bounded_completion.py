"""Bounded auxiliary calls must neither expire nor silently continue."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_bedrock import BedrockProvider


def response(stop_reason):
    return SimpleNamespace(
        model="fake",
        content=[SimpleNamespace(type="text", text="partial")],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=10,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=5,
        ),
    )


@pytest.mark.asyncio
async def test_bounded_call_uses_one_request_and_truthful_stop_reason():
    p = BedrockProvider(config={"aws_region": "us-east-1"})
    p.client.messages.create = AsyncMock(return_value=response("max_tokens"))
    req = ChatRequest(
        model="explicit-model",
        messages=[Message(role="user", content="Summary")],
        max_output_tokens=100,
        temperature=0,
    )
    result = await p.complete(req, request_options={"auto_continue": False})
    p.client.messages.create.assert_awaited_once()
    params = p.client.messages.create.call_args.kwargs
    assert params["model"] == "us.explicit-model"
    assert params["extra_body"]["temperature"] == 0 and params["max_tokens"] == 100
    assert "auto_continue" not in params and "request_options" not in params
    assert result.finish_reason == "length"
    assert result.usage.input_tokens == 120 and result.usage.cache_write_tokens == 5
    assert p.client.timeout.read is None and p.client.timeout.connect == 5
    assert p.auto_continue is True


@pytest.mark.asyncio
async def test_normal_continuation_sums_cache_and_usage():
    p = BedrockProvider(config={"aws_region": "us-east-1"})
    p.client.messages.create = AsyncMock(
        side_effect=[response("max_tokens"), response("end_turn")]
    )
    result = await p.complete(
        ChatRequest(messages=[Message(role="user", content="Hello")])
    )
    assert p.client.messages.create.await_count == 2
    assert result.usage.input_tokens == 240 and result.usage.output_tokens == 20
    assert (
        result.usage.cache_read_tokens == 40 and result.usage.cache_write_tokens == 10
    )


@pytest.mark.asyncio
async def test_explicit_timeout_and_cancellation_are_preserved():
    import asyncio

    p = BedrockProvider(config={"aws_region": "us-east-1", "timeout": 45})
    assert p.client.timeout.read == 45
    p.client.messages.create = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await p.complete(ChatRequest(messages=[Message(role="user", content="Hello")]))
    p.client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_raw_capture_does_not_require_debug_logging():
    hooks = SimpleNamespace(emit=AsyncMock())
    p = BedrockProvider(
        config={"aws_region": "us-east-1", "raw": True},
        coordinator=SimpleNamespace(hooks=hooks),
    )
    p.client.messages.create = AsyncMock(return_value=response("end_turn"))
    await p.complete(ChatRequest(messages=[Message(role="user", content="Summary")]))
    event = next(
        c.args[1] for c in hooks.emit.call_args_list if c.args[0] == "llm:request"
    )
    assert (
        event["raw"]["messages"]
        == p.client.messages.create.call_args.kwargs["messages"]
    )
