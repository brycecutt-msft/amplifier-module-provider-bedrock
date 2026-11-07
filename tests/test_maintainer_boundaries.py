"""Focused maintainer-review boundaries; no AWS requests or credentials needed."""
import asyncio
import importlib
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import botocore.session
from botocore.exceptions import (
    ConfigParseError, EndpointConnectionError, InvalidConfigError,
    ProfileNotFound, SSOTokenLoadError, UnauthorizedSSOTokenError,
)
import pytest
from amplifier_core import llm_errors as errors
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.utils import redact_secrets
from amplifier_module_provider_bedrock import BedrockProvider
from amplifier_module_provider_bedrock._catalog import fetch_catalog


def provider(**config):
    return BedrockProvider({"aws_region": "us-east-1", "aws_access_key": "fixture-key",
                            "aws_secret_key": "fixture-secret", **config})


@pytest.mark.parametrize("native,expected", [
    (UnauthorizedSSOTokenError(), errors.AuthenticationError),
    (SSOTokenLoadError(error_msg="PRIVATE"), errors.AuthenticationError),
    (ProfileNotFound(profile="PRIVATE"), errors.ConfigurationError),
    (ConfigParseError(path="PRIVATE"), errors.ConfigurationError),
    (InvalidConfigError(error_msg="PRIVATE"), errors.ConfigurationError),
])
@pytest.mark.parametrize("phase", ["session", "request"])
def test_local_credential_failures_are_nonretryable(monkeypatch, native, expected, phase):
    session = MagicMock()
    control = MagicMock()
    session.client.return_value = control
    if phase == "session":
        monkeypatch.setattr("amplifier_module_provider_bedrock._catalog.boto3.Session", MagicMock(side_effect=native))
    else:
        control.list_inference_profiles.side_effect = native
        monkeypatch.setattr("amplifier_module_provider_bedrock._catalog.boto3.Session", MagicMock(return_value=session))
    with pytest.raises(expected) as caught:
        fetch_catalog({}, True, Event())
    assert not caught.value.retryable
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    if phase == "request":
        control.close.assert_called_once()


def test_transport_failure_still_retryable_and_closes(monkeypatch):
    session, control = MagicMock(), MagicMock()
    session.client.return_value = control
    control.list_inference_profiles.side_effect = EndpointConnectionError(endpoint_url="https://PRIVATE.invalid")
    monkeypatch.setattr("amplifier_module_provider_bedrock._catalog.boto3.Session", MagicMock(return_value=session))
    with pytest.raises(errors.ProviderUnavailableError) as caught:
        fetch_catalog({}, True, Event())
    assert caught.value.retryable and "PRIVATE" not in str(caught.value)
    control.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured,default_env,profile_env,expected", [
    (None, "default-env", "profile-env", "default-env"),
    ("configured", "default-env", "profile-env", "configured"),
    (None, None, "profile-env", "profile-env"),
    (None, "default-env", None, "default-env"),
    (None, None, None, None),
])
async def test_profile_precedence_matches_botocore_and_binds_catalog(monkeypatch, configured, default_env, profile_env, expected):
    for key, value in (("AWS_DEFAULT_PROFILE", default_env), ("AWS_PROFILE", profile_env)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    # No credential lookup: configuration provider's profile variable only.
    assert botocore.session.Session().get_config_variable("profile") == (default_env or profile_env)
    p = provider(**({"aws_profile": configured} if configured else {}))
    try:
        assert p.client.aws_profile == p._catalog_session_kwargs["profile_name"] == expected
        assert p.client.aws_region == p._catalog_session_kwargs["region_name"] == "us-east-1"
        assert p.client.aws_access_key == p._catalog_session_kwargs["aws_access_key_id"] == "fixture-key"
        monkeypatch.setenv("AWS_DEFAULT_PROFILE", "later-change")
        assert p._catalog_session_kwargs["profile_name"] == expected
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("options,selected", [
    ({}, "anthropic.claude-request-v1:0"),
    ({"model": "anthropic.claude-keyword-v1:0"}, "anthropic.claude-keyword-v1:0"),
    ({"request_options": {"model": "anthropic.claude-mapped-v1:0"}}, "anthropic.claude-mapped-v1:0"),
    ({"model": "anthropic.claude-keyword-v1:0", "request_options": {"model": "anthropic.claude-mapped-v1:0"}}, "anthropic.claude-keyword-v1:0"),
])
async def test_model_override_reaches_native_sdk_and_keeps_request_fields(monkeypatch, options, selected):
    transport_module = "httpx2" if any(
        cls.__module__.startswith("httpx2")
        for cls in anthropic.DefaultAsyncHttpxClient.__mro__
    ) else "httpx"
    httpx = importlib.import_module(transport_module)
    p = provider(use_cross_region_inference=False, model_filters={"include": []})
    original = p.client
    seen = []
    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={
            "id": "fixture", "type": "message", "role": "assistant", "model": selected,
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
    p.client = original.with_options(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), max_retries=0)
    req = ChatRequest(model="anthropic.claude-request-v1:0", messages=[Message(role="user", content="test")], max_output_tokens=17, temperature=.25)
    try:
        result = await p.complete(req, **options)
        assert result.content[0].text == "ok" and len(seen) == 1
        assert seen[0].url.path == "/model/" + selected + "/invoke"
        assert "Credential=fixture-key/" in seen[0].headers["authorization"]
        body = json.loads(seen[0].content)
        assert body["max_tokens"] == 17 and body["temperature"] == .25
        assert req.model == "anthropic.claude-request-v1:0"
        assert seen[0].extensions["timeout"] == {"connect": 5, "pool": 5, "read": None, "write": None}
    finally:
        await p.client.close()
        await original.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("system_text", [None, "System instruction"])
async def test_legacy_nested_request_event_keys_remain(system_text):
    hooks = SimpleNamespace(emit=AsyncMock())
    p = BedrockProvider({"aws_region": "us-east-1"}, SimpleNamespace(hooks=hooks))
    p.client.messages.create = AsyncMock(return_value=SimpleNamespace(
        model="fixture", content=[], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    ))
    messages = [Message(role="user", content="Hi")]
    if system_text:
        messages.insert(0, Message(role="system", content=system_text))
    try:
        await p.complete(ChatRequest(messages=messages))
        event = next(call.args[1] for call in hooks.emit.call_args_list if call.args[0] == "llm:request")
        assert event["data"]["message_count"] == 1
        assert event["data"]["has_system"] is bool(system_text)
        assert event["data"]["messages_count"] == event["message_count"] == 1
    finally:
        await p.client.close()


def test_redaction_is_key_based_not_a_free_text_secret_scanner():
    result = redact_secrets({"api_key": "PRIVATE-KEY", "content": "PRIVATE-TEXT"})
    assert result["api_key"] != "PRIVATE-KEY"
    assert result["content"] == "PRIVATE-TEXT"


@pytest.mark.asyncio
async def test_cancelled_credential_lookup_can_linger_without_more_requests(monkeypatch):
    entered, release, closed = Event(), Event(), Event()
    control = MagicMock()
    control.close.side_effect = lambda: closed.set()
    session = MagicMock()
    def create_client(*args, **kwargs):
        entered.set()
        assert release.wait(5)  # test-only gate, simulates credential_process wait
        return control
    session.client.side_effect = create_client
    monkeypatch.setattr("amplifier_module_provider_bedrock._catalog.boto3.Session", MagicMock(return_value=session))
    p = provider()
    task = asyncio.create_task(p.list_models())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not closed.is_set()  # Cancellation isn't synchronous thread cleanup.
        release.set()
        assert await asyncio.to_thread(closed.wait, 5)
        control.list_inference_profiles.assert_not_called()
    finally:
        release.set()
        await p.client.close()
