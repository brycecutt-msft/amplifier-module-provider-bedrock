"""Offline AWS SDK boundary checks; names below are controlled fixtures, not entitlement."""
import asyncio
from datetime import datetime, timezone
from threading import Event
from unittest.mock import MagicMock

import boto3
from botocore.stub import Stubber
import pytest
from amplifier_core import llm_errors as errors
from amplifier_module_provider_bedrock import BedrockProvider
from amplifier_module_provider_bedrock._catalog import CatalogEntry, CatalogFilters, fetch_catalog, read_catalog

S45 = "anthropic.claude-sonnet-4-5-20250929-v1:0"
S55 = "anthropic.claude-sonnet-5-5-v1:0"
S55DATE = "anthropic.claude-sonnet-5-5-20261001-v1:0"
O55 = "anthropic.claude-opus-5-5-v1:0"
H55 = "anthropic.claude-haiku-5-5-v1:0"
F51 = "anthropic.claude-fable-5-1-v1:0"


def provider(**config):
    return BedrockProvider({"aws_region": "us-east-1", "aws_access_key": "fixture-key",
                            "aws_secret_key": "fixture-secret", **config})


def control():
    return boto3.client("bedrock", region_name="us-east-1", aws_access_key_id="fixture-key", aws_secret_access_key="fixture-secret")


def foundation(model, status="ACTIVE"):
    return {"modelId": model, "modelName": model, "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/"+model,
            "providerName": "Anthropic", "inputModalities": ["TEXT"], "outputModalities": ["TEXT"],
            "inferenceTypesSupported": ["ON_DEMAND"], "modelLifecycle": {"status": status}}


def profile(model, prefix="us.", *, identity=None, kind="SYSTEM_DEFINED"):
    identity = identity or prefix+model
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    arn = "arn:aws:bedrock:us-east-1:123456789012:" + ("application-inference-profile/" if kind == "APPLICATION" else "inference-profile/") + identity
    return {"inferenceProfileId": identity, "inferenceProfileArn": arn, "inferenceProfileName": identity,
            "models": [{"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/"+model}],
            "status": "ACTIVE", "type": kind, "createdAt": now, "updatedAt": now}


def bind(monkeypatch, client):
    session = MagicMock()
    session.client.return_value = client
    monkeypatch.setattr("amplifier_module_provider_bedrock._catalog.boto3.Session", lambda **kwargs: session)


@pytest.mark.asyncio
async def test_profiles_all_pages_latest_distinct_and_unfiltered(monkeypatch):
    client = control()
    rows = [profile(S45), profile(S55DATE), profile(S55), profile(O55), profile(H55), profile(F51),
            profile(S55, "global."), profile(S45, identity="production-chat", kind="APPLICATION"),
            profile(S45, identity="research-chat", kind="APPLICATION")]
    stub = Stubber(client)
    for _ in range(2):
        stub.add_response("list_inference_profiles", {"inferenceProfileSummaries": rows[:3], "nextToken": "next"}, {})
        stub.add_response("list_inference_profiles", {"inferenceProfileSummaries": rows[3:]}, {"nextToken": "next"})
    bind(monkeypatch, client)
    with stub:
        p = provider()
        models = await p.list_models()
        ids = {m.id for m in models}
        assert ids == {"us."+S55, "us."+O55, "us."+H55, "us."+F51, "global."+S55,
                       rows[7]["inferenceProfileArn"], rows[8]["inferenceProfileArn"]}
        assert all(m.context_window == m.max_output_tokens == 0 and m.capabilities == [] for m in models)
        assert len(await p.list_models(filters={})) == 9
        assert p._apply_inference_profile(rows[7]["inferenceProfileArn"]) == rows[7]["inferenceProfileArn"]
        assert p._apply_inference_profile("global."+S55) == "global."+S55
        assert p._apply_inference_profile("future."+S55) == "future."+S55
        stub.assert_no_pending_responses()
        await p.client.close()


@pytest.mark.asyncio
async def test_direct_route_native_params_active_and_replacing_filters(monkeypatch):
    client = control()
    rows = [foundation(S45, "LEGACY"), foundation(S55), foundation(O55)]
    stub = Stubber(client)
    expected = {"byProvider": "Anthropic", "byOutputModality": "TEXT", "byInferenceType": "ON_DEMAND"}
    for _ in range(4):
        stub.add_response("list_foundation_models", {"modelSummaries": rows}, expected)
    bind(monkeypatch, client)
    with stub:
        p = provider(use_cross_region_inference=False)
        assert {m.id for m in await p.list_models()} == {S55, O55}
        assert [m.id for m in await p.list_models(filters={"include": ["*SONNET*"], "exclude": ["*5-5*"]})] == [S45]
        assert len(await p.list_models(filters={})) == 3
        assert await p.list_models(filters={"include": []}) == []
        assert p._apply_inference_profile(S45) == S45
        stub.assert_no_pending_responses()
        await p.client.close()


@pytest.mark.asyncio
async def test_empty_and_sanitized_permission_failure_never_fallback(monkeypatch):
    client = control()
    stub = Stubber(client)
    stub.add_response("list_inference_profiles", {"inferenceProfileSummaries": []}, {})
    stub.add_client_error("list_inference_profiles", "AccessDeniedException", "SECRET account/credentials", expected_params={})
    bind(monkeypatch, client)
    with stub:
        p = provider()
        assert await p.list_models() == []
        with pytest.raises(errors.AccessDeniedError) as exc:
            await p.list_models()
        assert "SECRET" not in str(exc.value)
        assert exc.value.__cause__ is None and exc.value.__suppress_context__
        await p.client.close()


@pytest.mark.parametrize("code,kind", [("ThrottlingException", errors.RateLimitError), ("ExpiredTokenException", errors.AuthenticationError), ("InternalServerException", errors.ProviderUnavailableError)])
def test_service_errors_typed(monkeypatch, code, kind):
    client = control()
    stub = Stubber(client)
    stub.add_client_error("list_inference_profiles", code, "secret", expected_params={})
    bind(monkeypatch, client)
    with stub, pytest.raises(kind):
        fetch_catalog({}, True, Event())


@pytest.mark.parametrize("response", [{}, {"inferenceProfileSummaries": None}, {"inferenceProfileSummaries": [{}]}])
def test_malformed_is_error(response):
    client = MagicMock()
    client.list_inference_profiles.return_value = response
    with pytest.raises((ValueError, TypeError)):
        read_catalog(client, True, Event())


def test_repeated_token_is_error_and_no_partial_list():
    client = MagicMock()
    client.list_inference_profiles.return_value = {"inferenceProfileSummaries": [], "nextToken": "same"}
    with pytest.raises(ValueError, match="repeated"):
        read_catalog(client, True, Event())
    assert client.list_inference_profiles.call_count == 2


def test_natural_versions_alias_snapshots_and_unknown_names():
    def entry(model, route="us."):
        return CatalogEntry(route+model, model, True, model, route)
    ten = "anthropic.claude-sonnet-5-10-v1:0"
    future = "anthropic.claude-new-family-v1:0"
    rows = [entry(S55), entry(S55DATE), entry(ten), entry(F51), entry(future), entry(S55, "global.")]
    selected = CatalogFilters.parse({"latest_per_family": True}).select(rows)
    assert {e.id for e in selected} == {"us."+ten, "us."+F51, "us."+future, "global."+S55}
    assert [e.model for e in CatalogFilters.parse({"latest_per_family": True}).select([entry(S55DATE), entry(S55)])] == [S55]
    assert len(CatalogFilters.parse({}).select(rows)) == 6
    older = entry("anthropic.claude-3-5-sonnet-20240620-v1:0")
    assert [e.model for e in CatalogFilters.parse({"latest_per_family": True}).select([older, entry(S55)])] == [S55]


@pytest.mark.asyncio
async def test_config_snapshot_fresh_info_bound_connection(monkeypatch):
    config = {"include": ["*SONNET*"]}
    p = provider(model_filters=config)
    p.config["aws_region"] = "eu-west-1"
    config["include"].clear()
    info = p.get_model_list_info()
    assert info["source"] == "provider-api" and info["entitlement_verified"] is False
    assert info["region"] == "us-east-1"
    assert p._catalog_session_kwargs["region_name"] == "us-east-1"
    assert p._catalog_session_kwargs["aws_access_key_id"] == "fixture-key"
    info["default_filters"]["include"].clear()
    info["filters_schema"]["properties"].clear()
    monkeypatch.setattr("amplifier_module_provider_bedrock.fetch_catalog", lambda *args: [CatalogEntry(S45, "name", False, S45, "foundation")])
    assert len(await p.list_models()) == 1  # explicit configured policy did not inherit active/latest defaults
    assert p.get_model_list_info()["filters_schema"]["properties"]
    await p.client.close()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_owned_client(monkeypatch):
    entered, release, closed = Event(), Event(), Event()
    client = MagicMock()
    client.close.side_effect = lambda *args: closed.set()
    def page(**kwargs):
        entered.set()
        assert release.wait(5)
        return {"inferenceProfileSummaries": [], "nextToken": "later"}
    client.list_inference_profiles.side_effect = page
    bind(monkeypatch, client)
    p = provider()
    task = asyncio.create_task(p.list_models())
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(closed.wait, 5)
    assert client.list_inference_profiles.call_count == 1
    await p.client.close()


@pytest.mark.parametrize("invalid", [None, [], {"active_only": 1}, {"latest_per_family": "yes"}, {"unknown": True}])
def test_invalid_config_rejected_before_client_construction(invalid):
    with pytest.raises((TypeError, ValueError)):
        provider(model_filters=invalid)


def test_invalid_invocation_route_is_not_silently_truthy():
    with pytest.raises(ValueError, match="use_cross_region_inference"):
        provider(use_cross_region_inference="false")


@pytest.mark.asyncio
async def test_custom_endpoint_does_not_claim_unrelated_aws_catalog(monkeypatch):
    p = provider()
    p.client.base_url = "https://custom.example.invalid/"
    called = MagicMock()
    monkeypatch.setattr("amplifier_module_provider_bedrock.fetch_catalog", called)
    with pytest.raises(errors.LLMError, match="custom Bedrock runtime endpoint"):
        await p.list_models()
    called.assert_not_called()
    await p.client.close()


@pytest.mark.asyncio
async def test_hidden_exact_inference_uses_native_sdk_route_and_parameters():
    import json
    import importlib
    from anthropic import AsyncAnthropicBedrock, DefaultAsyncHttpxClient
    from amplifier_core.message_models import ChatRequest, Message

    transport_module = "httpx2" if any(
        cls.__module__.startswith("httpx2")
        for cls in DefaultAsyncHttpxClient.__mro__
    ) else "httpx"
    http = importlib.import_module(transport_module)
    seen = []
    def handle(request):
        seen.append(request)
        return http.Response(200, json={
            "id": "fixture", "type": "message", "role": "assistant", "model": S45,
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
    p = provider(model_filters={"include": []}, use_cross_region_inference=False)
    await p.client.close()
    p.client = AsyncAnthropicBedrock(
        aws_region="us-east-1", aws_access_key="fixture-key", aws_secret_key="fixture-secret",
        max_retries=0, http_client=http.AsyncClient(transport=http.MockTransport(handle)),
    )
    try:
        response = await p.complete(ChatRequest(
            model=S45, messages=[Message(role="user", content="test")],
            max_output_tokens=17, temperature=0.25,
        ))
        assert response.content[0].text == "ok"
        assert seen[0].url.path == "/model/"+S45+"/invoke"
        body = json.loads(seen[0].content)
        assert body["max_tokens"] == 17 and body["temperature"] == 0.25
        assert "model_filters" not in body
        assert "Credential=fixture-key/" in seen[0].headers["authorization"]
    finally:
        await p.client.close()
