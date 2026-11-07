"""AWS control-plane discovery for the adapter's selected invocation route.

Listing is not an InvokeModel entitlement check. Never borrow Anthropic's catalog
or invent profile IDs, model limits, tool support, or account authorization.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError, ClientError, ConfigParseError, InvalidConfigError,
    NoCredentialsError, PartialCredentialsError, ProfileNotFound,
    SSOTokenLoadError, UnauthorizedSSOTokenError,
)
from amplifier_core import ModelInfo
from amplifier_core import llm_errors as errors

from ._model_listing import ModelFilters, selector_schema


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    name: str
    active: bool
    model: str
    route: str


@dataclass(frozen=True)
class CatalogFilters:
    common: ModelFilters
    active_only: bool = False
    latest_per_family: bool = False

    @classmethod
    def parse(cls, value):
        if not isinstance(value, Mapping):
            raise TypeError("model filters must be a mapping")
        specific = {key: value.get(key, False) for key in ("active_only", "latest_per_family")}
        if any(type(flag) is not bool for flag in specific.values()):
            raise TypeError("active_only and latest_per_family must be booleans")
        return cls(ModelFilters.parse({key: val for key, val in value.items() if key not in specific}), **specific)

    def as_dict(self):
        return {**self.common.as_dict(), "active_only": self.active_only, "latest_per_family": self.latest_per_family}

    def select(self, entries):
        ids = set(self.common.select([entry.id for entry in entries]))
        selected = [entry for entry in entries if entry.id in ids and (entry.active or not self.active_only)]
        if not self.latest_per_family:
            return selected
        groups = {}
        keep = []
        for entry in selected:
            parsed = _family_version(entry.model)
            # Do not guess at unknown naming schemes or collapse application profiles.
            if parsed is None or entry.route.startswith("application:"):
                keep.append(entry)
                continue
            family, version, date, revision = parsed
            key = (entry.route, family)
            # Alias wins over its dated snapshot; newest snapshot otherwise.
            rank = (version, not bool(date), date, revision)
            previous = groups.get(key)
            if previous is None or rank > previous[0]:
                groups[key] = (rank, [entry])
            elif rank == previous[0]:
                previous[1].append(entry)
        return sorted(keep + [entry for _, rows in groups.values() for entry in rows], key=lambda entry: entry.id)


def _family_version(model):
    name = model.removeprefix("anthropic.claude-")
    match = re.fullmatch(r"(?:(opus|sonnet|haiku|fable)-(\d+(?:-\d{1,2})*)|(\d+(?:-\d{1,2})*)-(opus|sonnet|haiku|fable))(?:-(\d{8}))?-v(\d+):(\d+)", name)
    if not match:
        return None
    return (match[1] or match[4], tuple(map(int, (match[2] or match[3]).split("-"))),
            match[5] or "", (int(match[6]), int(match[7])))


def catalog_schema():
    schema = selector_schema()
    schema["properties"].update({
        "active_only": {"type": "boolean", "default": False, "description": "Recommend ACTIVE foundation models/profiles only."},
        "latest_per_family": {"type": "boolean", "default": False,
                              "description": "Latest numeric Claude family version per system route; retain distinct application profiles and unknown names."},
    })
    return schema


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid catalog string")
    return value


def _rows(response, key):
    if not isinstance(response, dict) or not isinstance(response.get(key), list):
        raise ValueError("invalid catalog envelope")
    if any(not isinstance(row, dict) for row in response[key]):
        raise ValueError("invalid catalog row")
    return response[key]


def read_catalog(client, cross_region, cancelled):
    """Validate whole responses; genuine [] differs from malformed or denied."""
    entries = []
    if not cross_region:
        response = client.list_foundation_models(byProvider="Anthropic", byOutputModality="TEXT", byInferenceType="ON_DEMAND")
        for row in _rows(response, "modelSummaries"):
            model = _text(row.get("modelId"))
            name = _text(row.get("modelName"))
            lifecycle = row.get("modelLifecycle")
            if not isinstance(lifecycle, dict):
                raise ValueError("missing model lifecycle")
            state = _text(lifecycle.get("status"))
            for key in ("inputModalities", "outputModalities", "inferenceTypesSupported"):
                if not isinstance(row.get(key), list):
                    raise ValueError("missing model modalities")
            if (model.startswith("anthropic.claude-") and "TEXT" in row["inputModalities"]
                    and "TEXT" in row["outputModalities"] and "ON_DEMAND" in row["inferenceTypesSupported"]):
                entries.append(CatalogEntry(model, name, state == "ACTIVE", model, "foundation"))
    else:
        tokens = set()
        params = {}
        while not cancelled.is_set():
            response = client.list_inference_profiles(**params)
            for row in _rows(response, "inferenceProfileSummaries"):
                identity = _text(row.get("inferenceProfileId"))
                arn = _text(row.get("inferenceProfileArn"))
                name = _text(row.get("inferenceProfileName"))
                status = _text(row.get("status"))
                kind = _text(row.get("type"))
                models = _rows(row, "models")
                if not models:
                    raise ValueError("empty profile targets")
                targets = {_text(item.get("modelArn")).rsplit("/", 1)[-1] for item in models}
                if not all(target.startswith("anthropic.claude-") for target in targets):
                    continue  # This adapter speaks Claude Messages, not all of Bedrock.
                if kind == "APPLICATION":
                    # ARN is a distinct, stable invocation target even for arbitrary aliases.
                    model = sorted(targets)[0]
                    route = "application:" + arn
                    identity = arn
                elif kind == "SYSTEM_DEFINED":
                    if len(targets) != 1:
                        raise ValueError("ambiguous system profile")
                    model = next(iter(targets))
                    route = identity[:-len(model)] if identity.endswith(model) else identity
                else:
                    raise ValueError("unknown profile type")
                entries.append(CatalogEntry(identity, name, status == "ACTIVE", model, route))
            token = response.get("nextToken")
            if token is None:
                break
            _text(token)
            if token in tokens:
                raise ValueError("repeated catalog pagination token")
            tokens.add(token)
            params = {"nextToken": token}
    unique = {}
    for entry in entries:
        if entry.id in unique and unique[entry.id] != entry:
            raise ValueError("conflicting duplicate catalog identity")
        unique[entry.id] = entry
    return list(unique.values())


def fetch_catalog(session_kwargs, cross_region, cancelled: Event):
    """Own/close this control client. Cancellation stops subsequent pages.

    The request's connect/read waits are bounded, not credential resolution
    (for example credential_process or MFA). Cancellation cannot interrupt that
    synchronous work; the thread may linger until it returns. An existing client
    is closed in finally when work settles, not necessarily when the caller cancels.
    """
    try:
        session = boto3.Session(**session_kwargs)
        client = session.client("bedrock", config=Config(
            connect_timeout=5, read_timeout=10, retries={"total_max_attempts": 1},
            ignore_configured_endpoint_urls=True,
        ))
        try:
            if cancelled.is_set():
                return []
            return read_catalog(client, cross_region, cancelled)
        finally:
            client.close()
    except (NoCredentialsError, PartialCredentialsError):
        raise errors.AuthenticationError("AWS credentials unavailable for Bedrock model listing", provider="bedrock") from None
    except (UnauthorizedSSOTokenError, SSOTokenLoadError):
        raise errors.AuthenticationError(
            "AWS SSO credentials unavailable for Bedrock model listing; sign in again",
            provider="bedrock", retryable=False,
        ) from None
    except (ProfileNotFound, ConfigParseError, InvalidConfigError):
        raise errors.ConfigurationError(
            "AWS profile or configuration invalid for Bedrock model listing",
            provider="bedrock", retryable=False,
        ) from None
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        error = {
            "AccessDeniedException": errors.AccessDeniedError,
            "UnrecognizedClientException": errors.AuthenticationError,
            "ExpiredTokenException": errors.AuthenticationError,
            "ThrottlingException": errors.RateLimitError,
            "InternalServerException": errors.ProviderUnavailableError,
        }.get(code, errors.LLMError)
        raise error("Bedrock model listing failed", provider="bedrock") from None
    except BotoCoreError:
        raise errors.ProviderUnavailableError("Bedrock model listing transport failed", provider="bedrock") from None
    except (TypeError, ValueError, KeyError):
        raise errors.LLMError("Invalid Bedrock model catalog", provider="bedrock") from None


def model_info(entry):
    # AWS catalog APIs do not publish context/output token limits. Core's required
    # integer fields use zero for unknown, matching the canonical listing contract.
    return ModelInfo(id=entry.id, display_name=entry.name, context_window=0,
                     max_output_tokens=0, capabilities=[], defaults={})
