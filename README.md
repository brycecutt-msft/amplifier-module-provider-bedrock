# Amplifier AWS Bedrock Provider Module

Access Anthropic's Claude models via AWS Bedrock as an AI provider for Amplifier.

## Prerequisites

- **[Amplifier](https://github.com/microsoft/amplifier)** - Install with `uv tool install git+https://github.com/microsoft/amplifier`
  - See [Amplifier Installation Guide](https://github.com/microsoft/amplifier#quick-start---zero-to-working-in-90-seconds) for full setup
- **AWS Account** with [Bedrock access](https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html)
- **AWS Credentials** - See [AWS Authentication](#aws-authentication)

## Installation

```bash
amplifier module add provider-bedrock --source git+https://github.com/brycecutt-msft/amplifier-module-provider-bedrock

amplifier provider use bedrock
```

The interactive setup will prompt for:
- **AWS Profile** - Your AWS profile name (or press Enter to use default)
- **AWS Region** - AWS region (default: us-east-1)
- **Cross-Region Inference** - Enable for better availability (recommended: yes)

> **Note:** By default, this installs for your current project. Use `--global`, `--project`, or `--local` flags to control scope. See [Amplifier Configuration](https://github.com/microsoft/amplifier#configuration) for details.

## Quick Test

```bash
amplifier run "What is 2+2?"
```

## Supported Models

`await provider.list_models()` discovers actual AWS targets for the resolved
region and selected invocation route. It does not use the direct Anthropic
catalog or invent regional prefixes:

- Cross-region enabled (default): all pages of `ListInferenceProfiles`, keeping
  profiles whose targets are Claude models. System profile IDs are byte-exact;
  application profile ARNs retain each distinct configured route.
- Cross-region disabled: `ListFoundationModels` with the supported native
  Anthropic/TEXT/ON_DEMAND filters, keeping text-input Claude models.

Default recommendation filters are `active_only: true` and
`latest_per_family: true`: latest numeric Opus/Sonnet/Haiku/Fable version per
system route, alias before a matching snapshot, newest snapshot otherwise.
Distinct application profiles and unrecognized Claude naming schemes are
retained conservatively rather than guessed away. Different tiers are not ranked
by intelligence or price.

`get_model_list_info()` reports `source: "provider-api"`, region, invocation route,
and the full v1 filter schema. Listing is **not proof of InvokeModel entitlement**.
AWS does not publish token limits or all capabilities in these catalog responses;
returned limits are `0` (unknown), with no invented capability tags.

`config.model_filters` is snapshotted at construction. `filters=None` uses it;
an explicit mapping replaces it. `filters={}` disables all recommendation filters
(not Claude/route compatibility), `filters={"include":[]}` returns none, and
`filters={"include":["*SONNET*"],"exclude":["global.*"]}` uses case-insensitive,
OS-independent globs. Exclusion wins and original returned IDs are unchanged.
Common `hide_dated` suppresses ISO YYYY-MM-DD alias snapshots; Bedrock's compact
dates and version suffixes are handled by `latest_per_family`. Neither policy
silently applies to explicit custom/unfiltered requests.

Discovery failures raise sanitized typed errors, never a cached/static list.
Custom runtime endpoints are not treated as AWS inventory: listing fails
explicitly rather than enumerating an unrelated standard endpoint. Control-plane
requests ignore ambient endpoint URL overrides and use the selected AWS region.
Control-plane request connect/read waits are bounded at 5/10 seconds, without
retries. These bounds do not cover credential resolution (for example,
`credential_process` or an MFA prompt) and are not a total elapsed deadline.
Cancellation returns to the caller and stops subsequent pages, but cannot
interrupt synchronous boto3 work. Its worker thread may linger until credential
lookup or an in-flight request returns. Any created client closes when that work
settles, not necessarily when cancellation returns. No inference call is made.
Local missing/invalid profiles and expired or missing SSO tokens produce
non-retryable configuration/authentication errors, not a retryable outage.
Existing exact selections and the configured inference default are not changed
or restricted by recommendations.

## Features

- ✅ **Multiple AWS Auth Methods** - SSO, profiles, environment variables, IAM roles
- ✅ **Extended Thinking** - Claude's internal reasoning (when supported by model)
- ✅ **Tool Calling** - Function calling with automatic validation
- ✅ **Streaming** - Real-time response streaming
- ✅ **Cross-Region Inference** - Automatic routing for best availability
- ✅ **Usage Tracking** - Token counting and limits

## Advanced Configuration

You can customize the provider in your Amplifier profile or settings:

```yaml
# .amplifier/settings.yaml
config:
  providers:
  - module: provider-bedrock
    config:
      aws_profile: my-profile           # Optional: AWS profile name
      aws_region: us-east-1             # Optional: defaults to profile/environment
      default_model: anthropic.claude-sonnet-4-5-20250929-v1:0
      use_cross_region_inference: true  # Default: true
      max_tokens: 8192
      temperature: 1.0
      debug: false                      # Enable debug logging
```

## AWS Authentication

### Recommended: AWS SSO

```bash
aws configure sso
aws sso login --profile your-profile
```

### Alternative Methods

The provider supports all standard AWS authentication:
- AWS profiles (SSO or access keys)
- Environment variables (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- IAM roles (EC2, ECS, Lambda)

See [AWS CLI Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).

Explicit `aws_profile` configuration wins. Otherwise `AWS_DEFAULT_PROFILE`
takes precedence over `AWS_PROFILE`, matching botocore. The resolved selection
is shared by inference and discovery.

### Required Permissions

Your AWS credentials need:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`
- `bedrock:ListInferenceProfiles` for cross-region catalog discovery, or
  `bedrock:ListFoundationModels` for direct on-demand catalog discovery

See [AWS Bedrock IAM](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html).

## Cross-Region Inference

AWS Bedrock's [cross-region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html) is **enabled by default**, routing requests to the nearest available region with capacity.

The provider automatically prefixes model IDs based on your region:

| Region Pattern | Prefix | Example |
|---------------|--------|---------|
| `us-*` | `us.` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `eu-*` | `eu.` | `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `ap-*` | `apac.` | `apac.anthropic.claude-sonnet-4-5-20250929-v1:0` |

Legacy explicit base-model selections still receive this automatic prefix.
Catalog selections instead use actual returned profile IDs/ARNs unchanged; an
existing inference profile prefix or ARN is never prefixed again.

## Differences from Anthropic Provider

- **Model IDs**: AWS-specific (e.g., `anthropic.claude-sonnet-4-5-20250929-v1:0`)
- **Authentication**: AWS credentials instead of Anthropic API key
- **Billing**: Through AWS, pricing may differ
- **Rate Limits**: AWS Bedrock quotas apply

## Updating

```bash
# Update to latest version
amplifier module update

# Or reinstall specific version
amplifier module remove provider-bedrock
amplifier module add provider-bedrock --source git+https://...@v1.1.0 --local
```

## Troubleshooting

### "Token has expired and refresh failed"
Your AWS SSO token expired:
```bash
aws sso login --profile your-profile
```

### "No module named 'anthropic'"
You installed from a local file source. Use git source instead:
```bash
amplifier module remove provider-bedrock
amplifier module add provider-bedrock --source git+https://...@main --local
```

### "Access Denied"
- Check [Bedrock permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html)
- Enable model access in [AWS Bedrock console](https://console.aws.amazon.com/bedrock/)

### "Model not found"
- Request access in [AWS Bedrock console](https://console.aws.amazon.com/bedrock/)
- Check [model availability](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) in your region

## Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/brycecutt-msft/amplifier-module-provider-bedrock.git
cd amplifier-module-provider-bedrock

# Create feature branch
git checkout -b feat/my-feature

# Make changes, commit, and push
git add .
git commit -m "feat: add new feature"
git push origin feat/my-feature

# Test your changes
amplifier module remove provider-bedrock
amplifier module add provider-bedrock \
  --source git+https://github.com/brycecutt-msft/amplifier-module-provider-bedrock@feat/my-feature \
  --local

amplifier run "test query"
```

### Testing

```bash
# Install dev dependencies
uv sync --locked --dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=amplifier_module_provider_bedrock
```

## License

MIT

## Contributing

See the main Amplifier repository for contribution guidelines.

### Bounded output and completion waits

Healthy requests have no default elapsed/read deadline. Explicit `timeout`
configuration remains supported, with a five-second connection/pool bound.
Cancellation still propagates. Slow model work must not be canceled by a default
deadline. Transport phase bounds do not bound credential lookup or a total
elapsed request. A stalled read may wait until the caller cancels or an explicit
timeout expires; this change does not introduce a default generation deadline.

An explicit nonempty `model=` keyword overrides `ChatRequest.model`; both take
precedence over the configured default. `request_options` may supply the same
override, with explicit keywords taking precedence over the mapping. Listing
filters do not restrict exact inference selection.

The adapter preserves its selected `temperature` in the AWS Claude Messages
JSON body via the SDK's documented `extra_body` extension. This works with older
SDKs and newer SDKs that no longer accept a named `temperature` keyword; it does
not remove, clamp, or retry without that value. The target AWS model still
determines which inference parameters it supports. Service rejection remains
an error, not permission to silently drop an explicitly selected parameter.

`auto_continue` defaults to `true`. Set it to `false` in configuration or pass
`request_options={"auto_continue": False}` to `complete()` to return a truncated
response without extra generation calls. Partial content, finish reason and usage
remain available; callers must not commit a truncated result as a complete
summary. The optional capability is `completion:auto_continue:v1`. This option is
consumed locally, never sent to the service, and does not impose a time limit.

`raw: true` enables opt-in request capture on `llm:request.raw`, independently of
debug logging. Redaction is best-effort and key-based: known sensitive keys are
redacted, but secrets embedded in message text or other strings are not. Treat
captured payloads as sensitive. It defaults to false; previous debug/raw_debug
behavior remains supported and may emit unredacted payloads.
