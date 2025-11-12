# Amplifier AWS Bedrock Provider Module

Claude model integration for Amplifier via AWS Bedrock.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager
- **AWS Account** with Bedrock access
- **AWS Credentials** - See [AWS Authentication](#aws-authentication)

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Purpose

Provides access to Anthropic's Claude models via AWS Bedrock as an LLM provider for Amplifier.

## Contract

**Module Type:** Provider  
**Mount Point:** `providers`  
**Entry Point:** `amplifier_module_provider_bedrock:mount`

## Supported Models

- `anthropic.claude-sonnet-4-5-20250929-v1:0` - Claude Sonnet 4.5 (recommended, default)
- `anthropic.claude-opus-4-1-20250805-v1:0` - Claude Opus 4.1 (most capable)
- `anthropic.claude-haiku-4-5-20251001-v1:0` - Claude Haiku 4.5 (fastest, cheapest)

> **Note:** Model availability varies by AWS region. See [AWS Bedrock Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for the complete list.

## Cross-Region Inference

AWS Bedrock supports [cross-region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html) which routes requests to the nearest available region with capacity. This feature is **enabled by default**.

When enabled, the provider automatically prefixes model IDs with the appropriate inference profile based on your region:

| Region Pattern | Inference Prefix | Example |
|---------------|------------------|---------|
| `us-east-1`, `us-west-2` | `us.` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `eu-west-1`, `eu-central-1` | `eu.` | `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `ap-southeast-1`, `ap-south-1` | `apac.` | `apac.anthropic.claude-sonnet-4-5-20250929-v1:0` |

You only need to specify the base model ID. The inference profile prefix is automatically applied.

To disable cross-region inference:

```toml
config = {
    use_cross_region_inference = false,
    aws_region = "us-east-1",
    default_model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
}
```

## Features

- ✅ **AWS Profile Support** - SSO and standard profiles
- ✅ **Multiple Auth Methods** - Profiles, environment variables, IAM roles
- ✅ **Extended Thinking** - Claude's internal reasoning (when supported)
- ✅ **Tool Calling** - Function calling capabilities
- ✅ **Streaming Support** - Real-time response streaming
- ✅ **Message Validation** - Defense-in-depth validation before API calls
- ✅ **Token Management** - Usage tracking and limits
- ✅ **Cross-Region Inference** - Automatic routing for best availability

## Message Validation

Before sending messages to Bedrock API, the provider validates tool_use/tool_result consistency:

- Each `tool_use` must have a corresponding `tool_result` in the next message
- Prevents API errors from malformed message sequences
- Provides actionable error messages for debugging

## Configuration

```toml
[[providers]]
module = "provider-bedrock"
name = "bedrock"
config = {
    aws_profile = "my-profile",  # Optional: AWS profile name
    aws_region = "us-east-1",    # Optional: defaults to profile/environment
    default_model = "anthropic.claude-sonnet-4-5-20250929-v1:0",
    use_cross_region_inference = true,  # Default: true
    max_tokens = 8192,
    temperature = 1.0,
    debug = false,      # Enable standard debug events
    raw_debug = false   # Enable ultra-verbose raw API I/O logging
}
```

### Debug Configuration

**Standard Debug** (`debug: true`):
- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with message counts, model info, usage stats
- Moderate log volume, suitable for development

**Raw Debug** (`debug: true, raw_debug: true`):
- Emits `llm:request:raw` and `llm:response:raw` events
- Contains complete, unmodified request params and response objects
- Extreme log volume, use only for deep provider integration debugging
- Captures the exact data sent to/from Bedrock API before any processing

**Example**:
```toml
[[providers]]
module = "provider-bedrock"
name = "bedrock"
config = {
    aws_profile = "my-profile",
    debug = true,      # Enable debug events
    raw_debug = true,  # Enable raw API I/O capture
    default_model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
}
```

## AWS Authentication

**Recommended: AWS IAM Identity Center (SSO)**

The most secure way to authenticate is using [AWS IAM Identity Center (SSO)](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html):

```bash
aws configure sso
```

This provides secure, temporary credentials without managing long-term access keys.

**Alternative Authentication Methods:**

The provider uses standard AWS credential resolution:
- AWS profiles (SSO or standard)
- Environment variables (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, etc.)
- IAM roles (EC2, ECS, Lambda, etc.)

For details, see [AWS CLI Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).

## AWS Permissions

Your AWS credentials need Bedrock permissions. See [AWS Bedrock IAM](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html) for details.

Minimum required actions:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`

## Differences from Direct Anthropic Provider

- **Model IDs**: Bedrock uses AWS-specific model IDs (e.g., `anthropic.claude-sonnet-4-5-20250929-v1:0`)
- **Authentication**: Uses AWS credentials instead of Anthropic API key
- **Regions**: Must specify AWS region
- **Pricing**: Billed through AWS, may differ from direct Anthropic API
- **Rate Limits**: Subject to AWS Bedrock quotas

## Dependencies

- `amplifier-core>=1.0.0`
- `anthropic[bedrock]>=0.72.0`
- `boto3>=1.40.0`
- `botocore>=1.40.0`

## Troubleshooting

### "No AWS region found"
- Set `aws_region` in config, or
- Configure region in your [AWS profile](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html), or
- Set `AWS_REGION` environment variable

### "Access Denied"
- Verify your AWS credentials have [Bedrock permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html)
- Ensure model access is enabled in [AWS Bedrock console](https://console.aws.amazon.com/bedrock/)

### "Profile not found"
- Verify profile exists in your [AWS configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)
- Check profile name spelling (case-sensitive)

### "Model not found"
- Request model access in [AWS Bedrock console](https://console.aws.amazon.com/bedrock/)
- Verify model ID is correct for your region
- See [supported models documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)

## Development

```bash
# Clone the repository
git clone https://github.com/brycecutt-msft/amplifier-module-provider-bedrock.git
cd amplifier-module-provider-bedrock

# Install dependencies
uv sync

# Run tests
uv run pytest
```

## License

MIT

## Contributing

See the main Amplifier repository for contribution guidelines.
