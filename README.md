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

- `anthropic.claude-sonnet-4-5-20250929-v1:0` - Claude Sonnet 4.5 (recommended, default)
- `anthropic.claude-opus-4-1-20250805-v1:0` - Claude Opus 4.1 (most capable)
- `anthropic.claude-haiku-4-5-20251001-v1:0` - Claude Haiku 4.5 (fastest, cheapest)

> **Note:** Model availability varies by AWS region. See [AWS Bedrock Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for the complete list.

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

### Required Permissions

Your AWS credentials need:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`

See [AWS Bedrock IAM](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html).

## Cross-Region Inference

AWS Bedrock's [cross-region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html) is **enabled by default**, routing requests to the nearest available region with capacity.

The provider automatically prefixes model IDs based on your region:

| Region Pattern | Prefix | Example |
|---------------|--------|---------|
| `us-*` | `us.` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `eu-*` | `eu.` | `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `ap-*` | `apac.` | `apac.anthropic.claude-sonnet-4-5-20250929-v1:0` |

You only specify the base model ID—the prefix is automatic.

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
uv sync --extra dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=amplifier_module_provider_bedrock
```

## License

MIT

## Contributing

See the main Amplifier repository for contribution guidelines.
