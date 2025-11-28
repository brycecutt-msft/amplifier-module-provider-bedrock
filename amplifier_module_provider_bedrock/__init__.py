"""AWS Bedrock provider module for Amplifier.

Integrates with AWS Bedrock for Claude models via AWS.
Supports AWS profiles, IAM credentials, streaming, tool calling, extended thinking, and ChatRequest format.
"""

__all__ = ["mount", "BedrockProvider"]

import asyncio
import logging
import os
import time
from typing import Any

from amplifier_core import ConfigField, ModelInfo, ModuleCoordinator, ProviderInfo
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    ImageBlock,
    Message,
    ReasoningBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    Usage,
)
from anthropic import AsyncAnthropicBedrock

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the AWS Bedrock provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including AWS credentials

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Register contribution channel for custom events
    coordinator.register_contributor(
        "observability.events",
        "bedrock",
        lambda: ["provider:tool_sequence_repaired"]
    )

    provider = BedrockProvider(config, coordinator)
    provider_name = config.get("name", "bedrock")
    await coordinator.mount("providers", provider, name=provider_name)
    logger.info(f"Mounted BedrockProvider as '{provider_name}'")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class BedrockProvider:
    """AWS Bedrock API integration.

    Provides Claude models with support for:
    - Text generation
    - Tool calling
    - Extended thinking
    - Streaming responses
    - AWS profile-based authentication
    """

    name = "bedrock"

    def __init__(
        self, config: dict[str, Any] | None = None, coordinator: ModuleCoordinator | None = None
    ):
        """
        Initialize Bedrock provider.

        Args:
            config: Configuration including AWS credentials
            coordinator: Module coordinator for event emission
        """
        self.config = config or {}
        self.coordinator = coordinator

        # AWS configuration
        aws_region = self.config.get("aws_region") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        aws_profile = self.config.get("aws_profile")
        aws_access_key = self.config.get("aws_access_key") or os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = self.config.get("aws_secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = self.config.get("aws_session_token") or os.environ.get("AWS_SESSION_TOKEN")
        
        # Cross-region inference configuration
        # AWS Bedrock Inference Profile mapping based on official documentation
        # https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
        self.use_cross_region_inference = self.config.get("use_cross_region_inference", True)  # Enabled by default
        self.aws_region = aws_region
        self.inference_profile_mapping = [
            # Ordered by pattern length (descending) to ensure more specific patterns match first
            ("ap-southeast-2", "au."),  # Australia (Sydney)
            ("ap-southeast-4", "au."),  # Australia (Melbourne)
            ("ap-northeast-", "jp."),   # Japan regions
            ("us-gov-", "ug."),         # US Government Cloud
            ("us-", "us."),             # Americas regions
            ("eu-", "eu."),             # Europe regions
            ("ap-", "apac."),           # Asia Pacific regions
            ("ca-", "ca."),             # Canada regions
            ("sa-", "sa."),             # South America regions
        ]

        # Build client kwargs
        client_kwargs = {}
        
        # Region (will be auto-inferred if not provided)
        if aws_region:
            client_kwargs["aws_region"] = aws_region
        
        # AWS Profile (if specified, will use credentials from ~/.aws/credentials)
        if aws_profile:
            client_kwargs["aws_profile"] = aws_profile
            logger.info(f"Using AWS profile: {aws_profile}")
        
        # Explicit credentials (take precedence over profile)
        if aws_access_key and aws_secret_key:
            client_kwargs["aws_access_key"] = aws_access_key
            client_kwargs["aws_secret_key"] = aws_secret_key
            if aws_session_token:
                client_kwargs["aws_session_token"] = aws_session_token
            logger.info("Using explicit AWS credentials")
        
        # Initialize AsyncAnthropicBedrock client
        self.client = AsyncAnthropicBedrock(**client_kwargs)

        # Model configuration
        self.default_model = self.config.get("default_model", "anthropic.claude-sonnet-4-5-20250929-v1:0")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        self.priority = self.config.get("priority", 100)
        self.debug = self.config.get("debug", False)
        self.raw_debug = self.config.get("raw_debug", False)
        self.timeout = self.config.get("timeout", 300.0)

        # Log configuration
        logger.info(f"BedrockProvider initialized - region: {self.client.aws_region}, default_model: {self.default_model}")
        if self.use_cross_region_inference:
            prefix = self._get_inference_profile_prefix(self.aws_region)
            if prefix:
                logger.info(f"Cross-region inference enabled - will use prefix '{prefix}' for model IDs in region '{self.aws_region}'")

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="bedrock",
            display_name="AWS Bedrock",
            credential_env_vars=["AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
            capabilities=["streaming", "tools", "thinking", "vision"],
            defaults={
                "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 300.0,
            },
            config_fields=[
                ConfigField(
                    id="aws_profile",
                    display_name="AWS Profile",
                    field_type="text",
                    prompt="Enter AWS profile name (optional, uses default if not specified)",
                    env_var="AWS_PROFILE",
                    required=False,
                ),
                ConfigField(
                    id="aws_region",
                    display_name="AWS Region",
                    field_type="text",
                    prompt="Enter AWS region (e.g., us-east-1)",
                    env_var="AWS_REGION",
                    default="us-east-1",
                    required=False,
                ),
                ConfigField(
                    id="use_cross_region_inference",
                    display_name="Cross-Region Inference",
                    field_type="boolean",
                    prompt="Enable cross-region inference for better availability?",
                    default="true",
                    required=False,
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available Claude models on AWS Bedrock.
        
        Returns hardcoded list since Bedrock doesn't provide dynamic model discovery.
        """
        return [
            ModelInfo(
                id="anthropic.claude-sonnet-4-5-20250929-v1:0",
                display_name="Claude Sonnet 4.5",
                context_window=200000,
                max_output_tokens=16000,
                capabilities=["tools", "vision", "thinking", "streaming"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="anthropic.claude-opus-4-1-20250805-v1:0",
                display_name="Claude Opus 4.1",
                context_window=200000,
                max_output_tokens=32000,
                capabilities=["tools", "vision", "thinking", "streaming"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="anthropic.claude-haiku-4-5-20251001-v1:0",
                display_name="Claude Haiku 4.5",
                context_window=200000,
                max_output_tokens=8192,
                capabilities=["tools", "vision", "streaming", "fast"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
        ]

    def _get_inference_profile_prefix(self, region: str) -> str | None:
        """
        Get the inference profile prefix for a given AWS region.
        
        Args:
            region: AWS region (e.g., "us-east-1")
            
        Returns:
            Inference profile prefix (e.g., "us.") or None if no mapping found
        """
        if not region:
            return None
            
        # Use AWS recommended inference profile prefixes
        # Array is pre-sorted by pattern length (descending) to ensure more specific patterns match first
        for region_pattern, inference_prefix in self.inference_profile_mapping:
            if region.startswith(region_pattern):
                return inference_prefix
        
        return None
    
    def _apply_inference_profile(self, model_id: str) -> str:
        """
        Apply cross-region inference profile prefix to model ID if enabled.
        
        Args:
            model_id: Base model ID (e.g., "anthropic.claude-sonnet-4-5-20250929-v1:0")
            
        Returns:
            Model ID with inference profile prefix if applicable (e.g., "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        """
        # Skip if cross-region inference is disabled
        if not self.use_cross_region_inference:
            return model_id
        
        # Skip if model already has an inference profile prefix
        if any(model_id.startswith(prefix) for _, prefix in self.inference_profile_mapping):
            return model_id
        
        # Skip if model has global inference prefix
        if model_id.startswith("global."):
            return model_id
        
        # Get inference profile prefix for the current region
        prefix = self._get_inference_profile_prefix(self.aws_region)
        if prefix:
            logger.debug(f"Applying inference profile prefix '{prefix}' to model '{model_id}'")
            return f"{prefix}{model_id}"
        
        logger.debug(f"No inference profile prefix found for region '{self.aws_region}'")
        return model_id


    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        # VALIDATE AND REPAIR: Check for missing tool results (backup safety net)
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] Bedrock: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for call_id, _, _ in missing]}"
            )

            # Inject synthetic results
            for call_id, tool_name, _ in missing:
                synthetic = self._create_synthetic_result(call_id, tool_name)
                request.messages.append(synthetic)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name} for call_id, tool_name, _ in missing
                        ],
                    },
                )

        return await self._complete_chat_request(request, **kwargs)

    def _find_missing_tool_results(self, messages: list) -> list[tuple[str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs.

        Returns:
            List of (call_id, tool_name, tool_arguments) tuples for unpaired calls
        """
        tool_calls = {}  # {call_id: (name, args)}
        tool_results = set()  # {call_id}

        for msg in messages:
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (block.name, block.input)

            # Check tool messages for tool_call_id
            elif msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                tool_results.add(msg.tool_call_id)

        return [(call_id, name, args) for call_id, (name, args) in tool_calls.items() if call_id not in tool_results]

    def _create_synthetic_result(self, call_id: str, tool_name: str):
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.
        """
        return Message(
            role="tool",
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.debug(f"Received ChatRequest with {len(request.messages)} messages (debug={self.debug})")

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant", "tool")]

        logger.debug(
            f"Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Combine system messages
        system = (
            "\n\n".join(m.content if isinstance(m.content, str) else "" for m in system_msgs) if system_msgs else None
        )

        if system:
            logger.info(f"[PROVIDER] Combined system message length: {len(system)}")
        else:
            logger.info("[PROVIDER] No system messages")

        # Convert developer messages to XML-wrapped user messages (at top)
        context_user_msgs = []
        for i, dev_msg in enumerate(developer_msgs):
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            content_preview = content[:100] + ("..." if len(content) > 100 else "")
            logger.info(f"[PROVIDER] Converting developer message {i + 1}/{len(developer_msgs)}: length={len(content)}")
            logger.debug(f"[PROVIDER] Developer message preview: {content_preview}")
            wrapped = f"<context_file>\n{content}\n</context_file>"
            context_user_msgs.append({"role": "user", "content": wrapped})

        logger.info(f"[PROVIDER] Created {len(context_user_msgs)} XML-wrapped context messages")

        # Convert conversation messages
        conversation_msgs = self._convert_messages([m.model_dump() for m in conversation])
        logger.info(f"[PROVIDER] Converted {len(conversation_msgs)} conversation messages")

        # Combine: context THEN conversation
        all_messages = context_user_msgs + conversation_msgs
        logger.info(f"[PROVIDER] Final message count for API: {len(all_messages)}")

        # Prepare request parameters
        base_model = kwargs.get("model", self.default_model)
        model_with_profile = self._apply_inference_profile(base_model)
        
        params = {
            "model": model_with_profile,
            "messages": all_messages,
            "max_tokens": request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature or kwargs.get("temperature", self.temperature),
        }

        if system:
            params["system"] = system

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)

        logger.info(
            f"[PROVIDER] Bedrock API call - model: {params['model']}, messages: {len(params['messages'])}, system: {bool(system)}"
        )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "data": {
                        "provider": "bedrock",
                        "model": params["model"],
                        "message_count": len(params["messages"]),
                        "has_system": bool(system),
                    }
                },
            )

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "data": {
                            "provider": "bedrock",
                            "request": {
                                "model": params["model"],
                                "messages": params["messages"],
                                "system": system,
                                "max_tokens": params["max_tokens"],
                                "temperature": params["temperature"],
                            },
                        },
                    },
                )

            # RAW level: Complete params dict as sent to Bedrock API (if debug AND raw_debug enabled)
            if self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": "bedrock",
                        "params": params,  # Complete untruncated params
                    },
                )

        start_time = time.time()

        # Call Bedrock API
        try:
            response = await asyncio.wait_for(self.client.messages.create(**params), timeout=self.timeout)
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from Bedrock API")
            logger.debug(f"[PROVIDER] Response type: {response.model}")

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "bedrock",
                            "model": params["model"],
                            "usage": {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                        },
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Full response (if debug enabled)
                if self.debug:
                    content_preview = str(response.content)[:500] if response.content else ""
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "data": {
                                "provider": "bedrock",
                                "response": {
                                    "content_preview": content_preview,
                                    "stop_reason": response.stop_reason,
                                },
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

                # RAW level: Complete response object from Bedrock API (if debug AND raw_debug enabled)
                if self.debug and self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": "bedrock",
                            "response": response.model_dump(),  # Complete untruncated response
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Bedrock API error: {e}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "bedrock",
                        "model": params["model"],
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
            raise

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from provider response.

        Filters out tool calls with empty/missing arguments to handle
        Anthropic API quirk where empty tool_use blocks are sometimes generated.

        Args:
            response: Provider response

        Returns:
            List of valid tool calls (with non-empty arguments)
        """
        if not response.tool_calls:
            return []

        # Filter out tool calls with empty arguments (Anthropic API quirk)
        # Claude sometimes generates tool_use blocks with empty input {}
        valid_calls = []
        for tc in response.tool_calls:
            # Skip tool calls with no arguments or empty dict
            if not tc.arguments:
                logger.debug(f"Filtering out tool '{tc.tool}' with empty arguments")
                continue
            valid_calls.append(tc)

        if len(valid_calls) < len(response.tool_calls):
            logger.info(f"Filtered {len(response.tool_calls) - len(valid_calls)} tool calls with empty arguments")

        return valid_calls

    def _validate_anthropic_tool_consistency(self, messages: list[dict[str, Any]]) -> None:
        """Validate tool_use/tool_result pairing per Anthropic API requirements.

        Anthropic requires: Each tool_use in message N must have matching tool_result
        in message N+1. This validation catches state corruption bugs before they hit
        the API (defense in depth).

        Args:
            messages: Anthropic-formatted messages to validate

        Raises:
            ValueError: If tool_use/tool_result pairs are inconsistent
        """
        i = 0
        while i < len(messages):
            msg = messages[i]

            # Anthropic format: assistant messages may have content blocks with tool_use
            if msg.get("role") == "assistant":
                content = msg.get("content", [])

                # Check if content is a list with tool_use blocks
                if isinstance(content, list):
                    tool_use_ids = {
                        block.get("id")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "tool_use"
                    }

                    if tool_use_ids:
                        # Next message MUST be user with tool_result blocks
                        if i + 1 >= len(messages):
                            raise ValueError(
                                f"Bedrock API invariant violated: Message {i} has tool_use blocks "
                                f"but no following message. Tool IDs: {tool_use_ids}. "
                                f"Each tool_use MUST have matching tool_result in next message."
                            )

                        next_msg = messages[i + 1]
                        if next_msg.get("role") != "user":
                            raise ValueError(
                                f"Bedrock API invariant violated: Message {i} has tool_use blocks "
                                f"but next message is role='{next_msg.get('role')}' (expected 'user' with tool_results)."
                            )

                        # Extract tool_result IDs from next message
                        next_content = next_msg.get("content", [])
                        if isinstance(next_content, list):
                            result_ids = {
                                block.get("tool_use_id")
                                for block in next_content
                                if isinstance(block, dict) and block.get("type") == "tool_result"
                            }

                            # Validate all tool_use IDs have matching tool_results
                            missing = tool_use_ids - result_ids
                            if missing:
                                raise ValueError(
                                    f"Bedrock API invariant violated: Message {i} has tool_use IDs "
                                    f"without matching tool_result blocks: {missing}. "
                                    f"This indicates a compaction bug or state corruption."
                                )

                            # Warn about orphaned tool_results (possible retry bug)
                            extra = result_ids - tool_use_ids
                            if extra:
                                raise ValueError(
                                    f"Bedrock API invariant violated: Message {i + 1} has tool_result blocks "
                                    f"without matching tool_use: {extra}. "
                                    f"This indicates stale results from a failed retry."
                                )

            i += 1

    def _clean_content_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """Clean a content block for API by removing fields not accepted by Bedrock API.

        Bedrock API may include extra fields (like 'visibility') in responses,
        but does NOT accept these fields when blocks are sent as input in messages.
        Also ensures text fields are valid strings.

        Args:
            block: Raw content block dict (may include visibility, nested text, etc.)

        Returns:
            Cleaned content block dict with only API-accepted fields
        """
        block_type = block.get("type")

        if block_type == "text":
            text_value = block.get("text", "")
            # Handle nested text objects like {"text": {"text": "..."}}
            while isinstance(text_value, dict) and "text" in text_value:
                text_value = text_value["text"]
            # Ensure it's a string
            if not isinstance(text_value, str):
                text_value = str(text_value) if text_value is not None else ""
            return {"type": "text", "text": text_value}
        if block_type == "thinking":
            cleaned = {"type": "thinking", "thinking": block.get("thinking", "")}
            if "signature" in block:
                cleaned["signature"] = block["signature"]
            return cleaned
        if block_type == "reasoning":
            return {
                "type": "reasoning",
                "content": block.get("content", []),
                "summary": block.get("summary", []),
            }
        if block_type == "redacted_thinking":
            return {
                "type": "redacted_thinking",
                "data": block.get("data", ""),
            }
        if block_type == "image":
            return {
                "type": "image",
                "source": block.get("source", {}),
            }
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }
        if block_type == "tool_result":
            content = block.get("content", "")
            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            return {
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id", ""),
                "content": content,
            }
        # Unknown block type - return as-is but remove visibility
        cleaned = dict(block)
        cleaned.pop("visibility", None)
        return cleaned

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format.

        CRITICAL: Anthropic requires ALL tool_result blocks from one assistant's tool_use
        to be batched into a SINGLE user message with multiple tool_result blocks in the
        content array. We cannot send separate user messages for each tool result.

        This method batches consecutive tool messages into one user message.
        """
        anthropic_messages = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled separately)
            if role == "system":
                i += 1
                continue

            # Batch consecutive tool messages into ONE user message
            if role == "tool":
                # Collect all consecutive tool results
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_use_id = tool_msg.get("tool_call_id")
                    if not tool_use_id:
                        logger.warning(f"Tool result missing tool_call_id: {tool_msg}")
                        tool_use_id = "unknown"  # Fallback

                    # Sanitize tool result content - ensure it's a string
                    tool_content = tool_msg.get("content", "")
                    if not isinstance(tool_content, str):
                        tool_content = str(tool_content) if tool_content is not None else ""

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_content,
                        }
                    )
                    i += 1

                # Add ONE user message with ALL tool results
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": tool_results,  # Array of tool_result blocks
                    }
                )
                continue  # i already advanced in while loop
            if role == "assistant":
                # Assistant messages - check for tool calls or thinking blocks
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    content_blocks = []

                    # CRITICAL: Check for thinking block and add it FIRST
                    if "thinking_block" in msg and msg["thinking_block"]:
                        # Clean thinking block (remove visibility field not accepted by API)
                        cleaned_thinking = self._clean_content_block(msg["thinking_block"])
                        content_blocks.append(cleaned_thinking)

                    # Add text content if present
                    if content:
                        if isinstance(content, list):
                            # Content is a list of blocks - extract text blocks only
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    cleaned_block = self._clean_content_block(block)
                                    content_blocks.append(cleaned_block)
                                elif not isinstance(block, dict) and hasattr(block, "type") and block.type == "text":
                                    content_blocks.append({"type": "text", "text": str(getattr(block, "text", ""))})
                        else:
                            # Content is a simple string
                            content_blocks.append({"type": "text", "text": str(content)})

                    # Add tool_use blocks
                    for tc in msg["tool_calls"]:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("tool", ""),
                                "input": tc.get("arguments", {}),
                            }
                        )

                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                elif "thinking_block" in msg and msg["thinking_block"]:
                    # Assistant message with thinking block
                    # Clean thinking block (remove visibility field not accepted by API)
                    cleaned_thinking = self._clean_content_block(msg["thinking_block"])
                    content_blocks = [cleaned_thinking]
                    if content:
                        if isinstance(content, list):
                            # Content is a list of blocks - extract text blocks only
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    cleaned_block = self._clean_content_block(block)
                                    content_blocks.append(cleaned_block)
                                elif not isinstance(block, dict) and hasattr(block, "type") and block.type == "text":
                                    content_blocks.append({"type": "text", "text": str(getattr(block, "text", ""))})
                        else:
                            # Content is a simple string
                            content_blocks.append({"type": "text", "text": str(content)})
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    # Regular assistant message - may have structured content blocks
                    if isinstance(content, list):
                        # Content is a list of blocks - clean each block
                        cleaned_blocks = [self._clean_content_block(block) for block in content]
                        anthropic_messages.append({"role": "assistant", "content": cleaned_blocks})
                    else:
                        # Content is a simple string
                        anthropic_messages.append({"role": "assistant", "content": str(content)})
                i += 1
            elif role == "developer":
                # Developer messages -> XML-wrapped user messages (context files)
                # Ensure content is a string
                content_str = str(content) if not isinstance(content, str) else content
                wrapped = f"<context_file>\n{content_str}\n</context_file>"
                anthropic_messages.append({"role": "user", "content": wrapped})
                i += 1
            else:
                # User messages - ensure content is a string
                content_str = str(content) if not isinstance(content, str) else content
                anthropic_messages.append({"role": "user", "content": content_str})
                i += 1

        return anthropic_messages

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format."""
        anthropic_tools = []

        for tool in tools:
            # Get schema from tool if available, otherwise use empty schema
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": input_schema,
                }
            )

        return anthropic_tools

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to Anthropic format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of Anthropic-formatted tool definitions
        """
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.parameters,
                }
            )
        return anthropic_tools

    def _convert_to_chat_response(self, response: Any) -> ChatResponse:
        """Convert Bedrock response to ChatResponse format.

        Args:
            response: Bedrock API response

        Returns:
            ChatResponse with content blocks
        """

        content_blocks = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "thinking":
                content_blocks.append(
                    ThinkingBlock(
                        thinking=block.thinking,
                        signature=getattr(block, "signature", None),
                        visibility="internal",
                    )
                )
            elif block.type == "reasoning":
                content_blocks.append(
                    ReasoningBlock(
                        content=block.content,
                        summary=block.summary,
                    )
                )
            elif block.type == "redacted_thinking":
                content_blocks.append(
                    RedactedThinkingBlock(
                        data=block.data,
                    )
                )
            elif block.type == "image":
                content_blocks.append(
                    ImageBlock(
                        source=block.source,
                    )
                )
            elif block.type == "tool_use":
                content_blocks.append(ToolCallBlock(id=block.id, name=block.name, input=block.input))
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=response.stop_reason,
        )
