"""
Tests for auto-continuation support.

Verifies that BedrockProvider handles truncated responses transparently
by automatically continuing the conversation until complete.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_module_provider_bedrock import BedrockProvider


class TestDetectIncompleteResponses:
    """Test detection of truncated responses."""

    @pytest.mark.asyncio
    async def test_detects_max_tokens_stop_reason(self):
        """Recognizes stop_reason='max_tokens' as incomplete."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        # First response: incomplete (max_tokens)
        incomplete_response = MagicMock()
        incomplete_response.content = [MagicMock(type="text", text="First part")]
        incomplete_response.stop_reason = "max_tokens"
        incomplete_response.usage = MagicMock(input_tokens=10, output_tokens=100)
        
        # Second response: complete
        complete_response = MagicMock()
        complete_response.content = [MagicMock(type="text", text="Second part")]
        complete_response.stop_reason = "end_turn"
        complete_response.usage = MagicMock(input_tokens=110, output_tokens=50)
        
        # Mock API to return incomplete then complete
        provider.client.messages.create = AsyncMock(
            side_effect=[incomplete_response, complete_response]
        )
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        # Should have made 2 API calls (initial + continuation)
        assert provider.client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_detects_length_stop_reason(self):
        """Recognizes stop_reason='length' as incomplete (alternative naming)."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        incomplete_response = MagicMock()
        incomplete_response.content = [MagicMock(type="text", text="Truncated")]
        incomplete_response.stop_reason = "length"  # Alternative naming
        incomplete_response.usage = MagicMock(input_tokens=10, output_tokens=100)
        
        complete_response = MagicMock()
        complete_response.content = [MagicMock(type="text", text="Complete")]
        complete_response.stop_reason = "end_turn"
        complete_response.usage = MagicMock(input_tokens=110, output_tokens=50)
        
        provider.client.messages.create = AsyncMock(
            side_effect=[incomplete_response, complete_response]
        )
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        assert provider.client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_ignores_complete_responses(self):
        """Doesn't continue on stop_reason='end_turn'."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        complete_response = MagicMock()
        complete_response.content = [MagicMock(type="text", text="Complete")]
        complete_response.stop_reason = "end_turn"
        complete_response.usage = MagicMock(input_tokens=10, output_tokens=50)
        
        provider.client.messages.create = AsyncMock(return_value=complete_response)
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        # Should only make 1 API call (no continuation needed)
        assert provider.client.messages.create.call_count == 1


class TestContinuationLoop:
    """Test continuation loop mechanics."""

    @pytest.mark.asyncio
    async def test_accumulates_output(self):
        """Combines content from multiple continuation calls."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        # Three responses building up content
        resp1 = MagicMock()
        resp1.content = [MagicMock(type="text", text="Part 1")]
        resp1.stop_reason = "max_tokens"
        resp1.usage = MagicMock(input_tokens=10, output_tokens=50)
        
        resp2 = MagicMock()
        resp2.content = [MagicMock(type="text", text=" Part 2")]
        resp2.stop_reason = "max_tokens"
        resp2.usage = MagicMock(input_tokens=60, output_tokens=50)
        
        resp3 = MagicMock()
        resp3.content = [MagicMock(type="text", text=" Part 3")]
        resp3.stop_reason = "end_turn"
        resp3.usage = MagicMock(input_tokens=110, output_tokens=30)
        
        provider.client.messages.create = AsyncMock(
            side_effect=[resp1, resp2, resp3]
        )
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        # Should have accumulated all 3 parts
        assert len(response.content) == 3
        assert response.content[0].text == "Part 1"
        assert response.content[1].text == " Part 2"
        assert response.content[2].text == " Part 3"

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self):
        """Stops after MAX_CONTINUATION_ATTEMPTS even if still incomplete."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        # Always return incomplete
        incomplete_response = MagicMock()
        incomplete_response.content = [MagicMock(type="text", text="Chunk")]
        incomplete_response.stop_reason = "max_tokens"
        incomplete_response.usage = MagicMock(input_tokens=10, output_tokens=50)
        
        provider.client.messages.create = AsyncMock(return_value=incomplete_response)
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        # Should stop at max iterations (default 5)
        # Initial + 5 continuations = 6 total calls
        assert provider.client.messages.create.call_count <= 6

    @pytest.mark.asyncio
    async def test_logs_continuation_event(self):
        """Emits provider:continuation event for observability."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        incomplete = MagicMock()
        incomplete.content = [MagicMock(type="text", text="Part 1")]
        incomplete.stop_reason = "max_tokens"
        incomplete.usage = MagicMock(input_tokens=10, output_tokens=50)
        
        complete = MagicMock()
        complete.content = [MagicMock(type="text", text="Part 2")]
        complete.stop_reason = "end_turn"
        complete.usage = MagicMock(input_tokens=60, output_tokens=30)
        
        provider.client.messages.create = AsyncMock(
            side_effect=[incomplete, complete]
        )
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        await provider.complete(request)
        
        # Should emit continuation event
        emitted_events = [call[0][0] for call in coordinator.hooks.emit.call_args_list]
        assert "provider:continuation" in emitted_events

    @pytest.mark.asyncio
    async def test_preserves_usage_tokens(self):
        """Sums usage tokens across all continuations."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        resp1 = MagicMock()
        resp1.content = [MagicMock(type="text", text="A")]
        resp1.stop_reason = "max_tokens"
        resp1.usage = MagicMock(input_tokens=100, output_tokens=50)
        
        resp2 = MagicMock()
        resp2.content = [MagicMock(type="text", text="B")]
        resp2.stop_reason = "end_turn"
        resp2.usage = MagicMock(input_tokens=150, output_tokens=30)
        
        provider.client.messages.create = AsyncMock(
            side_effect=[resp1, resp2]
        )
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        # Should sum usage from both calls
        # First: 100 input, 50 output
        # Second: 150 input, 30 output
        # Total: 250 input, 80 output
        assert response.usage.input_tokens == 250
        assert response.usage.output_tokens == 80
        assert response.usage.total_tokens == 330

    @pytest.mark.asyncio
    async def test_continuation_adds_to_messages(self):
        """Continuation call includes accumulated content in messages."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        incomplete = MagicMock()
        incomplete.content = [MagicMock(type="text", text="First")]
        incomplete.stop_reason = "max_tokens"
        incomplete.usage = MagicMock(input_tokens=10, output_tokens=50)
        
        complete = MagicMock()
        complete.content = [MagicMock(type="text", text="Second")]
        complete.stop_reason = "end_turn"
        complete.usage = MagicMock(input_tokens=60, output_tokens=30)
        
        provider.client.messages.create = AsyncMock(
            side_effect=[incomplete, complete]
        )
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        await provider.complete(request)
        
        # Second call should include assistant message with first response
        second_call_params = provider.client.messages.create.call_args_list[1][1]
        messages = second_call_params["messages"]
        
        # Should have: user message + assistant message (from first response)
        assert len(messages) >= 2
        assert any(m["role"] == "assistant" for m in messages)


class TestToolCallContinuationSafety:
    """Test that continuation doesn't violate tool_use/tool_result invariants."""

    @pytest.mark.asyncio
    async def test_stops_continuation_when_response_ends_with_tool_use(self):
        """Don't continue if response ENDS WITH tool_use block (incomplete tool sequence)."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        # Response truncated mid-tool-call sequence
        # Contains tool_use but stop_reason is max_tokens
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me call some tools"
        
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_test_123"
        tool_block.name = "search"
        tool_block.input = {"query": "test"}
        
        incomplete_with_tools = MagicMock()
        incomplete_with_tools.content = [text_block, tool_block]
        incomplete_with_tools.stop_reason = "max_tokens"
        incomplete_with_tools.usage = MagicMock(input_tokens=10, output_tokens=100)
        
        provider.client.messages.create = AsyncMock(return_value=incomplete_with_tools)
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        # Should NOT continue - would create invalid tool sequence
        # Only 1 API call should be made
        assert provider.client.messages.create.call_count == 1
        
        # Response should contain the partial tool_use blocks
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "toolu_test_123"
        
        # Should have stop_reason indicating truncation
        assert response.finish_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_continues_when_response_ends_with_text_after_tool_use(self):
        """Continue normally if response has tool_use but ENDS WITH text (complete tool sequence)."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        # Response has tool_use but ends with text (safe to continue)
        text_block1 = MagicMock()
        text_block1.type = "text"
        text_block1.text = "Let me search"
        
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_test_123"
        tool_block.name = "search"
        tool_block.input = {"query": "test"}
        
        text_block2 = MagicMock()
        text_block2.type = "text"
        text_block2.text = "Let me also check..."
        
        incomplete_mixed = MagicMock()
        incomplete_mixed.content = [text_block1, tool_block, text_block2]
        incomplete_mixed.stop_reason = "max_tokens"
        incomplete_mixed.usage = MagicMock(input_tokens=10, output_tokens=100)
        
        complete_text = MagicMock()
        complete_text.content = [MagicMock(type="text", text="continuation")]
        complete_text.stop_reason = "end_turn"
        complete_text.usage = MagicMock(input_tokens=113, output_tokens=50)
        
        provider.client.messages.create = AsyncMock(
            side_effect=[incomplete_mixed, complete_text]
        )
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        # Should continue - last block is text (safe)
        assert provider.client.messages.create.call_count == 2
        
        # Should accumulate all parts
        assert len(response.content) == 4

    @pytest.mark.asyncio
    async def test_continues_normally_for_text_only_truncation(self):
        """Continue normally if response only has text (no tool_use blocks)."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        # Text-only truncation (safe to continue)
        incomplete_text = MagicMock()
        incomplete_text.content = [MagicMock(type="text", text="Part 1")]
        incomplete_text.stop_reason = "max_tokens"
        incomplete_text.usage = MagicMock(input_tokens=10, output_tokens=100)
        
        complete_text = MagicMock()
        complete_text.content = [MagicMock(type="text", text="Part 2")]
        complete_text.stop_reason = "end_turn"
        complete_text.usage = MagicMock(input_tokens=110, output_tokens=50)
        
        provider.client.messages.create = AsyncMock(
            side_effect=[incomplete_text, complete_text]
        )
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        response = await provider.complete(request)
        
        # Should continue normally for text-only truncation
        assert provider.client.messages.create.call_count == 2
        
        # Should accumulate both parts
        assert len(response.content) == 2

    @pytest.mark.asyncio
    async def test_stops_when_only_tool_use_in_truncated_response(self):
        """Stop continuation when response contains ONLY tool_use blocks (most severe case)."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        # Response with ONLY tool_use block (truncated before any text)
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_test_456"
        tool_block.name = "calculator"
        tool_block.input = {"expression": "2+2"}
        
        incomplete_with_tools = MagicMock()
        incomplete_with_tools.content = [tool_block]
        incomplete_with_tools.stop_reason = "max_tokens"
        incomplete_with_tools.usage = MagicMock(input_tokens=10, output_tokens=100)
        
        provider.client.messages.create = AsyncMock(return_value=incomplete_with_tools)
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        
        # Should not raise exception and should stop after 1 call
        await provider.complete(request)
        assert provider.client.messages.create.call_count == 1
