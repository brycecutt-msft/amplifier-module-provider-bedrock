"""
Tests for auto-continuation support.

Verifies that BedrockProvider handles truncated responses transparently
by automatically continuing the conversation until complete.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
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
