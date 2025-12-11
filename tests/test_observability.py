"""
Tests for observability compliance.

Verifies that BedrockProvider implements complete observability patterns:
- Three-tier debug logging (info, debug, raw)
- Contribution channel registration for custom events
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from amplifier_module_provider_bedrock import BedrockProvider


class TestRawDebugEvents:
    """Test raw debug event emission (llm:request:raw, llm:response:raw)."""

    @pytest.mark.asyncio
    async def test_raw_debug_emits_request_event(self):
        """Emits llm:request:raw when raw_debug=true."""
        # Setup mock coordinator
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        # Create provider with raw_debug enabled
        provider = BedrockProvider(
            config={
                "aws_region": "us-east-1",
                "debug": True,
                "raw_debug": True,
            },
            coordinator=coordinator
        )
        
        # Mock the Bedrock API response
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        provider.client.messages.create = AsyncMock(return_value=mock_response)
        
        # Make a request
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        await provider.complete(request)
        
        # Verify llm:request:raw was emitted
        emitted_events = [call[0][0] for call in coordinator.hooks.emit.call_args_list]
        assert "llm:request:raw" in emitted_events, "Should emit llm:request:raw event"

    @pytest.mark.asyncio
    async def test_raw_debug_emits_response_event(self):
        """Emits llm:response:raw when raw_debug=true."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={
                "aws_region": "us-east-1",
                "debug": True,
                "raw_debug": True,
            },
            coordinator=coordinator
        )
        
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        provider.client.messages.create = AsyncMock(return_value=mock_response)
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        await provider.complete(request)
        
        emitted_events = [call[0][0] for call in coordinator.hooks.emit.call_args_list]
        assert "llm:response:raw" in emitted_events, "Should emit llm:response:raw event"

    @pytest.mark.asyncio
    async def test_raw_debug_includes_complete_params(self):
        """Raw request event includes complete untruncated params."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        provider = BedrockProvider(
            config={
                "aws_region": "us-east-1",
                "debug": True,
                "raw_debug": True,
            },
            coordinator=coordinator
        )
        
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        provider.client.messages.create = AsyncMock(return_value=mock_response)
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test message")])
        await provider.complete(request)
        
        # Find the raw request event
        raw_request_calls = [
            call for call in coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:request:raw"
        ]
        assert len(raw_request_calls) > 0, "Should have emitted llm:request:raw"
        
        # Check the event data
        event_data = raw_request_calls[0][0][1]
        assert "params" in event_data, "Should include params field"
        assert "model" in event_data["params"], "Should include model in params"

    @pytest.mark.asyncio
    async def test_raw_debug_disabled_by_default(self):
        """Doesn't emit raw events unless explicitly enabled."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        # raw_debug not set
        provider = BedrockProvider(
            config={"aws_region": "us-east-1"},
            coordinator=coordinator
        )
        
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        provider.client.messages.create = AsyncMock(return_value=mock_response)
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        await provider.complete(request)
        
        emitted_events = [call[0][0] for call in coordinator.hooks.emit.call_args_list]
        assert "llm:request:raw" not in emitted_events, "Should not emit raw events by default"
        assert "llm:response:raw" not in emitted_events, "Should not emit raw events by default"

    @pytest.mark.asyncio
    async def test_debug_flag_required_for_raw(self):
        """raw_debug=true without debug=true doesn't emit raw events."""
        coordinator = MagicMock()
        coordinator.hooks = AsyncMock()
        coordinator.hooks.emit = AsyncMock()
        
        # raw_debug=true but debug=false
        provider = BedrockProvider(
            config={
                "aws_region": "us-east-1",
                "debug": False,
                "raw_debug": True,
            },
            coordinator=coordinator
        )
        
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        provider.client.messages.create = AsyncMock(return_value=mock_response)
        
        from amplifier_core.message_models import ChatRequest, Message
        request = ChatRequest(messages=[Message(role="user", content="test")])
        await provider.complete(request)
        
        emitted_events = [call[0][0] for call in coordinator.hooks.emit.call_args_list]
        assert "llm:request:raw" not in emitted_events, "raw_debug requires debug=true"
        assert "llm:response:raw" not in emitted_events, "raw_debug requires debug=true"


class TestContributionChannel:
    """Test contribution channel registration for custom events."""

    @pytest.mark.asyncio
    async def test_mount_registers_contribution_channel(self):
        """mount() function registers contribution channel."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        from amplifier_module_provider_bedrock import mount
        await mount(coordinator, config={"aws_region": "us-east-1"})
        
        # Verify register_contributor was called
        assert coordinator.register_contributor.called, "Should register contribution channel"

    @pytest.mark.asyncio
    async def test_contribution_channel_name(self):
        """Registers on observability.events channel."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        from amplifier_module_provider_bedrock import mount
        await mount(coordinator, config={"aws_region": "us-east-1"})
        
        # Check the channel name
        call_args = coordinator.register_contributor.call_args
        assert call_args is not None, "Should have called register_contributor"
        assert call_args[0][0] == "observability.events", "Should use observability.events channel"

    @pytest.mark.asyncio
    async def test_contribution_module_name(self):
        """Uses 'bedrock' as module name in contribution."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        from amplifier_module_provider_bedrock import mount
        await mount(coordinator, config={"aws_region": "us-east-1"})
        
        call_args = coordinator.register_contributor.call_args
        assert call_args[0][1] == "bedrock", "Should use 'bedrock' as module name"

    @pytest.mark.asyncio
    async def test_contribution_declares_custom_events(self):
        """Returns list including provider:tool_sequence_repaired event."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        from amplifier_module_provider_bedrock import mount
        await mount(coordinator, config={"aws_region": "us-east-1"})
        
        call_args = coordinator.register_contributor.call_args
        # The third argument is a callable that returns the event list
        contributor_func = call_args[0][2]
        events = contributor_func()
        
        assert isinstance(events, list), "Should return list of events"
        assert "provider:tool_sequence_repaired" in events, "Should declare tool_sequence_repaired event"
