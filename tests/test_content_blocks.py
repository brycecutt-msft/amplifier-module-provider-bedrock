"""
Tests for content block support.

Verifies that BedrockProvider handles all content block types:
- text, thinking, tool_call, tool_result (existing)
- reasoning, redacted_thinking, image (new)
"""

import pytest
from unittest.mock import MagicMock
from amplifier_module_provider_bedrock import BedrockProvider


class TestResponseParsing:
    """Test _convert_to_chat_response() handles all content block types."""

    def test_convert_reasoning_block(self):
        """Handles ReasoningBlock from API response."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        # Mock Bedrock API response with reasoning block
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="reasoning",
                content=["step 1", "step 2"],
                summary=["conclusion"]
            )
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        
        chat_response = provider._convert_to_chat_response(mock_response)
        
        # Should have one ReasoningBlock
        assert len(chat_response.content) == 1
        block = chat_response.content[0]
        assert block.type == "reasoning"
        assert hasattr(block, "content")
        assert hasattr(block, "summary")

    def test_preserve_reasoning_content_and_summary(self):
        """Preserves content and summary arrays in ReasoningBlock."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="reasoning",
                content=["thought 1", "thought 2", "thought 3"],
                summary=["final answer"]
            )
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        
        chat_response = provider._convert_to_chat_response(mock_response)
        block = chat_response.content[0]
        
        assert block.content == ["thought 1", "thought 2", "thought 3"]
        assert block.summary == ["final answer"]

    def test_convert_redacted_thinking_block(self):
        """Handles RedactedThinkingBlock from API response."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="redacted_thinking",
                data="[REDACTED BY VENDOR POLICY]"
            )
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        
        chat_response = provider._convert_to_chat_response(mock_response)
        
        assert len(chat_response.content) == 1
        block = chat_response.content[0]
        assert block.type == "redacted_thinking"
        assert hasattr(block, "data")

    def test_preserve_redacted_data_field(self):
        """Preserves data field in RedactedThinkingBlock."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        redacted_content = "[CONTENT REDACTED]"
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="redacted_thinking", data=redacted_content)
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        
        chat_response = provider._convert_to_chat_response(mock_response)
        block = chat_response.content[0]
        
        assert block.data == redacted_content

    def test_convert_image_block(self):
        """Handles ImageBlock from API response."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="image",
                source={"type": "base64", "data": "abc123"}
            )
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        
        chat_response = provider._convert_to_chat_response(mock_response)
        
        assert len(chat_response.content) == 1
        block = chat_response.content[0]
        assert block.type == "image"
        assert hasattr(block, "source")

    def test_preserve_image_source(self):
        """Preserves source dict in ImageBlock."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        image_source = {"type": "url", "url": "https://example.com/image.png"}
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="image", source=image_source)
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        
        chat_response = provider._convert_to_chat_response(mock_response)
        block = chat_response.content[0]
        
        assert block.source == image_source

    def test_mixed_content_blocks(self):
        """Handles response with multiple different block types."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="Hello"),
            MagicMock(type="thinking", thinking="internal thought", signature="sig123"),
            MagicMock(type="reasoning", content=["step1"], summary=["result"]),
            MagicMock(type="image", source={"type": "base64", "data": "xyz"}),
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        
        chat_response = provider._convert_to_chat_response(mock_response)
        
        assert len(chat_response.content) == 4
        assert chat_response.content[0].type == "text"
        assert chat_response.content[1].type == "thinking"
        assert chat_response.content[2].type == "reasoning"
        assert chat_response.content[3].type == "image"


class TestMessageConversion:
    """Test _clean_content_block() handles all content block types."""

    def test_clean_reasoning_block(self):
        """_clean_content_block() handles reasoning block."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        block = {
            "type": "reasoning",
            "content": ["step 1", "step 2"],
            "summary": ["conclusion"],
            "visibility": "internal"  # Should be removed
        }
        
        cleaned = provider._clean_content_block(block)
        
        assert cleaned["type"] == "reasoning"
        assert cleaned["content"] == ["step 1", "step 2"]
        assert cleaned["summary"] == ["conclusion"]
        assert "visibility" not in cleaned

    def test_clean_redacted_thinking_block(self):
        """_clean_content_block() preserves data field."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        block = {
            "type": "redacted_thinking",
            "data": "[REDACTED]",
            "visibility": "internal"
        }
        
        cleaned = provider._clean_content_block(block)
        
        assert cleaned["type"] == "redacted_thinking"
        assert cleaned["data"] == "[REDACTED]"
        assert "visibility" not in cleaned

    def test_clean_image_block(self):
        """_clean_content_block() preserves source dict."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        source = {"type": "url", "url": "https://example.com/img.png"}
        block = {
            "type": "image",
            "source": source,
            "visibility": "user"
        }
        
        cleaned = provider._clean_content_block(block)
        
        assert cleaned["type"] == "image"
        assert cleaned["source"] == source
        assert "visibility" not in cleaned

    def test_round_trip_reasoning(self):
        """Message with reasoning block round-trips through conversion."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "content": ["analyze problem", "identify solution"],
                        "summary": ["answer is 42"],
                        "visibility": "internal"
                    },
                    {
                        "type": "text",
                        "text": "The answer is 42"
                    }
                ]
            }
        ]
        
        converted = provider._convert_messages(messages)
        
        # Should have one assistant message
        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        
        # Content should be list of blocks with visibility removed
        content = converted[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        
        # Reasoning block cleaned
        reasoning_block = content[0]
        assert reasoning_block["type"] == "reasoning"
        assert "visibility" not in reasoning_block
        assert reasoning_block["content"] == ["analyze problem", "identify solution"]
