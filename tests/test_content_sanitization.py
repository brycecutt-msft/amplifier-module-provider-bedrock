"""Test content sanitization and validation in Bedrock provider."""

import pytest

from amplifier_module_provider_bedrock import BedrockProvider


class TestCleanContentBlock:
    """Tests for _clean_content_block method."""

    def test_clean_text_block_normal(self):
        """Test cleaning a normal text block."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": "normal content"}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": "normal content"}

    def test_clean_text_block_nested_once(self):
        """Test cleaning text block with single level nesting."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": {"text": "nested content"}}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": "nested content"}

    def test_clean_text_block_nested_multiple(self):
        """Test cleaning text block with multiple levels of nesting."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": {"text": {"text": "deeply nested"}}}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": "deeply nested"}

    def test_clean_text_block_none(self):
        """Test cleaning text block with None value."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": None}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": ""}

    def test_clean_text_block_number(self):
        """Test cleaning text block with number value."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": 123}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": "123"}

    def test_clean_text_block_float(self):
        """Test cleaning text block with float value."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": 3.14}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": "3.14"}

    def test_clean_text_block_boolean(self):
        """Test cleaning text block with boolean value."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": True}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": "True"}

    def test_clean_text_block_empty_string(self):
        """Test cleaning text block with empty string."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": ""}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": ""}

    def test_clean_text_block_removes_visibility(self):
        """Test that visibility field is removed."""
        provider = BedrockProvider(config={})
        block = {"type": "text", "text": "content", "visibility": "internal"}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "text", "text": "content"}
        assert "visibility" not in cleaned

    def test_clean_thinking_block(self):
        """Test cleaning thinking block."""
        provider = BedrockProvider(config={})
        block = {
            "type": "thinking",
            "thinking": "reasoning content",
            "signature": "sig123"
        }
        cleaned = provider._clean_content_block(block)
        assert cleaned == {
            "type": "thinking",
            "thinking": "reasoning content",
            "signature": "sig123"
        }

    def test_clean_thinking_block_no_signature(self):
        """Test cleaning thinking block without signature."""
        provider = BedrockProvider(config={})
        block = {"type": "thinking", "thinking": "reasoning"}
        cleaned = provider._clean_content_block(block)
        assert cleaned == {"type": "thinking", "thinking": "reasoning"}
        assert "signature" not in cleaned

    def test_clean_tool_use_block(self):
        """Test cleaning tool_use block."""
        provider = BedrockProvider(config={})
        block = {
            "type": "tool_use",
            "id": "call_123",
            "name": "test_tool",
            "input": {"arg": "value"}
        }
        cleaned = provider._clean_content_block(block)
        assert cleaned == {
            "type": "tool_use",
            "id": "call_123",
            "name": "test_tool",
            "input": {"arg": "value"}
        }

    def test_clean_tool_result_block_string_content(self):
        """Test cleaning tool_result block with string content."""
        provider = BedrockProvider(config={})
        block = {
            "type": "tool_result",
            "tool_use_id": "call_123",
            "content": "result text"
        }
        cleaned = provider._clean_content_block(block)
        assert cleaned == {
            "type": "tool_result",
            "tool_use_id": "call_123",
            "content": "result text"
        }

    def test_clean_tool_result_block_dict_content(self):
        """Test cleaning tool_result block with dict content (should stringify)."""
        provider = BedrockProvider(config={})
        block = {
            "type": "tool_result",
            "tool_use_id": "call_123",
            "content": {"result": "data"}
        }
        cleaned = provider._clean_content_block(block)
        assert cleaned["type"] == "tool_result"
        assert cleaned["tool_use_id"] == "call_123"
        assert isinstance(cleaned["content"], str)
        assert "result" in cleaned["content"]

    def test_clean_tool_result_block_none_content(self):
        """Test cleaning tool_result block with None content."""
        provider = BedrockProvider(config={})
        block = {
            "type": "tool_result",
            "tool_use_id": "call_123",
            "content": None
        }
        cleaned = provider._clean_content_block(block)
        assert cleaned["content"] == ""

    def test_clean_unknown_block_type(self):
        """Test cleaning unknown block type removes visibility but keeps other fields."""
        provider = BedrockProvider(config={})
        block = {
            "type": "custom",
            "data": "value",
            "visibility": "internal"
        }
        cleaned = provider._clean_content_block(block)
        assert cleaned["type"] == "custom"
        assert cleaned["data"] == "value"
        assert "visibility" not in cleaned


class TestConvertMessagesContentSanitization:
    """Tests for content sanitization in _convert_messages method."""

    def test_convert_assistant_with_nested_text_blocks(self):
        """Test converting assistant message with nested text blocks."""
        provider = BedrockProvider(config={})
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": {"text": "nested response"}}
                ]
            }
        ]
        converted = provider._convert_messages(messages)
        
        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert isinstance(converted[0]["content"], list)
        assert converted[0]["content"][0] == {"type": "text", "text": "nested response"}

    def test_convert_assistant_with_string_content(self):
        """Test converting assistant message with simple string content."""
        provider = BedrockProvider(config={})
        messages = [
            {"role": "assistant", "content": "simple text"}
        ]
        converted = provider._convert_messages(messages)
        
        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "simple text"

    def test_convert_assistant_with_non_string_content(self):
        """Test converting assistant message with non-string content."""
        provider = BedrockProvider(config={})
        messages = [
            {"role": "assistant", "content": 123}
        ]
        converted = provider._convert_messages(messages)
        
        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "123"

    def test_convert_assistant_with_tool_calls_and_nested_text(self):
        """Test converting assistant with tool calls and nested text content."""
        provider = BedrockProvider(config={})
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": {"text": "nested"}}
                ],
                "tool_calls": [
                    {"id": "call_1", "tool": "test", "arguments": {}}
                ]
            }
        ]
        converted = provider._convert_messages(messages)
        
        assert len(converted) == 1
        assistant_msg = converted[0]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        # Should have text block and tool_use block
        assert len(assistant_msg["content"]) == 2
        assert assistant_msg["content"][0] == {"type": "text", "text": "nested"}
        assert assistant_msg["content"][1]["type"] == "tool_use"

    def test_convert_assistant_with_thinking_block(self):
        """Test converting assistant with thinking block."""
        provider = BedrockProvider(config={})
        messages = [
            {
                "role": "assistant",
                "content": "response",
                "thinking_block": {
                    "type": "thinking",
                    "thinking": "reasoning",
                    "visibility": "internal"
                }
            }
        ]
        converted = provider._convert_messages(messages)
        
        assert len(converted) == 1
        assert isinstance(converted[0]["content"], list)
        # Thinking should be first, then text
        assert converted[0]["content"][0]["type"] == "thinking"
        assert converted[0]["content"][0]["thinking"] == "reasoning"
        assert "visibility" not in converted[0]["content"][0]
        assert converted[0]["content"][1] == {"type": "text", "text": "response"}

    def test_convert_tool_messages_with_non_string_content(self):
        """Test that tool messages with non-string content are sanitized."""
        provider = BedrockProvider(config={})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "tool": "test", "arguments": {}}]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": {"error": "something", "code": 500}
            }
        ]
        converted = provider._convert_messages(messages)
        
        # Find user message with tool results
        tool_msg = next(
            msg for msg in converted
            if msg["role"] == "user" and isinstance(msg["content"], list)
        )
        tool_result = tool_msg["content"][0]
        assert tool_result["type"] == "tool_result"
        assert isinstance(tool_result["content"], str)

    def test_convert_tool_messages_with_none_content(self):
        """Test that tool messages with None content are converted to empty string."""
        provider = BedrockProvider(config={})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "tool": "test", "arguments": {}}]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": None
            }
        ]
        converted = provider._convert_messages(messages)
        
        tool_msg = next(
            msg for msg in converted
            if msg["role"] == "user" and isinstance(msg["content"], list)
        )
        tool_result = tool_msg["content"][0]
        assert tool_result["content"] == ""

    def test_convert_user_message_with_non_string_content(self):
        """Test that user messages with non-string content are converted."""
        provider = BedrockProvider(config={})
        messages = [
            {"role": "user", "content": ["list", "of", "items"]}
        ]
        converted = provider._convert_messages(messages)
        
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], str)

    def test_convert_developer_message_with_non_string_content(self):
        """Test that developer messages with non-string content are converted."""
        provider = BedrockProvider(config={})
        messages = [
            {"role": "developer", "content": {"file": "content"}}
        ]
        converted = provider._convert_messages(messages)
        
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], str)
        assert "<context_file>" in converted[0]["content"]

    def test_convert_multiple_tool_results_batched(self):
        """Test that multiple tool results are batched into one user message."""
        provider = BedrockProvider(config={})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "tool": "test1", "arguments": {}},
                    {"id": "call_2", "tool": "test2", "arguments": {}}
                ]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result 1"},
            {"role": "tool", "tool_call_id": "call_2", "content": 123}  # Non-string
        ]
        converted = provider._convert_messages(messages)
        
        # Should have assistant message and one user message with batched results
        assert len(converted) == 2
        
        tool_msg = converted[1]
        assert tool_msg["role"] == "user"
        assert isinstance(tool_msg["content"], list)
        assert len(tool_msg["content"]) == 2
        
        # Both should have string content
        assert tool_msg["content"][0]["content"] == "result 1"
        assert tool_msg["content"][1]["content"] == "123"

    def test_convert_complex_conversation_with_sanitization(self):
        """Test converting a complex conversation with various sanitization needs."""
        provider = BedrockProvider(config={})
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": [
                {"type": "text", "text": {"text": "nested"}}
            ]},
            {"role": "user", "content": 456},  # Non-string
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "tool": "test", "arguments": {}}]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": None},  # None
            {"role": "assistant", "content": True}  # Boolean
        ]
        converted = provider._convert_messages(messages)
        
        # All messages should be properly sanitized
        assert all(
            isinstance(msg["content"], (str, list))
            for msg in converted
        )
        
        # Verify specific sanitizations
        assert converted[0]["content"] == "start"
        assert converted[1]["content"][0]["text"] == "nested"
        assert converted[2]["content"] == "456"
        # Tool result should be in a user message with empty string
        tool_msg = next(
            msg for msg in converted
            if msg["role"] == "user" and isinstance(msg["content"], list)
        )
        assert tool_msg["content"][0]["content"] == ""
        # Last assistant message should have boolean converted
        assert converted[-1]["content"] == "True"
