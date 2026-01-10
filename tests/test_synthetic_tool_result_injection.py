"""
Tests for synthetic tool result injection when tool results are missing.

Verifies that the provider correctly detects missing tool results and
injects synthetic error messages that are properly converted to Anthropic format.
"""

import pytest
from amplifier_core.message_models import Message, ToolCallBlock, TextBlock
from amplifier_module_provider_bedrock import BedrockProvider


def test_find_missing_tool_results_with_toolcallblock_format():
    """Test that _find_missing_tool_results detects missing results in ToolCallBlock format."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    # Simulate ChatRequest format with ToolCallBlock in content
    messages = [
        Message(
            role="user",
            content="Run a command",
        ),
        Message(
            role="assistant",
            content=[
                TextBlock(text="I'll run that command"),
                ToolCallBlock(id="toolu_123", name="bash", input={"cmd": "ls"}),
            ],
        ),
        # Missing tool result message!
    ]

    missing = provider._find_missing_tool_results(messages)
    
    assert len(missing) == 1
    assert missing[0][0] == "toolu_123"
    assert missing[0][1] == "bash"
    assert missing[0][2] == {"cmd": "ls"}


def test_find_missing_tool_results_with_tool_calls_field():
    """Test that _find_missing_tool_results detects missing results in tool_calls field format."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    # Simulate dict format with tool_calls field (from model_dump)
    messages = [
        Message(
            role="user",
            content="Run a command",
        ),
        Message(
            role="assistant",
            content="I'll run that command",
            tool_calls=[
                {"id": "toolu_456", "tool": "bash", "arguments": {"cmd": "pwd"}},
            ],
        ),
        # Missing tool result message!
    ]

    missing = provider._find_missing_tool_results(messages)
    
    assert len(missing) == 1
    assert missing[0][0] == "toolu_456"
    assert missing[0][1] == "bash"
    assert missing[0][2] == {"cmd": "pwd"}


def test_find_missing_tool_results_with_multiple_missing():
    """Test detection of multiple missing tool results."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="toolu_1", name="bash", input={"cmd": "ls"}),
                ToolCallBlock(id="toolu_2", name="read_file", input={"path": "test.txt"}),
            ],
        ),
        # Both tool results missing!
    ]

    missing = provider._find_missing_tool_results(messages)
    
    assert len(missing) == 2
    missing_ids = {m[0] for m in missing}
    assert "toolu_1" in missing_ids
    assert "toolu_2" in missing_ids


def test_find_missing_tool_results_with_partial_results():
    """Test detection when some results are present but others are missing."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="toolu_1", name="bash", input={"cmd": "ls"}),
                ToolCallBlock(id="toolu_2", name="read_file", input={"path": "test.txt"}),
            ],
        ),
        Message(
            role="tool",
            content="file1.txt\nfile2.txt",
            tool_call_id="toolu_1",
            name="bash",
        ),
        # toolu_2 result is missing!
    ]

    missing = provider._find_missing_tool_results(messages)
    
    assert len(missing) == 1
    assert missing[0][0] == "toolu_2"
    assert missing[0][1] == "read_file"


def test_find_missing_tool_results_with_all_present():
    """Test that no missing results are detected when all are present."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="toolu_1", name="bash", input={"cmd": "ls"}),
            ],
        ),
        Message(
            role="tool",
            content="file1.txt",
            tool_call_id="toolu_1",
            name="bash",
        ),
    ]

    missing = provider._find_missing_tool_results(messages)
    
    assert len(missing) == 0


def test_convert_messages_handles_tool_call_blocks():
    """Test that _convert_messages properly converts tool_call blocks to tool_use blocks."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    # Simulate messages with tool_call blocks (from ToolCallBlock serialization)
    messages = [
        {
            "role": "user",
            "content": "Run a command",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll run that command"},
                {"type": "tool_call", "id": "toolu_123", "name": "bash", "input": {"cmd": "ls"}},
            ],
        },
    ]

    result = provider._convert_messages(messages)
    
    # Should have 2 messages
    assert len(result) == 2
    
    # Second message should be assistant with tool_use block (not tool_call)
    assistant_msg = result[1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["content"], list)
    assert len(assistant_msg["content"]) == 2
    
    # First block should be text
    assert assistant_msg["content"][0]["type"] == "text"
    assert assistant_msg["content"][0]["text"] == "I'll run that command"
    
    # Second block should be tool_use (converted from tool_call)
    tool_block = assistant_msg["content"][1]
    assert tool_block["type"] == "tool_use"
    assert tool_block["id"] == "toolu_123"
    assert tool_block["name"] == "bash"
    assert tool_block["input"] == {"cmd": "ls"}


def test_clean_content_block_converts_tool_call_to_tool_use():
    """Test that _clean_content_block converts tool_call blocks to tool_use format."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    tool_call_block = {
        "type": "tool_call",
        "id": "toolu_789",
        "name": "read_file",
        "input": {"path": "test.txt"},
    }

    result = provider._clean_content_block(tool_call_block)
    
    # Should be converted to tool_use
    assert result["type"] == "tool_use"
    assert result["id"] == "toolu_789"
    assert result["name"] == "read_file"
    assert result["input"] == {"path": "test.txt"}


def test_synthetic_result_creation():
    """Test that synthetic error results are properly formatted."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    synthetic = provider._create_synthetic_result("toolu_missing", "bash")
    
    assert synthetic.role == "tool"
    assert synthetic.tool_call_id == "toolu_missing"
    assert synthetic.name == "bash"
    assert "[SYSTEM ERROR: Tool result missing from conversation history]" in synthetic.content
    assert "toolu_missing" in synthetic.content


def test_convert_messages_batches_synthetic_tool_results():
    """Test that synthetic tool results are properly batched with other tool results."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    # Simulate scenario where synthetic result is added alongside regular result
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_call", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}},
                {"type": "tool_call", "id": "toolu_2", "name": "read_file", "input": {"path": "test.txt"}},
            ],
        },
        {
            "role": "tool",
            "content": "file1.txt",
            "tool_call_id": "toolu_1",
        },
        {
            "role": "tool",
            "content": "[SYSTEM ERROR: Tool result missing from conversation history]\n\nTool: read_file\nCall ID: toolu_2",
            "tool_call_id": "toolu_2",
        },
    ]

    result = provider._convert_messages(messages)
    
    # Should have 2 messages: assistant with tool_use blocks, user with tool_result blocks
    assert len(result) == 2
    
    # First should be assistant with 2 tool_use blocks
    assert result[0]["role"] == "assistant"
    assert len(result[0]["content"]) == 2
    assert all(block["type"] == "tool_use" for block in result[0]["content"])
    
    # Second should be user with 2 tool_result blocks (batched together)
    assert result[1]["role"] == "user"
    assert len(result[1]["content"]) == 2
    assert all(block["type"] == "tool_result" for block in result[1]["content"])
    assert result[1]["content"][0]["tool_use_id"] == "toolu_1"
    assert result[1]["content"][1]["tool_use_id"] == "toolu_2"
    assert "[SYSTEM ERROR" in result[1]["content"][1]["content"]


def test_end_to_end_synthetic_injection_flow():
    """Test the complete flow from detection to conversion with synthetic results."""
    provider = BedrockProvider(config={"aws_region": "us-east-1"})

    # Start with messages that have missing tool result
    messages = [
        Message(
            role="user",
            content="Run a command",
        ),
        Message(
            role="assistant",
            content=[
                TextBlock(text="I'll run that"),
                ToolCallBlock(id="toolu_missing", name="bash", input={"cmd": "ls"}),
            ],
        ),
    ]

    # Detect missing results
    missing = provider._find_missing_tool_results(messages)
    assert len(missing) == 1

    # Inject synthetic result
    for call_id, tool_name, _ in missing:
        synthetic = provider._create_synthetic_result(call_id, tool_name)
        messages.append(synthetic)

    # Convert to Anthropic format
    converted = provider._convert_messages([m.model_dump() for m in messages])
    
    # Should have 3 messages: user, assistant, user (with tool_result)
    assert len(converted) == 3
    
    # First: user message
    assert converted[0]["role"] == "user"
    
    # Second: assistant with tool_use
    assert converted[1]["role"] == "assistant"
    tool_use_blocks = [b for b in converted[1]["content"] if b["type"] == "tool_use"]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["id"] == "toolu_missing"
    
    # Third: user with tool_result (synthetic error)
    assert converted[2]["role"] == "user"
    assert len(converted[2]["content"]) == 1
    assert converted[2]["content"][0]["type"] == "tool_result"
    assert converted[2]["content"][0]["tool_use_id"] == "toolu_missing"
    assert "[SYSTEM ERROR" in converted[2]["content"][0]["content"]
