using OrtForge.AI.Agent.Agents;

namespace OrtForge.AI.UnitTests;

/// <summary>
/// Tests for the TOOL_CALL/END_TOOL_CALL pattern that matches the prompt format
/// </summary>
public class ToolCallStateNewPatternTests
{
    [Fact]
    public void AppendToken_WithToolCallMarkers_DetectsToolCall()
    {
        // Arrange
        var state = new ToolCallState();
        var text = @"Some text before
TOOL_CALL
name: search
args: {""query"": ""test""}
END_TOOL_CALL
Some text after";

        // Act
        state.AppendText(text);

        // Assert
        Assert.Single(state.Calls);
        Assert.Equal("search", state.Calls[0].Name);
        Assert.Equal(@"{""query"": ""test""}", state.Calls[0].Arguments);
    }

    [Fact]
    public void AppendToken_StreamingTokens_DetectsToolCall()
    {
        // Arrange
        var state = new ToolCallState();
        var tokens = new[]
        {
            "TOOL",
            "_CALL",
            "\n",
            "name: ",
            "calculator",
            "\nargs: ",
            "2+2",
            "\nEND",
            "_TOOL_CALL"
        };

        // Act
        foreach (var token in tokens)
        {
            state.AppendToken(token);
        }

        // Assert
        Assert.Single(state.Calls);
        Assert.Equal("calculator", state.Calls[0].Name);
        Assert.Equal("2+2", state.Calls[0].Arguments);
    }

    [Fact]
    public void AppendToken_PartialMarker_DoesNotDetectUntilComplete()
    {
        // Arrange
        var state = new ToolCallState();

        // Act - Append partial content
        state.AppendText("TOOL_CALL\nname: test\nargs: foo");

        // Assert - Should be in tool call but not complete
        Assert.True(state.InToolCall);
        Assert.Empty(state.Calls); // Not complete yet

        // Complete the tool call
        state.AppendText("\nEND_TOOL_CALL");

        // Assert - Now should be detected
        Assert.False(state.InToolCall);
        Assert.Single(state.Calls);
    }

    [Fact]
    public void ParseToolCallContent_ValidContent_ReturnsToolCall()
    {
        // Arrange
        var state = new ToolCallState();
        var content = @"TOOL_CALL
name: fetch_data
args: {""url"": ""https://example.com"", ""method"": ""GET""}
END_TOOL_CALL";

        // Act
        state.AppendText(content);

        // Assert
        Assert.Single(state.Calls);
        var call = state.Calls[0];
        Assert.Equal("fetch_data", call.Name);
        Assert.Equal(@"{""url"": ""https://example.com"", ""method"": ""GET""}", call.Arguments);
        Assert.Equal(ToolCallStatus.Pending, call.Status);
        Assert.NotEmpty(call.Id);
    }

    [Fact]
    public void AppendToken_MultipleToolCalls_DetectsAll()
    {
        // Arrange
        var state = new ToolCallState();
        var text = @"First tool:
TOOL_CALL
name: tool1
args: arg1
END_TOOL_CALL
Between tools
TOOL_CALL
name: tool2
args: arg2
END_TOOL_CALL
After tools";

        // Act
        state.AppendText(text);

        // Assert
        Assert.Equal(2, state.Calls.Count);
        Assert.Equal("tool1", state.Calls[0].Name);
        Assert.Equal("tool2", state.Calls[1].Name);
    }

    [Fact]
    public void AppendToken_NameOnly_NoArgs_ReturnsEmptyArgs()
    {
        // Arrange
        var state = new ToolCallState();
        var text = @"TOOL_CALL
name: no_args_tool
END_TOOL_CALL";

        // Act
        state.AppendText(text);

        // Assert
        Assert.Single(state.Calls);
        Assert.Equal("no_args_tool", state.Calls[0].Name);
        Assert.Equal(string.Empty, state.Calls[0].Arguments);
    }

    [Fact]
    public void Reset_ClearsAllState()
    {
        // Arrange
        var state = new ToolCallState();
        state.AppendText(@"TOOL_CALL
name: test
args: data
END_TOOL_CALL");
        Assert.Single(state.Calls);

        // Act
        state.Reset();

        // Assert
        Assert.Empty(state.Calls);
        Assert.False(state.InToolCall);
    }

    [Fact]
    public void GetNextPendingCall_ReturnsPendingCall()
    {
        // Arrange
        var state = new ToolCallState();
        state.AppendText(@"TOOL_CALL
name: pending_test
args: test
END_TOOL_CALL");

        // Act
        var pending = state.GetNextPendingCall();

        // Assert
        Assert.NotNull(pending);
        Assert.Equal("pending_test", pending.Name);
        Assert.Equal(ToolCallStatus.Pending, pending.Status);
    }
}

