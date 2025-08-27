using OrtForge.AI.Agent.Agents;
using OrtForge.AI.Agent.Generation;

namespace OrtForge.AI.UnitTests;

public class AgentOrchestratorHelpersTests
{
    [Fact]
    public void BuildPrompt_WithoutTools_IncludesContextAndHistory()
    {
        var history = new List<(string role, string content)>
        {
            ("user", "hi"),
            ("assistant", "hello")
        };
        var retrieved = new List<string> { "ctx1", "ctx2" };
        var prompt = AgentOrchestrator.BuildPrompt(history, "what?", retrieved, enableTools: false);
        
        // Check for proper Llama 3.1 chat template format
        Assert.Contains("<|begin_of_text|>", prompt);
        Assert.Contains("<|start_header_id|>system<|end_header_id|>", prompt);
        
        // Check for enhanced system prompt structure
        Assert.Contains("## Core Instructions:", prompt);
        Assert.Contains("**ONLY respond as the assistant**", prompt);
        Assert.Contains("**Always format your response in markdown**", prompt);
        Assert.Contains("**Base your answers primarily on the provided context**", prompt);
        
        // Check for context section
        Assert.Contains("## Available Context:", prompt);
        Assert.Contains("**Source 1:**", prompt);
        Assert.Contains("> ctx1", prompt);
        Assert.Contains("**Source 2:**", prompt);
        Assert.Contains("> ctx2", prompt);
        
        // Check for conversation history in proper Llama 3.1 format
        Assert.Contains("<|start_header_id|>user<|end_header_id|>", prompt);
        Assert.Contains("hi", prompt);
        Assert.Contains("<|start_header_id|>assistant<|end_header_id|>", prompt);
        Assert.Contains("hello", prompt);
        
        // Check for current user message and assistant start
        Assert.Contains("what?", prompt);
        Assert.Contains("<|eot_id|>", prompt);
        
        // Should not contain tool instructions when tools are disabled
        Assert.DoesNotContain("## Tool Usage:", prompt);
        Assert.DoesNotContain("TOOL_CALL", prompt);
    }

    [Fact]
    public void BuildPrompt_WithTools_IncludesToolInstructions()
    {
        var history = new List<(string role, string content)>();
        var retrieved = new List<string>();
        var prompt = AgentOrchestrator.BuildPrompt(history, "test", retrieved, enableTools: true);
        
        // Check for proper Llama 3.1 chat template format
        Assert.Contains("<|begin_of_text|>", prompt);
        Assert.Contains("<|start_header_id|>system<|end_header_id|>", prompt);
        
        // Check for system prompt
        Assert.Contains("## Core Instructions:", prompt);
        Assert.Contains("**ONLY respond as the assistant**", prompt);
        
        // Check for tool instructions section
        Assert.Contains("## Tool Usage:", prompt);
        Assert.Contains("When you need to use a tool", prompt);
        Assert.Contains("TOOL_CALL", prompt);
        Assert.Contains("name: tool_name", prompt);
        Assert.Contains("args: tool_arguments", prompt);
        Assert.Contains("END_TOOL_CALL", prompt);
        Assert.Contains("TOOL_RESULT...END_TOOL_RESULT", prompt);
        
        // Check for proper section endings and user message format
        Assert.Contains("<|eot_id|>", prompt);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>", prompt);
        Assert.Contains("test", prompt);
        Assert.Contains("<|start_header_id|>assistant<|end_header_id|>", prompt);
    }

    [Fact]
    public void IsStopToken_RecognizesConfiguredTokens()
    {
        var config = InferenceConfig.Default;
        Assert.True(AgentOrchestrator.IsStopToken(2, config));
        Assert.True(AgentOrchestrator.IsStopToken(0, config));
        Assert.False(AgentOrchestrator.IsStopToken(5, config));
    }

    [Fact]
    public void IsStopSequence_DetectsConfiguredSequences()
    {
        var config = new InferenceConfig { StopSequences = new[] { "</s>", "<|end|>" } };
        Assert.True(AgentOrchestrator.IsStopSequence("hello</s>world", config));
        Assert.True(AgentOrchestrator.IsStopSequence("test<|end|>", config));
        Assert.False(AgentOrchestrator.IsStopSequence("nothing here", config));
    }
}

public class ToolCallStateTests
{
    [Fact]
    public void ToolCallState_DetectsCompleteToolCall()
    {
        var state = new ToolCallState();
        state.AppendText("<|tool_call|>\nname: test_tool\nargs: test_args\n<|/tool_call|>");
        
        Assert.True(state.HasPendingCalls);
        var call = state.GetNextPendingCall();
        Assert.NotNull(call);
        Assert.Equal("test_tool", call.Name);
        Assert.Equal("test_args", call.Arguments);
        Assert.Equal(ToolCallStatus.Pending, call.Status);
    }

    [Fact]
    public void ToolCallState_HandlesIncompleteCall()
    {
        var state = new ToolCallState();
        state.AppendToken("<|tool_call|>");
        state.AppendToken("\nname: ");
        state.AppendToken("test");
        
        Assert.False(state.HasPendingCalls);
        Assert.True(state.InToolCall);
    }

    [Fact]
    public void ToolCallState_UpdatesCallStatus()
    {
        var state = new ToolCallState();
        state.AppendText("<|tool_call|>\nname: test\nargs: args\n<|/tool_call|>");
        
        var call = state.GetNextPendingCall();
        Assert.NotNull(call);
        
        state.UpdateCallStatus(call, ToolCallStatus.Executing);
        Assert.Equal(ToolCallStatus.Executing, state.Calls[0].Status);
        
        state.UpdateCallStatus(call, ToolCallStatus.Completed, "result");
        Assert.Equal(ToolCallStatus.Completed, state.Calls[0].Status);
        Assert.Equal("result", state.Calls[0].Result);
    }

    [Fact]
    public void ToolCallState_ResetClearsState()
    {
        var state = new ToolCallState();
        state.AppendText("<|tool_call|>\nname: test\nargs: args\n<|/tool_call|>");
        
        Assert.True(state.HasPendingCalls);
        state.Reset();
        Assert.False(state.HasPendingCalls);
        Assert.False(state.InToolCall);
    }
}
