using OrtAgent.Core.Agents;
using OrtAgent.Core.Generation;

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
        Assert.Contains("<|system|>", prompt);
        Assert.Contains("<|context|>", prompt);
        Assert.Contains("ctx1", prompt);
        Assert.Contains("ctx2", prompt);
        Assert.Contains("</context>", prompt);
        Assert.Contains("<|user|>hi</s>", prompt);
        Assert.Contains("<|assistant|>hello</s>", prompt);
        Assert.Contains("<|user|>what?</s>", prompt);
        Assert.Contains("<|assistant|>", prompt);
        Assert.DoesNotContain("<|tool_call|>", prompt);
    }

    [Fact]
    public void BuildPrompt_WithTools_IncludesToolInstructions()
    {
        var history = new List<(string role, string content)>();
        var retrieved = new List<string>();
        var prompt = AgentOrchestrator.BuildPrompt(history, "test", retrieved, enableTools: true);
        Assert.Contains("<|system|>", prompt);
        Assert.Contains("When you need to use a tool", prompt);
        Assert.Contains("<|tool_call|>", prompt);
        Assert.Contains("name: tool_name", prompt);
        Assert.Contains("args: tool_arguments", prompt);
        Assert.Contains("<|/tool_call|>", prompt);
        Assert.Contains("<|tool_result|>", prompt);
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
