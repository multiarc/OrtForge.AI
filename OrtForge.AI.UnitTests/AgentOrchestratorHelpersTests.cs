using OrtForge.AI.Agent.Agents;
using OrtForge.AI.Agent.Generation;

namespace OrtForge.AI.UnitTests;

public class AgentOrchestratorHelpersTests
{
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
        var config = new InferenceConfig { StopSequences = ["</s>", "<|end|>"] };
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
