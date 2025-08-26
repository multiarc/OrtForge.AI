using System.Collections.Generic;
using OrtAgent.Core.Agents;

namespace OrtForge.AI.UnitTests;

public class AgentOrchestratorHelpersTests
{
    [Fact]
    public void BuildPrompt_IncludesContextAndHistory()
    {
        var history = new List<(string role, string content)>
        {
            ("user", "hi"),
            ("assistant", "hello")
        };
        var retrieved = new List<string> { "ctx1", "ctx2" };
        var prompt = AgentOrchestrator.BuildPrompt(history, "what?", retrieved);
        Assert.Contains("<|system|>", prompt);
        Assert.Contains("<|context|>", prompt);
        Assert.Contains("ctx1", prompt);
        Assert.Contains("ctx2", prompt);
        Assert.Contains("</context>", prompt);
        Assert.Contains("<|user|>hi</s>", prompt);
        Assert.Contains("<|assistant|>hello</s>", prompt);
        Assert.Contains("<|user|>what?</s>", prompt);
        Assert.Contains("<|assistant|>", prompt);
    }

    [Fact]
    public void ParseToolCall_ExtractsBody()
    {
        var text = "prefix [T-CALL]{\"a\":1}[/T-CALL] suffix";
        var parsed = AgentOrchestrator.ParseToolCall(text);
        Assert.Equal("tool", parsed.name);
        Assert.Equal("{\"a\":1}", parsed.args);
    }

    [Fact]
    public void ParseToolCall_NoTags_ReturnsEmpty()
    {
        var parsed = AgentOrchestrator.ParseToolCall("nothing here");
        Assert.Equal("", parsed.name);
        Assert.Equal("", parsed.args);
    }

    [Fact]
    public void IsToolCallStart_DetectsTag()
    {
        Assert.True(AgentOrchestrator.IsToolCallStart("[T-CALL]"));
        Assert.False(AgentOrchestrator.IsToolCallStart("nope"));
    }

    [Fact]
    public void IsStopToken_RecognizesEos()
    {
        Assert.True(AgentOrchestrator.IsStopToken(2));
        Assert.True(AgentOrchestrator.IsStopToken(0));
        Assert.False(AgentOrchestrator.IsStopToken(5));
    }
}
