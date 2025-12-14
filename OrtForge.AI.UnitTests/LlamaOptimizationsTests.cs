using OrtForge.AI.Agent.LLM;

namespace OrtForge.AI.UnitTests;

public class LlamaOptimizationsTests
{
    [Fact]
    public void CreateOptimalPositionIds_InitialPrompt_ReturnsSequentialIds()
    {
        // Arrange - initial prompt of 5 tokens, total length 5
        var totalSequenceLength = 5;
        var newTokenCount = 5;
        
        // Act
        var positionIds = LlamaOptimizations.CreateOptimalPositionIds(totalSequenceLength, newTokenCount);
        
        // Assert
        Assert.Equal(5, positionIds.Length);
        Assert.Equal(new long[] { 0, 1, 2, 3, 4 }, positionIds);
    }
    
    [Fact]
    public void CreateOptimalPositionIds_SingleNewToken_ReturnsSinglePosition()
    {
        // Arrange - 10 tokens already, adding 1 more
        var totalSequenceLength = 10;
        var newTokenCount = 1;
        
        // Act
        var positionIds = LlamaOptimizations.CreateOptimalPositionIds(totalSequenceLength, newTokenCount);
        
        // Assert - should return single position ID = 9 (the 10th position, 0-indexed)
        Assert.Single(positionIds);
        Assert.Equal(9, positionIds[0]);
    }
    
    [Fact]
    public void CreateOptimalPositionIds_LengthMatchesNewTokenCount()
    {
        // Arrange - 50 tokens already, adding 10 more (e.g., new prompt)
        var totalSequenceLength = 60;
        var newTokenCount = 10;
        
        // Act
        var positionIds = LlamaOptimizations.CreateOptimalPositionIds(totalSequenceLength, newTokenCount);
        
        // Assert - should be positions 50, 51, 52, ... 59
        Assert.Equal(newTokenCount, positionIds.Length);
        for (int i = 0; i < newTokenCount; i++)
        {
            Assert.Equal(50 + i, positionIds[i]);
        }
    }
    
    [Fact]
    public void CreateOptimalPositionIds_MultipleGenerationSteps_ReturnsCorrectPositions()
    {
        // Simulate generation steps
        // Step 0: Initial prompt of 5 tokens
        var step0Ids = LlamaOptimizations.CreateOptimalPositionIds(5, 5);
        Assert.Equal(new long[] { 0, 1, 2, 3, 4 }, step0Ids);
        
        // Step 1: Generate 1 token, total is 6
        var step1Ids = LlamaOptimizations.CreateOptimalPositionIds(6, 1);
        Assert.Single(step1Ids);
        Assert.Equal(5, step1Ids[0]);
        
        // Step 2: Generate 1 token, total is 7
        var step2Ids = LlamaOptimizations.CreateOptimalPositionIds(7, 1);
        Assert.Single(step2Ids);
        Assert.Equal(6, step2Ids[0]);
        
        // New turn: Add 3-token prompt, total is 10
        var newTurnIds = LlamaOptimizations.CreateOptimalPositionIds(10, 3);
        Assert.Equal(3, newTurnIds.Length);
        Assert.Equal(new long[] { 7, 8, 9 }, newTurnIds);
    }
}

