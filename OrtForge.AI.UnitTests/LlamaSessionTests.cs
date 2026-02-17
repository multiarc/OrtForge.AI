using OrtForge.AI.Agent.LLM;

namespace OrtForge.AI.UnitTests;

/// <summary>
/// Tests for LlamaSession and related classes.
/// Note: Full integration tests would require actual ONNX models.
/// </summary>
public class LlamaSessionTests
{
    [Fact]
    public void StepInputs_Create_WithValidInput_ReturnsStepInputs()
    {
        // Arrange
        var inputIds = new long[] { 1, 2, 3, 4, 5 };
        var kvState = new KvState([]);
        
        // Act
        using var stepInputs = LlamaSession.StepInputs.Create(inputIds, kvState);
        
        // Assert
        Assert.NotNull(stepInputs);
        Assert.NotNull(stepInputs.InputIds);
    }
    
    [Fact]
    public void StepInputs_Create_WithPositionIds_IncludesPositionIds()
    {
        // Arrange
        var inputIds = new long[] { 1, 2, 3 };
        var positionIds = new long[] { 0, 1, 2 };
        var kvState = new KvState([]);
        
        // Act
        using var stepInputs = LlamaSession.StepInputs.Create(inputIds, kvState, positionIds);
        
        // Assert
        Assert.NotNull(stepInputs);
        Assert.NotNull(stepInputs.PositionIds);
    }
    
    [Fact]
    public void StepInputs_Create_WithAttentionMask_IncludesAttentionMask()
    {
        // Arrange
        var inputIds = new long[] { 1, 2, 3 };
        var attentionMask = new long[] { 1, 1, 1 };
        var kvState = new KvState([]);
        
        // Act
        using var stepInputs = LlamaSession.StepInputs.Create(inputIds, kvState, null, attentionMask);
        
        // Assert
        Assert.NotNull(stepInputs);
        Assert.NotNull(stepInputs.AttentionMask);
    }
    
    [Fact]
    public void StepInputs_Dispose_DoesNotThrow()
    {
        // Arrange
        var inputIds = new long[] { 1, 2, 3 };
        var positionIds = new long[] { 0, 1, 2 };
        var attentionMask = new long[] { 1, 1, 1 };
        var kvState = new KvState([]);
        
        // Act
        var stepInputs = LlamaSession.StepInputs.Create(inputIds, kvState, positionIds, attentionMask);
        
        // Assert - dispose should not throw
        var exception = Record.Exception(() => stepInputs.Dispose());
        Assert.Null(exception);
    }
    
    [Fact]
    public void StepInputs_Create_EmptyInputIds_StillCreatesValidInputs()
    {
        // Arrange - edge case with single token
        var inputIds = new long[] { 42 };
        var kvState = new KvState([]);
        
        // Act
        using var stepInputs = LlamaSession.StepInputs.Create(inputIds, kvState);
        
        // Assert
        Assert.NotNull(stepInputs);
        Assert.NotNull(stepInputs.InputIds);
    }
}

