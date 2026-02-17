using OrtForge.AI.Agent.LLM;

namespace OrtForge.AI.UnitTests;

public class KvStateTests
{
    [Fact]
    public void KvState_Dispose_ClearsTensorsList()
    {
        // Arrange - create empty KvState (no actual OrtValues needed for this test)
        var tensors = new List<LlamaSession.OutputKvTensor>();
        var kvState = new KvState(tensors, initialSequenceLength: 10);
        
        // Act
        kvState.Dispose();
        
        // Assert
        Assert.Empty(kvState.Tensors);
    }
    
    [Fact]
    public void KvState_CalculateTotalLengthAfterTokens_ReturnsCorrectLength()
    {
        // Arrange
        var kvState = new KvState([], initialSequenceLength: 10);
        
        // Act
        var totalLength = kvState.CalculateTotalLengthAfterTokens(5);
        
        // Assert
        Assert.Equal(15, totalLength);
    }
    
    [Fact]
    public void KvState_AccumulatedSequenceLength_IsSetCorrectly()
    {
        // Arrange & Act
        var kvState = new KvState([], initialSequenceLength: 42);
        
        // Assert
        Assert.Equal(42, kvState.AccumulatedSequenceLength);
    }
    
    [Fact]
    public void KvState_CalculateTotalLengthAfterTokens_ThrowsForNegativeTokenCount()
    {
        // Arrange
        var kvState = new KvState([], initialSequenceLength: 10);
        
        // Act & Assert
        Assert.Throws<ArgumentException>(() => kvState.CalculateTotalLengthAfterTokens(-1));
    }
    
    [Fact]
    public void KvState_Constructor_ThrowsForNegativeSequenceLength()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new KvState([], initialSequenceLength: -1));
    }
    
    [Fact]
    public void KvState_EmptyState_HasZeroSequenceLength()
    {
        // Arrange & Act
        var kvState = new KvState([]);
        
        // Assert
        Assert.Equal(0, kvState.AccumulatedSequenceLength);
        Assert.Empty(kvState.Tensors);
    }
}

