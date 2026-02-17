using OrtForge.AI.Agent.Generation;

namespace OrtForge.AI.UnitTests;

/// <summary>
/// Tests for sliding window token history maintained across conversation turns
/// for repetition penalty purposes.
/// </summary>
public class SlidingWindowTests
{
    [Fact]
    public void TokenHistory_MaintainsAcrossTurns()
    {
        // Arrange
        var history = new TokenHistory(maxSize: 10);
        
        // Simulate turn 1
        history.AddTokens([1, 2, 3]);
        
        // Simulate turn 2
        history.AddTokens([4, 5, 6]);
        
        // Assert - all tokens should be in history
        var tokens = history.GetTokens();
        Assert.Equal(6, tokens.Count);
        Assert.Contains(1, tokens);
        Assert.Contains(6, tokens);
    }
    
    [Fact]
    public void TokenHistory_EnforcesMaxSize()
    {
        // Arrange
        var history = new TokenHistory(maxSize: 5);
        
        // Add more tokens than max size
        history.AddTokens([1, 2, 3, 4, 5, 6, 7]);
        
        // Assert - should only keep last 5
        var tokens = history.GetTokens();
        Assert.Equal(5, tokens.Count);
        Assert.DoesNotContain(1, tokens);
        Assert.DoesNotContain(2, tokens);
        Assert.Contains(7, tokens);
    }
    
    [Fact]
    public void TokenHistory_AddToken_UpdatesHistory()
    {
        // Arrange
        var history = new TokenHistory(maxSize: 3);
        
        // Act
        history.AddToken(1);
        history.AddToken(2);
        history.AddToken(3);
        history.AddToken(4); // Should push out 1
        
        // Assert
        var tokens = history.GetTokens();
        Assert.Equal(3, tokens.Count);
        Assert.DoesNotContain(1, tokens);
        Assert.Contains(4, tokens);
    }
    
    [Fact]
    public void TokenHistory_Clear_ResetsHistory()
    {
        // Arrange
        var history = new TokenHistory(maxSize: 10);
        history.AddTokens([1, 2, 3, 4, 5]);
        
        // Act
        history.Clear();
        
        // Assert
        Assert.Empty(history.GetTokens());
    }
    
    [Fact]
    public void TokenHistory_DefaultMaxSize()
    {
        // Arrange & Act - default should be reasonable (128)
        var history = new TokenHistory();
        
        // Assert
        Assert.Equal(128, history.MaxSize);
    }
    
    [Fact]
    public void TokenHistory_CountReflectsActualTokens()
    {
        // Arrange
        var history = new TokenHistory(maxSize: 100);
        
        // Act
        history.AddTokens([1, 2, 3]);
        
        // Assert
        Assert.Equal(3, history.Count);
    }
}

