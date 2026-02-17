using OrtForge.AI.Agent.Generation;

namespace OrtForge.AI.UnitTests;

public class SamplingTests
{
    [Fact]
    public void Greedy_SelectsMaxIndex()
    {
        var logits = new float[] { -1f, 0.5f, 3.2f, 3.19f };
        var idx = Sampling.Greedy(logits);
        Assert.Equal(2, idx);
    }

    [Fact]
    public void Sample_WithGreedyConfig_EqualsGreedy()
    {
        var logits = new float[] { 0.1f, 2.5f, -0.5f, 1.0f };
        var greedy = Sampling.Greedy(logits);
        var config = InferenceConfig.Greedy;
        var idx = Sampling.Sample(logits, config, [], new Random(42));
        Assert.Equal(greedy, idx);
    }

    [Fact]
    public void Sample_TopK_SamplesOnlyFromTopK()
    {
        var logits = new float[] { 1f, 2f, 3f, 4f, 5f };
        var config = new InferenceConfig { TopK = 3, Temperature = 1.0, Seed = 123 };
        var rng = new Random(123);
        for (int t = 0; t < 100; t++)
        {
            var idx = Sampling.Sample(logits, config, [], rng);
            Assert.Contains(idx, new[] { 2, 3, 4 });
        }
    }

    [Fact]
    public void Sample_LowTemperature_PrefersMax()
    {
        var logits = new float[] { 1f, 2f, 3f, 4f, 5f };
        var config = new InferenceConfig { TopK = 5, Temperature = 0.01, Seed = 7 };
        int favored = 0;
        var rng = new Random(7);
        for (int t = 0; t < 50; t++)
        {
            var idx = Sampling.Sample(logits, config, [], rng);
            if (idx == 4) favored++;
        }
        Assert.True(favored > 40);
    }

    [Fact]
    public void Sample_WithRepetitionPenalty_ReducesRepeatedTokens()
    {
        var logits = new float[] { 1f, 2f, 3f, 4f, 5f };
        var previousTokens = new int[] { 4, 4, 4 };
        var config = new InferenceConfig { RepetitionPenalty = 1.2, TopK = 5, Temperature = 0.1, Seed = 42 };
        
        var idx = Sampling.Sample(logits, config, previousTokens.ToList(), new Random(42));
        
        Assert.NotEqual(4, idx);
    }

    [Fact]
    public void Sample_WithTopP_LimitsTokenSelection()
    {
        var logits = new float[] { 1f, 1f, 1f, 10f, 10f };
        var config = new InferenceConfig { TopP = 0.5, Temperature = 1.0, Seed = 123 };
        var rng = new Random(123);
        
        for (int t = 0; t < 50; t++)
        {
            var idx = Sampling.Sample(logits, config, [], rng);
            Assert.Contains(idx, new[] { 3, 4 });
        }
    }
    
    [Fact]
    public void Sample_WithRepetitionPenaltyOfOne_DoesNotModifyLogits()
    {
        // Arrange - penalty of 1.0 should be a no-op
        var logits = new float[] { 1f, 2f, 3f, 4f, 5f };
        var previousTokens = new int[] { 4, 4, 4 };
        
        // Config with penalty = 1.0 (should be no-op)
        var configWithPenalty = new InferenceConfig { RepetitionPenalty = 1.0, TopK = 5, Temperature = 0.01, Seed = 42 };
        
        // Config without penalty
        var configWithoutPenalty = new InferenceConfig { RepetitionPenalty = 0.0, TopK = 5, Temperature = 0.01, Seed = 42 };
        
        // Both should behave the same - select token 4 (highest logit)
        var idxWithPenalty = Sampling.Sample(logits, configWithPenalty, previousTokens.ToList(), new Random(42));
        var idxWithoutPenalty = Sampling.Sample(logits, configWithoutPenalty, previousTokens.ToList(), new Random(42));
        
        Assert.Equal(idxWithoutPenalty, idxWithPenalty);
        Assert.Equal(4, idxWithPenalty); // Both should select token 4 (highest logit, unpenalized)
    }
}
