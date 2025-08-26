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
        var idx = Sampling.Sample(logits, config, ReadOnlySpan<int>.Empty, new Random(42));
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
            var idx = Sampling.Sample(logits, config, ReadOnlySpan<int>.Empty, rng);
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
            var idx = Sampling.Sample(logits, config, ReadOnlySpan<int>.Empty, rng);
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
        
        var idx = Sampling.Sample(logits, config, previousTokens.AsSpan(), new Random(42));
        
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
            var idx = Sampling.Sample(logits, config, ReadOnlySpan<int>.Empty, rng);
            Assert.Contains(idx, new[] { 3, 4 });
        }
    }
}
