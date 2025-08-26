using System;
using OrtAgent.Core.Generation;

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
    public void TopK_WithK1_EqualsGreedy()
    {
        var logits = new float[] { 0.1f, 2.5f, -0.5f, 1.0f };
        var greedy = Sampling.Greedy(logits);
        var idx = Sampling.TopK(logits, k: 1, temperature: 1.0, rng: new Random(42));
        Assert.Equal(greedy, idx);
    }

    [Fact]
    public void TopK_SamplesOnlyFromTopK()
    {
        var logits = new float[] { 1f, 2f, 3f, 4f, 5f };
        var k = 3;
        var rng = new Random(123);
        for (int t = 0; t < 100; t++)
        {
            var idx = Sampling.TopK(logits, k: k, temperature: 1.0, rng: rng);
            Assert.Contains(idx, new[] { 2, 3, 4 });
        }
    }

    [Fact]
    public void TopK_LowTemperature_PrefersMax()
    {
        var logits = new float[] { 1f, 2f, 3f, 4f, 5f };
        int favored = 0;
        var rng = new Random(7);
        for (int t = 0; t < 50; t++)
        {
            var idx = Sampling.TopK(logits, k: 5, temperature: 0.01, rng: rng);
            if (idx == 4) favored++;
        }
        Assert.True(favored > 40);
    }
}
