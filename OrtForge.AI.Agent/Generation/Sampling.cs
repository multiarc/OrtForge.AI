using System;
using System.Collections.Generic;
using System.Linq;

namespace OrtAgent.Core.Generation;

public static class Sampling
{
    public static int Greedy(ReadOnlySpan<float> logits)
    {
        var maxIdx = 0;
        var maxVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
        }
        return maxIdx;
    }

    public static int TopK(ReadOnlySpan<float> logits, int k = 40, double temperature = 1.0, Random? rng = null)
    {
        rng ??= Random.Shared;
        k = Math.Max(1, k);
        var indices = Enumerable.Range(0, logits.Length).ToArray();
        Array.Sort(indices, (a, b) => logits[b].CompareTo(logits[a]));
        var top = indices.Take(k).ToArray();

        // softmax with temperature over top-k
        var probs = new double[top.Length];
        double sum = 0;
        for (int i = 0; i < top.Length; i++)
        {
            var v = Math.Exp(logits[top[i]] / Math.Max(1e-6, temperature));
            probs[i] = v; sum += v;
        }
        for (int i = 0; i < probs.Length; i++) probs[i] /= sum;
        var choice = SampleCategorical(probs, rng);
        return top[choice];
    }

    private static int SampleCategorical(IReadOnlyList<double> probs, Random rng)
    {
        var r = rng.NextDouble();
        double c = 0;
        for (int i = 0; i < probs.Count; i++)
        {
            c += probs[i];
            if (r <= c) return i;
        }
        return probs.Count - 1;
    }
}


