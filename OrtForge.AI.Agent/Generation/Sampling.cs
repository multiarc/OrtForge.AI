namespace OrtForge.AI.Agent.Generation;

public static class Sampling
{
    public static int Sample(ReadOnlySpan<float> logits, InferenceConfig config, ReadOnlySpan<int> previousTokens = default, Random? rng = null)
    {
        rng ??= config.Seed.HasValue ? new Random(config.Seed.Value) : Random.Shared;

        if (config.UseGreedy || config.Temperature <= 1e-6)
        {
            return Greedy(logits);
        }

        var logitsArray = logits.ToArray();
        
        if (config.RepetitionPenalty != 1.0 && !previousTokens.IsEmpty)
        {
            ApplyRepetitionPenalty(logitsArray, previousTokens, config.RepetitionPenalty);
        }
        
        if (config.FrequencyPenalty != 0.0 && !previousTokens.IsEmpty)
        {
            ApplyFrequencyPenalty(logitsArray, previousTokens, config.FrequencyPenalty);
        }
        
        if (config.PresencePenalty != 0.0 && !previousTokens.IsEmpty)
        {
            ApplyPresencePenalty(logitsArray, previousTokens, config.PresencePenalty);
        }

        var probs = Softmax(logitsArray, config.Temperature);
        
        if (config.MinP > 0.0)
        {
            ApplyMinP(probs, config.MinP);
        }
        
        if (config.TopK > 0)
        {
            ApplyTopK(probs, config.TopK);
        }
        
        if (config.TopP < 1.0)
        {
            ApplyTopP(probs, config.TopP);
        }
        
        if (config.TfsZ < 1.0)
        {
            ApplyTailFreeSampling(probs, config.TfsZ);
        }
        
        if (config.TypicalP < 1.0)
        {
            ApplyTypicalSampling(probs, config.TypicalP);
        }

        return SampleCategorical(probs, rng);
    }

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

    private static double[] Softmax(float[] logits, double temperature)
    {
        var probs = new double[logits.Length];
        var maxLogit = logits.Max();
        double sum = 0;
        
        for (int i = 0; i < logits.Length; i++)
        {
            var scaled = (logits[i] - maxLogit) / Math.Max(1e-6, temperature);
            probs[i] = Math.Exp(scaled);
            sum += probs[i];
        }
        
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= sum;
        }
        
        return probs;
    }

    private static void ApplyRepetitionPenalty(float[] logits, ReadOnlySpan<int> previousTokens, double penalty)
    {
        if (penalty == 1.0) return;
        
        var tokenCounts = new Dictionary<int, int>();
        foreach (var token in previousTokens)
        {
            tokenCounts[token] = tokenCounts.GetValueOrDefault(token, 0) + 1;
        }
        
        foreach (var (token, count) in tokenCounts)
        {
            if (token >= 0 && token < logits.Length)
            {
                var penaltyFactor = Math.Pow(penalty, count);
                if (logits[token] > 0)
                {
                    logits[token] /= (float)penaltyFactor;
                }
                else
                {
                    logits[token] *= (float)penaltyFactor;
                }
            }
        }
    }

    private static void ApplyFrequencyPenalty(float[] logits, ReadOnlySpan<int> previousTokens, double penalty)
    {
        if (penalty == 0.0) return;
        
        var tokenCounts = new Dictionary<int, int>();
        foreach (var token in previousTokens)
        {
            tokenCounts[token] = tokenCounts.GetValueOrDefault(token, 0) + 1;
        }
        
        foreach (var (token, count) in tokenCounts)
        {
            if (token >= 0 && token < logits.Length)
            {
                logits[token] -= (float)(count * penalty);
            }
        }
    }

    private static void ApplyPresencePenalty(float[] logits, ReadOnlySpan<int> previousTokens, double penalty)
    {
        if (penalty == 0.0) return;
        
        var presentTokens = new HashSet<int>();
        foreach (var token in previousTokens)
        {
            presentTokens.Add(token);
        }
        
        foreach (var token in presentTokens)
        {
            if (token >= 0 && token < logits.Length)
            {
                logits[token] -= (float)penalty;
            }
        }
    }

    private static void ApplyMinP(double[] probs, double minP)
    {
        var maxProb = probs.Max();
        var threshold = maxProb * minP;
        
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] < threshold)
            {
                probs[i] = 0.0;
            }
        }
        
        var sum = probs.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
    }

    private static void ApplyTopK(double[] probs, int k)
    {
        if (k <= 0 || k >= probs.Length) return;
        
        var indices = Enumerable.Range(0, probs.Length).ToArray();
        Array.Sort(indices, (a, b) => probs[b].CompareTo(probs[a]));
        
        for (int i = k; i < indices.Length; i++)
        {
            probs[indices[i]] = 0.0;
        }
        
        var sum = probs.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
    }

    private static void ApplyTopP(double[] probs, double p)
    {
        if (p >= 1.0) return;
        
        var indices = Enumerable.Range(0, probs.Length).ToArray();
        Array.Sort(indices, (a, b) => probs[b].CompareTo(probs[a]));
        
        double cumulative = 0.0;
        int cutoff = probs.Length;
        
        for (int i = 0; i < indices.Length; i++)
        {
            cumulative += probs[indices[i]];
            if (cumulative >= p)
            {
                cutoff = i + 1;
                break;
            }
        }
        
        for (int i = cutoff; i < indices.Length; i++)
        {
            probs[indices[i]] = 0.0;
        }
        
        var sum = probs.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
    }

    private static void ApplyTailFreeSampling(double[] probs, double z)
    {
        if (z >= 1.0) return;
        
        var indices = Enumerable.Range(0, probs.Length).ToArray();
        Array.Sort(indices, (a, b) => probs[b].CompareTo(probs[a]));
        
        var derivatives = new double[probs.Length - 1];
        for (int i = 0; i < derivatives.Length; i++)
        {
            derivatives[i] = Math.Abs(probs[indices[i]] - probs[indices[i + 1]]);
        }
        
        var normDerivatives = derivatives.Select(d => d / derivatives.Sum()).ToArray();
        
        double cumulative = 0.0;
        int cutoff = probs.Length;
        
        for (int i = 0; i < normDerivatives.Length; i++)
        {
            cumulative += normDerivatives[i];
            if (cumulative >= z)
            {
                cutoff = i + 1;
                break;
            }
        }
        
        for (int i = cutoff; i < indices.Length; i++)
        {
            probs[indices[i]] = 0.0;
        }
        
        var sum = probs.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
    }

    private static void ApplyTypicalSampling(double[] probs, double p)
    {
        if (p >= 1.0) return;
        
        var entropy = -probs.Where(x => x > 0).Sum(x => x * Math.Log(x));
        var surprisals = probs.Select(x => x > 0 ? -Math.Log(x) : double.PositiveInfinity).ToArray();
        var deviations = surprisals.Select(s => Math.Abs(s - entropy)).ToArray();
        
        var indices = Enumerable.Range(0, probs.Length).ToArray();
        Array.Sort(indices, (a, b) => deviations[a].CompareTo(deviations[b]));
        
        double cumulative = 0.0;
        int cutoff = 0;
        
        for (int i = 0; i < indices.Length; i++)
        {
            if (probs[indices[i]] > 0)
            {
                cumulative += probs[indices[i]];
                if (cumulative >= p)
                {
                    cutoff = i + 1;
                    break;
                }
            }
        }
        
        for (int i = cutoff; i < indices.Length; i++)
        {
            probs[indices[i]] = 0.0;
        }
        
        var sum = probs.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
    }

    private static int SampleCategorical(double[] probs, Random rng)
    {
        var r = rng.NextDouble();
        double cumulative = 0.0;
        
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        
        return probs.Length - 1;
    }
}


