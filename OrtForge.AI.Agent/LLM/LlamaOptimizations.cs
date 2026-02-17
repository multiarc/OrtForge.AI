using OrtForge.AI.Agent.Generation;

namespace OrtForge.AI.Agent.LLM;

public static class LlamaOptimizations
{
    public static readonly Dictionary<ModelType, int[]> ModelStopTokens = new()
    {
        [ModelType.Llama3_1] = [128001, 128009],
        [ModelType.Llama3_2] = [128001, 128009],
        [ModelType.Llama3] = [128001, 128009],
        [ModelType.Llama2] = [2],
        [ModelType.Default] = [0, 2]
    };

    public static readonly Dictionary<ModelType, string[]> ModelStopSequences = new()
    {
        [ModelType.Llama3_1] = ["<|eot_id|>", "<|end_of_text|>"],
        [ModelType.Llama3_2] = ["<|eot_id|>", "<|end_of_text|>"],
        [ModelType.Llama3] = ["<|eot_id|>", "<|end_of_text|>"],
        [ModelType.Llama2] = ["</s>"],
        [ModelType.Default] = []
    };

    public static InferenceConfig GetOptimalConfigForModel(ModelType modelType, InferenceConfig? baseConfig = null)
    {
        baseConfig ??= InferenceConfig.Default;
        
        var stopTokenIds = ModelStopTokens.GetValueOrDefault(modelType, ModelStopTokens[ModelType.Default]);
        var stopSequences = ModelStopSequences.GetValueOrDefault(modelType, ModelStopSequences[ModelType.Default]);

        // Use model-specific stop tokens, only add base config tokens if they're non-empty and valid
        var mergedStopTokens = new HashSet<int>(stopTokenIds);
        foreach (var token in baseConfig.StopTokenIds)
        {
            // Only add tokens > 127999 (special tokens range for Llama 3) or explicitly set
            if (token >= 128000)
            {
                mergedStopTokens.Add(token);
            }
        }

        return baseConfig with
        {
            StopTokenIds = mergedStopTokens,
            StopSequences = stopSequences.Concat(baseConfig.StopSequences).Distinct().ToArray(),
            Temperature = modelType.IsLlama3Family() ? Math.Max(0.1, baseConfig.Temperature) : baseConfig.Temperature,
            TopP = modelType.IsLlama3Family() ? Math.Min(0.95, baseConfig.TopP) : baseConfig.TopP
        };
    }

    /// <summary>
    /// Creates position IDs for the current inference step.
    /// </summary>
    /// <param name="totalSequenceLength">Total sequence length after adding new tokens</param>
    /// <param name="newTokenCount">Number of new tokens being added</param>
    /// <returns>Position IDs array of length newTokenCount</returns>
    public static long[] CreateOptimalPositionIds(int totalSequenceLength, int newTokenCount)
    {
        // Position IDs should be [startPos, startPos+1, ..., startPos+newTokenCount-1]
        // where startPos = totalSequenceLength - newTokenCount
        var startPosition = totalSequenceLength - newTokenCount;
        var positionIds = new long[newTokenCount];
        for (int i = 0; i < newTokenCount; i++)
        {
            positionIds[i] = startPosition + i;
        }
        return positionIds;
    }

    public static long[]? CreateOptimalAttentionMask(int totalSequenceLength)
    {
        var attentionMask = new long[totalSequenceLength];
        Array.Fill(attentionMask, 1L);
        return attentionMask;
    }
}
