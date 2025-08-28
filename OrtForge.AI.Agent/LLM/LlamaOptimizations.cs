using OrtForge.AI.Agent.Generation;

namespace OrtForge.AI.Agent.LLM;

public static class LlamaOptimizations
{
    public static readonly Dictionary<ModelType, int[]> ModelStopTokens = new()
    {
        [ModelType.Llama3_1] = new[] { 128001, 128009 },
        [ModelType.Llama3_2] = new[] { 128001, 128009 },
        [ModelType.Llama3] = new[] { 128001, 128009 },
        [ModelType.Llama2] = new[] { 2 },
        [ModelType.Default] = new[] { 0, 2 }
    };

    public static readonly Dictionary<ModelType, string[]> ModelStopSequences = new()
    {
        [ModelType.Llama3_1] = new[] { "<|eot_id|>", "<|end_of_text|>" },
        [ModelType.Llama3_2] = new[] { "<|eot_id|>", "<|end_of_text|>" },
        [ModelType.Llama3] = new[] { "<|eot_id|>", "<|end_of_text|>" },
        [ModelType.Llama2] = new[] { "</s>" },
        [ModelType.Default] = Array.Empty<string>()
    };

    public static InferenceConfig GetOptimalConfigForModel(ModelType modelType, InferenceConfig? baseConfig = null)
    {
        baseConfig ??= InferenceConfig.Default;
        
        var stopTokenIds = ModelStopTokens.GetValueOrDefault(modelType, ModelStopTokens[ModelType.Default]);
        var stopSequences = ModelStopSequences.GetValueOrDefault(modelType, ModelStopSequences[ModelType.Default]);

        return baseConfig with
        {
            StopTokenIds = new HashSet<int>(stopTokenIds.Concat(baseConfig.StopTokenIds)),
            StopSequences = stopSequences.Concat(baseConfig.StopSequences).ToArray(),
            Temperature = modelType.IsLlama3Family() ? Math.Max(0.1, baseConfig.Temperature) : baseConfig.Temperature,
            TopP = modelType.IsLlama3Family() ? Math.Min(0.95, baseConfig.TopP) : baseConfig.TopP
        };
    }
    
    /// <summary>
    /// Backwards compatibility method - converts string to enum and calls optimized version
    /// </summary>
    [Obsolete("Use GetOptimalConfigForModel(ModelType, InferenceConfig) instead for better performance")]
    public static InferenceConfig GetOptimalConfigForModel(string modelName, InferenceConfig? baseConfig = null)
    {
        var modelType = ModelTypeExtensions.ParseFromString(modelName);
        return GetOptimalConfigForModel(modelType, baseConfig);
    }

    public static long[]? CreateOptimalPositionIds(int sequenceLength, int currentStep, ModelType modelType)
    {
        if (!RequiresPositionIds(modelType))
        {
            return null;
        }

        if (currentStep == 0)
        {
            // First step: create position IDs for all tokens in the sequence [0, 1, 2, ..., sequenceLength-1]
            var positionIds = new long[sequenceLength];
            for (int i = 0; i < sequenceLength; i++)
            {
                positionIds[i] = i;
            }
            return positionIds;
        }
        else
        {
            // FIXED: For subsequent steps, the position ID should be the current sequence length
            // The sequenceLength parameter already includes the step count
            var posId = new long[] { sequenceLength - 1 };
            return posId;
        }
    }

    public static long[]? CreateOptimalAttentionMask(int totalSequenceLength, ModelType modelType)
    {
        if (!RequiresAttentionMask(modelType))
        {
            return null;
        }

        var attentionMask = new long[totalSequenceLength];
        Array.Fill(attentionMask, 1L);
        return attentionMask;
    }

    public static int GetOptimalKvCacheSize(ModelType modelType, int maxSequenceLength)
    {
        return modelType switch
        {
            ModelType.Llama3_1 or ModelType.Llama3_2 => Math.Min(maxSequenceLength, 131072),
            ModelType.Llama3 => Math.Min(maxSequenceLength, 8192),
            ModelType.Llama2 => Math.Min(maxSequenceLength, 4096),
            _ => maxSequenceLength
        };
    }

    public static bool ShouldUseGQA(ModelType modelType)
    {
        return modelType.IsLlama3Family();
    }

    public static int GetOptimalBatchSize(ModelType modelType)
    {
        return modelType switch
        {
            ModelType.Llama3_1 or ModelType.Llama3_2 => 1,
            ModelType.Llama3 => 1,
            ModelType.Llama2 => 2,
            _ => 1
        };
    }

    // Legacy methods kept for backwards compatibility
    [Obsolete("Use ModelType enum instead of string parsing")]
    private static string GetModelKey(string modelName)
    {
        return ModelTypeExtensions.ParseFromString(modelName).ToModelKey();
    }

    [Obsolete("Use ModelType.IsLlama3Family() extension method instead")]
    private static bool IsLlama3Family(string modelKey)
    {
        var modelType = ModelTypeExtensions.ParseFromString(modelKey);
        return modelType.IsLlama3Family();
    }

    private static bool RequiresPositionIds(ModelType modelType)
    {
        return modelType switch
        {
            ModelType.Llama3_1 or ModelType.Llama3_2 => true,  // Provide position IDs for proper generation
            ModelType.Llama3 => true,                           // Provide position IDs for proper generation
            ModelType.Llama2 => true,
            _ => true  // Default to providing position IDs
        };
    }

    private static bool RequiresAttentionMask(ModelType modelType)
    {
        return modelType switch
        {
            ModelType.Llama3_1 or ModelType.Llama3_2 => true,
            ModelType.Llama3 => true,
            ModelType.Llama2 => true,
            _ => true
        };
    }
}
