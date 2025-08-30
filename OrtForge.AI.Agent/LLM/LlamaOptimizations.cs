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

        return baseConfig with
        {
            StopTokenIds = [..stopTokenIds.Concat(baseConfig.StopTokenIds)],
            StopSequences = stopSequences.Concat(baseConfig.StopSequences).ToArray(),
            Temperature = modelType.IsLlama3Family() ? Math.Max(0.1, baseConfig.Temperature) : baseConfig.Temperature,
            TopP = modelType.IsLlama3Family() ? Math.Min(0.95, baseConfig.TopP) : baseConfig.TopP
        };
    }

    public static long[] CreateOptimalPositionIds(int sequenceLength, int currentStep)
    {
        if (currentStep == 0)
        {
            var positionIds = new long[sequenceLength];
            for (int i = 0; i < sequenceLength; i++)
            {
                positionIds[i] = i;
            }
            return positionIds;
        }
        else
        {
            var posId = new long[] { sequenceLength - 1 };
            return posId;
        }
    }

    public static long[]? CreateOptimalAttentionMask(int totalSequenceLength)
    {
        var attentionMask = new long[totalSequenceLength];
        Array.Fill(attentionMask, 1L);
        return attentionMask;
    }
}
