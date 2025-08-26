using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtAgent.Core.Generation;

namespace OrtAgent.Core.LLM;

public static class LlamaOptimizations
{
    public static readonly Dictionary<string, int[]> ModelStopTokens = new()
    {
        ["llama-3.1"] = new[] { 128001, 128009 },
        ["llama-3.2"] = new[] { 128001, 128009 },
        ["llama-3"] = new[] { 128001, 128009 },
        ["llama-2"] = new[] { 2 },
        ["default"] = new[] { 0, 2 }
    };

    public static readonly Dictionary<string, string[]> ModelStopSequences = new()
    {
        ["llama-3.1"] = new[] { "<|eot_id|>", "<|end_of_text|>" },
        ["llama-3.2"] = new[] { "<|eot_id|>", "<|end_of_text|>" },
        ["llama-3"] = new[] { "<|eot_id|>", "<|end_of_text|>" },
        ["llama-2"] = new[] { "</s>" },
        ["default"] = Array.Empty<string>()
    };

    public static InferenceConfig GetOptimalConfigForModel(string modelName, InferenceConfig? baseConfig = null)
    {
        baseConfig ??= InferenceConfig.Default;
        
        var modelKey = GetModelKey(modelName);
        var stopTokenIds = ModelStopTokens.GetValueOrDefault(modelKey, ModelStopTokens["default"]);
        var stopSequences = ModelStopSequences.GetValueOrDefault(modelKey, ModelStopSequences["default"]);

        return baseConfig with
        {
            StopTokenIds = new HashSet<int>(stopTokenIds.Concat(baseConfig.StopTokenIds)),
            StopSequences = stopSequences.Concat(baseConfig.StopSequences).ToArray(),
            Temperature = IsLlama3Family(modelKey) ? Math.Max(0.1, baseConfig.Temperature) : baseConfig.Temperature,
            TopP = IsLlama3Family(modelKey) ? Math.Min(0.95, baseConfig.TopP) : baseConfig.TopP
        };
    }

    public static DenseTensor<long>? CreateOptimalPositionIds(int sequenceLength, int currentStep, string modelName)
    {
        var modelKey = GetModelKey(modelName);
        
        if (!RequiresPositionIds(modelKey))
        {
            return null;
        }

        var positionIds = new DenseTensor<long>(new[] { 1, 1 });
        positionIds[0, 0] = sequenceLength + currentStep;
        return positionIds;
    }

    public static DenseTensor<int>? CreateOptimalAttentionMask(int totalSequenceLength, string modelName)
    {
        var modelKey = GetModelKey(modelName);
        
        if (!RequiresAttentionMask(modelKey))
        {
            return null;
        }

        var attentionMask = new DenseTensor<int>(new[] { 1, totalSequenceLength });
        for (int i = 0; i < totalSequenceLength; i++)
        {
            attentionMask[0, i] = 1;
        }
        return attentionMask;
    }

    public static int GetOptimalKvCacheSize(string modelName, int maxSequenceLength)
    {
        var modelKey = GetModelKey(modelName);
        
        return modelKey switch
        {
            "llama-3.1" or "llama-3.2" => Math.Min(maxSequenceLength, 131072),
            "llama-3" => Math.Min(maxSequenceLength, 8192),
            "llama-2" => Math.Min(maxSequenceLength, 4096),
            _ => maxSequenceLength
        };
    }

    public static bool ShouldUseGQA(string modelName)
    {
        var modelKey = GetModelKey(modelName);
        return IsLlama3Family(modelKey);
    }

    public static int GetOptimalBatchSize(string modelName)
    {
        var modelKey = GetModelKey(modelName);
        
        return modelKey switch
        {
            "llama-3.1" or "llama-3.2" => 1,
            "llama-3" => 1,
            "llama-2" => 2,
            _ => 1
        };
    }

    private static string GetModelKey(string modelName)
    {
        var lower = modelName.ToLowerInvariant();
        
        if (lower.Contains("llama-3.2") || lower.Contains("llama3.2"))
            return "llama-3.2";
        if (lower.Contains("llama-3.1") || lower.Contains("llama3.1"))
            return "llama-3.1";
        if (lower.Contains("llama-3") || lower.Contains("llama3"))
            return "llama-3";
        if (lower.Contains("llama-2") || lower.Contains("llama2"))
            return "llama-2";
        
        return "default";
    }

    private static bool IsLlama3Family(string modelKey)
    {
        return modelKey is "llama-3" or "llama-3.1" or "llama-3.2";
    }

    private static bool RequiresPositionIds(string modelKey)
    {
        return modelKey switch
        {
            "llama-3.1" or "llama-3.2" => false,
            "llama-3" => false,
            "llama-2" => true,
            _ => false
        };
    }

    private static bool RequiresAttentionMask(string modelKey)
    {
        return modelKey switch
        {
            "llama-3.1" or "llama-3.2" => false,
            "llama-3" => false,
            "llama-2" => true,
            _ => true
        };
    }
}
