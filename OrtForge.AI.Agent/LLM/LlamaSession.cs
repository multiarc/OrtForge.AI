using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OrtAgent.Core.LLM;

public sealed class LlamaSession : IDisposable
{
    private readonly InferenceSession _session;

    public LlamaSession(InferenceSession session)
    {
        _session = session;
    }

    public void Dispose() => _session.Dispose();

    public sealed record StepInputs(
        DenseTensor<int> InputIds,
        Dictionary<string, DenseTensor<float>>? KvCache,
        DenseTensor<long>? PositionIds,
        DenseTensor<int>? AttentionMask);

    public sealed record StepOutputs(
        DenseTensor<float> Logits,
        Dictionary<string, DenseTensor<float>> KvCache);

    public StepOutputs RunStep(StepInputs inputs)
    {
        var inputNames = _session.InputMetadata.Keys.ToArray();
        var container = new List<NamedOnnxValue>();

        if (!inputNames.Contains("input_ids"))
            throw new InvalidOperationException("Model expects 'input_ids'.");
        container.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputs.InputIds));

        if (inputs.PositionIds != null && inputNames.Contains("position_ids"))
            container.Add(NamedOnnxValue.CreateFromTensor("position_ids", inputs.PositionIds));
        if (inputs.AttentionMask != null && inputNames.Contains("attention_mask"))
            container.Add(NamedOnnxValue.CreateFromTensor("attention_mask", inputs.AttentionMask));

        // Feed KV cache if provided, with normalization for common present->past naming
        if (inputs.KvCache != null && inputs.KvCache.Count > 0)
        {
            foreach (var kv in inputs.KvCache)
            {
                // 1) Exact match
                if (inputNames.Contains(kv.Key))
                {
                    container.Add(NamedOnnxValue.CreateFromTensor(kv.Key, kv.Value));
                    continue;
                }

                // 2) Try mapping common output "present" names to input "past" names
                var mapped = MapKvNameToInput(kv.Key, inputNames);
                if (mapped != null)
                {
                    container.Add(NamedOnnxValue.CreateFromTensor(mapped, kv.Value));
                }
                // else: silently ignore non-matching cache entries
            }
        }

        using var results = _session.Run(container);

        DenseTensor<float>? logits = null;
        var newKv = new Dictionary<string, DenseTensor<float>>();
        foreach (var r in results)
        {
            if (string.Equals(r.Name, "logits", StringComparison.OrdinalIgnoreCase))
            {
                logits = (DenseTensor<float>)r.AsTensor<float>();
            }
            else
            {
                var t = r.AsTensor<float>();
                if (t is DenseTensor<float> dt)
                {
                    newKv[r.Name] = dt;

                    // Also store an alias for the next step if inputs expect "past_*" but outputs gave "present_*"
                    var alias = MapKvOutputToPastAlias(r.Name);
                    if (alias != null && !newKv.ContainsKey(alias))
                    {
                        newKv[alias] = dt;
                    }
                }
            }
        }

        if (logits is null)
            throw new InvalidOperationException("Model did not return 'logits'.");

        return new StepOutputs(logits, newKv);
    }

    private static string? MapKvNameToInput(string outputLikeName, string[] inputNames)
    {
        // Try several common mappings used by Llama ONNX exports
        // present_key_values.* -> past_key_values.*
        if (outputLikeName.StartsWith("present_key_values", StringComparison.Ordinal))
        {
            var candidate = "past_" + outputLikeName.Substring("present_".Length);
            if (inputNames.Contains(candidate)) return candidate;
        }
        // present.* -> past_key_values.*
        if (outputLikeName.StartsWith("present.", StringComparison.Ordinal))
        {
            var candidate = "past_key_values" + outputLikeName.Substring("present".Length);
            if (inputNames.Contains(candidate)) return candidate;
        }
        // Generic swap of "present"->"past"
        if (outputLikeName.Contains("present"))
        {
            var candidate = outputLikeName.Replace("present", "past_key_values");
            if (inputNames.Contains(candidate)) return candidate;
        }
        return null;
    }

    private static string? MapKvOutputToPastAlias(string outputName)
    {
        if (outputName.StartsWith("present_key_values", StringComparison.Ordinal))
        {
            return "past_" + outputName.Substring("present_".Length);
        }
        if (outputName.StartsWith("present.", StringComparison.Ordinal))
        {
            return "past_key_values" + outputName.Substring("present".Length);
        }
        if (outputName.Contains("present"))
        {
            return outputName.Replace("present", "past_key_values");
        }
        return null;
    }
}


