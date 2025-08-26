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

        // input_ids (cast to expected dtype)
        if (!inputNames.Contains("input_ids"))
            throw new InvalidOperationException("Model expects 'input_ids'.");
        var idsMeta = _session.InputMetadata["input_ids"];
        if (idsMeta.ElementType == typeof(long))
        {
            var cast = CastIntToLong(inputs.InputIds);
            container.Add(NamedOnnxValue.CreateFromTensor("input_ids", cast));
        }
        else
        {
            container.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputs.InputIds));
        }

        // position_ids (optional, ensure dtype)
        if (inputs.PositionIds != null && inputNames.Contains("position_ids"))
        {
            var posMeta = _session.InputMetadata["position_ids"];
            if (posMeta.ElementType == typeof(int))
            {
                var cast = CastLongToInt(inputs.PositionIds);
                container.Add(NamedOnnxValue.CreateFromTensor("position_ids", cast));
            }
            else
            {
                container.Add(NamedOnnxValue.CreateFromTensor("position_ids", inputs.PositionIds));
            }
        }

        // attention_mask (optional, ensure dtype)
        if (inputs.AttentionMask != null && inputNames.Contains("attention_mask"))
        {
            var maskMeta = _session.InputMetadata["attention_mask"];
            if (maskMeta.ElementType == typeof(long))
            {
                var cast = CastIntToLong(inputs.AttentionMask);
                container.Add(NamedOnnxValue.CreateFromTensor("attention_mask", cast));
            }
            else
            {
                container.Add(NamedOnnxValue.CreateFromTensor("attention_mask", inputs.AttentionMask));
            }
        }

        // Feed KV cache if provided, with normalization for common present->past naming
        if (inputs.KvCache != null && inputs.KvCache.Count > 0)
        {
            foreach (var kv in inputs.KvCache)
            {
                string? targetName = null;
                // 1) Exact match
                if (inputNames.Contains(kv.Key))
                {
                    targetName = kv.Key;
                }
                else
                {
                    // 2) Try mapping common output "present" names to input "past" names
                    targetName = MapKvNameToInput(kv.Key, inputNames);
                }
                if (targetName == null) continue; // silently ignore non-matching cache entries

                // dtype-aware bind: convert to fp16 if required
                var meta = _session.InputMetadata[targetName];
                if (meta.ElementType == typeof(System.Half))
                {
                    var halfTensor = CastFloatToHalf(kv.Value);
                    container.Add(NamedOnnxValue.CreateFromTensor(targetName, halfTensor));
                }
                else
                {
                    container.Add(NamedOnnxValue.CreateFromTensor(targetName, kv.Value));
                }
            }
        }

        using var results = _session.Run(container);

        DenseTensor<float>? logits = null;
        DenseTensor<float>? logitsLast = null;
        var newKv = new Dictionary<string, DenseTensor<float>>();
        foreach (var r in results)
        {
            // Prefer logits_last_token if present
            if (string.Equals(r.Name, "logits_last_token", StringComparison.OrdinalIgnoreCase))
            {
                logitsLast = ReadFloatTensorFromOutput(r);
                continue;
            }
            if (string.Equals(r.Name, "logits", StringComparison.OrdinalIgnoreCase))
            {
                logits = ReadFloatTensorFromOutput(r);
                continue;
            }

            // KV tensors: convert to float32 for storage
            var kvFloat = ReadFloatTensorFromOutput(r);
            if (kvFloat != null)
            {
                newKv[r.Name] = kvFloat;
                // Also store an alias for the next step if inputs expect "past_*" but outputs gave "present_*"
                var alias = MapKvOutputToPastAlias(r.Name);
                if (alias != null && !newKv.ContainsKey(alias))
                {
                    newKv[alias] = kvFloat;
                }
            }
        }

        var finalLogits = logitsLast ?? logits;
        if (finalLogits is null)
            throw new InvalidOperationException("Model did not return 'logits' or 'logits_last_token'.");

        return new StepOutputs(finalLogits, newKv);
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

    private static DenseTensor<long> CastIntToLong(DenseTensor<int> src)
    {
        var dims = src.Dimensions.ToArray();
        var dst = new DenseTensor<long>(dims);
        var s = src.Buffer.Span;
        var d = dst.Buffer.Span;
        for (int i = 0; i < s.Length; i++) d[i] = s[i];
        return dst;
    }

    private static DenseTensor<int> CastLongToInt(DenseTensor<long> src)
    {
        var dims = src.Dimensions.ToArray();
        var dst = new DenseTensor<int>(dims);
        var s = src.Buffer.Span;
        var d = dst.Buffer.Span;
        for (int i = 0; i < s.Length; i++) d[i] = checked((int)s[i]);
        return dst;
    }

    private static DenseTensor<System.Half> CastFloatToHalf(DenseTensor<float> src)
    {
        var dims = src.Dimensions.ToArray();
        var dst = new DenseTensor<System.Half>(dims);
        var s = src.Buffer.Span;
        var d = dst.Buffer.Span;
        for (int i = 0; i < s.Length; i++) d[i] = (System.Half)s[i];
        return dst;
    }

    private DenseTensor<float>? ReadFloatTensorFromOutput(NamedOnnxValue r)
    {
        // Use output metadata to decide element type
        var meta = _session.OutputMetadata.ContainsKey(r.Name) ? _session.OutputMetadata[r.Name] : null;
        if (meta != null && meta.ElementType == typeof(System.Half))
        {
            var tHalf = r.AsTensor<System.Half>();
            return CastHalfToFloat(tHalf);
        }
        if (meta != null && meta.ElementType == typeof(float))
        {
            return (DenseTensor<float>)r.AsTensor<float>();
        }
        // Fallback attempts
        try { return (DenseTensor<float>)r.AsTensor<float>(); } catch { }
        try { var th = r.AsTensor<System.Half>(); return CastHalfToFloat(th); } catch { }
        return null;
    }

    private static DenseTensor<float> CastHalfToFloat(Tensor<System.Half> src)
    {
        var dims = src.Dimensions.ToArray();
        var dst = new DenseTensor<float>(dims);
        var d = dst.Buffer.Span;
        int i = 0;
        foreach (var v in src.ToArray())
        {
            d[i++] = (float)v;
        }
        return dst;
    }
}


