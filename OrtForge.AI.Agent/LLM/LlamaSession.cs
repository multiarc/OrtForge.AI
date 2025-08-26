using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OrtAgent.Core.LLM;

public sealed class LlamaSession : IDisposable
{
    public enum KvStorageType { Float32, Float16, Int4 }

    private readonly InferenceSession _session;
    private readonly KvStorageType _kvType;

    public LlamaSession(InferenceSession session, KvStorageType kvType = KvStorageType.Float32)
    {
        _session = session;
        _kvType = kvType;
    }

    public void Dispose() => _session.Dispose();

    public sealed class KvBlock
    {
        public enum Kind { F32, F16, I4 }
        public Kind Type { get; }
        public DenseTensor<float>? F32 { get; }
        public DenseTensor<System.Half>? F16 { get; }
        public byte[]? I4Packed { get; }
        public float I4Scale { get; }
        public int[] Shape { get; }
        private KvBlock(DenseTensor<float> f32)
        {
            Type = Kind.F32; F32 = f32; Shape = f32.Dimensions.ToArray();
        }
        private KvBlock(DenseTensor<System.Half> f16)
        {
            Type = Kind.F16; F16 = f16; Shape = f16.Dimensions.ToArray();
        }
        private KvBlock(byte[] data, float scale, int[] shape)
        {
            Type = Kind.I4; I4Packed = data; I4Scale = scale; Shape = shape;
        }
        public static KvBlock FromFloat(DenseTensor<float> src, KvStorageType t)
        {
            if (t == KvStorageType.Float32) return new KvBlock(src);
            if (t == KvStorageType.Float16) return new KvBlock(CastFloatToHalf(src));
            var packed = QuantizeInt4(src, out var scale);
            return new KvBlock(packed, scale, src.Dimensions.ToArray());
        }
        public Tensor<float> AsFloatTensor()
        {
            if (Type == Kind.F32 && F32 != null) return F32;
            if (Type == Kind.F16 && F16 != null) return CastHalfToFloat(F16);
            return DequantizeInt4(I4Packed!, I4Scale, Shape);
        }
        public Tensor<System.Half> AsHalfTensor()
        {
            if (Type == Kind.F16 && F16 != null) return F16;
            if (Type == Kind.F32 && F32 != null) return CastFloatToHalf(F32);
            var f32 = DequantizeInt4(I4Packed!, I4Scale, Shape);
            return CastFloatToHalf((DenseTensor<float>)f32);
        }
    }

    public sealed class KvState
    {
        public readonly Dictionary<string, KvBlock> Blocks = new();
        public static KvState Empty => new();
    }

    public sealed record StepInputs(
        DenseTensor<int> InputIds,
        KvState Kv,
        DenseTensor<long>? PositionIds,
        DenseTensor<int>? AttentionMask);

    public sealed record StepOutputs(
        DenseTensor<float> Logits,
        KvState KvCache);

    public StepOutputs RunStep(StepInputs inputs)
    {
        var inputNames = _session.InputMetadata.Keys.ToArray();
        var container = new List<NamedOnnxValue>();
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
        if (inputs.Kv != null && inputs.Kv.Blocks.Count > 0)
        {
            foreach (var kv in inputs.Kv.Blocks)
            {
                string? targetName = null;
                if (inputNames.Contains(kv.Key))
                {
                    targetName = kv.Key;
                }
                else
                {
                    targetName = MapKvNameToInput(kv.Key, inputNames);
                }
                if (targetName == null) continue;
                var meta = _session.InputMetadata[targetName];
                if (meta.ElementType == typeof(System.Half))
                {
                    var halfTensor = kv.Value.AsHalfTensor();
                    container.Add(NamedOnnxValue.CreateFromTensor(targetName, halfTensor));
                }
                else
                {
                    var floatTensor = kv.Value.AsFloatTensor();
                    container.Add(NamedOnnxValue.CreateFromTensor(targetName, floatTensor));
                }
            }
        }

        using var results = _session.Run(container);

        DenseTensor<float>? logits = null;
        DenseTensor<float>? logitsLast = null;
        var newKv = new KvState();
        foreach (var r in results)
        {
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
            var kvFloat = ReadFloatTensorFromOutput(r);
            if (kvFloat != null)
            {
                var block = KvBlock.FromFloat(kvFloat, _kvType);
                newKv.Blocks[r.Name] = block;
                var alias = MapKvOutputToPastAlias(r.Name);
                if (alias != null && !newKv.Blocks.ContainsKey(alias))
                {
                    newKv.Blocks[alias] = block;
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
        if (outputLikeName.StartsWith("present_key_values", StringComparison.Ordinal))
        {
            var candidate = "past_" + outputLikeName.Substring("present_".Length);
            if (inputNames.Contains(candidate)) return candidate;
        }
        if (outputLikeName.StartsWith("present.", StringComparison.Ordinal))
        {
            var candidate = "past_key_values" + outputLikeName.Substring("present".Length);
            if (inputNames.Contains(candidate)) return candidate;
        }
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

    private static byte[] QuantizeInt4(DenseTensor<float> src, out float scale)
    {
        var s = src.Buffer.Span;
        float maxAbs = 0f;
        for (int i = 0; i < s.Length; i++) { var a = Math.Abs(s[i]); if (a > maxAbs) maxAbs = a; }
        scale = maxAbs <= 0 ? 1f : maxAbs / 7f;
        var n = s.Length;
        var bytes = new byte[(n + 1) / 2];
        for (int i = 0; i < n; i += 2)
        {
            int q0 = (int)Math.Round(s[i] / scale);
            if (q0 < -8) q0 = -8; if (q0 > 7) q0 = 7;
            int q1 = 0;
            if (i + 1 < n)
            {
                q1 = (int)Math.Round(s[i + 1] / scale);
                if (q1 < -8) q1 = -8; if (q1 > 7) q1 = 7;
            }
            byte nib0 = (byte)(q0 & 0x0F);
            byte nib1 = (byte)(q1 & 0x0F);
            bytes[i / 2] = (byte)(nib1 << 4 | nib0);
        }
        return bytes;
    }

    private static DenseTensor<float> DequantizeInt4(byte[] data, float scale, int[] shape)
    {
        int n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var dst = new DenseTensor<float>(shape);
        var d = dst.Buffer.Span;
        for (int i = 0; i < n; i += 2)
        {
            var b = data[i / 2];
            int q0 = (sbyte)((b & 0x0F) << 4) >> 4;
            d[i] = q0 * scale;
            if (i + 1 < n)
            {
                int q1 = (sbyte)(b & 0xF0) >> 4;
                d[i + 1] = q1 * scale;
            }
        }
        return dst;
    }
}


