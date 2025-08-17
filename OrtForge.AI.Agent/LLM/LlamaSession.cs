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
        Dictionary<string, OrtValue>? KvCache,
        DenseTensor<long>? PositionIds,
        DenseTensor<int>? AttentionMask);

    public sealed record StepOutputs(
        DenseTensor<float> Logits,
        Dictionary<string, OrtValue> KvCache);

    public StepOutputs RunStep(StepInputs inputs)
    {
        var inputNames = _session.InputMetadata.Keys.ToArray();
        var container = new List<NamedOnnxValue>();

        // Common inputs
        if (TryBind(inputNames, "input_ids", OrtValue.CreateFromTensor(inputs.InputIds), container) == false)
            throw new InvalidOperationException("Model expects 'input_ids'.");

        if (TryBind(inputNames, "position_ids", inputs.PositionIds is null ? null : OrtValue.CreateFromTensor(inputs.PositionIds), container)) { }
        if (TryBind(inputNames, "attention_mask", inputs.AttentionMask is null ? null : OrtValue.CreateFromTensor(inputs.AttentionMask), container)) { }

        if (inputs.KvCache != null)
        {
            foreach (var kv in inputs.KvCache)
            {
                if (inputNames.Contains(kv.Key))
                {
                    container.Add(NamedOnnxValue.CreateFromOrtValue(kv.Key, kv.Value));
                }
            }
        }

        using var results = _session.Run(container);

        DenseTensor<float>? logits = null;
        var newKv = new Dictionary<string, OrtValue>();
        foreach (var r in results)
        {
            if (string.Equals(r.Name, "logits", StringComparison.OrdinalIgnoreCase))
            {
                logits = (DenseTensor<float>)r.AsTensor<float>();
            }
            else if (r.Value is OrtValue ov)
            {
                newKv[r.Name] = ov; // kv-cache tensors come as OrtValue with device placement; keep reference
            }
        }

        if (logits is null)
            throw new InvalidOperationException("Model did not return 'logits'.");

        return new StepOutputs(logits, newKv);
    }

    private static bool TryBind(string[] inputNames, string name, OrtValue? value, List<NamedOnnxValue> dst)
    {
        if (!inputNames.Contains(name) || value is null) return false;
        dst.Add(NamedOnnxValue.CreateFromOrtValue(name, value));
        return true;
    }
}


