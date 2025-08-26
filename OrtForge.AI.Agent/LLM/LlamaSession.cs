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

        if (inputs.KvCache != null)
        {
            foreach (var kv in inputs.KvCache)
            {
                if (inputNames.Contains(kv.Key))
                {
                    container.Add(NamedOnnxValue.CreateFromTensor(kv.Key, kv.Value));
                }
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
                    newKv[r.Name] = dt;
            }
        }

        if (logits is null)
            throw new InvalidOperationException("Model did not return 'logits'.");

        return new StepOutputs(logits, newKv);
    }
}


