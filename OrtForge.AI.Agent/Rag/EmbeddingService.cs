using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OrtAgent.Core.Rag;

public sealed class EmbeddingService : IDisposable
{
    private readonly InferenceSession _session;

    public EmbeddingService(InferenceSession session)
    {
        _session = session;
    }

    public void Dispose() => _session.Dispose();

    public float[] EmbedTokenIds(int[] tokenIds)
    {
        var inputIds = new DenseTensor<int>(new[] { 1, tokenIds.Length });
        for (int i = 0; i < tokenIds.Length; i++) inputIds[0, i] = tokenIds[i];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds)
        };
        using var results = _session.Run(inputs);
        var first = results.First();
        var tensor = (DenseTensor<float>)first.AsTensor<float>();
        // assume [1, D] or [1, 1, D]
        return tensor.Buffer.Span.ToArray();
    }
}


