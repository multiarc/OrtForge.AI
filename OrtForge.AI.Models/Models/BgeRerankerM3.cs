using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using OrtForge.AI.Models.Astractions;
using OrtForge.AI.Models.Options;

namespace OrtForge.AI.Models.Models;

/// <summary>
/// Provides functionality for working with the BGE-M3 embedding model using ONNX Runtime.
/// </summary>
public sealed class BgeRerankerM3 : ModelHostBase
{
    public BgeRerankerM3(BgeM3Options options) : base(options)
    {
        OutputTensorType = options.TensorElementType;
    }

    public override Tokenizer CreateTokenizer(Stream tokenizerModelStream)
    {
        return SentencePieceTokenizer.Create(tokenizerModelStream);
    }

    protected override (OrtValue inputTokens, OrtValue mask) CreateInputTensor(IReadOnlyList<EncodedToken> tokens)
    {
        var ids = new long[tokens.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = tokens[i].Id;
        }

        var attentionMask = new long[tokens.Count];
        Array.Fill(attentionMask, 1);

        var inputs = OrtValue.CreateTensorValueFromMemory(
            ids,
            [1, tokens.Count]
        );

        var mask = OrtValue.CreateTensorValueFromMemory(
            attentionMask,
            [1, tokens.Count]
        );

        return (inputs, mask);
    }
    
    protected override TensorElementType OutputTensorType { get; }

    public async Task<float> GetRerankingScoreAsync(string query, string document, string modelOutputName = "logits", CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Text cannot be null or empty", nameof(query));
        }
        if (string.IsNullOrWhiteSpace(document))
        {
            throw new ArgumentException("Text cannot be null or empty", nameof(document));
        }
        
        var input = $"{query}[SEP]{document}";
        
        cancellationToken.ThrowIfCancellationRequested();
        
        var (output, _) = await ExecuteModelAsync(input, modelOutputName, false, cancellationToken);
        
        return GetSigmoid(output.First());
    }
    
    private static float GetSigmoid(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }
}