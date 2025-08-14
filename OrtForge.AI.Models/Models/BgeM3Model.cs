using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using OrtForge.AI.Models.Astractions;
using OrtForge.AI.Models.Options;

namespace OrtForge.AI.Models.Models;

/// <summary>
/// Provides functionality for working with the BGE-M3 embedding model using ONNX Runtime.
/// </summary>
public sealed class BgeM3Model : ModelHostBase
{
    /// <summary>
    /// Creates a new instance of the BgeM3Model.
    /// </summary>
    /// <param name="options"></param>
    public BgeM3Model(BgeM3Options options) : base(options)
    {
        OutputTensorType = options.TensorElementType;
    }

    /// <summary>
    /// Creates embeddings from text input asynchronously.
    /// </summary>
    /// <param name="text">Input text to embed</param>
    /// <param name="modelOutputName"></param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Normalized embedding vector</returns>
    /// <remarks>The majority of models exported onto ONNX with normalization enforced by default, so normalize parameter should be kept as false.</remarks>
    public async Task<float[]> CreateEmbeddingAsync(string text, string modelOutputName = "sentence_embedding", CancellationToken cancellationToken = default)
    {
        var (result, _) = await ExecuteModelAsync(text, modelOutputName, false, cancellationToken);
        return result;
    }

    /// <summary>
    /// Creates embeddings for multiple texts in batch asynchronously
    /// </summary>
    /// <param name="texts">Input texts to embed</param>
    /// <param name="modelOutputName"></param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Array of normalized embedding vectors</returns>
    /// <remarks>The majority of models exported onto ONNX with normalization enforced by default, so normalize parameter should be kept as false.</remarks>
    public async Task<float[][]> CreateEmbeddingsAsync(IReadOnlyCollection<string> texts, string modelOutputName = "sentence_embedding", CancellationToken cancellationToken = default)
    {
        if (texts.Count == 0)
        {
            return [];
        }

        var results = new float[texts.Count][];

        var index = 0;
        foreach (var text in texts) {
            results[index] = await CreateEmbeddingAsync(text, modelOutputName, cancellationToken);
            index++;
        }
        
        return results;
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
}