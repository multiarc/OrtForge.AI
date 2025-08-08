using System.Runtime.InteropServices;
using OrtForge.AI.Models.Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace OrtForge.AI.Models.Models;

/// <summary>
/// Provides functionality for working with the BGE-M3 embedding model using ONNX Runtime.
/// </summary>
public sealed class BgeM3Model : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _modelPath;
    private readonly Tokenizer _tokenizer;
    private readonly int _maxLength;

    /// <summary>
    /// Creates a new instance of the BgeM3Model.
    /// </summary>
    /// <param name="tokenizerModelPath">Path to the Sentence Piece model file (tokenization)</param>
    /// <param name="modelPath">Path to the ONNX model file (embedding)</param>
    /// <param name="maxLength">Maximum input sequence length, actual limit is 8192 tokens that is not directly mappable to characters</param>
    public BgeM3Model(string tokenizerModelPath, string modelPath, int maxLength = 51200)
    {
        _modelPath = modelPath;
        _maxLength = maxLength;
        
        if (!File.Exists(_modelPath))
        {
            throw new FileNotFoundException($"Model file not found at path: {_modelPath}");
        }
        
        using var file = File.OpenRead(tokenizerModelPath);
        _tokenizer = SentencePieceTokenizer.Create(file);
        
        using var sessionOptions = SessionOptions.MakeSessionOptionWithRocmProvider();
        sessionOptions.AppendExecutionProvider_CPU();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
        _session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <summary>
    /// Creates embeddings from text input asynchronously.
    /// </summary>
    /// <param name="text">Input text to embed</param>
    /// <param name="postNormalize">Whether to normalize the resulting embedding vector</param>
    /// <param name="modelOutputName"></param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Normalized embedding vector</returns>
    /// <remarks>The majority of models exported onto ONNX with normalization enforced by default, so normalize parameter should be kept as false.</remarks>
    //TODO: output normalized text as well
    public async Task<float[]> CreateEmbeddingAsync(string text, bool postNormalize = false, string modelOutputName = "sentence_embedding", CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("Text cannot be null or empty", nameof(text));
        }
        cancellationToken.ThrowIfCancellationRequested();
        
        var tokens = _tokenizer.EncodeToTokens(text, out var normalizedText);

        var ids = new long[tokens.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = tokens[i].Id;
        }
        
        var attentionMask = new long[tokens.Count];
        Array.Fill(attentionMask, 1);
        
        cancellationToken.ThrowIfCancellationRequested();

        using var inputs = OrtValue.CreateTensorValueFromMemory(
            ids,
            [1, tokens.Count]
        );
        
        using var mask = OrtValue.CreateTensorValueFromMemory(
            attentionMask,
            [1, tokens.Count]
        );
        
        var inputNames = new[] { "input_ids", "attention_mask" };
        var inputValues = new[] { inputs, mask };
        
        var outputMetadata = _session.OutputMetadata[modelOutputName];
        var outputShape = outputMetadata.Dimensions;
        var totalSize = outputShape[1];

        using var outputTensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance, TensorElementType.Float16,
            [1, totalSize]
        );


        try {
            using var runOptions = new RunOptions();
            var values = await _session.RunAsync(runOptions, inputNames, inputValues, [modelOutputName], [outputTensor]);

            Half[] result;
            if (postNormalize) {
                result = MemoryMarshal.Cast<byte, Half>(outputTensor.GetTensorMutableDataAsSpan<byte>()).Normalize().ToArray();
            }
            else {
                result = MemoryMarshal.Cast<byte, Half>(outputTensor.GetTensorMutableDataAsSpan<byte>()).ToArray();
            }

            foreach (var value in values) {
                value.Dispose();
            }

            return result.Select(x => (float)x).ToArray();
        }
        catch (Exception ex) {
            throw new InvalidOperationException($"Error running the model: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Creates embeddings for multiple texts in batch asynchronously
    /// </summary>
    /// <param name="texts">Input texts to embed</param>
    /// <param name="postNormalize">Whether to normalize the resulting embedding vector</param>
    /// <param name="modelOutputName"></param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Array of normalized embedding vectors</returns>
    /// <remarks>The majority of models exported onto ONNX with normalization enforced by default, so normalize parameter should be kept as false.</remarks>
    public async Task<float[][]> CreateEmbeddingsAsync(IReadOnlyCollection<string> texts, bool postNormalize = false, string modelOutputName = "sentence_embedding", CancellationToken cancellationToken = default)
    {
        if (texts.Count == 0)
        {
            return [];
        }

        var results = new float[texts.Count][];

        var index = 0;
        foreach (var text in texts) {
            results[index] = await CreateEmbeddingAsync(text, postNormalize, modelOutputName, cancellationToken);
            index++;
        }
        
        return results;
    }

    /// <summary>
    /// Get model information
    /// </summary>
    public ModelInfo GetModelInfo()
    {
        return new ModelInfo
        {
            ModelPath = _modelPath,
            MaxLength = _maxLength,
            InputNames = _session.InputMetadata.Keys.ToArray(),
            OutputNames = _session.OutputMetadata.Keys.ToArray()
        };
    }

    /// <summary>
    /// Dispose of resources
    /// </summary>
    public void Dispose() {
        _session.Dispose();
    }
}