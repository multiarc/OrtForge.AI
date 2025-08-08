using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace OrtForge.AI.Models.Models;

/// <summary>
/// Provides functionality for working with the BGE-M3 embedding model using ONNX Runtime.
/// </summary>
public sealed class BgeRerankerM3 : IDisposable
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
    public BgeRerankerM3(string tokenizerModelPath, string modelPath, int maxLength = 51200)
    {
        _modelPath = modelPath;
        _maxLength = maxLength;
        
        if (!File.Exists(_modelPath))
        {
            throw new FileNotFoundException($"Model file not found at path: {_modelPath}");
        }
        
        using var file = File.OpenRead(tokenizerModelPath);
        _tokenizer = SentencePieceTokenizer.Create(file);
        //var providers = OrtEnv.Instance().GetAvailableProviders();
        
        using var sessionOptions = SessionOptions.MakeSessionOptionWithRocmProvider();
        sessionOptions.AppendExecutionProvider_CPU();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
        _session = new InferenceSession(modelPath, sessionOptions);
    }

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
        
        var tokens = _tokenizer.EncodeToTokens(input, out _);

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

        using var runOptions = new RunOptions();
        await _session.RunAsync(runOptions, inputNames, inputValues, [modelOutputName], [outputTensor]);
        return GetSigmoid((float)MemoryMarshal.Cast<byte, Half>(outputTensor.GetTensorMutableDataAsSpan<byte>())[0]);
    }
    
    private static float GetSigmoid(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
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