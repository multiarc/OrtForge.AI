using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using OrtForge.AI.Models.Astractions.Extensions;

namespace OrtForge.AI.Models.Astractions;

public abstract class ModelHostBase : IDisposable
{
    private readonly BaseModelOptions _options;
    private InferenceSession? _session;
    private Tokenizer? _tokenizer;

    /// <summary>
    /// Creates a new instance of the BgeM3Model.
    /// </summary>
    /// <param name="options">Model Options</param>
    protected ModelHostBase(BaseModelOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <exception cref="FileNotFoundException"></exception>
    public virtual void Initialize(ExecutionMode mode = ExecutionMode.ORT_PARALLEL, OrtLoggingLevel loggingLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, ExecutionProvider providers = ExecutionProvider.CPU)
    {
        var options = CreateDefaultOptions(mode, loggingLevel, optimizationLevel, providers);
        using var file = File.OpenRead(_options.TokenizerModelPath);
        _tokenizer = CreateTokenizer(file);
        _session = new InferenceSession(_options.ModelPath, options);
    }

    public abstract Tokenizer CreateTokenizer(Stream tokenizerModelStream);

    public virtual SessionOptions CreateDefaultOptions(ExecutionMode mode, OrtLoggingLevel loggingLevel, GraphOptimizationLevel optimizationLevel, ExecutionProvider providers)
    {
        var sessionOptions = new SessionOptions();
        if (providers.HasFlag(ExecutionProvider.ROCm))
        {
            sessionOptions.AppendExecutionProvider_ROCm();
        }
        if (providers.HasFlag(ExecutionProvider.CUDA))
        {
            sessionOptions.AppendExecutionProvider_CUDA();
        }
        if (providers.HasFlag(ExecutionProvider.OpenVINO))
        {
            sessionOptions.AppendExecutionProvider_OpenVINO();
        }
        if (providers.HasFlag(ExecutionProvider.MIGraphX))
        {
            sessionOptions.AppendExecutionProvider_MIGraphX();
        }
        if (providers.HasFlag(ExecutionProvider.CoreML))
        {
            sessionOptions.AppendExecutionProvider_CoreML();
        }
        if (providers.HasFlag(ExecutionProvider.DirectML))
        {
            sessionOptions.AppendExecutionProvider_DML();
        }
        if (providers.HasFlag(ExecutionProvider.TensorRT))
        {
            sessionOptions.AppendExecutionProvider_Tensorrt();
        }
        if (providers.HasFlag(ExecutionProvider.NNAPI))
        {
            sessionOptions.AppendExecutionProvider_Nnapi();
        }
        if (providers.HasFlag(ExecutionProvider.oneDNN))
        {
            sessionOptions.AppendExecutionProvider_Dnnl();
        }
        if (providers.HasFlag(ExecutionProvider.CPU))
        {
            sessionOptions.AppendExecutionProvider_CPU();
        }
        sessionOptions.GraphOptimizationLevel = optimizationLevel;
        sessionOptions.ExecutionMode = mode;
        sessionOptions.LogSeverityLevel = loggingLevel;

        return sessionOptions;
    }
    
    protected abstract (OrtValue inputTokens, OrtValue mask) CreateInputTensor(IReadOnlyList<EncodedToken> tokens);
    protected virtual TensorElementType OutputTensorType => TensorElementType.Float;

    public async Task<(float[], string? normalizedText)> ExecuteModelAsync(string text, string modelOutputName, bool postNormalize = false, CancellationToken cancellationToken = default)
    {
        if (_tokenizer == null)
        {
            throw new InvalidOperationException("Tokenizer not initialized");
        }
        
        if (_session == null)
        {
            throw new InvalidOperationException("Session not initialized");
        }
        
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("Text cannot be null or empty", nameof(text));
        }

        if (text.Length > _options.MaxInputLength)
        {
            throw new ArgumentException($"Text length cannot exceed {_options.MaxInputLength}", nameof(text));
        }
        
        cancellationToken.ThrowIfCancellationRequested();
        
        var tokens = _tokenizer.EncodeToTokens(text, out var normalizedText);

        var (inputs, mask) = CreateInputTensor(tokens);
        using (inputs)
        {
            using (mask)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var inputNames = new[] { "input_ids", "attention_mask" };
                var inputValues = new[] { inputs, mask };

                var outputMetadata = _session.OutputMetadata[modelOutputName];
                var outputShape = outputMetadata.Dimensions;
                var totalSize = outputShape[1];

                using var outputTensor = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, OutputTensorType,
                    [1, totalSize]
                );

                try
                {
                    using var runOptions = new RunOptions();
                    await _session.RunAsync(runOptions, inputNames, inputValues, [modelOutputName], [outputTensor]);

                    return (MapOutputTensor(outputTensor, postNormalize), normalizedText);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Error running the model: {ex.Message}", ex);
                }
            }
        }
    }
    
    protected virtual float[] MapOutputTensor(OrtValue outputTensor, bool postNormalize)
    {
        switch (OutputTensorType)
        {
            case TensorElementType.Float:
                var span = outputTensor.GetTensorMutableDataAsSpan<float>();
                if (postNormalize)
                {
                    span.Normalize();
                }
                return span.ToArray();
            case TensorElementType.Float16:
                var fbyteSpan = outputTensor.GetTensorMutableDataAsSpan<byte>();
                var span16 = MemoryMarshal.Cast<byte, Half>(fbyteSpan);
                if (postNormalize)
                {
                    span16.Normalize();
                }
                var dst16 = GC.AllocateUninitializedArray<float>(span16.Length);
                for (int i = 0; i < span16.Length; i++)
                {
                    dst16[i] = (float)span16[i];
                }
                return dst16;
            case TensorElementType.BFloat16:
                var bfbyteSpan = outputTensor.GetTensorMutableDataAsSpan<byte>();
                var spanb16 = MemoryMarshal.Cast<byte, BFloat16>(bfbyteSpan);
                if (postNormalize)
                {
                    spanb16.Normalize();
                }
                var dstb16 = GC.AllocateUninitializedArray<float>(spanb16.Length);
                for (int i = 0; i < spanb16.Length; i++)
                {
                    dstb16[i] = (float)spanb16[i];
                }
                return dstb16;
            case TensorElementType.Double:
                var span64 = outputTensor.GetTensorMutableDataAsSpan<double>();
                if (postNormalize)
                {
                    span64.Normalize();
                }
                var dst64 = GC.AllocateUninitializedArray<float>(span64.Length);
                for (int i = 0; i < span64.Length; i++)
                {
                    dst64[i] = (float)span64[i];
                }
                return dst64;
            case TensorElementType.UInt8:
            case TensorElementType.Int8:
            case TensorElementType.UInt16:
            case TensorElementType.Int16:
            case TensorElementType.Int32:
            case TensorElementType.Int64:
            case TensorElementType.UInt32:
            case TensorElementType.UInt64:
                throw new NotSupportedException($"{OutputTensorType} is not supported.");
            default:
                throw new ArgumentOutOfRangeException();
        }
    }

    /// <summary>
    /// Get model information
    /// </summary>
    public ModelInfo GetModelInfo()
    {
        return new ModelInfo
        {
            ModelPath = _options.ModelPath,
            MaxLength = _options.MaxInputLength,
            InputNames = _session?.InputMetadata.Keys.ToArray(),
            OutputNames = _session?.OutputMetadata.Keys.ToArray()
        };
    }

    /// <summary>
    /// Dispose of resources
    /// </summary>
    public void Dispose() {
        _session?.Dispose();
    }
}