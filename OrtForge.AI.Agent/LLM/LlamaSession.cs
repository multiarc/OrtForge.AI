using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Agent.Generation;

namespace OrtForge.AI.Agent.LLM;

public sealed class LlamaSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly KvTensorMappingStrategy _kvMapping;
    private string[] _outputNames;
    private string[] _inputNames;
    private readonly Dictionary<string, KvTensorInfo> _kvOutputs = new();
    private readonly Dictionary<string, KvTensorInfo> _kvInputs = new();

    public LlamaSession(InferenceSession session, ModelType modelType = ModelType.Default)
    {
        _session = session;
        ModelType = modelType;
        
        OptimalConfig = LlamaOptimizations.GetOptimalConfigForModel(modelType);
        
        _kvMapping = KvTensorMappingStrategy.Create(_session.InputMetadata.Keys, _session.OutputMetadata.Keys);

        DiscoverModelInputsAndOutputs();
    }

    public ModelType ModelType { get; }
    public InferenceConfig OptimalConfig { get; }
    
    public void MapInputs(StepInputs inputs, OrtValue[] modelInputs)
    {
        var inputShape = inputs.InputIds.GetTensorTypeAndShape().Shape;
        var batchSize = inputShape[0];
        var currentInputLength = inputShape[1];  // Length of current input tokens
        
        var totalSequenceLength = inputs.Kv.CalculateTotalLengthAfterTokens((int)currentInputLength);
        modelInputs[0] = inputs.InputIds;
        //modelInputs[1] = inputs.PositionIds;
        if (inputs.AttentionMask != null)
        {
            modelInputs[1] = inputs.AttentionMask;
        }
        else
        {
            var defaultAttentionMask = new long[totalSequenceLength];
            Array.Fill(defaultAttentionMask, 1L);
            var attentionMaskOrt = OrtValue.CreateTensorValueFromMemory(defaultAttentionMask, [1, totalSequenceLength]);
            modelInputs[1] = attentionMaskOrt;
        }
        
        if (inputs.Kv.Tensors.Count > 0)
        {
            foreach (var kv in inputs.Kv.Tensors)
            {
                modelInputs[kv.Info.Offset] = kv.Tensor;
            }
        }
        else
        {
            foreach (var kv in _kvInputs.Values)
            {
                kv.Dimensions[0] = batchSize;
                kv.Dimensions[2] = 0L;
                modelInputs[kv.Offset] =
                    OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, kv.ElementType, kv.Dimensions);
            }
        }
    }

    public async Task<StepOutputs> RunStepAsync(StepInputs inputs, CancellationToken cancellationToken = default)
    {
        var inputShape = inputs.InputIds.GetTensorTypeAndShape().Shape;
        var batchSize = inputShape[0];
        var currentInputLength = inputShape[1];
        
        var inputValues = new OrtValue[_inputNames.Length];
        var outputValues = new OrtValue[_outputNames.Length];

        MapInputs(inputs, inputValues);
        var stepOutputs = MapOutputs(inputs, outputValues, batchSize, currentInputLength);

        cancellationToken.ThrowIfCancellationRequested();

        try
        {
            using var runOptions = new RunOptions();
            await _session.RunAsync(runOptions, _inputNames, inputValues, _outputNames, outputValues);
        }
        catch (Exception ex)
        {
            stepOutputs.Dispose();
            throw new InvalidOperationException($"Error running the model: {ex.Message}", ex);
        }

        return stepOutputs;
    }

    private StepOutputs MapOutputs(StepInputs inputs,
        OrtValue[] outputValues, long batchSize, long currentInputLength)
    {
        var logitsMeta = _session.OutputMetadata["logits"];
        var vocabSize = logitsMeta.Dimensions[^1];
        var logitsTensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            logitsMeta.ElementDataType,
            [batchSize, currentInputLength, vocabSize]);
        
        var totalSequenceLength = inputs.Kv.CalculateTotalLengthAfterTokens((int)currentInputLength);
        List<OutputKvTensor> mappedKvTensors = [];
        var newKv = new KvState(mappedKvTensors, totalSequenceLength);
        var outputs = new StepOutputs(logitsTensor, newKv);
        
        outputValues[0] = logitsTensor;
        foreach (var output in _kvOutputs.Values)
        {
            var kvDims = output.Dimensions.Select(d => (long)d).ToArray();
            kvDims[0] = batchSize;
            if (inputs.Kv.Tensors.Count == 0)
            {
                kvDims[2] = currentInputLength;
            }
            else
            {
                kvDims[2] = totalSequenceLength;
            }
            
            var kvTensor = OrtValue.CreateAllocatedTensorValue(
                OrtAllocator.DefaultInstance, 
                output.ElementType, 
                kvDims);
            outputValues[output.Offset] = kvTensor;
            mappedKvTensors.Add(new OutputKvTensor
            {
                Tensor = kvTensor,
                Info = _kvInputs[_kvMapping.MapOutputToInput(output.Name)]
            });
        }

        return outputs;
    }

    private void DiscoverModelInputsAndOutputs()
    {
        var inputMetadata = _session.InputMetadata;
        var outputMetadata = _session.OutputMetadata;
        
        if (!inputMetadata.ContainsKey("input_ids"))
            throw new InvalidOperationException("Model has to have 'input_ids'.");
        
        // if (!inputMetadata.ContainsKey("position_ids"))
        //     throw new InvalidOperationException("Model has to have 'position_ids'.");
        
        if (!inputMetadata.ContainsKey("attention_mask"))
            throw new InvalidOperationException("Model has to have 'attention_mask'.");
        
        if (!outputMetadata.ContainsKey("logits"))
            throw new InvalidOperationException("Model has to have 'logits' as its output.");
        
        var inputNames = new List<string>
        {
            "input_ids",
            //"position_ids",
            "attention_mask"
        };

        var inputOffset = 2;
        foreach (var inputName in inputMetadata.Keys)
        {
            if (_kvMapping.IsKvInput(inputName))
            {
                var inputMeta = inputMetadata[inputName];
                var dimensions = inputMeta.Dimensions.Select(d => (long)d).ToArray();
                _kvInputs.Add(inputName, new KvTensorInfo
                {
                    Name = inputName,
                    Dimensions = dimensions,
                    ElementType = inputMeta.ElementDataType,
                    Offset = inputOffset
                });
                inputOffset++;
                inputNames.Add(inputName);
            }
        }
        
        _inputNames = inputNames.ToArray();
        
        var outputNames = new List<string> { "logits" };
        
        var outputOffset = 1;

        foreach (var outputName in outputMetadata.Keys)
        {
            if (_kvMapping.IsKvOutput(outputName))
            {
                var outputMeta = outputMetadata[outputName];
                var dimensions = outputMeta.Dimensions.Select(d => (long)d).ToArray();
                _kvOutputs.Add(outputName, new KvTensorInfo
                {
                    Name = outputName,
                    Dimensions = dimensions,
                    ElementType = outputMeta.ElementDataType,
                    Offset = outputOffset
                });
                outputOffset++;
                outputNames.Add(outputName);
            }
        }
        
        _outputNames = outputNames.ToArray();
    }

    public async Task<StepOutputs> RunOptimizedStepAsync(long[] inputIds, KvState kv, int currentStep, int sequenceLength, CancellationToken cancellationToken = default)
    {
        //var positionIds = LlamaOptimizations.CreateOptimalPositionIds(sequenceLength, currentStep);
        var attentionMask = LlamaOptimizations.CreateOptimalAttentionMask(sequenceLength);
        
        using var inputs = StepInputs.Create(inputIds, kv, null, attentionMask);
        return await RunStepAsync(inputs, cancellationToken);
    }

    public async Task<StepOutputs> RunOptimizedStep(long[] inputIds, KvState kv, int currentStep, int sequenceLength)
    {
        return await RunOptimizedStepAsync(inputIds, kv, currentStep, sequenceLength, CancellationToken.None);
    }
    
    public void Dispose()
    {
        _session.Dispose();
    }
    
    private static TensorElementType GetTensorElementType(Type type)
    {
        if (type == typeof(float)) return TensorElementType.Float;
        if (type == typeof(Half)) return TensorElementType.Float16;
        if (type.Name == "Float16" || type.FullName?.Contains("OnnxRuntime.Float16") == true) 
            return TensorElementType.Float16;
        if (type == typeof(byte)) return TensorElementType.UInt8;
        if (type == typeof(sbyte)) return TensorElementType.Int8;
        if (type == typeof(int)) return TensorElementType.Int32;
        if (type == typeof(long)) return TensorElementType.Int64;
        return TensorElementType.Float;
    }
    
    public sealed record StepInputs(
        OrtValue InputIds,
        KvState Kv,
        OrtValue? PositionIds,
        OrtValue? AttentionMask) : IDisposable
    {
        public void Dispose()
        {
            InputIds.Dispose();
            PositionIds?.Dispose();
            AttentionMask?.Dispose();
            Kv.Dispose();
        }
        
        public static StepInputs Create(
            long[] inputIds,
            KvState kv,
            long[]? positionIds = null,
            long[]? attentionMask = null)
        {
            var inputIdsOrt = OrtValue.CreateTensorValueFromMemory(
                inputIds, 
                [1, inputIds.Length]);
                
            OrtValue? positionIdsOrt = null;
            if (positionIds != null)
            {
                positionIdsOrt = OrtValue.CreateTensorValueFromMemory(
                    positionIds,
                    [1, positionIds.Length]);
            }
            
            OrtValue? attentionMaskOrt = null;
            if (attentionMask != null)
            {
                attentionMaskOrt = OrtValue.CreateTensorValueFromMemory(
                    attentionMask,
                    [1, attentionMask.Length]);
            }
            
            return new StepInputs(inputIdsOrt, kv, positionIdsOrt, attentionMaskOrt);
        }
    }

    public sealed record StepOutputs(
        OrtValue Logits,
        KvState KvCache) : IDisposable
    {
        public void Dispose()
        {
            Logits.Dispose();
        }
        
        public Span<float> GetLogitsSpan()
        {
            var typeInfo = Logits.GetTensorTypeAndShape();
            switch (typeInfo.ElementDataType)
            {
                case TensorElementType.Float:
                    return Logits.GetTensorMutableDataAsSpan<float>();
                
                case TensorElementType.Float16:
                case TensorElementType.BFloat16:
                    return GetLogitsArray().AsSpan();
                
                default:
                    throw new NotSupportedException($"Unsupported tensor element type: {typeInfo.ElementDataType}");
            }
        }
        
        public float[] GetLogitsArray()
        {
            var typeInfo = Logits.GetTensorTypeAndShape();
            switch (typeInfo.ElementDataType)
            {
                case TensorElementType.Float:
                    {
                        var span = Logits.GetTensorMutableDataAsSpan<float>();
                        var array = new float[span.Length];
                        span.CopyTo(array);
                        return array;
                    }
                case TensorElementType.Float16:
                    {
                        var byteSpan = Logits.GetTensorMutableDataAsSpan<byte>();
                        var halfSpan = MemoryMarshal.Cast<byte, Half>(byteSpan);
                        var array = GC.AllocateUninitializedArray<float>(halfSpan.Length);
                        for (int i = 0; i < halfSpan.Length; i++)
                        {
                            array[i] = (float)halfSpan[i];
                        }
                        
                        return array;
                    }
                case TensorElementType.BFloat16:
                    {
                        var byteSpan = Logits.GetTensorMutableDataAsSpan<byte>();
                        var bfloatSpan = MemoryMarshal.Cast<byte, BFloat16>(byteSpan);
                        var array = GC.AllocateUninitializedArray<float>(bfloatSpan.Length);
                        for (int i = 0; i < bfloatSpan.Length; i++)
                        {
                            array[i] = (float)bfloatSpan[i];
                        }
                        return array;
                    }
                default:
                    throw new NotSupportedException($"Unsupported tensor element type: {typeInfo.ElementDataType}");
            }
        }
    }

    public sealed class OutputKvTensor
    {
        public KvTensorInfo Info { get; init; }
        public OrtValue Tensor { get; set; }
    }

    public sealed class KvTensorInfo
    {
        public string Name { get; init; }
        public TensorElementType ElementType { get; init; }
        public long[] Dimensions { get; init; }
        public int Offset { get; init; }
    }
}