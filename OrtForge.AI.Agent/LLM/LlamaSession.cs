using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Agent.Generation;

namespace OrtForge.AI.Agent.LLM;

public sealed class LlamaSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly KvMappingFormat _kvMappingFormat;
    private readonly KvMappingValidationResult _kvMappingValidation;

    public LlamaSession(InferenceSession session, ModelType modelType = ModelType.Default)
    {
        _session = session;
        ModelType = modelType;
        OptimalConfig = LlamaOptimizations.GetOptimalConfigForModel(modelType);
        
        _kvMappingFormat = KvTensorMappingStrategy.DetectFormat(_session.InputMetadata, _session.OutputMetadata);
        _kvMappingValidation = KvTensorMappingStrategy.ValidateMapping(
            _session.InputMetadata, _session.OutputMetadata, _kvMappingFormat);
            
        LogKvMappingValidation();
    }

    public ModelType ModelType { get; }
    public InferenceConfig OptimalConfig { get; }
    public void Dispose()
    {
        _session.Dispose();
    }
    
    private void LogKvMappingValidation()
    {
        Console.WriteLine($"KV Mapping Format Detected: {_kvMappingFormat}");
        
        if (!_kvMappingValidation.IsValid)
        {
            Console.WriteLine("⚠️  KV Mapping Validation Issues:");
            foreach (var issue in _kvMappingValidation.Issues)
            {
                Console.WriteLine($"   - {issue}");
            }
        }
        else
        {
            Console.WriteLine($"✅ KV Mapping Validated: {_kvMappingValidation.MappedPairs.Count} tensor pairs mapped successfully");
        }
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

    public async Task<StepOutputs> RunStepAsync(StepInputs inputs, CancellationToken cancellationToken = default)
    {
        var inputMetadataKeys = _session.InputMetadata.Keys;
        var outputMetadata = _session.OutputMetadata;
        
        var inputShape = inputs.InputIds.GetTensorTypeAndShape().Shape;
        var batchSize = inputShape[0];
        var currentInputLength = inputShape[1];  // Length of current input tokens
        
        var totalSequenceLength = inputs.Kv.CalculateTotalLengthAfterTokens((int)currentInputLength);
        

        
        var inputValues = new List<OrtValue>();
        var inputNamesList = new List<string>();
        var outputCount = outputMetadata.Count;
        var outputNames = new List<string>(outputCount);
        var outputValues = new List<OrtValue>(outputCount);

        bool hasInputIds = false;
        foreach (var key in inputMetadataKeys)
        {
            if (key == "input_ids")
            {
                hasInputIds = true;
                break;
            }
        }
        
        if (!hasInputIds)
            throw new InvalidOperationException("Model expects 'input_ids'.");
            
        inputValues.Add(inputs.InputIds);
        inputNamesList.Add("input_ids");

        bool hasPositionIds = false;
        if (inputs.PositionIds != null)
        {
            foreach (var key in inputMetadataKeys)
            {
                if (key == "position_ids")
                {
                    hasPositionIds = true;
                    break;
                }
            }
        }

        if (hasPositionIds && inputs.PositionIds != null)
        {
            inputValues.Add(inputs.PositionIds);
            inputNamesList.Add("position_ids");
        }

        if (inputMetadataKeys.Contains("attention_mask"))
        {
            if (inputs.AttentionMask != null)
            {
                inputValues.Add(inputs.AttentionMask);
            }
            else
            {
                var defaultAttentionMask = new long[totalSequenceLength];
                Array.Fill(defaultAttentionMask, 1L);
                var attentionMaskOrt = OrtValue.CreateTensorValueFromMemory(defaultAttentionMask, [1, totalSequenceLength]);
                inputValues.Add(attentionMaskOrt);
            }
            inputNamesList.Add("attention_mask");
        }

        var providedKvInputs = new HashSet<string>();
        
        if (inputs.Kv.Tensors.Count > 0)
        {
            foreach (var kv in inputs.Kv.Tensors)
            {
                string? targetName = null;
                
                foreach (var inputName in inputMetadataKeys)
                {
                    if (inputName == kv.Key)
                    {
                        targetName = kv.Key;
                        break;
                    }
                }
                
                if (targetName == null)
                {
                    targetName = KvTensorMappingStrategy.MapOutputToInput(
                        kv.Key, _kvMappingFormat, inputMetadataKeys.ToList());
                    
                    if (targetName != null)
                    {
                        Console.WriteLine($"🔗 Mapped KV tensor: {kv.Key} → {targetName}");
                    }
                    else
                    {
                        Console.WriteLine($"❌ Failed to map KV tensor: {kv.Key}");
                    }
                }
                
                if (targetName == null) continue;


                
                inputValues.Add(kv.Value);
                inputNamesList.Add(targetName);
                providedKvInputs.Add(targetName);
            }
        }
        
        foreach (var inputName in inputMetadataKeys)
        {
            if ((inputName.Contains("past") || inputName.Contains("key") || inputName.Contains("value")) && 
                !providedKvInputs.Contains(inputName) &&
                inputName != "input_ids" && inputName != "position_ids" && inputName != "attention_mask")
            {

                var inputMeta = _session.InputMetadata[inputName];
                var kvDims = inputMeta.Dimensions.ToArray();
                
                for (int i = 0; i < kvDims.Length; i++)
                {
                    if (kvDims[i] < 0)
                    {
                        if (i == 0) 
                            kvDims[i] = (int)batchSize;
                        else if (i == 2) 
                            kvDims[i] = 0;
                    }
                }
                
                var longDims = kvDims.Select(d => (long)d).ToArray();
                var emptyKvTensor = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, 
                    GetTensorElementType(inputMeta.ElementType), 
                    longDims);
                
                inputValues.Add(emptyKvTensor);
                inputNamesList.Add(inputName);
            }
        }
        
        foreach (var output in outputMetadata)
        {
            outputNames.Add(output.Key);
            
            if (output.Key.ToLower().Contains("logits"))
            {
                var vocabSize = output.Value.Dimensions[^1];
                
                var tensorElementType = GetTensorElementType(output.Value.ElementType);
                
                var logitsTensor = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, 
                    tensorElementType, 
                    [batchSize, currentInputLength, vocabSize]);
                outputValues.Add(logitsTensor);
            }
            else
            {
                var kvDims = output.Value.Dimensions.ToArray();
                for (int i = 0; i < kvDims.Length; i++)
                {
                    if (kvDims[i] < 0)
                    {
                        if (i == 0) kvDims[i] = (int)batchSize;
                        else if (i == 2) 
                        {
                            if (inputs.Kv.Tensors.Count == 0)
                            {
                                kvDims[i] = (int)currentInputLength;

                            }
                            else
                            {
                                kvDims[i] = (int)totalSequenceLength;
                            }
                        }
                    }
                }
                var longDims = kvDims.Select(d => (long)d).ToArray();
                

                
                var kvTensor = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, 
                    GetTensorElementType(output.Value.ElementType), 
                    longDims);
                outputValues.Add(kvTensor);
            }
        }

        var inputNamesArray = inputNamesList.ToArray();
        var inputValuesArray = inputValues.ToArray();
        var outputNamesArray = outputNames.ToArray();
        var outputValuesArray = outputValues.ToArray();

        cancellationToken.ThrowIfCancellationRequested();

        try
        {
            using var runOptions = new RunOptions();
            await _session.RunAsync(runOptions, inputNamesArray, inputValuesArray, outputNamesArray, outputValuesArray);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Error running the model: {ex.Message}", ex);
        }

        var newKv = new KvState((int)totalSequenceLength);

        OrtValue? logits = null;
        
        for (int i = 0; i < outputNamesArray.Length; i++)
        {
            var outputName = outputNamesArray[i];
            var outputTensor = outputValuesArray[i];
            
            if (outputName.ToLower().Contains("logits"))
            {
                logits = outputTensor;
            }
            else
            {
                newKv.AddTensor(outputName, outputTensor);
                var availableInputNames = _session.InputMetadata.Keys.Where(name => 
                    name.Contains("past") || name.Contains("key") || name.Contains("value")).ToList();
                var alias = KvTensorMappingStrategy.MapInputToOutput(outputName, _kvMappingFormat, availableInputNames);
                
                if (alias != null)
                {
                    newKv.AddTensor(alias, outputTensor);
                    Console.WriteLine($"🔗 Created KV alias: {outputName} → {alias}");
                }
            }
        }

        if (logits is null)
            throw new InvalidOperationException("Model did not return logits.");

        return new StepOutputs(logits, newKv);
    }

    public async Task<StepOutputs> RunOptimizedStepAsync(long[] inputIds, KvState kv, int currentStep, int sequenceLength, CancellationToken cancellationToken = default)
    {
        var positionIds = LlamaOptimizations.CreateOptimalPositionIds(sequenceLength, currentStep);
        var attentionMask = LlamaOptimizations.CreateOptimalAttentionMask(sequenceLength);
        
        using var inputs = StepInputs.Create(inputIds, kv, positionIds, attentionMask);
        return await RunStepAsync(inputs, cancellationToken);
    }

    public async Task<StepOutputs> RunOptimizedStep(long[] inputIds, KvState kv, int currentStep, int sequenceLength)
    {
        return await RunOptimizedStepAsync(inputIds, kv, currentStep, sequenceLength, CancellationToken.None);
    }
}