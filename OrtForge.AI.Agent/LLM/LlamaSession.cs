using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OrtForge.AI.Agent.LLM;

public sealed class LlamaSession : IDisposable
{
    private readonly InferenceSession _session;
    
    public LlamaSession(InferenceSession session)
    {
        _session = session;
    }

    public string ModelName { get; init; } = "default";

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
            InputIds?.Dispose();
            PositionIds?.Dispose();
            AttentionMask?.Dispose();
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
            Logits?.Dispose();
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
                    // For 16-bit types, we need to convert to float first
                    // This requires allocation, so performance is similar to GetLogitsArray()
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
                        // Follow ModelHostBase pattern for Float16 handling
                        var byteSpan = Logits.GetTensorMutableDataAsSpan<byte>();
                        var halfSpan = MemoryMarshal.Cast<byte, Half>(byteSpan);
                        var array = GC.AllocateUninitializedArray<float>(halfSpan.Length);
                        for (int i = 0; i < halfSpan.Length; i++)
                        {
                            array[i] = (float)halfSpan[i];
                        }
                        
                        // Debug: Check for NaN/Inf values in logits
                        var nanCount = array.Count(f => float.IsNaN(f));
                        var infCount = array.Count(f => float.IsInfinity(f));
                        if (nanCount > 0 || infCount > 0)
                        {
                            Console.WriteLine($"WARNING: Logits contain {nanCount} NaN and {infCount} Inf values");
                        }
                        
                        return array;
                    }
                case TensorElementType.BFloat16:
                    {
                        // Follow ModelHostBase pattern for BFloat16 handling
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
        
        // Get input dimensions used throughout the method
        var inputShape = inputs.InputIds.GetTensorTypeAndShape().Shape;
        var batchSize = inputShape[0];
        var sequenceLength = inputShape[1];
        
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
                // Create default attention mask (all 1s)
                var defaultAttentionMask = new long[sequenceLength];
                Array.Fill(defaultAttentionMask, 1L);
                var attentionMaskOrt = OrtValue.CreateTensorValueFromMemory(defaultAttentionMask, [1, sequenceLength]);
                inputValues.Add(attentionMaskOrt);
            }
            inputNamesList.Add("attention_mask");
        }

        // Handle KV cache inputs - create empty tensors for missing ones on first step
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
                    targetName = MapKvNameToInput(kv.Key, inputMetadataKeys);
                }
                
                if (targetName == null) continue;

                inputValues.Add(kv.Value);
                inputNamesList.Add(targetName);
                providedKvInputs.Add(targetName);
            }
        }
        
        // Create empty KV cache tensors for any missing KV inputs (first step)
        
        foreach (var inputName in inputMetadataKeys)
        {
            if ((inputName.Contains("past") || inputName.Contains("key") || inputName.Contains("value")) && 
                !providedKvInputs.Contains(inputName) &&
                inputName != "input_ids" && inputName != "position_ids" && inputName != "attention_mask")
            {

                var inputMeta = _session.InputMetadata[inputName];
                var kvDims = inputMeta.Dimensions.ToArray();
                
                // Replace symbolic dimensions
                for (int i = 0; i < kvDims.Length; i++)
                {
                    if (kvDims[i] < 0)
                    {
                        if (i == 0) 
                            kvDims[i] = (int)batchSize;
                        else if (i == 2) 
                            kvDims[i] = 0; // Sequence length starts at 0 for empty cache
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
                    [batchSize, sequenceLength, vocabSize]);
                outputValues.Add(logitsTensor);
            }
            else
            {
                var kvDims = output.Value.Dimensions.ToArray();
                for (int i = 0; i < kvDims.Length; i++)
                {
                    if (kvDims[i] < 0) // Replace symbolic dimensions
                    {
                        if (i == 0) kvDims[i] = (int)batchSize;
                        else if (i == 2) kvDims[i] = inputs.Kv.AccumulatedSequenceLength + (int)sequenceLength; // Total KV sequence length
                    }
                }
                var longDims = kvDims.Select(d => (long)d).ToArray();
                // Direct allocation - let ONNX Runtime handle memory pooling efficiently
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

        // Create new KvState with updated sequence length
        var newKv = new KvState(inputs.Kv.AccumulatedSequenceLength + (int)sequenceLength);
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
                var alias = MapKvOutputToPastAlias(outputName);
                if (alias != null)
                {
                    newKv.AddTensor(alias, outputTensor);
                }
            }
        }

        if (logits is null)
            throw new InvalidOperationException("Model did not return logits.");

        return new StepOutputs(logits, newKv);
    }

    public async Task<StepOutputs> RunOptimizedStepAsync(long[] inputIds, KvState kv, int currentStep, int sequenceLength, CancellationToken cancellationToken = default)
    {
        var positionIds = LlamaOptimizations.CreateOptimalPositionIds(sequenceLength, currentStep, ModelName);
        // Always provide attention mask since model requires it - must match current input length for KV cache
        var attentionMask = LlamaOptimizations.CreateOptimalAttentionMask(inputIds.Length, ModelName);
        
        using var inputs = StepInputs.Create(inputIds, kv, positionIds, attentionMask);
        return await RunStepAsync(inputs, cancellationToken);
    }

    public async Task<StepOutputs> RunOptimizedStep(long[] inputIds, KvState kv, int currentStep, int sequenceLength)
    {
        return await RunOptimizedStepAsync(inputIds, kv, currentStep, sequenceLength, CancellationToken.None);
    }

    
    private sealed class DisposableOrtValueList : IDisposable
    {
        private readonly IEnumerable<OrtValue> _values;
        
        public DisposableOrtValueList(IEnumerable<OrtValue> values)
        {
            _values = values;
        }
        
        public void Dispose()
        {
            foreach (var value in _values)
            {
                value?.Dispose();
            }
        }
    }

    private static string? MapKvNameToInput(string outputLikeName, IEnumerable<string> inputNames)
    {
        var inputNamesSet = inputNames.ToHashSet();
        
        if (outputLikeName.StartsWith("present_key_values", StringComparison.Ordinal))
        {
            var candidate = "past_" + outputLikeName.Substring("present_".Length);
            if (inputNamesSet.Contains(candidate)) return candidate;
        }
        
        if (outputLikeName.StartsWith("present.", StringComparison.Ordinal))
        {
            var candidate = "past_key_values" + outputLikeName.Substring("present".Length);
            if (inputNamesSet.Contains(candidate)) return candidate;
            
            candidate = "past" + outputLikeName.Substring("present".Length);
            if (inputNamesSet.Contains(candidate)) return candidate;
        }
        
        if (outputLikeName.Contains("present"))
        {
            var candidate = outputLikeName.Replace("present", "past");
            if (inputNamesSet.Contains(candidate)) return candidate;
            
            candidate = outputLikeName.Replace("present", "past_key_values");
            if (inputNamesSet.Contains(candidate)) return candidate;
        }
        
        foreach (var inputName in inputNamesSet)
        {
            if (inputName.Contains("past") && outputLikeName.Contains("present"))
            {
                var baseName = outputLikeName.Replace("present", "").Replace("_", "").Replace(".", "");
                var inputBaseName = inputName.Replace("past", "").Replace("_", "").Replace(".", "").Replace("key", "").Replace("values", "");
                if (baseName.Contains(inputBaseName) || inputBaseName.Contains(baseName))
                {
                    return inputName;
                }
            }
        }
        
        return null;
    }

    private static string? MapKvOutputToPastAlias(string outputName)
    {
        if (outputName.StartsWith("present_key_values", StringComparison.Ordinal))
        {
            return "past_" + outputName.Substring("present_".Length);
        }
        
        if (outputName.StartsWith("present.", StringComparison.Ordinal))
        {
            return "past" + outputName.Substring("present".Length);
        }
        
        if (outputName.Contains("present"))
        {
            return outputName.Replace("present", "past");
        }
        
        return null;
    }



}