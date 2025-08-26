using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OrtAgent.Core.LLM;

public sealed class LlamaSession : IDisposable
{
    public enum KvStorageType { Float32, Float16, Int4 }

    private readonly InferenceSession _session;
    private readonly KvStorageType _kvType;
    
    private readonly Dictionary<string, OrtValue> _kvTensorPool = new();
    private readonly Dictionary<string, TensorElementType> _kvTensorTypes = new();
    private readonly object _tensorLock = new object();

    public LlamaSession(InferenceSession session, KvStorageType kvType = KvStorageType.Float32)
    {
        _session = session;
        _kvType = kvType;
        DetectModelQuantization();
    }

    public string ModelName { get; init; } = "default";
    
    private void DetectModelQuantization()
    {
        foreach (var output in _session.OutputMetadata)
        {
            if (output.Value.ElementType == typeof(byte) || 
                output.Value.ElementType == typeof(sbyte) || 
                output.Value.ElementType.Name == "Int4")
            {
                Console.WriteLine($"Detected quantized model output: {output.Key} with type {output.Value.ElementType}");
            }
        }
    }

    public void Dispose()
    {
        lock (_tensorLock)
        {
            foreach (var tensor in _kvTensorPool.Values)
            {
                tensor?.Dispose();
            }
            _kvTensorPool.Clear();
            _kvTensorTypes.Clear();
        }
        _session.Dispose();
    }
    
    private OrtValue GetOrCreateKvTensor(string name, long[] shape, TensorElementType elementType)
    {
        lock (_tensorLock)
        {
            if (_kvTensorPool.TryGetValue(name, out var existingTensor))
            {
                return existingTensor;
            }
            
            var tensor = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, elementType, shape);
            _kvTensorPool[name] = tensor;
            _kvTensorTypes[name] = elementType;
            return tensor;
        }
    }
    
    
    private static TensorElementType GetTensorElementType(Type type)
    {
        if (type == typeof(float)) return TensorElementType.Float;
        if (type == typeof(System.Half)) return TensorElementType.Float16;
        if (type == typeof(byte)) return TensorElementType.UInt8;
        if (type == typeof(sbyte)) return TensorElementType.Int8;
        if (type == typeof(int)) return TensorElementType.Int32;
        if (type == typeof(long)) return TensorElementType.Int64;
        return TensorElementType.Float;
    }
    
    private static long[] ConvertToLongArray(ReadOnlySpan<int> dimensions)
    {
        var result = new long[dimensions.Length];
        for (int i = 0; i < dimensions.Length; i++)
        {
            result[i] = dimensions[i];
        }
        return result;
    }
    
    private static int[] ConvertToIntArray(ReadOnlySpan<long> dimensions)
    {
        var result = new int[dimensions.Length];
        for (int i = 0; i < dimensions.Length; i++)
        {
            result[i] = (int)dimensions[i];
        }
        return result;
    }

    public sealed class KvState : IDisposable
    {
        public readonly Dictionary<string, OrtValue> Tensors = new();
        public static KvState Empty => new();
        private bool _disposed = false;
        
        public void AddTensor(string name, OrtValue tensor)
        {
            Tensors[name] = tensor;
        }
        
        public OrtValue? GetTensor(string name)
        {
            return Tensors.TryGetValue(name, out var tensor) ? tensor : null;
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                foreach (var tensor in Tensors.Values)
                {
                    tensor?.Dispose();
                }
                Tensors.Clear();
                _disposed = true;
            }
        }
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
            DenseTensor<int> inputIds,
            KvState kv,
            DenseTensor<long>? positionIds = null,
            DenseTensor<int>? attentionMask = null)
        {
            var inputIdsOrt = OrtValue.CreateTensorValueFromMemory(
                inputIds.Buffer.ToArray(), 
                ConvertToLongArray(inputIds.Dimensions));
                
            OrtValue? positionIdsOrt = null;
            if (positionIds != null)
            {
                positionIdsOrt = OrtValue.CreateTensorValueFromMemory(
                    positionIds.Buffer.ToArray(),
                    ConvertToLongArray(positionIds.Dimensions));
            }
            
            OrtValue? attentionMaskOrt = null;
            if (attentionMask != null)
            {
                attentionMaskOrt = OrtValue.CreateTensorValueFromMemory(
                    attentionMask.Buffer.ToArray(),
                    ConvertToLongArray(attentionMask.Dimensions));
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
            KvCache?.Dispose();
        }
        
        public Span<float> GetLogitsSpan() => Logits.GetTensorMutableDataAsSpan<float>();
        
        public float[] GetLogitsArray()
        {
            var span = GetLogitsSpan();
            var array = new float[span.Length];
            span.CopyTo(array);
            return array;
        }
        
        public DenseTensor<float> GetLogitsTensor()
        {
            var span = GetLogitsSpan();
            var shape = Logits.GetTensorTypeAndShape().Shape;
            var dims = ConvertToIntArray(shape);
            var array = new float[span.Length];
            span.CopyTo(array);
            return new DenseTensor<float>(array, dims);
        }
    }

    public async Task<StepOutputs> RunStepAsync(StepInputs inputs, CancellationToken cancellationToken = default)
    {
        var inputMetadataKeys = _session.InputMetadata.Keys;
        var outputMetadata = _session.OutputMetadata;
        
        var maxInputs = 3 + (inputs.Kv?.Tensors.Count ?? 0);
        var inputValues = new List<OrtValue>(maxInputs);
        var inputNamesList = new List<string>(maxInputs);
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

        bool hasAttentionMask = false;
        if (inputs.AttentionMask != null)
        {
            foreach (var key in inputMetadataKeys)
            {
                if (key == "attention_mask")
                {
                    hasAttentionMask = true;
                    break;
                }
            }
        }

        if (hasAttentionMask && inputs.AttentionMask != null)
        {
            inputValues.Add(inputs.AttentionMask);
            inputNamesList.Add("attention_mask");
        }

        if (inputs.Kv != null && inputs.Kv.Tensors.Count > 0)
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
            }
        }

        foreach (var output in outputMetadata)
        {
            outputNames.Add(output.Key);
            if (output.Key.ToLower().Contains("logits"))
            {
                var longDims = ConvertToLongArray(output.Value.Dimensions);
                var logitsTensor = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float, longDims);
                outputValues.Add(logitsTensor);
            }
            else
            {
                var longDims = ConvertToLongArray(output.Value.Dimensions);
                var kvTensor = GetOrCreateKvTensor(output.Key, longDims, GetTensorElementType(output.Value.ElementType));
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

        var newKv = new KvState();
        OrtValue? logits = null;
        
        using (var disposableInputs = new DisposableOrtValueList(inputValuesArray.Where(t => !_kvTensorPool.ContainsValue(t))))
        {
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
        }

        if (logits is null)
            throw new InvalidOperationException("Model did not return logits.");

        return new StepOutputs(logits, newKv);
    }
    
    public StepOutputs RunStep(StepInputs inputs)
    {
        return RunStepAsync(inputs, CancellationToken.None).GetAwaiter().GetResult();
    }

    public async Task<StepOutputs> RunOptimizedStepAsync(DenseTensor<int> inputIds, KvState kv, int currentStep, int sequenceLength, CancellationToken cancellationToken = default)
    {
        var positionIds = LlamaOptimizations.CreateOptimalPositionIds(sequenceLength, currentStep, ModelName);
        var attentionMask = currentStep == 0 ? LlamaOptimizations.CreateOptimalAttentionMask(inputIds.Dimensions[1], ModelName) : null;
        
        using var inputs = StepInputs.Create(inputIds, kv, positionIds, attentionMask);
        return await RunStepAsync(inputs, cancellationToken);
    }

    public StepOutputs RunOptimizedStep(DenseTensor<int> inputIds, KvState kv, int currentStep, int sequenceLength)
    {
        return RunOptimizedStepAsync(inputIds, kv, currentStep, sequenceLength, CancellationToken.None).GetAwaiter().GetResult();
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


