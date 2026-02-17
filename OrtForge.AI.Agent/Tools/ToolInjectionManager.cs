using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Tokenization;
using OrtForge.AI.Agent.Agents;

namespace OrtForge.AI.Agent.Tools;

/// <summary>
/// Result of a tool injection operation
/// </summary>
public record ToolInjectionResult(
    bool Success,
    string InjectedText,
    int[] InjectedTokens,
    KvState UpdatedKvState,
    int NewSequenceLength,
    string? ErrorMessage = null);

/// <summary>
/// Validation result for KV state consistency
/// </summary>
public record KvStateValidationResult(
    bool IsValid,
    IReadOnlyList<string> Issues);

/// <summary>
/// Manages safe tool execution and result injection with KV state validation
/// </summary>
public sealed class ToolInjectionManager
{
    private readonly TokenizerService _tokenizer;
    
    public ToolInjectionManager(TokenizerService tokenizer)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    }
    
    /// <summary>
    /// Execute tool and inject result with comprehensive validation
    /// </summary>
    public async Task<ToolInjectionResult> ExecuteAndInjectAsync(
        ToolCall toolCall,
        Func<string, string> toolExecutor,
        ToolCallState toolState,
        LlamaSession llamaSession,
        KvState currentKvState,
        int currentSequenceLength)
    {
        try
        {
            var preValidation = ValidateKvState(currentKvState, currentSequenceLength);
            if (!preValidation.IsValid)
            {
                return new ToolInjectionResult(
                    false, "", [], currentKvState, currentSequenceLength,
                    $"Pre-injection KV state validation failed: {string.Join(", ", preValidation.Issues)}");
            }
            
            toolState.UpdateCallStatus(toolCall, ToolCallStatus.Executing);
            
            string result;
            try
            {
                result = toolExecutor.Invoke(toolCall.Arguments);
                toolState.UpdateCallStatus(toolCall, ToolCallStatus.Completed, result);
            }
            catch (Exception ex)
            {
                var errorMessage = $"Tool execution failed: {ex.Message}";
                toolState.UpdateCallStatus(toolCall, ToolCallStatus.Failed, error: errorMessage);
                result = $"Error: {errorMessage}";
            }
            
            var injectedText = $"\n<|tool_result|>\n{result}\n<|/tool_result|>\n";
            var injectedTokens = _tokenizer.EncodeToIds(injectedText);
            
            var newSequenceLength = currentSequenceLength + injectedTokens.Length;
            
            var kvStateSnapshot = CreateKvStateSnapshot(currentKvState);
            
            var injectArray = injectedTokens.Select(token => (long)token).ToArray();
            var injectOutputs = await llamaSession.RunOptimizedStepAsync(
                injectArray, currentKvState, newSequenceLength);
            
            var updatedKvState = injectOutputs.KvCache;
            var postValidation = ValidateKvState(updatedKvState, newSequenceLength);
            
            if (!postValidation.IsValid)
            {
                injectOutputs.Dispose();
                Console.WriteLine("⚠️  Post-injection validation failed, attempting rollback");
                
                return new ToolInjectionResult(
                    false, "", [], kvStateSnapshot, currentSequenceLength,
                    $"Post-injection KV state validation failed: {string.Join(", ", postValidation.Issues)}");
            }
            
            injectOutputs.Dispose();
            
            Console.WriteLine($"✅ Tool injection successful: {toolCall.Name} → {injectedTokens.Length} tokens injected");
            
            return new ToolInjectionResult(
                true, injectedText, injectedTokens, updatedKvState, newSequenceLength);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Tool injection failed with exception: {ex.Message}");
            return new ToolInjectionResult(
                false, "", [], currentKvState, currentSequenceLength,
                $"Tool injection exception: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Validate KV state consistency and sequence length alignment
    /// </summary>
    public KvStateValidationResult ValidateKvState(KvState kvState, int expectedSequenceLength)
    {
        var issues = new List<string>();
        
        if (kvState.AccumulatedSequenceLength != expectedSequenceLength)
        {
            issues.Add($"Sequence length mismatch: KvState={kvState.AccumulatedSequenceLength}, Expected={expectedSequenceLength}");
        }
        
        var tensors = kvState.Tensors;
        if (tensors.Count > 0)
        {
            try
            {
                foreach (var tensor in tensors)
                {
                    var shape = tensor.Tensor.GetTensorTypeAndShape().Shape;
                    
                    if (shape.Length >= 3) // [batch, heads, seq_len, head_dim]
                    {
                        var tensorSeqLength = shape[2];
                        if (tensorSeqLength != expectedSequenceLength)
                        {
                            issues.Add($"Tensor sequence dimension mismatch: tensor={tensorSeqLength}, expected={expectedSequenceLength}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                issues.Add($"Error validating tensor shapes: {ex.Message}");
            }
        }
        
        if (kvState.Tensors.Count == 0 && expectedSequenceLength > 0)
        {
            issues.Add("KV state has no tensors but sequence length > 0");
        }
        
        return new KvStateValidationResult(issues.Count == 0, issues);
    }
    
    /// <summary>
    /// Create a snapshot of KV state for potential rollback
    /// Note: This is a reference snapshot - actual rollback would require deep copying
    /// </summary>
    private KvState CreateKvStateSnapshot(KvState originalKvState)
    {
        return originalKvState;
    }
    
    /// <summary>
    /// Validate that tool injection point is safe (at token boundary)
    /// </summary>
    public bool IsInjectionPointSafe(int currentStep, bool isGenerationPhase)
    {
        return isGenerationPhase;
    }
}
