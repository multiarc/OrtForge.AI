namespace OrtForge.AI.Agent.LLM;

/// <summary>
/// Supported LLM model types with optimized configurations
/// </summary>
public enum ModelType
{
    /// <summary>
    /// Default/Unknown model type with basic configuration
    /// </summary>
    Default = 0,
    
    /// <summary>
    /// Llama 2 model family
    /// </summary>
    Llama2 = 1,
    
    /// <summary>
    /// Llama 3 base model
    /// </summary>
    Llama3 = 2,
    
    /// <summary>
    /// Llama 3.1 model
    /// </summary>
    Llama3_1 = 3,
    
    /// <summary>
    /// Llama 3.2 model
    /// </summary>
    Llama3_2 = 4
}

/// <summary>
/// Extension methods for ModelType enum
/// </summary>
public static class ModelTypeExtensions
{
    /// <summary>
    /// Check if the model is part of the Llama 3 family
    /// </summary>
    public static bool IsLlama3Family(this ModelType modelType)
    {
        return modelType is ModelType.Llama3 or ModelType.Llama3_1 or ModelType.Llama3_2;
    }
    
    /// <summary>
    /// Get the string representation for backwards compatibility
    /// </summary>
    public static string ToModelKey(this ModelType modelType)
    {
        return modelType switch
        {
            ModelType.Llama2 => "llama-2",
            ModelType.Llama3 => "llama-3", 
            ModelType.Llama3_1 => "llama-3.1",
            ModelType.Llama3_2 => "llama-3.2",
            _ => "default"
        };
    }
    
    /// <summary>
    /// Parse model type from string (for backwards compatibility and auto-detection)
    /// </summary>
    public static ModelType ParseFromString(string modelName)
    {
        var lower = modelName.ToLowerInvariant();
        
        if (lower.Contains("llama-3.2") || lower.Contains("llama3.2"))
            return ModelType.Llama3_2;
        if (lower.Contains("llama-3.1") || lower.Contains("llama3.1"))
            return ModelType.Llama3_1;
        if (lower.Contains("llama-3") || lower.Contains("llama3"))
            return ModelType.Llama3;
        if (lower.Contains("llama-2") || lower.Contains("llama2"))
            return ModelType.Llama2;
        
        return ModelType.Default;
    }
}
