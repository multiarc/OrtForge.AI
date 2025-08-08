namespace OrtForge.AI.Models.Models;

/// <summary>
/// Model information structure
/// </summary>
public class ModelInfo
{
    public string ModelPath { get; set; } = string.Empty;
    public int MaxLength { get; set; }
    public string[] InputNames { get; set; } = [];
    public string[] OutputNames { get; set; } = [];
}