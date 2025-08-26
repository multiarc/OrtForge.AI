namespace OrtForge.AI.Models.Astractions;

public class BaseModelOptions
{
    /// <summary>
    /// Path to the ML model file
    /// </summary>
    public required string ModelPath { get; init; }
    /// <summary>
    /// Path to the tokenizer model file
    /// </summary>
    public required string TokenizerModelPath { get; init; }

    /// <summary>
    /// Maximum input sequence length, actual limit is 8192 tokens that is not directly mappable to length in characters
    /// </summary>
    public int MaxInputLength { get; init; } = 51200;
}