namespace OrtForge.AI.Models.Astractions;

public class BaseModelOptions
{
    /// <summary>
    /// Path to the ML model file
    /// </summary>
    public string ModelPath { get; set; }
    /// <summary>
    /// Path to the tokenizer model file
    /// </summary>
    public string TokenizerModelPath { get; set; }

    /// <summary>
    /// Maximum input sequence length, actual limit is 8192 tokens that is not directly mappable to length in characters
    /// </summary>
    public int MaxInputLength { get; set; } = 51200;
}