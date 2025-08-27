namespace OrtForge.AI.Agent.Generation;

public sealed record InferenceConfig
{
    public double Temperature { get; init; } = 0.7;
    public int TopK { get; init; } = 40;
    public double TopP { get; init; } = 0.95;
    public double RepetitionPenalty { get; init; } = 1.0;
    public double FrequencyPenalty { get; init; } = 0.0;
    public double PresencePenalty { get; init; } = 0.0;
    public int MaxTokens { get; init; } = 2048;
    public int? Seed { get; init; }
    public bool UseGreedy { get; init; } = false;
    public double MinP { get; init; } = 0.0;
    public double TfsZ { get; init; } = 1.0;
    public double TypicalP { get; init; } = 1.0;
    public HashSet<int> StopTokenIds { get; init; } = new() { 0, 2 };
    public string[] StopSequences { get; init; } = Array.Empty<string>();
    
    public static InferenceConfig Default => new()
    {
        Temperature = 0.5,
        TopK = 40,
        TopP = 0.95,
        RepetitionPenalty = 1.1,  // FIXED: Add repetition penalty to prevent loops
        FrequencyPenalty = 0.1,   // FIXED: Add frequency penalty to reduce repetition
        PresencePenalty = 0.1     // FIXED: Add presence penalty to encourage diversity
    };
    
    public static InferenceConfig Greedy => new()
    {
        UseGreedy = true,
        Temperature = 0.0,
        RepetitionPenalty = 1.05  // Even for greedy, prevent repetition
    };
    
    public static InferenceConfig Creative => new()
    {
        Temperature = 0.8,
        TopK = 50,
        TopP = 0.9,
        RepetitionPenalty = 1.15,
        FrequencyPenalty = 0.2,
        PresencePenalty = 0.2
    };
    
    public static InferenceConfig Precise => new()
    {
        Temperature = 0.3,
        TopK = 20,
        TopP = 0.8,
        RepetitionPenalty = 1.1,
        FrequencyPenalty = 0.15,
        PresencePenalty = 0.1
    };
}
