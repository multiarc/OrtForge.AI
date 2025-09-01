using System.Text.RegularExpressions;

namespace OrtForge.AI.Agent.LLM;

public class KvTensorMappingStrategy
{
    private static readonly Regex InputRegex = new("^past.*?([0-9]+)(.*)$", RegexOptions.Compiled);
    private static readonly Regex OutputRegex = new("^present.*?([0-9]+)(.*)$", RegexOptions.Compiled);
    
    private readonly Dictionary<string, string> _inputMappingCache = new();
    private readonly Dictionary<string, string> _outpuMappingCache = new();

    public bool IsKvInput(string name)
    {
        return _inputMappingCache.ContainsKey(name);
    }
    
    public bool IsKvOutput(string name)
    {
        return _outpuMappingCache.ContainsKey(name);
    }
    
    public static KvTensorMappingStrategy Create(IEnumerable<string> inputMetadata, IEnumerable<string> outputMetadata)
    {
        var outputSet = outputMetadata.ToHashSet();

        var result = new KvTensorMappingStrategy();

        var inputs = new Dictionary<(int, string), string>();

        foreach (var input in inputMetadata)
        {
            var match = InputRegex.Match(input);
            if (match.Success)
            {
                inputs[(int.Parse(match.Groups[1].Value), match.Groups[2].Value)] = input;
            }
        }

        foreach (var output in outputSet)
        {
            var match = OutputRegex.Match(output);
            if (match.Success)
            {
                var outputIndex = int.Parse(match.Groups[1].Value);
                var outputName = match.Groups[2].Value;
                if (inputs.TryGetValue((outputIndex, outputName), out var input))
                {
                    result._inputMappingCache[input] = output;
                    result._outpuMappingCache[output] = input;
                }
            }
        }

        return result;
    }
    
    public string MapOutputToInput(string output)
    {
        return _outpuMappingCache.GetValueOrDefault(output) ?? throw new InvalidOperationException($"Cannot map output tensor '{output}'");;
    }
}
