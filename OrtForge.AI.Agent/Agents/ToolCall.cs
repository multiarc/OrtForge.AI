namespace OrtForge.AI.Agent.Agents;

public sealed record ToolCall(
    string Name,
    string Arguments,
    string Id = "",
    string? Result = null,
    ToolCallStatus Status = ToolCallStatus.Pending,
    string? Error = null
);

public enum ToolCallStatus
{
    Pending,
    Parsing,
    Executing,
    Completed,
    Failed
}

public sealed class ToolCallState
{
    private readonly List<ToolCall> _calls = new();
    private string _currentBuffer = string.Empty;
    private bool _inToolCall = false;
    private int _toolCallStart = -1;

    public IReadOnlyList<ToolCall> Calls => _calls;
    public bool InToolCall => _inToolCall;
    public bool HasPendingCalls => _calls.Exists(c => c.Status == ToolCallStatus.Pending);

    public void AppendToken(string token)
    {
        _currentBuffer += token;
        CheckForToolCallPatterns();
    }

    public void AppendText(string text)
    {
        _currentBuffer += text;
        CheckForToolCallPatterns();
    }

    public ToolCall? GetNextPendingCall()
    {
        return _calls.Find(c => c.Status == ToolCallStatus.Pending);
    }

    public void UpdateCallStatus(ToolCall call, ToolCallStatus status, string? result = null, string? error = null)
    {
        var index = _calls.FindIndex(c => c.Id == call.Id);
        if (index >= 0)
        {
            _calls[index] = call with { Status = status, Result = result, Error = error };
        }
    }

    public void Reset()
    {
        _calls.Clear();
        _currentBuffer = string.Empty;
        _inToolCall = false;
        _toolCallStart = -1;
    }

    private void CheckForToolCallPatterns()
    {
        if (!_inToolCall)
        {
            var startIndex = _currentBuffer.IndexOf("<|tool_call|>", StringComparison.Ordinal);
            if (startIndex >= 0)
            {
                _inToolCall = true;
                _toolCallStart = startIndex;
            }
        }

        if (_inToolCall)
        {
            var endIndex = _currentBuffer.IndexOf("<|/tool_call|>", _toolCallStart, StringComparison.Ordinal);
            if (endIndex >= 0)
            {
                var callContent = _currentBuffer.Substring(_toolCallStart + 14, endIndex - (_toolCallStart + 14));
                var toolCall = ParseToolCallContent(callContent);
                if (toolCall != null)
                {
                    _calls.Add(toolCall);
                }
                
                _inToolCall = false;
                _toolCallStart = -1;
            }
        }
    }

    private static ToolCall? ParseToolCallContent(string content)
    {
        try
        {
            var lines = content.Trim().Split('\n', StringSplitOptions.RemoveEmptyEntries);
            string? name = null;
            string? args = null;

            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (trimmed.StartsWith("name:", StringComparison.OrdinalIgnoreCase))
                {
                    name = trimmed.Substring(5).Trim();
                }
                else if (trimmed.StartsWith("args:", StringComparison.OrdinalIgnoreCase))
                {
                    args = trimmed.Substring(5).Trim();
                }
            }

            if (!string.IsNullOrEmpty(name))
            {
                return new ToolCall(name, args ?? string.Empty, Guid.NewGuid().ToString());
            }
        }
        catch
        {
        }

        return null;
    }
}
