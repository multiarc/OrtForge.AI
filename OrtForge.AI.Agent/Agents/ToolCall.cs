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
    private const int MaxBufferSize = 8192; // Limit buffer to prevent unbounded growth
    
    private readonly List<ToolCall> _calls = [];
    private string _currentBuffer = string.Empty;
    private bool _inToolCall = false;
    private int _toolCallStart = -1;

    public IReadOnlyList<ToolCall> Calls => _calls;
    public bool InToolCall => _inToolCall;
    public bool HasPendingCalls => _calls.Exists(c => c.Status == ToolCallStatus.Pending);

    public void AppendToken(string token)
    {
        _currentBuffer += token;
        TrimBufferIfNeeded();
        CheckForToolCallPatterns();
    }

    public void AppendText(string text)
    {
        _currentBuffer += text;
        TrimBufferIfNeeded();
        CheckForToolCallPatterns();
    }
    
    private void TrimBufferIfNeeded()
    {
        // If buffer exceeds max size and we're not in a tool call, keep only the tail
        if (_currentBuffer.Length > MaxBufferSize && !_inToolCall)
        {
            // Keep the last portion that might contain a partial TOOL_CALL marker
            var keepSize = Math.Min(MaxBufferSize / 2, _currentBuffer.Length);
            _currentBuffer = _currentBuffer.Substring(_currentBuffer.Length - keepSize);
        }
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

    private const string StartMarker = "TOOL_CALL";
    private const string EndMarker = "END_TOOL_CALL";
    
    private void CheckForToolCallPatterns()
    {
        // Keep checking for tool calls until no more complete ones are found
        while (true)
        {
            if (!_inToolCall)
            {
                var startIndex = _currentBuffer.IndexOf(StartMarker, StringComparison.Ordinal);
                if (startIndex >= 0)
                {
                    _inToolCall = true;
                    _toolCallStart = startIndex;
                }
                else
                {
                    break; // No more tool call starts found
                }
            }

            if (_inToolCall)
            {
                var endIndex = _currentBuffer.IndexOf(EndMarker, _toolCallStart, StringComparison.Ordinal);
                if (endIndex >= 0)
                {
                    // Extract content between TOOL_CALL and END_TOOL_CALL
                    var contentStart = _toolCallStart + StartMarker.Length;
                    var callContent = _currentBuffer.Substring(contentStart, endIndex - contentStart);
                    var toolCall = ParseToolCallContent(callContent);
                    if (toolCall != null)
                    {
                        _calls.Add(toolCall);
                    }
                    
                    // Remove processed content from buffer to allow finding next tool call
                    _currentBuffer = _currentBuffer.Substring(endIndex + EndMarker.Length);
                    _inToolCall = false;
                    _toolCallStart = -1;
                }
                else
                {
                    break; // Incomplete tool call, wait for more tokens
                }
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
