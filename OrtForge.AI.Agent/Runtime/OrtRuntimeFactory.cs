using Microsoft.ML.OnnxRuntime;

namespace OrtForge.AI.Agent.Runtime;

public static class OrtRuntimeFactory
{
    private static readonly Lazy<OrtEnv> s_env = new(() => OrtEnv.Instance());

    public static OrtEnv Env => s_env.Value;

    public static InferenceSession CreateSession(string modelPath, SessionOptions? options = null)
    {
        var opts = options ?? CreateDefaultSessionOptions();
        return new InferenceSession(modelPath, opts);
    }

    public static SessionOptions CreateDefaultSessionOptions()
    {
        var so = new SessionOptions();
        so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        so.AppendExecutionProvider_ROCm();
        so.AppendExecutionProvider_CPU();
        return so;
    }
}


