using Microsoft.ML.OnnxRuntime;

namespace OrtAgent.Core.Runtime;

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
        so.EnableCpuMemArena = true;
        so.IntraOpNumThreads = Environment.ProcessorCount;
        so.InterOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2);
        so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        // EPs can be appended externally by caller for CUDA/DirectML etc.
        return so;
    }
}


