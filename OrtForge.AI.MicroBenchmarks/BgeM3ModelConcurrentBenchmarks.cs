#if !WINDOWS //Windows does not support running DirectML in parallel, CPU tests are worthless for this case
#if ROCM || CUDA
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Models.Astractions;
using OrtForge.AI.Models.Models;
using OrtForge.AI.Models.Options;

namespace OrtForge.AI.MicroBenchmarks;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput)]
[MaxIterationCount(16)]
public class BgeM3ModelConcurrentBenchmarks
{
    private BgeM3Model _model = null!;
    private string _text = null!;

    [GlobalSetup]
    public async Task Initialize()
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        _model = new BgeM3Model(new BgeM3Options
        {
            TokenizerModelPath = Path.Combine(home, "LLM/bge_m3_onnx_gpu/sentencepiece.bpe.model"), 
            ModelPath = Path.Combine(home, "LLM/bge_m3_onnx_gpu/model.onnx"),
            TensorElementType = TensorElementType.Float16
        });
        _model.Initialize(Mode, optimizationLevel: OptimizationLevel, providers: Providers);
        _text = await File.ReadAllTextAsync("test_docs/data_science.txt");
    }

    [Params(GraphOptimizationLevel.ORT_ENABLE_ALL)]
    public GraphOptimizationLevel OptimizationLevel { get; set; }

#if ROCM
    [Params(ExecutionProvider.ROCm | ExecutionProvider.CPU)]
#elif CUDA
    [Params(ExecutionProvider.CUDA | ExecutionProvider.CPU)]
#endif
    public ExecutionProvider Providers { get; set; }

    [Params(ExecutionMode.ORT_SEQUENTIAL)]
    public ExecutionMode Mode { get; set; }

    [Params(1, 8, 16, 64, 256, 512)]
    public int NumTasks { get; set; }

    [GlobalCleanup]
    public async Task Teardown()
    {
        var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        // Dispose the model in a separate thread to work around deadlocks
        new Thread(_ =>
        {
            _model.Dispose();
            tcs.SetResult();
        }).Start();
        await tcs.Task;
    }

    //[Benchmark]
    public async Task<float[]> CreateEmbeddingAsync()
    {
        return await _model.CreateEmbeddingAsync(_text);
    }
    
    [Benchmark]
    public async Task<float[]> CreateEmbeddingConcurrentlyAsync()
    {
        float[] result = [];
        var tasks = new List<Task>();
        for (var i = 0; i < NumTasks; i++)
        {
            tasks.Add(Task.Run(async () =>
            {
                result = await _model.CreateEmbeddingAsync(_text);
            }));
        }

        await Task.WhenAll(tasks);
        return result;
    }
}
#endif
#endif