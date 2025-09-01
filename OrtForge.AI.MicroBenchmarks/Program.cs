using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace OrtForge.AI.MicroBenchmarks;

class Program
{
    static void Main(string[] args)
    {
        var config = DefaultConfig.Instance
#if ROCM                    
            .AddJob(Job.Default.WithArguments([new MsBuildArgument("/p:OrtTarget=ROCM")]))
#elif CUDA           
            .AddJob(Job.Default.WithArguments([new MsBuildArgument("/p:OrtTarget=CUDA")]))
#endif
        ;
        
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
    }
}