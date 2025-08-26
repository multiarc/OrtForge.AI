#!/bin/bash
cd OrtForge.AI.MicroBenchmarks
dotnet restore /property:OrtTarget=ROCM
dotnet build -c Release --no-restore /property:OrtTarget=ROCM
dotnet run -c Release --no-build --no-restore /property:OrtTarget=ROCM