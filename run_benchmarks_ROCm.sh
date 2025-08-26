#!/bin/bash
cd OrtForge.AI.MicroBenchmarks
dotnet restore /property:OrtTarget=ROCM
dotnet build --no-restore -c Release /property:OrtTarget=ROCM
dotnet run -c Release --no-build --no-restore /property:OrtTarget=ROCM