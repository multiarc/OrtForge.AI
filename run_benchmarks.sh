#!/bin/bash
cd OrtForge.AI.MicroBenchmarks
dotnet restore
dotnet build --no-restore -c Release
dotnet run --no-build --no-restore