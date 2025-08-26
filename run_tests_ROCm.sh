#!/bin/bash
dotnet restore /property:OrtTarget=ROCM
dotnet build -c Release --no-restore /property:OrtTarget=ROCM
dotnet test -c Release --no-build --no-restore --logger "console;verbosity=detailed"