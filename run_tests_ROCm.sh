#!/bin/bash
dotnet restore /property:OrtTarget=ROCM
dotnet build --no-restore -c Release /property:OrtTarget=ROCM
dotnet test -c Release --no-build --no-restore --logger "console;verbosity=detailed"