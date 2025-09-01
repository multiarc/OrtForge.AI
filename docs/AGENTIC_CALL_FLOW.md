### ONNX Runtime GenAI: C# to Native Call Flow

This document diagrams the high-level flow from the C# API down to the native layers, stopping at ONNX Runtime. File and symbol names are shown for orientation.

Key C# entry points:
- `Model` (loads model/config)
- `Generator`, `GeneratorParams` (token generation)
- `Tokenizer`, `TokenizerStream` (text <-> tokens)
- `MultiModalProcessor` (image/audio preprocessing)

Native boundaries:
- P/Invoke to `onnxruntime-genai` via `src/csharp/NativeMethods.cs`
- C API in `src/ort_genai_c.h` implemented by `src/ort_genai_c.cpp`
- C++ implementation in `src/models/*.cpp`, `src/generators.cpp`, etc.
- ONNX Runtime boundary: `OrtSession::Create`, `OrtSession::Run`, allocators in `src/models/onnxruntime_api.h`, `src/models/model.cpp`

---

### Component Map (C# → P/Invoke → C API → C++ → ONNX Runtime)

```mermaid
flowchart LR
  subgraph CSharp["C# (Microsoft.ML.OnnxRuntimeGenAI)"]
    CS_Model["Model"]
    CS_Gen["Generator / GeneratorParams"]
    CS_Tok["Tokenizer / TokenizerStream"]
    CS_MMP["MultiModalProcessor"]
  end

  subgraph PInvoke["P/Invoke (src/csharp/NativeMethods.cs)"]
    PINV["[DllImport('onnxruntime-genai')] Oga* functions"]
  end

  subgraph CAPI["C API (src/ort_genai_c.h/.cpp)"]
    C_OgaCreateModel["OgaCreateModel"]
    C_OgaCreateGenerator["OgaCreateGenerator"]
    C_OgaTokenizer["OgaCreateTokenizer / OgaTokenizer*"]
    C_OgaProcessor["OgaCreateMultiModalProcessor / OgaProcessor*"]
    C_OgaGenOps["OgaGenerator_* (AppendTokens/GenerateNextToken/GetLogits)"]
  end

  subgraph CPP["C++ Impl (namespace Generators)"]
    CPP_Model["Model (src/models/model.cpp)"]
    CPP_Gen["Generator (src/generators.cpp)"]
    CPP_Tok["Tokenizer"]
    CPP_Proc["MultiModalProcessor"]
  end

  subgraph ORT["ONNX Runtime Boundary"]
    ORT_Session["OrtSession::Create / Run"]
    ORT_Allocs["Ort::Allocator, OrtMemoryInfo"]
  end

  CS_Model --> PINV --> C_OgaCreateModel --> CPP_Model --> ORT_Session
  CS_Gen --> PINV --> C_OgaCreateGenerator --> CPP_Gen --> ORT_Session
  CS_Tok --> PINV --> C_OgaTokenizer --> CPP_Tok
  CS_MMP --> PINV --> C_OgaProcessor --> CPP_Proc
  CS_Gen -. runtime ops .-> PINV -.-> C_OgaGenOps -.-> CPP_Gen -.-> ORT_Allocs
```

---

### Model Construction Flow

Relevant code:
- `src/csharp/Model.cs` → `NativeMethods.OgaCreateModel`
- `src/ort_genai_c.cpp: OgaCreateModel` → `OgaCreateModelWithRuntimeSettings`
- `Generators::CreateModel` → `Model::CreateSessionOptions`, `Model::CreateSession`, `OrtSession::Create`

```mermaid
sequenceDiagram
  autonumber
  participant C# as C# Model
  participant P as P/Invoke OgaCreateModel
  participant C as C API (ort_genai_c.cpp)
  participant CPP as Generators::Model
  participant ORT as ONNX Runtime

  C#->>P: OgaCreateModel(configPath)
  P->>C: OgaCreateModel
  C->>C: OgaCreateModelWithRuntimeSettings(...)
  C->>CPP: Generators::CreateModel(GetOrtEnv(), ...)
  CPP->>CPP: CreateSessionOptionsFromConfig(...)
  CPP->>ORT: OrtSession::Create(...)
  ORT-->>CPP: OrtSession*
  CPP-->>C: shared OgaModel
  C-->>C#: IntPtr model handle
```

---

### Generation Loop Flow

Relevant code:
- `src/csharp/Generator.cs` → `NativeMethods.OgaCreateGenerator`
- `src/ort_genai_c.cpp: OgaCreateGenerator`, `OgaGenerator_*`
- `Generators::Generator::GenerateNextToken`, `Model::Run`
- ORT calls: `OrtSession::Run`

```mermaid
sequenceDiagram
  autonumber
  participant C# as C# Generator
  participant P as P/Invoke Oga*
  participant C as C API (ort_genai_c.cpp)
  participant CPP as Generators::Generator/Model
  participant ORT as ONNX Runtime

  C#->>P: OgaCreateGenerator(model, params)
  P->>C: OgaCreateGenerator
  C->>CPP: CreateGenerator(model, params)
  CPP-->>C#: IntPtr generator handle

  loop per step
    C#->>P: OgaGenerator_AppendTokens / _SetInputs
    P->>C: OgaGenerator_*
    C->>CPP: generator->AppendTokens / SetInputs
    C#->>P: OgaGenerator_GenerateNextToken
    P->>C: OgaGenerator_GenerateNextToken
    C->>CPP: generator->GenerateNextToken()
    CPP->>CPP: model->Run(...)
    CPP->>ORT: OrtSession::Run(inputs, outputs)
    ORT-->>CPP: logits/output OrtValues
    CPP-->>C: expose logits/next tokens via accessors
    C#->>P: OgaGenerator_GetNextTokens / _GetLogits
    P->>C: OgaGenerator_* getters
    C-->>C#: tokens/logits (CPU memory)
  end
```

---

### Tokenizer Encode/Decode Flow

Relevant code:
- `src/csharp/Tokenizer.cs` → `OgaCreateTokenizer`, `OgaTokenizerEncode`, `OgaTokenizerDecode`
- `src/ort_genai_c.cpp: OgaCreateTokenizer`, `OgaTokenizer*`
- `Generators::Tokenizer`

```mermaid
sequenceDiagram
  autonumber
  participant C# as C# Tokenizer
  participant P as P/Invoke OgaTokenizer*
  participant C as C API (ort_genai_c.cpp)
  participant CPP as Generators::Tokenizer

  C#->>P: OgaCreateTokenizer(model)
  P->>C: OgaCreateTokenizer
  C->>CPP: model->CreateTokenizer()
  CPP-->>C#: IntPtr tokenizer handle

  C#->>P: OgaTokenizerEncode(str)
  P->>C: OgaTokenizerEncode
  C->>CPP: tokenizer->Encode(str)
  CPP-->>C#: token ids

  C#->>P: OgaTokenizerDecode(tokens)
  P->>C: OgaTokenizerDecode
  C->>CPP: tokenizer->Decode(tokens)
  CPP-->>C#: string
```

---

### MultiModal Processor (Images/Audio → NamedTensors)

Relevant code:
- `src/csharp/MultiModalProcessor.cs` → `OgaCreateMultiModalProcessor`, `OgaProcessorProcess*`
- `src/ort_genai_c.cpp: OgaCreateMultiModalProcessor`, `OgaProcessorProcess*`
- `Generators::MultiModalProcessor`

```mermaid
sequenceDiagram
  autonumber
  participant C# as C# MultiModalProcessor
  participant P as P/Invoke OgaProcessor*
  participant C as C API (ort_genai_c.cpp)
  participant CPP as Generators::MultiModalProcessor

  C#->>P: OgaCreateMultiModalProcessor(model)
  P->>C: OgaCreateMultiModalProcessor
  C->>CPP: model->CreateMultiModalProcessor()
  CPP-->>C#: IntPtr processor handle

  C#->>P: OgaProcessorProcessImages(prompt, images)
  P->>C: OgaProcessorProcessImages
  C->>CPP: processor->Process(...)
  CPP-->>C#: NamedTensors
  C-->>C#: IntPtr named tensors
```

---

### Error Handling (Result pattern)

Errors from native calls surface via `OgaResult`:
- C# wrappers call `Result.VerifySuccess(NativeMethods.Oga...(...))`
- C API returns `OgaResult*` on failure; message via `OgaResultGetError`
- Typical C entry: `OGA_TRY`/`OGA_CATCH` in `src/ort_genai_c.cpp`

```mermaid
flowchart LR
  CAPI["C API call"] -->|throw std::exception| CATCH["OGA_CATCH → make OgaResult(error)"]
  CATCH --> CS["C# Result.VerifySuccess → throw OnnxRuntimeGenAIException(message)"]
```

---

### Stopping Boundary

These diagrams stop at ONNX Runtime calls within the native layer:
- `OrtSession::Create` and `OrtSession::Run` in `src/models/model.cpp`
- Allocators and device interfaces in `src/models/onnxruntime_api.h`


