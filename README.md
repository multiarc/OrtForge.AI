# OrtForge.AI (Active work in progress!)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OrtForge.AI is a comprehensive .NET library suite for building advanced Retrieval Augmented Generation (RAG) applications with fully local hosting capabilities. It leverages ONNX Runtime to provide efficient, cross-platform AI model deployment without cloud dependencies.

## 🌟 Features

- **Fully Local AI Processing**: Run LLMs (TODO), embedding models, and rerankers entirely on your own infrastructure
- **High-Performance Inference**: Optimized execution through ONNX Runtime integration
- **Cross-Platform Support**: Works across Windows, Linux, and macOS environments (TBD on acceleration technics)
- **Advanced RAG Capabilities**: Build sophisticated document retrieval and generation workflows
- **Modular Architecture**: Mix and match components to suit your specific use case
- **Production-Ready**: Designed for reliability and performance in real-world applications

## 📋 Components

OrtForge.AI consists of several specialized libraries:

- **OrtForge.AI.Core**: Core abstractions and utilities for the entire framework
- **OrtForge.AI.Models**: Implementation of various AI models including embedding generators and rerankers
- **OrtForge.AI.LLM**: Integration with large language models for text generation
- **OrtForge.AI.Rag**: Building blocks for implementing RAG pipelines
- **OrtForge.PgSql**: PostgreSQL query generator library to enable vector search

## 🚀 Getting Started

### Prerequisites

- .NET 8.0 SDK or later
- Sufficient CPU/GPU resources for running inference (requirements vary by model)

### Installation

```bash
#TBD
```

### Basic Usage

```csharp
// Initialize an embedding model
var embeddingModel = new BgeM3Model("path/to/tokenizer.bpe.model", "path/to/model.onnx");

// Generate embeddings for text
var embeddings = await embeddingModel.CreateEmbeddingAsync("Your text here");

// Generate embeddings for multiple texts
var batchEmbeddings = await embeddingModel.CreateEmbeddingsAsync(new[] { "First text", "Second text" });

// Initialize a reranker
var reranker = new BgeRerankerM3("path/to/tokenizer.bpe.model", "path/to/reranker.onnx");

// Get reranking score between query and document
float score = await reranker.GetRerankingScoreAsync("query", "document");

// Initialize and use LLM (implementation may vary)
var llmOptions = new LlmOptions { ModelPath = "path/to/llm.onnx" };
var llm = new LlmService(llmOptions);
var response = await llm.GenerateResponseAsync("Your prompt here");
```

## 📖 Documentation

Comprehensive documentation is available in the `/docs` directory and includes:

- Detailed API references
- Architecture overview
- Performance optimization guides
- Example applications and use cases

## 🔍 Examples

The `/examples` directory contains complete sample applications demonstrating:

- Document indexing and retrieval
- Question-answering systems
- Custom RAG pipeline construction
- Performance benchmarking

## 🛠️ Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/OrtForge.AI.git
cd OrtForge.AI

# Build the solution
dotnet build

# Run tests
dotnet test
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📊 Benchmarks

Performance benchmarks for various models and configurations can be found in the `/benchmarks` directory.

## 🔗 Related Projects

- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [LangChain](https://github.com/hwchase17/langchain)
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
