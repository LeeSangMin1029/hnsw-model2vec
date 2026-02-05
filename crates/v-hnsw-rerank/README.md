# v-hnsw-rerank

Cross-encoder reranking for v-hnsw search results.

## Overview

This crate provides cross-encoder models for reranking search results. Cross-encoders process (query, document) pairs together, providing more accurate relevance scores than bi-encoder retrieval alone.

## Features

- **Cross-Encoder Models**: MS-MARCO MiniLM and BGE reranker base
- **ONNX Runtime**: Fast inference with CPU/CUDA/DirectML support
- **Batched Scoring**: Efficient batch processing of candidates
- **Reranker Trait**: Implements `v_hnsw_search::Reranker` for integration

## Usage

```rust
use v_hnsw_rerank::{CrossEncoderReranker, CrossEncoderConfig, RerankerModel};
use v_hnsw_search::Reranker;

// Create cross-encoder reranker
let config = CrossEncoderConfig::new(RerankerModel::MsMiniLM)
    .with_batch_size(32)
    .with_max_length(512);
let reranker = CrossEncoderReranker::new(config)?;

// Rerank search results
let candidates = vec![
    (1, 0.8, "First document text".to_string()),
    (2, 0.7, "Second document text".to_string()),
];
let reranked = reranker.rerank("query text", &candidates)?;
```

## Models

### MS-MARCO MiniLM
- **Model**: microsoft/ms-marco-MiniLM-L-6-v2
- **Size**: 33M parameters
- **Max Length**: 512 tokens
- **Use Case**: Fast reranking with good accuracy

### BGE Reranker Base
- **Model**: BAAI/bge-reranker-base
- **Size**: 110M parameters
- **Max Length**: 512 tokens
- **Use Case**: Higher accuracy, slower inference

## Features

- `cuda`: Enable CUDA acceleration via ONNX Runtime
- `directml`: Enable DirectML acceleration (Windows)

## Integration with HybridSearcher

```rust
use v_hnsw_search::{HybridSearcher, SimpleHybridSearcher};
use v_hnsw_rerank::{CrossEncoderReranker, CrossEncoderConfig};

// Create hybrid searcher with cross-encoder reranking
let config = CrossEncoderConfig::default();
let reranker = CrossEncoderReranker::new(config)?;
let searcher = SimpleHybridSearcher::with_reranker(hnsw, bm25, config, reranker);

// Search with automatic reranking
let results = searcher.search(&query_vector, "query text", 10)?;
```

## Performance

Cross-encoder reranking is slower than bi-encoder retrieval but provides significantly better accuracy. Best practices:

1. Use bi-encoder retrieval to get top-100 candidates
2. Use cross-encoder to rerank to top-10
3. Use batch processing for efficiency
4. Enable CUDA/DirectML for faster inference

## License

MIT OR Apache-2.0
