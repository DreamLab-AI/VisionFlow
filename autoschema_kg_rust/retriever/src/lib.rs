//! High-Performance RAG Retriever Module
//!
//! This module provides advanced retrieval capabilities for RAG systems including:
//! - Multi-hop query processing with graph traversal
//! - Semantic search with vector embeddings
//! - Query expansion and rewriting
//! - Result ranking and filtering
//! - Context window management
//! - Caching layer for performance optimization

pub mod cache;
pub mod config;
pub mod context;
pub mod embeddings;
pub mod error;
pub mod graph;
pub mod metrics;
pub mod query;
pub mod ranking;
pub mod retriever;
pub mod search;
mod utils;

pub use cache::*;
pub use config::*;
pub use context::*;
pub use embeddings::*;
pub use error::*;
pub use graph::*;
pub use metrics::*;
pub use query::*;
pub use ranking::*;
pub use retriever::*;
pub use search::*;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{
        cache::{CacheManager, CacheStats},
        config::RetrieverConfig,
        context::{ContextManager, ContextWindow},
        embeddings::{EmbeddingModel, VectorIndex},
        error::{RetrieverError, Result},
        graph::{GraphTraverser, HopResult, TraversalStrategy},
        metrics::{MetricsCollector, RetrievalMetrics},
        query::{QueryExpander, QueryProcessor, QueryRewriter},
        ranking::{RankingModel, RankingStrategy, ScoredDocument},
        retriever::{MultiHopRetriever, RetrieverBuilder},
        search::{SearchEngine, SearchResult, SemanticSearch},
    };
}
