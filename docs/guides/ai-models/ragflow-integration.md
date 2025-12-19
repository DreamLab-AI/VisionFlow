---
title: RAGFlow Knowledge Management Integration
description: RAGFlow provides document ingestion, vector storage, and semantic search capabilities for building knowledge bases and RAG (Retrieval-Augmented Generation) applications.  It runs as a separate Dock...
category: guide
tags:
  - architecture
  - patterns
  - api
  - api
  - endpoints
related-docs:
  - guides/ai-models/README.md
  - guides/features/MOVED.md
  - guides/ai-models/INTEGRATION_SUMMARY.md
  - guides/ai-models/deepseek-deployment.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Docker installation
---

# RAGFlow Knowledge Management Integration

**Status**: Active
**Service**: RAGFlow (Docker-based)
**Network**: `docker_ragflow`

## Overview

RAGFlow provides document ingestion, vector storage, and semantic search capabilities for building knowledge bases and RAG (Retrieval-Augmented Generation) applications. It runs as a separate Docker service and integrates with the main system via REST API and shared Docker network.

## Architecture

```
Main System (Rust)
    ↓ Docker Network (docker_ragflow)
RAGFlow Service
    ↓
Vector Storage + Document Processing
```

**Network Configuration**:
- Network: `docker_ragflow` (bridge)
- Hostname: `turbo-devpod.ragflow`
- Aliases: `turbo-devpod.ragflow`, `turbo-unified.local`

## Components

### 1. RAGFlow Service (Docker)

**Configuration**: `docker-compose.unified-with-neo4j.yml`

```yaml
services:
  turbo-devpod:
    networks:
      - docker_ragflow
    hostname: turbo-devpod
    aliases:
      - turbo-devpod.ragflow
      - turbo-unified.local

networks:
  docker_ragflow:
    external: true
```

### 2. Rust Service Layer

**Location**: `src/services/ragflow_service.rs`

**Main Components**:
```rust
pub struct RAGFlowService {
    client: Client,
    settings: Arc<RwLock<AppFullSettings>>,
}

pub enum RAGFlowError {
    ReqwestError(reqwest::Error),
    StatusError(StatusCode, String),
    ParseError(String),
    IoError(std::io::Error),
}
```

**Key Methods**:
- Document ingestion
- Vector search
- Chat completions
- Streaming responses

### 3. API Handler

**Location**: `src/handlers/ragflow_handler.rs`

**Endpoints**: (See API reference for complete list)
- Document management
- Vector search
- Chat interface

### 4. Data Models

**Location**: `src/models/ragflow_chat.rs`

```rust
pub struct RAGFlowChatMessage {
    pub role: String,
    pub content: String,
}

pub struct RAGFlowChatRequest {
    pub conversation_id: String,
    pub messages: Vec<RAGFlowChatMessage>,
    pub stream: bool,
}
```

## Features

### 1. Document Ingestion

**Supported Formats**:
- PDF documents
- Markdown files
- Plain text
- (Check RAGFlow docs for full list)

**Processing**:
- Text extraction
- Chunking and segmentation
- Embedding generation
- Vector storage

### 2. Semantic Vector Search

**Capabilities**:
- Similarity search across documents
- Semantic meaning matching (not just keywords)
- Relevance ranking
- Context retrieval

### 3. Chat Interface

**Conversational Q&A**:
- Query knowledge base via natural language
- Context-aware responses
- Source citation
- Conversation history

### 4. Streaming Responses

**Real-time Generation**:
- Stream responses as they generate
- Lower perceived latency
- Better UX for long responses

## Integration Points

### From Rust API

```rust
// Example: Ingest document
let service = RAGFlowService::new();
service.ingest_document(file_path, metadata).await?;

// Example: Search knowledge base
let results = service.vector_search("query", top_k=5).await?;

// Example: Chat query
let response = service.chat_completion(conversation_id, messages).await?;
```

### From REST API

```bash
# Document ingestion
curl -X POST http://localhost:4000/api/ragflow/ingest \
  -F "file=@document.pdf" \
  -F "metadata={\"title\":\"Document Title\"}"

# Vector search
curl -X POST http://localhost:4000/api/ragflow/search \
  -H "Content-Type: application/json" \
  -d '{"query": "semantic search query", "top_k": 5}'

# Chat query
curl -X POST http://localhost:4000/api/ragflow/chat \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv-123",
    "messages": [{"role": "user", "content": "What is the main topic?"}],
    "stream": false
  }'
```

### Docker Network Access

```bash
# Check network connectivity
docker network inspect docker_ragflow

# Verify hostname resolution
docker exec turbo-flow-unified ping turbo-devpod.ragflow

# Test RAGFlow service
docker exec turbo-flow-unified curl http://ragflow-service:port/health
```

## Use Cases

### 1. Knowledge Base Management

**Scenario**: Build searchable knowledge base from documentation

```rust
// Ingest documentation
for doc in documentation_files {
    ragflow.ingest_document(&doc).await?;
}

// Query knowledge base
let answer = ragflow.chat(
    "How do I configure authentication?"
).await?;
```

### 2. Document Q&A System

**Scenario**: Answer questions about specific documents

```rust
// Upload document
ragflow.ingest_document("contract.pdf").await?;

// Ask questions
let response = ragflow.chat_completion(
    "contract-conv",
    vec![ChatMessage {
        role: "user",
        content: "What are the payment terms?"
    }]
).await?;
```

### 3. Semantic Search

**Scenario**: Find relevant information across large document sets

```rust
// Vector search
let results = ragflow.vector_search(
    "machine learning best practices",
    top_k=10
).await?;

// Results ranked by semantic relevance
for result in results {
    println!("{}: {}", result.title, result.score);
}
```

### 4. RAG Workflows

**Scenario**: Combine retrieval with generation

```rust
// 1. Retrieve relevant context
let context = ragflow.vector_search(query, top_k=5).await?;

// 2. Generate response with context
let prompt = format!(
    "Context: {}\n\nQuestion: {}\n\nAnswer:",
    context.join("\n"),
    query
);
let response = llm.generate(prompt).await?;
```

## Performance Characteristics

### Document Processing

| Document Size | Processing Time | Notes |
|--------------|-----------------|-------|
| Small (<1MB) | 1-5 seconds | Fast indexing |
| Medium (1-10MB) | 5-30 seconds | Depends on complexity |
| Large (>10MB) | 30s-5min | May require chunking |

### Vector Search

| Knowledge Base Size | Search Latency | Notes |
|--------------------|---------------|-------|
| Small (<1000 docs) | <100ms | Sub-second |
| Medium (1K-10K) | 100-500ms | Fast enough |
| Large (>10K) | 500ms-2s | May need optimization |

### Memory Usage

- **Base**: ~500MB-1GB (RAGFlow service)
- **Per Document**: ~1-10MB (depends on size and embeddings)
- **Vector Index**: Scales with document count

**Recommendation**: Allocate at least 2GB RAM for RAGFlow service.

## Configuration

### Docker Compose

**File**: `docker-compose.unified-with-neo4j.yml`

```yaml
networks:
  docker_ragflow:
    external: true
    driver: bridge
```

### Environment Variables

```bash
# RAGFlow configuration (if needed)
RAGFLOW_API_URL=http://ragflow-service:port
RAGFLOW_API_KEY=xxxxx  # If authentication required
```

### Service Settings

**Location**: Rust service configuration

```rust
// In src/services/ragflow_service.rs
// Timeout, retries, batch size, etc.
const TIMEOUT: Duration = Duration::from_secs(30);
```

## Best Practices

### 1. Document Chunking

- **Small chunks**: Better precision, more storage
- **Large chunks**: Better context, less storage
- **Recommended**: 512-1024 tokens per chunk

### 2. Metadata Management

Include useful metadata for filtering:
```json
{
  "title": "Document Title",
  "author": "Author Name",
  "date": "2025-12-02",
  "category": "technical",
  "tags": ["api", "authentication", "security"]
}
```

### 3. Query Optimization

- **Specific queries**: Better than vague questions
- **Context inclusion**: Add relevant background
- **Iterative refinement**: Refine based on results

### 4. Memory Management

- Monitor vector storage size
- Clean up old/unused documents
- Batch ingestion for large datasets

## Limitations

### Docker Network Dependency

**Issue**: RAGFlow must run in same Docker network
**Impact**: Cannot access from outside Docker environment
**Solution**: Ensure both services in `docker_ragflow` network

### Resource Requirements

**Issue**: RAGFlow requires significant memory
**Impact**: May impact performance on low-resource systems
**Solution**: Allocate minimum 2GB RAM, monitor usage

### Processing Time

**Issue**: Large documents take time to process
**Impact**: Latency for document ingestion
**Solution**: Batch process during off-peak hours

### Embedding Model

**Issue**: Fixed embedding model (RAGFlow default)
**Impact**: Quality depends on model capabilities
**Solution**: Check RAGFlow docs for model configuration

## Troubleshooting

### Cannot Connect to RAGFlow

**Symptom**: Connection refused or network errors

**Solutions**:
1. Check Docker network: `docker network inspect docker_ragflow`
2. Verify service running: `docker ps | grep ragflow`
3. Test connectivity: `ping turbo-devpod.ragflow`
4. Check hostname resolution in `/etc/hosts`

### Document Processing Slow

**Symptom**: Ingestion takes long time

**Solutions**:
1. Check CPU/memory usage: `docker stats`
2. Process smaller batches
3. Increase Docker resource allocation
4. Consider pre-processing documents

### Search Quality Poor

**Symptom**: Irrelevant results returned

**Solutions**:
1. Improve query specificity
2. Include context in search
3. Adjust `top_k` parameter (more results)
4. Review document chunking strategy

### Memory Issues

**Symptom**: Service crashes or slow performance

**Solutions**:
1. Monitor memory: `docker stats ragflow-service`
2. Clean up old documents
3. Increase Docker memory limit
4. Consider document archival strategy

## Security

### Network Isolation

- RAGFlow in separate Docker network
- Not exposed to external network (unless configured)
- Internal communication only

### Data Privacy

- Documents stored in RAGFlow internal storage
- Vector embeddings not reversible (privacy-preserving)
- No external API calls (self-hosted)

### Best Practices

1. **Authentication**: Add API key if exposing externally
2. **Network Security**: Keep in isolated Docker network
3. **Data Encryption**: Consider encryption at rest (Docker volumes)
4. **Access Control**: Limit which services can access RAGFlow

## Integration with Other AI Services

### With Perplexity

```rust
// 1. Research with Perplexity
let research = perplexity.research("AI safety 2025").await?;

// 2. Store in RAGFlow
ragflow.ingest_text(
    &research.content,
    metadata![
        "source": "perplexity",
        "topic": "AI safety",
        "date": "2025-12-02"
    ]
).await?;

// 3. Query knowledge base
let answer = ragflow.chat("What are the main AI safety concerns?").await?;
```

### With DeepSeek

```rust
// 1. Query RAGFlow for context
let context = ragflow.vector_search("algorithm optimization", 5).await?;

// 2. Deep reasoning with DeepSeek
let analysis = deepseek.reason(
    format!("Given context: {}\n\nAnalyze optimization strategies", context)
).await?;
```

## Monitoring

### Metrics to Track

- **Document count**: Total documents in knowledge base
- **Storage size**: Vector storage size
- **Query latency**: Search response time
- **Ingestion rate**: Documents processed per hour
- **Error rate**: Failed operations

### Health Checks

```bash
# Docker health
docker ps --filter name=ragflow

# Network connectivity
docker exec turbo-flow-unified curl http://ragflow-service/health

# Storage usage
docker exec ragflow-service df -h
```

### Logging

```bash
# Service logs
docker logs ragflow-service

# Rust service logs
tail -f /var/log/turbo-flow.log | grep ragflow
```

---

---

## Related Documentation

- [Perplexity AI Integration](perplexity-integration.md)
- [Adding Features](../developer/04-adding-features.md)
- [Working with Agents](../../archive/docs/guides/user/working-with-agents.md)
- [ComfyUI Service Integration - Automatic Startup](../../comfyui-service-integration.md)
- [Skills Documentation](../../multi-agent-docker/SKILLS.md)

## Future Enhancements

### Planned

1. **Multi-tenant Support** - Separate knowledge bases per user/project
2. **Advanced Filtering** - Filter by metadata, date, category
3. **Analytics Dashboard** - Query patterns, usage statistics
4. **Backup/Restore** - Automated backups of knowledge base

### Under Consideration

1. **Custom Embeddings** - Support for different embedding models
2. **Hybrid Search** - Combine vector + keyword search
3. **Document Versioning** - Track document changes over time
4. **Export/Import** - Knowledge base portability
