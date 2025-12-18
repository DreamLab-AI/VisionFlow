---
title: AI Models & Services Integration
description: This system integrates multiple AI models and services for diverse use cases: real-time research, reasoning, knowledge management, and document processing.  Each service is isolated for security an...
category: howto
tags:
  - architecture
  - patterns
  - structure
  - api
  - rest
related-docs:
  - guides/ai-models/deepseek-verification.md
  - guides/ai-models/deepseek-deployment.md
  - guides/ai-models/perplexity-integration.md
  - guides/ai-models/ragflow-integration.md
  - README.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Docker installation
---

# AI Models & Services Integration

**Status**: Active
**Last Updated**: December 2, 2025
**Maintainer**: Development Team

## Overview

This system integrates multiple AI models and services for diverse use cases: real-time research, reasoning, knowledge management, and document processing. Each service is isolated for security and configured for specific workloads.

## Integrated Services

### 1. DeepSeek Reasoning (deepseek-reasoner)

**Status**: ✅ Active
**Purpose**: Advanced multi-step reasoning and complex problem-solving
**Endpoint**: `https://api.deepseek.com`

#### Models
- **deepseek-chat** - Fast general-purpose chat (standard endpoint)
- **deepseek-reasoner** - Advanced reasoning with chain-of-thought traces

#### Performance Metrics
- **Response Time**: 2-5 seconds for reasoning queries
- **Token Efficiency**: 200-500 tokens per query (includes reasoning tokens)
- **Quality**: Excellent for multi-step logic, math, algorithm analysis
- **Cost**: Significantly lower than GPT-4

#### Integration Points
- **MCP Skill**: `/multi-agent-docker/skills/deepseek-reasoning/`
- **CLI Tool**: `/usr/local/bin/deepseek-chat` (container-only)
- **User Isolation**: `deepseek-user` (UID 1004)
- **Configuration**: `/home/deepseek-user/.config/deepseek/config.json`

#### MCP Tools
1. `deepseek_reason` - Complex reasoning with thinking mode
2. `deepseek_analyze` - Code/system analysis with root cause reasoning
3. `deepseek_plan` - Task planning with dependency analysis

#### Use Cases
- Multi-step logical analysis
- Mathematical proofs and algorithm complexity
- Bug root cause identification
- Architecture evaluation
- Feature decomposition and planning

#### Limitations
- **agentic-flow**: Does not natively support DeepSeek (use custom CLI or MCP skill)
- **Special endpoint**: `v3.2_speciale_expires_on_20251215` returns empty content (use standard endpoint)
- **JSON parsing**: Some complex prompts may fail (escape special characters)

**Documentation**:
- [Verification Guide](deepseek-verification.md)
- [Deployment Guide](deepseek-deployment.md)
- [Skill Guide](/multi-agent-docker/skills/deepseek-reasoning/SKILL.md)

---

### 2. Perplexity AI (Sonar API)

**Status**: ✅ Active
**Purpose**: Real-time web search, research, and source-cited responses
**Endpoint**: Perplexity Sonar API

#### Models
- **sonar** - Fast, balanced (default)
- **sonar-pro** - Deep research with more sources
- **sonar-reasoning** - Complex analysis with extended context

#### Performance Metrics
- **Response Time**: 3-8 seconds depending on depth
- **Sources**: 5-15 cited sources per response
- **Quality**: Excellent for current events, market research, technical documentation
- **Cost**: Usage-based, competitive pricing

#### Integration Points
- **MCP Skill**: `/multi-agent-docker/skills/perplexity/`
- **Rust Service**: `src/services/perplexity_service.rs`
- **API Handler**: `src/handlers/perplexity_handler.rs`
- **Configuration**: Environment variable `PERPLEXITY_API_KEY`

#### MCP Tools
1. `perplexity_search` - Quick factual search with citations
2. `perplexity_research` - Deep research analysis with multi-source synthesis
3. `perplexity_generate_prompt` - Prompt optimization for maximum quality

#### Use Cases
- Real-time web data access
- Current events and news research
- Market research and competitive analysis
- Technical documentation lookup
- UK/European-centric research queries

#### Limitations
- **Rate Limits**: API rate limits apply (configure in service)
- **Context Window**: Limited by model (check Perplexity docs)
- **Prompt Engineering**: Quality depends on prompt structure (use `generate_prompt` tool)

**Documentation**:
- [Integration Guide](perplexity-integration.md)
- [Skill Guide](/multi-agent-docker/skills/perplexity/SKILL.md)
- [Templates](/multi-agent-docker/skills/perplexity/docs/templates.md)

---

### 3. RAGFlow Knowledge Management

**Status**: ✅ Active
**Purpose**: Document ingestion, vector search, and knowledge base management
**Endpoint**: Docker network `docker_ragflow`

#### Components
- **RAGFlow Server**: Document processing and vector storage
- **Rust Service**: `src/services/ragflow_service.rs`
- **API Handler**: `src/handlers/ragflow_handler.rs`
- **Chat Models**: `src/models/ragflow_chat.rs`

#### Performance Metrics
- **Document Processing**: Depends on document size and complexity
- **Vector Search**: Sub-second for typical knowledge bases
- **Quality**: Excellent for semantic search and document retrieval
- **Cost**: Self-hosted, no API costs

#### Integration Points
- **Docker Network**: `docker_ragflow` bridge network
- **Hostname**: `turbo-devpod.ragflow`
- **API Endpoints**: Via REST API (see API reference)
- **Configuration**: `docker-compose.unified-with-neo4j.yml`

#### Features
1. **Document Ingestion**: PDF, Markdown, text file processing
2. **Vector Storage**: Semantic embeddings for search
3. **Chat Interface**: Conversational knowledge base queries
4. **Streaming Responses**: Real-time response generation

#### Use Cases
- Knowledge base management
- Document Q&A systems
- Semantic search across large document sets
- RAG (Retrieval-Augmented Generation) workflows

#### Limitations
- **Docker Network**: Must run in same Docker network
- **Memory**: Vector storage requires sufficient RAM
- **Processing Time**: Large documents take time to index

**Documentation**:
- [Integration Guide](ragflow-integration.md)
- [Docker Configuration](/docker-compose.unified-with-neo4j.yml)
- [API Reference](/docs/reference/api-complete-reference.md)

---

### 4. Claude via Z.AI Service (Cost-Effective)

**Status**: ✅ Active
**Purpose**: Cost-effective Claude API access with worker pool
**Endpoint**: `http://localhost:9600` (internal only)

#### Configuration
- **User**: `zai-user` (UID 1003)
- **Port**: 9600 (internal, not exposed)
- **Worker Pool**: 4 concurrent workers (configurable)
- **Max Queue**: 50 requests (configurable)
- **Base URL**: `ANTHROPIC_BASE_URL` environment variable

#### Performance Metrics
- **Response Time**: Depends on Z.AI API latency
- **Concurrency**: 4 parallel requests
- **Queue Management**: Auto-throttling at 50 requests
- **Timeout**: 30 seconds default
- **Cost**: Z.AI pricing (typically lower than direct Anthropic)

#### Integration Points
- **Service**: `/opt/claude-zai/wrapper/server.js`
- **Supervisord**: Priority 500
- **User Isolation**: `zai-user` with separate credentials
- **Configuration**: `/home/zai-user/.config/zai/config.json`

#### API Endpoints
- `POST /chat` - Chat completion request
- `GET /health` - Service health check

#### Use Cases
- Cost-effective Claude access for batch operations
- Background processing tasks
- Skills that need Claude without direct API costs
- Internal automation workflows

#### Limitations
- **Internal Only**: Not exposed outside container
- **Z.AI Dependency**: Requires valid Z.AI account
- **Worker Pool**: Limited concurrency (configurable)
- **No Streaming**: Responses are complete (not streamed)

**Documentation**:
- Service Configuration: `/multi-agent-docker/unified-config/supervisord.unified.conf`
- User Setup: `/multi-agent-docker/unified-config/entrypoint-unified.sh`

---

### 5. Google Gemini (via gemini-flow)

**Status**: ⚠️ Experimental
**Purpose**: Multi-agent coordination with Gemini models
**Endpoint**: Google Gemini API

#### Configuration
- **User**: `gemini-user` (UID 1001)
- **Service**: `gemini-flow` (supervisord priority 600)
- **Configuration**: `/multi-agent-docker/config/gemini-flow.config.ts`

#### Commands
- `gf-init` - Initialize with protocols (a2a, mcp) and topology
- `gf-swarm` - Spawn 66 agents with intelligent coordination
- `gf-architect` - 5 system architects
- `gf-coder` - 12 master coders
- `gf-status` - Swarm status
- `gf-monitor` - Protocols and performance
- `gf-health` - Health check

#### Use Cases
- Multi-agent workflows requiring Gemini's long context
- Experimental agent coordination patterns
- Alternative to Claude for specific tasks

#### Limitations
- **Experimental**: Still in development, may have stability issues
- **API Costs**: Google Gemini API pricing applies
- **Integration**: Not fully integrated with main system
- **Documentation**: Limited compared to other services

**Documentation**:
- Configuration: `/multi-agent-docker/config/gemini-flow.config.ts`
- User Setup: Check gemini-user home directory

---

### 6. OpenAI (Isolated User)

**Status**: ⚠️ Configured but Inactive
**Purpose**: OpenAI API access with user isolation
**Endpoint**: OpenAI API

#### Configuration
- **User**: `openai-user` (UID 1002)
- **Configuration**: `/home/openai-user/.config/openai/config.json`
- **API Key**: Configured via environment variable

#### Status
Currently configured but not actively used in the system. User isolation is in place for potential future integration.

**Recommendation**: Activate only if specific OpenAI models (GPT-4, DALL-E, Whisper) are required for features not covered by other services.

---

## Service Architecture

### Multi-User Isolation

Four isolated Linux users with credential separation:

| User | UID | Purpose | API Keys | Access |
|------|-----|---------|----------|--------|
| devuser | 1000 | Primary development | Claude (direct) | sudo, all tools |
| gemini-user | 1001 | Google Gemini | Gemini API | `as-gemini` |
| openai-user | 1002 | OpenAI tools | OpenAI API | `as-openai` |
| zai-user | 1003 | Z.AI service | Claude (via Z.AI) | `as-zai` |
| deepseek-user | 1004 | DeepSeek reasoning | DeepSeek API | MCP bridge |

### Service Priority (Supervisord)

| Service | Priority | User | Purpose |
|---------|----------|------|---------|
| management-api | 300 | devuser | HTTP API |
| claude-zai | 500 | zai-user | Cost-effective Claude |
| gemini-flow | 600 | gemini-user | Gemini coordination |

### MCP Skills Architecture

```
Claude Code (devuser)
    ↓ MCP Protocol
MCP Skill Server
    ↓ User bridge (sudo)
Isolated AI Service User
    ↓ HTTPS API
AI Service Provider
```

**Skills with MCP Servers**:
- `deepseek-reasoning` - DeepSeek reasoning bridge
- `perplexity` - Perplexity research assistant
- `web-summary` - YouTube + web summarization (uses Z.AI)

---

## Hybrid AI Workflows

### Pattern 1: DeepSeek as Planner, Claude as Executor

**Best for**: Complex features requiring reasoning + polished implementation

```
User Query
    ↓
Claude detects complexity
    ↓
deepseek_plan() or deepseek_reason()
    ↓
DeepSeek returns structured reasoning
    ↓
Claude implements with polished code
    ↓
Production-ready result
```

### Pattern 2: Perplexity for Research, RAGFlow for Knowledge

**Best for**: Research-heavy tasks with knowledge persistence

```
Research Query
    ↓
perplexity_research() with citations
    ↓
Store results in RAGFlow knowledge base
    ↓
Semantic search for future queries
    ↓
Reuse knowledge without re-research
```

### Pattern 3: Z.AI for Batch Processing

**Best for**: Background tasks, summaries, non-critical workloads

```
Background Task
    ↓
Queue request to Z.AI service (port 9600)
    ↓
Worker pool processes with cost-effective Claude
    ↓
Results stored or returned
    ↓
Cost savings on high-volume operations
```

---

## Configuration

### Environment Variables

```bash
# DeepSeek
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-reasoner

# Perplexity
PERPLEXITY_API_KEY=pplx-xxxxx

# Z.AI (Claude via Z.AI)
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxx
CLAUDE_WORKER_POOL_SIZE=4
CLAUDE_MAX_QUEUE_SIZE=50

# Gemini
GOOGLE_GEMINI_API_KEY=xxxxx

# OpenAI (if activated)
OPENAI_API_KEY=sk-xxxxx
```

### User Configuration Files

- **devuser**: `/home/devuser/.config/claude/config.json`
- **gemini-user**: `/home/gemini-user/.config/gemini/config.json`
- **openai-user**: `/home/openai-user/.config/openai/config.json`
- **zai-user**: `/home/zai-user/.config/zai/config.json`
- **deepseek-user**: `/home/deepseek-user/.config/deepseek/config.json`

Configuration is distributed at container startup via `entrypoint-unified.sh`.

---

## Cost Optimization

### Service Cost Comparison (Approximate)

| Service | Use Case | Relative Cost | Best For |
|---------|----------|---------------|----------|
| DeepSeek | Reasoning | 💰 Low | Complex logic, planning |
| Perplexity | Research | 💰💰 Medium | Real-time web data |
| RAGFlow | Knowledge | 💰 Self-hosted | Document storage/search |
| Z.AI (Claude) | Batch | 💰💰 Medium | Background tasks |
| Claude Direct | Primary | 💰💰💰 High | Interactive dev work |
| Gemini | Experimental | 💰💰 Medium | Long context tasks |

### Recommendations

1. **Use DeepSeek for planning** - Cheaper than Claude for reasoning-heavy tasks
2. **Use Perplexity for research** - Avoid manual web searches, get cited sources
3. **Use RAGFlow for persistence** - Store knowledge, avoid re-processing
4. **Use Z.AI for batch** - Cost-effective Claude for non-interactive tasks
5. **Use Claude Direct sparingly** - Reserve for interactive development where quality matters most

---

## Testing & Validation

### DeepSeek
```bash
# CLI test
docker exec -u deepseek-user agentic-workstation deepseek-chat "What is 2+2?"

# MCP test
docker exec agentic-workstation \
  node /home/devuser/.claude/skills/deepseek-reasoning/tools/deepseek_client.js \
  --tool deepseek_reason --params '{"query":"Explain quicksort","format":"steps"}'
```

### Perplexity
```bash
# MCP test
docker exec agentic-workstation \
  python /home/devuser/.claude/skills/perplexity/tools/perplexity_client.py \
  search "current UK mortgage rates"
```

### Z.AI
```bash
# Health check
curl http://localhost:9600/health

# Chat request
curl -X POST http://localhost:9600/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","timeout":30000}'
```

### RAGFlow
```bash
# Check network connectivity
docker network inspect docker_ragflow
ping turbo-devpod.ragflow
```

---

## Troubleshooting

### DeepSeek Issues

**Problem**: Empty content field in response
**Solution**: Use standard endpoint with `deepseek-reasoner` model (not special endpoint)

**Problem**: JSON parsing errors
**Solution**: Escape special characters in prompts, simplify complex queries

**Problem**: Permissions errors
**Solution**: Ensure skill directories have `o+rx` permissions for cross-user access

### Perplexity Issues

**Problem**: Rate limit errors
**Solution**: Add delay between requests, check API quota

**Problem**: Poor quality responses
**Solution**: Use `perplexity_generate_prompt` tool to optimize query structure

### Z.AI Issues

**Problem**: Connection refused
**Solution**: Check supervisord status (`sudo supervisorctl status claude-zai`)

**Problem**: Timeout errors
**Solution**: Increase timeout in request body, check worker pool size

### RAGFlow Issues

**Problem**: Cannot connect to RAGFlow
**Solution**: Ensure containers are in same Docker network (`docker_ragflow`)

**Problem**: Document processing slow
**Solution**: Check container resources, increase memory allocation

---

## Security

### Credential Isolation

- Each AI service user has separate credentials
- API keys never shared between users
- MCP skills use sudo bridge for cross-user access
- Z.AI service is internal-only (port 9600 not exposed)

### Best Practices

1. **Never commit API keys** - Use environment variables
2. **Rotate keys regularly** - Especially for production
3. **Monitor usage** - Set up billing alerts
4. **User isolation** - Keep users separate with `sudo -u`
5. **Internal services** - Z.AI and RAGFlow not exposed to internet

---

## Future Enhancements

### Planned

1. **DeepSeek MCP Server Auto-Start** - Add to supervisord for automatic launch
2. **Perplexity Caching** - Cache common queries to reduce API calls
3. **RAGFlow Integration** - Deeper integration with main API
4. **Gemini Stabilization** - Move from experimental to active status
5. **OpenAI Activation** - Activate if specific models needed

### Under Consideration

1. **Local LLM Integration** - Ollama or LM Studio for fully offline operation
2. **Multi-Model Router** - Intelligent routing based on query complexity
3. **Cost Tracking Dashboard** - Real-time API cost monitoring
4. **Performance Benchmarks** - Automated quality and speed testing

---

## Related Documentation

### Core Documentation
- [Multi-Agent Skills Overview](/docs/guides/multi-agent-skills.md)
- [Agent Orchestration](/docs/guides/orchestrating-agents.md)
- [Configuration Guide](/docs/guides/configuration.md)

### AI Service Documentation
- [DeepSeek Verification](/docs/guides/features/deepseek-verification.md)
- [DeepSeek Deployment](/docs/guides/features/deepseek-deployment.md)
- [Perplexity Skill](/multi-agent-docker/skills/perplexity/SKILL.md)
- [DeepSeek Reasoning Skill](/multi-agent-docker/skills/deepseek-reasoning/SKILL.md)

### Infrastructure Documentation
- [Docker Environment Setup](/docs/guides/docker-environment-setup.md)
- [Multi-Agent Docker README](/multi-agent-docker/README.md)
- [Supervisord Configuration](/multi-agent-docker/unified-config/supervisord.unified.conf)

---

**Last Updated**: December 2, 2025
**Maintained By**: Development Team
**Status**: Living Document - Update as services evolve
