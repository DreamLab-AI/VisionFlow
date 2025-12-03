# Perplexity AI Integration

**Status**: Active
**Service**: Perplexity Sonar API
**MCP Skill**: `/multi-agent-docker/skills/perplexity/`

## Overview

Perplexity AI provides real-time web search and research capabilities with source citations. This integration allows Claude Code to access current information, perform market research, and generate comprehensive reports backed by verified web sources.

## Architecture

```
Claude Code
    ↓
MCP Skill (perplexity)
    ↓
Perplexity Sonar API
    ↓
Real-time Web Sources
```

## Available Models

| Model | Speed | Depth | Sources | Use Case |
|-------|-------|-------|---------|----------|
| **sonar** | Fast | Balanced | 5-10 | Quick factual lookups |
| **sonar-pro** | Medium | Deep | 10-15 | Comprehensive research |
| **sonar-reasoning** | Slower | Extended | 15+ | Complex analysis |

**Default**: `sonar` (balanced speed and quality)

## MCP Tools

### 1. perplexity_search

Quick factual search with citations.

**Parameters**:
- `query` (string, required): Search query
- `model` (string, optional): Model to use (default: sonar)
- `max_sources` (number, optional): Maximum sources to include (default: 5)

**Example**:
```javascript
perplexity_search({
  query: "current UK mortgage rates major banks",
  max_sources: 10
})
```

**Returns**:
```json
{
  "content": "As of December 2025, UK mortgage rates...",
  "sources": [
    {"title": "Bank of England", "url": "https://..."},
    {"title": "Financial Times", "url": "https://..."}
  ]
}
```

### 2. perplexity_research

Deep research analysis with multi-source synthesis.

**Parameters**:
- `topic` (string, required): Research topic
- `format` (string, optional): Output format (summary|detailed|executive)
- `depth` (string, optional): Research depth (quick|balanced|deep)
- `max_sources` (number, optional): Maximum sources (default: 10)

**Example**:
```javascript
perplexity_research({
  topic: "AI trends UK enterprise 2025",
  format: "executive summary",
  depth: "deep",
  max_sources: 15
})
```

**Returns**:
```json
{
  "summary": "Executive summary of findings...",
  "key_points": ["Point 1", "Point 2"],
  "sources": [...],
  "confidence": "high"
}
```

### 3. perplexity_generate_prompt

Optimize prompts for maximum research quality using five-element framework.

**Parameters**:
- `goal` (string, required): Research goal
- `context` (string, optional): Additional context
- `constraints` (string, optional): Constraints or requirements

**Example**:
```javascript
perplexity_generate_prompt({
  goal: "market research for renewable energy ETFs",
  context: "UK retail investor £10K budget",
  constraints: "focus on 2024-2025 performance"
})
```

**Returns**:
```json
{
  "optimized_prompt": "Comprehensive structured prompt...",
  "framework": {
    "instruction": "...",
    "context": "...",
    "input": "...",
    "keywords": ["renewable", "ETF", "UK"],
    "output_format": "..."
  }
}
```

## Integration Points

### Rust Service

**Location**: `src/services/perplexity_service.rs`

```rust
pub struct PerplexityService {
    client: Client,
    settings: Arc<RwLock<AppFullSettings>>,
}

// Main methods
pub async fn query(&self, query: &str) -> Result<PerplexityResponse>
pub async fn research(&self, topic: &str, depth: ResearchDepth) -> Result<ResearchReport>
```

### API Handler

**Location**: `src/handlers/perplexity_handler.rs`

**Endpoints**:
- `POST /api/perplexity/search` - Quick search
- `POST /api/perplexity/research` - Deep research

### Configuration

**Environment Variable**:
```bash
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxxxxxxx
```

**Service Configuration**:
```rust
// In src/services/perplexity_service.rs
const MARKDOWN_DIR: &str = "/app/data/markdown";
// Timeout: 30 seconds
// Default model: sonar
```

## Usage Examples

### From Claude Code

#### Quick Factual Search
```bash
# Natural language
"Use Perplexity to search for current UK mortgage rates"

# Direct tool call
perplexity_search "UK mortgage rates December 2025"
```

#### Deep Research
```bash
# Research report
perplexity_research \
  --topic "Enterprise AI adoption UK 2025" \
  --format "detailed" \
  --depth "deep"
```

#### Optimize Complex Query
```bash
# Generate optimized prompt
perplexity_generate_prompt \
  --goal "Compare cloud providers for UK startups" \
  --context "Under 50 employees, budget £500/month" \
  --constraints "Must have UK data centers"
```

### From Rust API

```bash
# Search endpoint
curl -X POST http://localhost:4000/api/perplexity/search \
  -H "Content-Type: application/json" \
  -d '{"query": "current AI trends", "max_sources": 10}'

# Research endpoint
curl -X POST http://localhost:4000/api/perplexity/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "cloud security best practices", "depth": "deep"}'
```

## Prompt Optimization Framework

Perplexity works best with structured prompts following this five-element framework:

### 1. Instruction
Clear, direct goal statement.
```
"Provide a comprehensive analysis of..."
```

### 2. Context
Background information and framing.
```
"For a UK retail investor with £10,000 to invest in 2025..."
```

### 3. Input
Specific constraints, requirements, or data.
```
"Focus on renewable energy ETFs with expense ratios under 0.5%..."
```

### 4. Keywords
Focus terms for search emphasis.
```
Keywords: renewable, ETF, UK, 2025, performance
```

### 5. Output Format
Desired structure of response.
```
"Format as executive summary with key findings, risks, and sources."
```

Use `perplexity_generate_prompt` to automatically apply this framework.

## Performance Characteristics

### Response Times

| Model | Typical Response | Max Response |
|-------|-----------------|--------------|
| sonar | 3-5 seconds | 10 seconds |
| sonar-pro | 5-8 seconds | 15 seconds |
| sonar-reasoning | 8-12 seconds | 20 seconds |

### Source Quality

- **Citation Rate**: 100% (all responses include sources)
- **Source Count**: 5-15 per response
- **Recency**: Real-time web data (updated continuously)
- **Quality**: Prioritizes authoritative sources (news, academic, government)

### Token Usage

- **Input**: 50-200 tokens (depends on query complexity)
- **Output**: 200-1000 tokens (depends on depth)
- **Cost**: Usage-based, see Perplexity pricing

## Best Practices

### 1. Query Optimization
- Be specific: "UK mortgage rates 2025" > "mortgage rates"
- Include context: Add location, timeframe, constraints
- Use keywords: Important terms help focus search

### 2. Model Selection
- **Quick facts**: Use `sonar` (fast, good enough)
- **Research reports**: Use `sonar-pro` (more sources)
- **Complex analysis**: Use `sonar-reasoning` (extended context)

### 3. Source Management
- Request more sources for controversial topics
- Verify sources for critical decisions
- Cross-reference with domain expertise

### 4. Rate Limiting
- Add delays between requests (avoid rapid-fire)
- Cache results when possible
- Use batch processing for multiple queries

## Use Cases

### Market Research
```javascript
perplexity_research({
  topic: "UK SaaS market 2025 - company formation trends",
  format: "detailed",
  depth: "deep",
  max_sources: 15
})
```
**Output**: Comprehensive market analysis with statistics, trends, and forecasts.

### Technical Documentation Lookup
```javascript
perplexity_search({
  query: "Rust async tokio best practices 2025",
  max_sources: 5
})
```
**Output**: Current best practices with code examples and source documentation.

### Competitive Analysis
```javascript
perplexity_research({
  topic: "Cloud IDE market - competitors to GitHub Codespaces",
  format: "executive summary"
})
```
**Output**: Feature comparison, pricing, and market positioning.

### Current Events
```javascript
perplexity_search({
  query: "UK tech policy changes December 2025"
})
```
**Output**: Recent news with government and media sources.

## Limitations

### API Constraints
- **Rate Limits**: Depends on API tier (check Perplexity account)
- **Context Window**: Limited by model (typically 4K-8K tokens)
- **Language**: Best results in English (other languages supported but variable)

### Content Limitations
- **Paywall Content**: May not access full text behind paywalls
- **Real-time Data**: Some data sources may have slight delays
- **Source Availability**: Quality depends on available web sources

### Technical Limitations
- **No Streaming**: Responses are complete (not streamed)
- **Timeout**: 30-second timeout in service configuration
- **Concurrency**: Limited by API rate limits

## Troubleshooting

### Rate Limit Errors

**Symptom**: HTTP 429 responses
**Solution**:
1. Add delay between requests (500ms-1s)
2. Check API quota in Perplexity dashboard
3. Consider upgrading API tier

### Poor Quality Responses

**Symptom**: Generic or off-topic results
**Solution**:
1. Use `perplexity_generate_prompt` to optimize query
2. Add specific context and constraints
3. Include keywords and timeframe
4. Try `sonar-pro` for deeper research

### Timeout Errors

**Symptom**: Request times out
**Solution**:
1. Simplify query (reduce complexity)
2. Use faster model (`sonar` instead of `sonar-reasoning`)
3. Increase timeout in service config (if needed)

### Missing Sources

**Symptom**: Few or no sources returned
**Solution**:
1. Increase `max_sources` parameter
2. Broaden query (too specific may have limited sources)
3. Check if topic is too recent (sources may not exist yet)

## Security

### API Key Management
- Stored in environment variable (`PERPLEXITY_API_KEY`)
- Not logged or exposed in responses
- User isolation via devuser account

### Data Privacy
- Queries sent to Perplexity API (external service)
- Responses cached in `/app/data/markdown` (optional)
- No persistent storage of API keys in code

### Best Practices
1. **Rotate keys regularly** - Update API key periodically
2. **Monitor usage** - Set up billing alerts in Perplexity dashboard
3. **Sanitize inputs** - Avoid sending sensitive data in queries
4. **Review sources** - Verify source credibility for critical decisions

## Integration with Other Services

### With RAGFlow
```javascript
// 1. Research with Perplexity
const research = await perplexity_research({
  topic: "AI safety best practices 2025"
});

// 2. Store in RAGFlow knowledge base
await ragflow.ingest({
  content: research.summary,
  sources: research.sources,
  metadata: {type: "research", date: "2025-12-02"}
});

// 3. Future queries use cached knowledge
```

### With DeepSeek Reasoning
```javascript
// 1. Get research data from Perplexity
const data = await perplexity_search("quantum computing breakthroughs 2025");

// 2. Deep analysis with DeepSeek
const analysis = await deepseek_analyze({
  code: data.content,
  issue: "Evaluate feasibility for commercial applications"
});

// Result: Research + deep reasoning
```

## Monitoring

### Metrics to Track
- **Request count**: Total API calls
- **Response time**: Average latency
- **Error rate**: Failed requests
- **Token usage**: API consumption
- **Cost**: Monthly spending

### Logging
```rust
// Service logs to standard output
info!("Perplexity query: {}", query);
info!("Response time: {}ms", duration);
error!("Perplexity API error: {}", error);
```

Check logs:
```bash
# Service logs
docker logs agentic-workstation | grep perplexity

# Rust logs
tail -f /var/log/turbo-flow.log | grep perplexity
```

## Future Enhancements

### Planned
1. **Response Caching** - Cache common queries to reduce API calls
2. **Batch Processing** - Process multiple queries in parallel
3. **Source Filtering** - Filter sources by domain, date, credibility
4. **Cost Tracking** - Real-time API cost monitoring

### Under Consideration
1. **Local Caching Layer** - Redis cache for frequently accessed data
2. **Source Quality Scoring** - Automatic evaluation of source credibility
3. **Multi-language Support** - Optimized prompts for non-English queries
4. **Streaming Responses** - Real-time response generation

## Related Documentation

- [AI Models Overview](/docs/guides/ai-models/README.md)
- [Perplexity MCP Skill](/multi-agent-docker/skills/perplexity/SKILL.md)
- [Perplexity Templates](/multi-agent-docker/skills/perplexity/docs/templates.md)
- [API Complete Reference](/docs/reference/api-complete-reference.md)

---

**Last Updated**: December 2, 2025
**Status**: Active Production Service
