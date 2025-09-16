# External Integrations

Integration documentation for GitHub, OpenAI, RAGFlow, Claude Flow MCP, and other external services.

## Overview

VisionFlow integrates with multiple external services to provide comprehensive functionality:
- **GitHub** - Repository and issue management
- **Claude Flow MCP** - AI agent orchestration
- **RAGFlow** - Knowledge processing and analysis
- **Nostr** - Decentralized authentication
- **OpenAI API** - AI-powered features
- **Perplexity** - Enhanced search capabilities

## AI Agent Integration

### Claude Flow MCP
Direct TCP integration for AI agent orchestration.

#### Connection Details
- **Protocol**: JSON-RPC 2.0 over TCP
- **Port**: 9500
- **Connection**: Direct backend WebSocket
- **Encoding**: UTF-8 JSON

#### Features
- Multi-agent swarm coordination
- Real-time agent telemetry
- Task orchestration
- Performance monitoring

#### Integration Architecture
```rust
// MCP Integration
pub struct MCPIntegration {
    connection: TcpStream,
    agents: HashMap<String, AgentState>,
    swarms: Vec<SwarmConfig>,
}

impl MCPIntegration {
    pub async fn orchestrate_task(&mut self, task: TaskRequest) -> Result<TaskResponse> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": "task.orchestrate",
            "params": task,
            "id": generate_id()
        });

        self.send_message(message).await
    }
}
```

### OpenAI Integration
AI-powered analysis and generation features.

#### Capabilities
- Semantic analysis of graph data
- Natural language queries
- Content generation
- Code analysis and suggestions

#### Configuration
```toml
[ai.openai]
api_key = "${OPENAI_API_KEY}"
model = "gpt-4"
max_tokens = 4000
temperature = 0.7
```

### RAGFlow Integration
Knowledge processing and semantic analysis.

#### Features
- Document processing
- Semantic relationship extraction
- Knowledge graph generation
- Content enrichment

#### API Integration
```rust
pub struct RAGFlowClient {
    base_url: String,
    api_key: String,
    http_client: reqwest::Client,
}

impl RAGFlowClient {
    pub async fn process_documents(&self, docs: Vec<Document>) -> Result<ProcessedKnowledge> {
        let response = self.http_client
            .post(&format!("{}/api/v1/process", self.base_url))
            .json(&docs)
            .send()
            .await?;

        response.json().await
    }
}
```

## Version Control Integration

### GitHub Integration
Comprehensive GitHub API integration for repository management.

#### Features
- Repository analysis and metrics
- Issue tracking and management
- Pull request automation
- Webhook handling
- Release management

#### Authentication
```toml
[github]
token = "${GITHUB_TOKEN}"
webhook_secret = "${GITHUB_WEBHOOK_SECRET}"
```

#### Repository Analysis
```rust
pub struct GitHubAnalyzer {
    client: octocrab::Octocrab,
}

impl GitHubAnalyzer {
    pub async fn analyze_repository(&self, owner: &str, repo: &str) -> Result<RepoAnalysis> {
        let repo_data = self.client.repos(owner, repo).get().await?;
        let issues = self.client.issues(owner, repo).list().send().await?;
        let prs = self.client.pulls(owner, repo).list().send().await?;

        Ok(RepoAnalysis {
            repository: repo_data,
            issues: issues.items,
            pull_requests: prs.items,
            metrics: self.calculate_metrics(&issues, &prs).await?,
        })
    }
}
```

## Authentication Integration

### Nostr Protocol
Decentralized authentication using the Nostr protocol.

#### Features
- NIP-07 browser extension integration
- Cryptographic signature verification
- Decentralized identity
- Event-based authentication

#### Implementation
```rust
pub struct NostrAuth {
    relay_urls: Vec<String>,
    client: nostr_sdk::Client,
}

impl NostrAuth {
    pub async fn verify_signature(&self, event: &Event) -> Result<bool> {
        // Verify event signature
        let is_valid = event.verify()?;

        if is_valid {
            // Check against relay network
            self.verify_with_relays(event).await
        } else {
            Ok(false)
        }
    }
}
```

## Search and Discovery

### Perplexity Integration
Enhanced search capabilities with AI-powered insights.

#### Features
- Semantic search queries
- Real-time information retrieval
- Context-aware responses
- Multi-source aggregation

#### Configuration
```toml
[search.perplexity]
api_key = "${PERPLEXITY_API_KEY}"
model = "llama-3.1-sonar-large-128k-online"
max_tokens = 4000
```

## WebSocket Relay Services

### MCP Relay Architecture
Backend-only MCP integration with frontend relay.

```
Client ←→ WebSocket Relay ←→ Backend ←→ MCP TCP Connection
```

#### Relay Features
- Protocol translation (WebSocket ↔ TCP)
- Message routing and filtering
- Connection pooling
- Error handling and recovery

#### Implementation
```rust
pub struct MCPRelay {
    mcp_connection: TcpStream,
    websocket_connections: HashMap<Uuid, WebSocket>,
}

impl MCPRelay {
    pub async fn handle_websocket_message(&mut self,
        client_id: Uuid,
        message: WebSocketMessage
    ) -> Result<()> {
        match message {
            WebSocketMessage::MCPRequest(req) => {
                let response = self.forward_to_mcp(req).await?;
                self.send_to_client(client_id, response).await?;
            }
            _ => {
                // Handle other message types
            }
        }
        Ok(())
    }
}
```

## Configuration Management

### Environment Variables
```bash
# AI Services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
PERPLEXITY_API_KEY=pplx-...

# Version Control
GITHUB_TOKEN=ghp_...
GITHUB_WEBHOOK_SECRET=...

# Authentication
NOSTR_RELAY_URLS=wss://relay1.com,wss://relay2.com

# External Services
RAGFLOW_API_URL=https://api.ragflow.io
RAGFLOW_API_KEY=...
```

### Service Configuration
```toml
[integrations]
enabled_services = ["github", "claude_flow", "ragflow", "nostr"]

[integrations.github]
api_base_url = "https://api.github.com"
rate_limit = 5000
timeout = 30

[integrations.claude_flow]
tcp_host = "localhost"
tcp_port = 9500
connection_timeout = 10
retry_attempts = 3

[integrations.ragflow]
base_url = "https://api.ragflow.io"
timeout = 60
batch_size = 100
```

## Error Handling and Resilience

### Retry Logic
```rust
pub struct IntegrationManager {
    retry_config: RetryConfig,
}

impl IntegrationManager {
    pub async fn call_with_retry<F, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> BoxFuture<'_, Result<T, E>>,
        E: std::fmt::Debug,
    {
        let mut attempts = 0;
        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) if attempts < self.retry_config.max_attempts => {
                    attempts += 1;
                    let delay = Duration::from_millis(
                        self.retry_config.base_delay_ms * 2_u64.pow(attempts - 1)
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }
    }
}
```

### Circuit Breaker
```rust
pub struct CircuitBreaker {
    state: CircuitBreakerState,
    failure_count: AtomicU32,
    last_failure_time: AtomicU64,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub async fn call<F, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        match self.state {
            CircuitBreakerState::Open => Err(CircuitBreakerError::Open),
            CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen => {
                match operation.await {
                    Ok(result) => {
                        self.on_success();
                        Ok(result)
                    }
                    Err(e) => {
                        self.on_failure();
                        Err(CircuitBreakerError::CallFailed(e))
                    }
                }
            }
        }
    }
}
```

## Monitoring and Observability

### Integration Metrics
- Request/response latencies
- Success/failure rates
- Connection pool utilization
- Rate limit consumption

### Health Checks
```rust
pub struct HealthChecker {
    integrations: Vec<Box<dyn HealthCheckable>>,
}

#[async_trait]
pub trait HealthCheckable {
    async fn health_check(&self) -> HealthStatus;
}

impl HealthChecker {
    pub async fn check_all(&self) -> OverallHealth {
        let mut results = Vec::new();

        for integration in &self.integrations {
            let status = integration.health_check().await;
            results.push(status);
        }

        OverallHealth::from_individual(results)
    }
}
```

## Related Documentation

- [MCP Integration Architecture](../architecture/mcp-integration.md)
- [Claude Flow MCP Integration](../server/features/claude-flow-mcp-integration.md)
- [Authentication Guide](../security/authentication.md)
- [API Documentation](../api/README.md)

---

[← Back to Documentation](../README.md)