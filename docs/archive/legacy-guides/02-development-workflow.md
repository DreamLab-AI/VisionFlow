# Development Workflow Guide

*[← Back to Guides](index.md)*

This guide covers best practices for developing with VisionFlow, including environment setup, coding standards, testing strategies, and debugging techniques.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Strategies](#testing-strategies)
6. [Debugging Techniques](#debugging-techniques)
7. [Contributing Guidelines](#contributing-guidelines)
8. [Performance Optimization](#performance-optimisation)

## Development Environment Setup

### Prerequisites

Ensure you have the following tools installed:

```bash
# Required tools
docker --version          # 20.10+
docker compose version    # 2.0+
git --version            # 2.30+
node --version           # 18+
rust --version           # 1.70+
python --version         # 3.10+

# Optional but recommended
code --version           # VS Code
gh --version            # GitHub CLI
```

### Initial Setup

1. **Clone and Configure Repository**
```bash
# Clone with submodules
git clone --recursive https://github.com/your-org/VisionFlow.git
cd VisionFlow

# Set up Git hooks
./scripts/setup-git-hooks.sh

# Configure environment
cp .env.example .env.development
nano .env.development
```

2. **Install Development Dependencies**
```bash
# Frontend dependencies
cd client
npm install
npm run dev

# Backend dependencies
cd ../server
cargo build
cargo test

# Python tools
cd ../multi-agent-docker
pip install -r requirements-dev.txt
```

3. **VS Code Configuration**
```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "rust-analyser.cargo.features": ["all"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "typescript.tsdk": "client/node_modules/typescript/lib"
}
```

### Docker Development Environment

Use Docker Compose profiles for different development scenarios:

```bash
# Start core services only
docker-compose --profile core up -d

# Start with AI agents
docker-compose --profile agents up -d

# Start everything including GUI tools
docker-compose --profile full up -d

# Hot reload development
docker-compose --profile dev up
```

## Project Structure

### Directory Organization

```
VisionFlow/
├── client/                    # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/           # Custom React hooks
│   │   ├── services/        # API services
│   │   ├── store/          # State management
│   │   └── utils/          # Utility functions
│   └── public/             # Static assets
├── server/                   # Rust backend
│   ├── src/
│   │   ├── api/           # API endpoints
│   │   ├── models/        # Data models
│   │   ├── services/      # Business logic
│   │   └── utils/         # Utilities
│   └── tests/             # Integration tests
├── multi-agent-docker/      # Agent system
│   ├── core-assets/        # MCP tools
│   ├── gui-based-tools/    # GUI integrations
│   └── scripts/           # Helper scripts
├── docs/                   # Documentation
│   ├── guides/            # User guides
│   ├── reference/         # API reference
│   └── _archive/          # Archived docs
└── scripts/               # Development scripts
```

### Code Organization Best Practices

1. **Module Structure**
```rust
// server/src/services/agent_manager.rs
pub mod agent_manager {
    use crate::models::Agent;
    use crate::utils::Result;

    pub struct AgentManager {
        agents: Vec<Agent>,
    }

    impl AgentManager {
        pub fn new() -> Self {
            Self { agents: Vec::new() }
        }

        pub fn spawn_agent(&mut self, config: AgentConfig) -> Result<Agent> {
            // Implementation
        }
    }
}
```

2. **Component Structure**
```typescript
// client/src/components/AgentGraph/AgentGraph.tsx
import React, { useState, useEffect } from 'react';
import { useAgentStore } from '../../store/agentStore';
import { AgentNode } from './AgentNode';
import { GraphCanvas } from '../GraphCanvas';

interface AgentGraphProps {
  className?: string;
  onNodeClick?: (nodeId: string) => void;
}

export const AgentGraph: React.FC<AgentGraphProps> = ({
  className,
  onNodeClick
}) => {
  const agents = useAgentStore(state => state.agents);

  return (
    <GraphCanvas className={className}>
      {agents.map(agent => (
        <AgentNode
          key={agent.id}
          agent={agent}
          onClick={onNodeClick}
        />
      ))}
    </GraphCanvas>
  );
};
```

## Development Workflow

### Branch Strategy

Follow Git Flow for organized development:

```bash
# Feature development
git checkout -b feature/agent-telemetry
# ... make changes ...
git commit -m "feat: add agent telemetry visualization"
git push origin feature/agent-telemetry

# Bug fixes
git checkout -b fix/websocket-reconnect
# ... fix bug ...
git commit -m "fix: handle websocket reconnection properly"
git push origin fix/websocket-reconnect

# Hotfixes for production
git checkout -b hotfix/memory-leak
# ... apply fix ...
git commit -m "hotfix: fix memory leak in graph renderer"
```

### Commit Convention

Use conventional commits for clear history:

```
feat: add new feature
fix: bug fix
docs: documentation changes
style: formatting, missing semicolons, etc
refactor: code restructuring
perf: performance improvements
test: adding tests
chore: maintenance tasks
```

### Development Cycle

1. **Start New Feature**
```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature

# Start development services
docker-compose --profile dev up
```

2. **Develop Iteratively**
```bash
# Frontend development with hot reload
cd client
npm run dev

# Backend development with cargo watch
cd server
cargo watch -x run

# Test changes
npm test
cargo test
```

3. **Submit Pull Request**
```bash
# Run pre-commit checks
./scripts/pre-commit.sh

# Push changes
git push origin feature/your-feature

# Create PR with template
gh pr create --template .github/pull_request_template.md
```

## Coding Standards

### TypeScript/React Standards

1. **Use TypeScript Strictly**
```typescript
// tsconfig.json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true
  }
}
```

2. **Component Guidelines**
```typescript
// Use functional components with hooks
export const MyComponent: React.FC<Props> = ({ prop1, prop2 }) => {
  // Use custom hooks for logic
  const { data, loading } = useCustomHook();

  // Early returns for conditionals
  if (loading) return <Loading />;
  if (!data) return null;

  return <div>{/* Component JSX */}</div>;
};

// Export types alongside components
export interface MyComponentProps {
  prop1: string;
  prop2?: number;
}
```

### Rust Standards

1. **Follow Rust Conventions**
```rust
// Use descriptive names
pub struct AgentManager {
    active_agents: HashMap<AgentId, Agent>,
    task_queue: VecDeque<Task>,
}

// Prefer iterators over loops
let active_count = self.active_agents
    .values()
    .filter(|agent| agent.is_active())
    .count();

// Handle errors explicitly
pub fn process_task(&mut self, task: Task) -> Result<(), ProcessError> {
    let agent = self.active_agents
        .get_mut(&task.agent_id)
        .ok_or(ProcessError::AgentNotFound)?;

    agent.execute(task)?;
    Ok(())
}
```

2. **Error Handling**
```rust
// Define custom error types
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Agent not found: {0}")]
    NotFound(String),

    #[error("Task execution failed")]
    ExecutionFailed(#[from] ExecutionError),

    #[error("Network error")]
    Network(#[from] reqwest::Error),
}

// Use Result type alias
pub type Result<T> = std::result::Result<T, AgentError>;
```

### Python Standards (MCP Tools)

1. **Follow PEP 8**
```python
# mcp_tools/custom_tool.py
import json
import sys
from typing import Dict, Any, Optional

class MCPTool:
    """Base class for MCP tools."""

    def __init__(self, name: str):
        self.name = name

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an MCP request."""
        method = request.get('method')
        params = request.get('params', {})

        try:
            result = self._execute(method, params)
            return {'result': result}
        except Exception as e:
            return {'error': str(e)}

    def _execute(self, method: str, params: Dict[str, Any]) -> Any:
        """Execute the requested method."""
        raise NotImplementedError
```

## Testing Strategies

### Unit Testing

1. **Frontend Testing**
```typescript
// client/src/components/__tests__/AgentNode.test.tsx
import { render, fireEvent } from '@testing-library/react';
import { AgentNode } from '../AgentNode';

describe('AgentNode', () => {
  it('renders agent information', () => {
    const agent = { id: '1', name: 'Test Agent', status: 'active' };
    const { getByText } = render(<AgentNode agent={agent} />);

    expect(getByText('Test Agent')).toBeInTheDocument();
    expect(getByText('active')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const onClick = jest.fn();
    const agent = { id: '1', name: 'Test Agent' };
    const { container } = render(
      <AgentNode agent={agent} onClick={onClick} />
    );

    fireEvent.click(container.firstChild);
    expect(onClick).toHaveBeenCalledWith('1');
  });
});
```

2. **Backend Testing**
```rust
// server/src/services/tests.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_spawn() {
        let mut manager = AgentManager::new();
        let config = AgentConfig::default();

        let agent = manager.spawn_agent(config).await.unwrap();
        assert_eq!(agent.status, AgentStatus::Active);
        assert_eq!(manager.agent_count(), 1);
    }

    #[tokio::test]
    async fn test_concurrent_agents() {
        let manager = Arc::new(Mutex::new(AgentManager::new()));
        let mut handles = vec![];

        for i in 0..10 {
            let mgr = manager.clone();
            handles.push(tokio::spawn(async move {
                let mut mgr = mgr.lock().await;
                mgr.spawn_agent(AgentConfig::default()).await
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles).await;
        assert_eq!(results.len(), 10);
    }
}
```

### Integration Testing

1. **API Integration Tests**
```rust
// server/tests/api_integration.rs
use reqwest::StatusCode;

#[tokio::test]
async fn test_health_endpoint() {
    let app = spawn_app().await;

    let response = app.client
        .get(&format!("{}/health", app.address))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let health: HealthResponse = response.json().await.unwrap();
    assert_eq!(health.status, "healthy");
}
```

2. **E2E Testing**
```typescript
// e2e/agent-workflow.spec.ts
import { test, expect } from '@playwright/test';

test('complete agent workflow', async ({ page }) => {
  // Navigate to app
  await page.goto('http://localhost:3002');

  // Create new agent
  await page.click('button[aria-label="Create Agent"]');
  await page.fill('input[name="agentName"]', 'Test Agent');
  await page.click('button[type="submit"]');

  // Verify agent appears
  await expect(page.locator('text=Test Agent')).toBeVisible();

  // Execute task
  await page.click('text=Test Agent');
  await page.click('button[aria-label="Execute Task"]');

  // Verify task completion
  await expect(page.locator('text=Task completed')).toBeVisible();
});
```

### Performance Testing

```bash
# Load testing with k6
cat > load-test.js << 'EOF'
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '30s', target: 20 },
    { duration: '1m', target: 20 },
    { duration: '30s', target: 0 },
  ],
};

export default function() {
  let response = http.get('http://localhost:3001/api/agents');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });
}
EOF

k6 run load-test.js
```

## Debugging Techniques

### Frontend Debugging

1. **React DevTools**
```typescript
// Enable React DevTools profiler
if (process.env.NODE_ENV === 'development') {
  if (typeof window !== 'undefined') {
    window.__REACT_DEVTOOLS_GLOBAL_HOOK__.supportsFiber = true;
  }
}
```

2. **Debug Logging**
```typescript
// utils/debug.ts
const DEBUG = process.env.REACT_APP_DEBUG === 'true';

export const debugLog = (category: string, ...args: any[]) => {
  if (DEBUG) {
    console.log(`[${category}]`, ...args);
  }
};

// Usage
debugLog('AgentStore', 'Updating agent:', agentId, updates);
```

### Backend Debugging

1. **Rust Debugging**
```rust
// Enable debug logging
env_logger::init();

// Use debug macros
debug!("Processing agent request: {:?}", request);
info!("Agent {} spawned successfully", agent.id);
warn!("High memory usage detected: {}MB", memory_mb);
error!("Failed to connect to agent: {}", error);
```

2. **Remote Debugging**
```bash
# Debug in Docker container
docker-compose exec server bash
cargo build
RUST_LOG=debug cargo run

# Attach debugger
docker-compose exec server gdbserver :9999 target/debug/visionflow
# Connect from host
gdb target/debug/visionflow
(gdb) target remote localhost:9999
```

### MCP Tool Debugging

1. **Interactive Testing**
```bash
# Test MCP tool directly
echo '{"method": "test", "params": {}}' | python mcp_tools/my_tool.py

# Debug with logging
PYTHONUNBUFFERED=1 python -u mcp_tools/my_tool.py
```

2. **MCP Helper Debugging**
```bash
# List all tools
./mcp-helper.sh list-tools

# Test specific tool
./mcp-helper.sh test-tool imagemagick-mcp

# Debug mode
DEBUG=1 ./mcp-helper.sh run-tool blender-mcp '{"tool": "get_scene_info"}'
```

## Contributing Guidelines

### Before Contributing

1. **Check Existing Issues**
```bash
# Search for related issues
gh issue list --search "keyword"

# Check pull requests
gh pr list --search "keyword"
```

2. **Discuss Major Changes**
- Open an issue for discussion
- Join Discord for real-time chat
- Check roadmap for alignment

### Making Contributions

1. **Fork and Clone**
```bash
# Fork repository on GitHub
gh repo fork your-org/VisionFlow

# Clone your fork
git clone https://github.com/YOUR_USERNAME/VisionFlow.git
cd VisionFlow

# Add upstream remote
git remote add upstream https://github.com/your-org/VisionFlow.git
```

2. **Create Feature Branch**
```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create branch
git checkout -b feature/your-feature
```

3. **Make Changes**
- Follow coding standards
- Add tests for new features
- Update documentation
- Run linters and formatters

4. **Submit PR**
```bash
# Run all checks
npm run lint
cargo clippy
cargo test
npm test

# Commit with conventional message
git commit -m "feat: add agent telemetry dashboard"

# Push and create PR
git push origin feature/your-feature
gh pr create
```

### Code Review Process

1. **Automated Checks**
- CI/CD pipeline runs tests
- Code coverage requirements
- Linting and formatting

2. **Manual Review**
- Architecture alignment
- Performance impact
- Security considerations
- Documentation completeness

3. **Feedback Integration**
```bash
# Address review comments
git commit -m "address review: improve error handling"

# Update PR
git push origin feature/your-feature
```

## Performance Optimization

### Frontend Optimization

1. **React Performance**
```typescript
// Use React.memo for expensive components
export const ExpensiveComponent = React.memo(({ data }) => {
  return <ComplexVisualization data={data} />;
});

// Use useMemo for expensive calculations
const processedData = useMemo(() => {
  return heavyProcessing(rawData);
}, [rawData]);

// Use useCallback for stable references
const handleClick = useCallback((id: string) => {
  dispatch({ type: 'SELECT_NODE', payload: id });
}, [dispatch]);
```

2. **Bundle Optimization**
```javascript
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom'],
          'three': ['three', '@react-three/fiber'],
        }
      }
    }
  }
}
```

### Backend Optimization

1. **Async Performance**
```rust
// Use tokio for concurrent operations
use tokio::task::JoinSet;

pub async fn process_agents(agents: Vec<Agent>) -> Vec<Result<ProcessedAgent>> {
    let mut set = JoinSet::new();

    for agent in agents {
        set.spawn(async move {
            process_single_agent(agent).await
        });
    }

    let mut results = Vec::new();
    while let Some(result) = set.join_next().await {
        results.push(result?);
    }

    results
}
```

2. **Database Optimization**
```rust
// Use connection pooling
let pool = sqlx::postgres::PgPoolOptions::new()
    .max_connections(5)
    .connect(&database_url)
    .await?;

// Use prepared statements
let agent = sqlx::query_as!(
    Agent,
    "SELECT * FROM agents WHERE id = $1",
    agent_id
)
.fetch_one(&pool)
.await?;
```

## Development Tools

### Recommended Extensions

**VS Code Extensions:**
- Rust Analyzer
- ESLint
- Prettier
- Docker
- GitLens
- Thunder Client (API testing)

**Chrome Extensions:**
- React Developer Tools
- Redux DevTools
- Lighthouse

### Useful Scripts

```bash
# scripts/dev-setup.sh
#!/bin/bash
set -e

echo "Setting up development environment..."

# Install dependencies
npm install
cd server && cargo build
cd ../multi-agent-docker && pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Initialize database
docker-compose up -d postgres
sleep 5
diesel migration run

echo "Development environment ready!"
```

## Next Steps

- Continue to [Using the GUI Sandbox](03-using-the-gui-sandbox.md) for MCP tool usage
- See [Orchestrating Agents](04-orchestrating-agents.md) for agent management
- Check [Extending the System](05-extending-the-system.md) for customization

---

*[← Deployment](01-deployment.md) | [Back to Guides](index.md) | [Using the GUI Sandbox →](03-using-the-gui-sandbox.md)*