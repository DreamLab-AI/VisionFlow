# Development Workflow

[← Knowledge Base](../index.md) > [Guides](./index.md) > Development Workflow

This guide provides comprehensive best practices for developing with VisionFlow, covering git workflow, branch strategies, code review processes, manual testing procedures, and contribution guidelines.

## Table of Contents

1. [Overview](#overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Git Workflow](#git-workflow)
4. [Branch Strategy](#branch-strategy)
5. [Development Cycle](#development-cycle)
6. [Coding Standards](#coding-standards)
7. [Testing Procedures](#testing-procedures)
8. [Code Review Process](#code-review-process)
9. [Contributing Guidelines](#contributing-guidelines)
10. [Performance Optimisation](#performance-optimisation)

## Overview

VisionFlow follows a structured development workflow designed to maintain code quality, architectural integrity, and system stability. This workflow balances rapid iteration with careful validation through manual testing procedures.

### Key Principles

- **Incremental Development**: Build features in small, testable increments
- **Manual Validation**: Comprehensive manual testing replaces automated testing (see [Testing Procedures](#testing-procedures))
- **Architecture Alignment**: All changes must align with documented architectural decisions (see [ADRs](../concepts/decisions/))
- **Documentation First**: Update documentation alongside code changes
- **Review Culture**: All changes undergo peer review before merging

### Related Documentation

- [Contributing Guide](../contributing.md) - Quick start for contributors
- [Extending the System](./extending-the-system.md) - Creating custom extensions
- [ADR-001: Unified API Client](../concepts/decisions/adr-001-unified-api-client.md) - API architecture
- [ADR-003: Code Pruning](../concepts/decisions/adr-003-code-pruning-2025-10.md) - Codebase maintenance

## Development Environment Setup

### Prerequisites

Ensure the following tools are installed:

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

1. **Clone Repository**
```bash
# Clone with submodules
git clone --recursive https://github.com/your-org/VisionFlow.git
cd VisionFlow

# Verify repository structure
ls -la
```

2. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env.development

# Configure for local development
nano .env.development

# Key settings to configure:
# - Database connection strings
# - API endpoints
# - Feature flags
# - Debug logging levels
```

3. **Install Dependencies**
```bash
# Frontend dependencies
cd client
npm install

# Verify installation
npm run build

# Backend dependencies
cd ../server
cargo build

# Verify compilation
cargo check

# Python tools (MCP)
cd ../multi-agent-docker
pip install -r requirements-dev.txt
```

4. **Start Development Services**
```bash
# Return to project root
cd ..

# Start core services
docker compose --profile core up -d

# Verify services are running
docker compose ps

# Check service health
curl http://localhost:3030/health
```

### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "rust-analyzer.cargo.features": ["all"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "typescript.tsdk": "client/node_modules/typescript/lib",
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  },
  "files.watcherExclude": {
    "**/target/**": true,
    "**/node_modules/**": true,
    "**/dist/**": true
  }
}
```

### Recommended VS Code Extensions

- **Rust Analyser** - Rust language support
- **ESLint** - TypeScript/JavaScript linting
- **Prettier** - Code formatting
- **Docker** - Container management
- **GitLens** - Enhanced Git integration
- **Thunder Client** - API testing

## Git Workflow

VisionFlow uses a modified Git Flow workflow optimised for continuous integration and feature development.

### Branch Types

| Branch Type | Purpose | Naming Convention | Base Branch |
|-------------|---------|-------------------|-------------|
| `main` | Production-ready code | `main` | N/A |
| Feature | New features | `feature/description` | `main` |
| Fix | Bug fixes | `fix/description` | `main` |
| Hotfix | Urgent production fixes | `hotfix/description` | `main` |
| Refactor | Code improvements | `refactor/description` | `main` |
| Docs | Documentation updates | `docs/description` | `main` |
| Chore | Maintenance tasks | `chore/description` | `main` |

### Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) for clear, semantic commit history:

```bash
# Format
<type>(<scope>): <subject>

# Types
feat:      # New feature
fix:       # Bug fix
docs:      # Documentation changes
style:     # Formatting, missing semicolons, etc
refactor:  # Code restructuring without behaviour change
perf:      # Performance improvements
test:      # Adding or updating tests
chore:     # Maintenance tasks (deps, build, etc)

# Examples
feat(graph): add force-directed layout algorithm
fix(api): handle websocket reconnection properly
docs(guides): update development workflow guide
refactor(client): consolidate API client layers
perf(rendering): optimise node batching
chore(deps): update three.js to v0.159.0
```

### Commit Message Guidelines

1. **Subject Line** (first line):
   - Maximum 72 characters
   - Imperative mood ("add" not "added" or "adds")
   - No full stop at end
   - Capitalise first word after type

2. **Body** (optional):
   - Separate from subject with blank line
   - Wrap at 72 characters
   - Explain *what* and *why*, not *how*

3. **Footer** (optional):
   - Reference issues: `Closes #123`, `Related to #456`
   - Note breaking changes: `BREAKING CHANGE: description`

Example commit message:

```
feat(agents): add data pipeline agent for ETL operations

Implement BaseCustomAgent subclass for data extraction, transformation,
and loading (ETL) operations. Supports multiple data sources including
databases, files, and APIs with configurable transformation rules.

Key features:
- Async data processing with pandas
- Configurable pipeline stages
- Built-in validation and error handling
- Metrics tracking for pipeline performance

Related to #234
```

### Git Hooks

VisionFlow uses git hooks to maintain code quality. A `prepare-commit-msg` hook is already configured:

```bash
# Verify hooks
ls -la .git/hooks/

# The prepare-commit-msg hook helps format commit messages
```

## Branch Strategy

### Creating Feature Branches

```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/agent-telemetry

# Alternative: use GitHub CLI
gh repo fork  # If contributing from fork
gh pr create --draft  # Create draft PR early
```

### Branch Naming Best Practices

```bash
# Good branch names
feature/multi-agent-coordination
fix/websocket-reconnection-timeout
refactor/api-client-consolidation
docs/development-workflow
chore/upgrade-dependencies

# Avoid
feature/new-stuff
fix/bug
update
johns-branch
```

### Working with Branches

```bash
# List branches
git branch -a

# Switch branches
git checkout feature/my-feature

# Create and switch in one command
git checkout -b feature/new-feature

# Delete local branch
git branch -d feature/old-feature

# Delete remote branch
git push origin --delete feature/old-feature

# Sync with main frequently
git checkout main
git pull origin main
git checkout feature/my-feature
git merge main  # or git rebase main
```

### Branch Protection Rules

The `main` branch has protection rules:

- ✅ Require pull request reviews (minimum 1 approval)
- ✅ Require status checks to pass
- ✅ Require branches to be up to date before merging
- ✅ Include administrators in restrictions
- ❌ No direct pushes to main

## Development Cycle

### 1. Plan Your Work

```bash
# Check existing issues
gh issue list --search "label:feature"

# Create new issue if needed
gh issue create --title "Add agent telemetry dashboard" \
                --body "Implement real-time telemetry dashboard for agent monitoring"

# Assign to yourself
gh issue edit 123 --add-assignee @me
```

### 2. Create Feature Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/agent-telemetry-dashboard

# Push branch to remote (enables collaboration)
git push -u origin feature/agent-telemetry-dashboard
```

### 3. Iterative Development

```bash
# Start development services
cd client
npm run dev  # Frontend with hot reload

# In separate terminal
cd server
cargo watch -x run  # Backend with auto-restart

# In separate terminal
docker compose --profile dev up  # Supporting services
```

**Development Loop**:

1. Make changes to code
2. Verify compilation/transpilation
3. Perform manual testing (see [Testing Procedures](#testing-procedures))
4. Commit changes with conventional commit message
5. Push to remote branch regularly

```bash
# Commit workflow
git add src/features/telemetry/
git commit -m "feat(telemetry): add agent metrics collection"

# Push changes
git push origin feature/agent-telemetry-dashboard
```

### 4. Sync with Main

```bash
# Regularly sync with main to avoid conflicts
git checkout main
git pull origin main
git checkout feature/agent-telemetry-dashboard
git merge main

# Or use rebase for cleaner history
git rebase main

# Resolve conflicts if any
git status
# Edit conflicting files
git add .
git rebase --continue
```

### 5. Prepare for Review

Before creating a pull request:

```bash
# Verify build succeeds
cd client && npm run build
cd ../server && cargo build --release

# Run linters
cd client && npm run lint
cd ../server && cargo clippy -- -D warnings

# Format code
cd client && npm run format
cd ../server && cargo fmt

# Perform manual testing checklist (see Testing Procedures section)

# Update documentation
# - Update relevant docs in docs/
# - Update CHANGELOG if significant change
# - Add comments for complex logic
```

### 6. Create Pull Request

```bash
# Using GitHub CLI (recommended)
gh pr create \
  --title "feat(telemetry): add agent telemetry dashboard" \
  --body "$(cat <<'EOF'
## Summary

Implements real-time telemetry dashboard for monitoring agent performance.

## Changes

- Add `AgentTelemetry` component for metrics visualisation
- Implement WebSocket subscription for live metrics
- Add metrics aggregation in backend
- Update API with `/telemetry/agents` endpoint

## Testing

- ✅ Manual testing: Dashboard displays metrics correctly
- ✅ WebSocket connection maintains state
- ✅ Metrics update in real-time (< 100ms latency)
- ✅ Tested with 10+ concurrent agents
- ✅ CPU usage remains under 5%

## Documentation

- Updated `docs/guides/orchestrating-agents.md`
- Added telemetry section to architecture docs

## Related Issues

Closes #123

## Screenshots

(attach screenshots if UI changes)
EOF
)" \
  --label "feature" \
  --assignee @me

# Or via web interface
gh pr view --web
```

### 7. Address Review Feedback

```bash
# Make requested changes
git add .
git commit -m "refactor(telemetry): address review feedback on error handling"

# Push updates
git push origin feature/agent-telemetry-dashboard

# PR automatically updates with new commits

# Respond to review comments on GitHub
gh pr view --web
```

### 8. Merge Pull Request

Once approved:

```bash
# Update branch with latest main
git checkout main
git pull origin main
git checkout feature/agent-telemetry-dashboard
git merge main

# Push final update
git push origin feature/agent-telemetry-dashboard

# Merge via GitHub CLI or web interface
gh pr merge --squash --delete-branch
# or
gh pr merge --merge --delete-branch
# or
gh pr merge --rebase --delete-branch

# Update local main
git checkout main
git pull origin main

# Delete local feature branch
git branch -d feature/agent-telemetry-dashboard
```

## Coding Standards

### TypeScript/React Standards

**Use TypeScript Strictly**:

```typescript
// tsconfig.json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true
  }
}
```

**Component Guidelines**:

```typescript
// ✅ Good: Functional component with proper typing
interface AgentNodeProps {
  agent: Agent;
  onSelect?: (agentId: string) => void;
  className?: string;
}

export const AgentNode: React.FC<AgentNodeProps> = ({
  agent,
  onSelect,
  className
}) => {
  const { status, metrics } = useAgentStatus(agent.id);

  // Early returns for loading/error states
  if (!status) return <LoadingSpinner />;
  if (status.error) return <ErrorDisplay error={status.error} />;

  return (
    <div className={cn("agent-node", className)} onClick={() => onSelect?.(agent.id)}>
      <AgentStatus status={status} />
      <AgentMetrics metrics={metrics} />
    </div>
  );
};

// ❌ Bad: Implicit any, no proper typing
export const AgentNode = ({ agent, onSelect }) => {
  // ...
};
```

**Custom Hooks Pattern**:

```typescript
// ✅ Good: Typed custom hook with proper dependencies
export function useAgentStatus(agentId: string) {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const subscription = agentService.subscribeToStatus(agentId, {
      onUpdate: setStatus,
      onError: setError
    });

    return () => subscription.unsubscribe();
  }, [agentId]);

  return { status, error, isLoading: !status && !error };
}
```

**State Management**:

```typescript
// Use Zustand for global state
import { create } from 'zustand';

interface AgentStore {
  agents: Map<string, Agent>;
  selectedAgentId: string | null;

  addAgent: (agent: Agent) => void;
  removeAgent: (agentId: string) => void;
  selectAgent: (agentId: string | null) => void;
}

export const useAgentStore = create<AgentStore>((set) => ({
  agents: new Map(),
  selectedAgentId: null,

  addAgent: (agent) => set((state) => ({
    agents: new Map(state.agents).set(agent.id, agent)
  })),

  removeAgent: (agentId) => set((state) => {
    const agents = new Map(state.agents);
    agents.delete(agentId);
    return { agents };
  }),

  selectAgent: (agentId) => set({ selectedAgentId: agentId })
}));
```

### Rust Standards

**Follow Rust Conventions**:

```rust
// ✅ Good: Idiomatic Rust with proper error handling
use thiserror::Error;
use std::collections::HashMap;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Agent not found: {0}")]
    NotFound(String),

    #[error("Agent operation failed: {0}")]
    OperationFailed(String),

    #[error("Network error")]
    Network(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;

pub struct AgentManager {
    agents: HashMap<String, Agent>,
    max_agents: usize,
}

impl AgentManager {
    pub fn new(max_agents: usize) -> Self {
        Self {
            agents: HashMap::new(),
            max_agents,
        }
    }

    pub fn spawn_agent(&mut self, config: AgentConfig) -> Result<Agent> {
        if self.agents.len() >= self.max_agents {
            return Err(AgentError::OperationFailed(
                format!("Maximum agents ({}) reached", self.max_agents)
            ));
        }

        let agent = Agent::new(config);
        self.agents.insert(agent.id.clone(), agent.clone());

        Ok(agent)
    }

    pub fn get_agent(&self, agent_id: &str) -> Result<&Agent> {
        self.agents
            .get(agent_id)
            .ok_or_else(|| AgentError::NotFound(agent_id.to_string()))
    }
}

// ❌ Bad: Using unwrap, no error handling
impl AgentManager {
    pub fn get_agent(&self, agent_id: &str) -> &Agent {
        self.agents.get(agent_id).unwrap() // Panic on error!
    }
}
```

**Async Rust Patterns**:

```rust
use tokio::task::JoinSet;

pub async fn process_agents_concurrently(
    agents: Vec<Agent>
) -> Vec<Result<ProcessedAgent>> {
    let mut set = JoinSet::new();

    for agent in agents {
        set.spawn(async move {
            process_single_agent(agent).await
        });
    }

    let mut results = Vec::new();
    while let Some(result) = set.join_next().await {
        match result {
            Ok(processed) => results.push(processed),
            Err(e) => results.push(Err(AgentError::OperationFailed(e.to_string()))),
        }
    }

    results
}
```

### Python Standards (MCP Tools)

**Follow PEP 8 and Type Hints**:

```python
#!/usr/bin/env python3
"""
Custom MCP Tool for data processing.

This module implements an MCP tool that provides data transformation
and analysis capabilities for the VisionFlow system.
"""
import json
import logging
import sys
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DataProcessingTool:
    """MCP tool for data processing operations."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise data processing tool.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if not config_path:
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}")
            return {}

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process MCP request.

        Args:
            request: MCP request with method and params

        Returns:
            Response dictionary with result or error
        """
        try:
            method = request.get('method', 'default')
            params = request.get('params', {})

            handler = self._get_handler(method)
            result = handler(params)

            return {'result': result}

        except Exception as e:
            self.logger.error(f"Request processing failed: {e}", exc_info=True)
            return {'error': str(e)}

    def _get_handler(self, method: str):
        """Get handler function for method."""
        handlers = {
            'transform': self._handle_transform,
            'analyse': self._handle_analyse,
            'validate': self._handle_validate,
        }
        return handlers.get(method, self._handle_unknown)

    def _handle_transform(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data transformation requests."""
        data = params.get('data', [])
        operations = params.get('operations', [])

        transformed = data
        for operation in operations:
            transformed = self._apply_operation(transformed, operation)

        return {
            'transformed': transformed,
            'operations_applied': len(operations)
        }

    def _apply_operation(
        self,
        data: List[Any],
        operation: Dict[str, Any]
    ) -> List[Any]:
        """Apply single transformation operation."""
        op_type = operation.get('type')

        if op_type == 'filter':
            return [item for item in data if self._matches_filter(item, operation)]
        elif op_type == 'map':
            return [self._transform_item(item, operation) for item in data]
        else:
            raise ValueError(f"Unknown operation: {op_type}")


def main():
    """Main entry point for MCP tool."""
    tool = DataProcessingTool()

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = tool.process_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError as e:
            error = {'error': f'Invalid JSON: {e}'}
            print(json.dumps(error), flush=True)


if __name__ == '__main__':
    main()
```

### API Architecture

VisionFlow uses a **two-layer API architecture** (see [ADR-001](../concepts/decisions/adr-001-unified-api-client.md)):

**Layer 1: UnifiedApiClient** (Transport Layer)
```typescript
// Low-level HTTP client with retry, auth, interceptors
const user = await unifiedApiClient.getData<User>('/users/123');
```

**Layer 2: Domain APIs** (Business Logic Layer)
```typescript
// High-level APIs with debouncing, batching, domain logic
import { settingsApi } from '@/api/settingsApi';

// Automatic debouncing, priority handling, batching
await settingsApi.updateSetting('physics.gravity', 9.81);
```

**Never bypass this architecture**:
```typescript
// ❌ Bad: Direct fetch bypasses architecture
const response = await fetch('/api/users/123');

// ❌ Bad: Using old apiService (deprecated)
const response = await apiService.get('/api/users/123');

// ✅ Good: Use domain API when available
const user = await userApi.getUser('123');

// ✅ Good: Use UnifiedApiClient for new endpoints
const data = await unifiedApiClient.getData<CustomType>('/custom/endpoint');
```

## Testing Procedures

**Important**: VisionFlow has disabled automated testing due to supply chain security concerns (see [ADR-003](../concepts/decisions/adr-003-code-pruning-2025-10.md)). All testing is performed manually.

### Manual Testing Checklist

**Before Creating Pull Request**:

#### 1. Build Verification
```bash
# Frontend build
cd client
npm run build
# ✅ Verify: No compilation errors
# ✅ Verify: No TypeScript errors
# ✅ Verify: Bundle size reasonable

# Backend build
cd ../server
cargo build --release
# ✅ Verify: No compilation errors
# ✅ Verify: No clippy warnings (cargo clippy)

# Format check
cd ../client && npm run format
cd ../server && cargo fmt --check
# ✅ Verify: No formatting issues
```

#### 2. Functional Testing

**Core Functionality**:
- ✅ Application starts without errors
- ✅ Main UI loads and renders correctly
- ✅ Navigation works between panels
- ✅ Data loads from backend

**Feature-Specific**:
- ✅ New feature works as intended
- ✅ Edge cases handled (empty data, max limits, etc.)
- ✅ Error states display correctly
- ✅ Loading states display correctly

**Regression Testing**:
- ✅ Existing features still work
- ✅ No visual regressions
- ✅ No performance degradation

#### 3. Integration Testing

```bash
# Start all services
docker compose --profile full up -d

# Test points
- ✅ Frontend connects to backend
- ✅ WebSocket connections establish
- ✅ Database queries execute
- ✅ Agent communication works
- ✅ MCP tools respond
- ✅ External integrations functional
```

#### 4. Performance Testing

```bash
# Monitor resource usage
docker stats

# Check:
- ✅ CPU usage remains reasonable (< 50% idle, < 90% under load)
- ✅ Memory usage stable (no leaks)
- ✅ Response times acceptable (< 200ms for API calls)
- ✅ UI remains responsive (60 FPS target)
```

#### 5. Browser Testing

Test in multiple browsers:
- ✅ Chrome/Chromium (primary target)
- ✅ Firefox (secondary target)
- ✅ Safari (if available)
- ✅ Edge (Chromium-based)

Check:
- ✅ UI renders correctly
- ✅ No console errors
- ✅ WebSocket connections work
- ✅ Local storage/IndexedDB works

#### 6. Cross-Browser Console Check

```javascript
// Open browser console (F12) and verify:
// ✅ No red errors
// ✅ No unhandled promise rejections
// ✅ Warnings are expected and documented

// Optional: Monitor network requests
// ✅ No failed requests (except expected 404s)
// ✅ Reasonable request sizes
// ✅ Proper caching headers
```

### Testing Environments

**Development**:
```bash
docker compose --profile dev up
# - Hot reload enabled
# - Debug logging enabled
# - Development databases
```

**Staging**:
```bash
docker compose --profile staging up
# - Production build
# - Staging databases
# - Closer to production config
```

**Local Production Simulation**:
```bash
docker compose --profile prod up
# - Production builds
# - Optimised settings
# - Minimal logging
```

### Load Testing

For performance-critical changes:

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:3030/api/agents

# Using curl in loop
for i in {1..100}; do
  curl http://localhost:3030/api/health
done

# Monitor with docker stats during load
docker stats --no-stream
```

### Manual Test Report Template

Include in PR description:

```markdown
## Manual Testing Report

### Environment
- OS: Linux/macOS/Windows
- Browser: Chrome 120
- Docker: 24.0.7
- Node: 18.19.0

### Build Verification
- [x] Client builds without errors
- [x] Server builds without errors
- [x] No linting issues
- [x] Code formatted correctly

### Functional Testing
- [x] Feature works as intended
- [x] Edge cases handled:
  - Empty data: ✅ Shows empty state
  - Max limits: ✅ Shows appropriate message
  - Invalid input: ✅ Validation errors displayed
- [x] Error handling verified

### Integration Testing
- [x] API endpoints respond correctly
- [x] WebSocket connection maintains state
- [x] Database operations succeed
- [x] No console errors

### Performance Testing
- [x] Response times < 200ms
- [x] Memory usage stable
- [x] CPU usage acceptable
- [x] No memory leaks detected

### Browser Testing
- [x] Chrome: Works correctly
- [x] Firefox: Works correctly
- [ ] Safari: Not tested (no access)

### Regression Testing
- [x] Existing features work
- [x] No visual regressions
- [x] Settings persist correctly
- [x] Agent operations functional
```

## Code Review Process

### Review Checklist

**For Reviewers**:

#### Architecture Alignment
- ✅ Changes align with system architecture
- ✅ Follows established patterns (see [ADRs](../concepts/decisions/))
- ✅ API layer usage correct (UnifiedApiClient + domain APIs)
- ✅ No architectural anti-patterns

#### Code Quality
- ✅ Code is readable and well-structured
- ✅ Naming is clear and consistent
- ✅ Comments explain "why" not "what"
- ✅ No unnecessary complexity
- ✅ Error handling is comprehensive
- ✅ Type safety maintained (TypeScript/Rust)

#### Testing Coverage
- ✅ Manual testing checklist completed
- ✅ Test report in PR description
- ✅ Edge cases covered
- ✅ Regression testing performed

#### Documentation
- ✅ Code changes documented
- ✅ API changes documented
- ✅ User-facing changes documented
- ✅ ADR created if architectural decision
- ✅ CHANGELOG updated if needed

#### Performance
- ✅ No obvious performance issues
- ✅ Database queries optimised
- ✅ No unnecessary re-renders (React)
- ✅ Async operations efficient
- ✅ Resource usage acceptable

#### Security
- ✅ Input validation present
- ✅ No SQL injection vulnerabilities
- ✅ Authentication/authorisation correct
- ✅ No sensitive data in logs
- ✅ Dependencies secure

### Review Process

1. **Automated Checks**
   - ✅ Build succeeds
   - ✅ Linting passes
   - ✅ Format check passes

2. **Manual Review**
   - Read through code changes
   - Check against review checklist
   - Test changes locally if significant
   - Verify documentation updates

3. **Provide Feedback**
   ```markdown
   ## Review Comments

   ### Must Fix (Blocking)
   - [ ] API layer: Should use settingsApi instead of direct UnifiedApiClient call
   - [ ] Error handling: Missing validation for null agent ID

   ### Suggestions (Non-blocking)
   - Consider extracting this logic into a custom hook
   - Could simplify this with Array.prototype.flatMap

   ### Questions
   - Why was this approach chosen over using the existing utility?
   - Have you tested this with a large dataset (1000+ nodes)?

   ### Praise
   - Excellent documentation in the comments
   - Great test coverage in the manual testing report
   ```

4. **Request Changes or Approve**
   ```bash
   # Approve if ready
   gh pr review --approve

   # Request changes if issues
   gh pr review --request-changes --body "See inline comments"

   # Comment without blocking
   gh pr review --comment --body "Minor suggestions, but LGTM overall"
   ```

### For Pull Request Authors

**Responding to Feedback**:

```bash
# Make requested changes
git add .
git commit -m "refactor: address review feedback on error handling"
git push origin feature/my-feature

# Respond to comments
# - Click "Resolve conversation" when addressed
# - Reply to explain decisions
# - Ask for clarification if unclear
```

**Handling Disagreements**:

- Discuss technical merits, not preferences
- Reference documentation (ADRs, architecture docs)
- Bring in third-party reviewer if needed
- Escalate to architecture review if necessary

## Contributing Guidelines

### Before Contributing

1. **Review Documentation**
   - Read [Contributing Guide](../contributing.md)
   - Review [Architecture Documentation](../concepts/architecture/)
   - Check [ADRs](../concepts/decisions/) for relevant decisions

2. **Check Existing Work**
   ```bash
   # Search for related issues
   gh issue list --search "keyword"

   # Check open pull requests
   gh pr list --search "keyword"
   ```

3. **Discuss Major Changes**
   - Open an issue for discussion
   - Propose architectural changes in issue first
   - Get consensus before implementation

### Contribution Process

1. **Fork and Clone** (external contributors)
   ```bash
   # Fork on GitHub
   gh repo fork your-org/VisionFlow --clone

   # Add upstream remote
   cd VisionFlow
   git remote add upstream https://github.com/your-org/VisionFlow.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/my-contribution
   ```

3. **Develop Feature**
   - Follow coding standards
   - Add documentation
   - Perform manual testing

4. **Submit Pull Request**
   ```bash
   gh pr create --title "feat: description" --body "..."
   ```

5. **Address Review Feedback**
   - Respond to comments
   - Make requested changes
   - Update documentation

6. **Merge**
   - Maintainer merges when approved
   - Branch deleted automatically

### Contribution Areas

**High-Impact Areas**:
- Agent orchestration improvements
- Performance optimisations
- Documentation improvements
- Bug fixes

**Extension Development**:
- Custom MCP tools
- New agent types
- Custom visualisations
- See [Extending the System](./extending-the-system.md)

### Getting Help

- Check [Troubleshooting Guide](./troubleshooting.md)
- Search GitHub issues
- Join community Discord
- Tag maintainers in issue/PR

## Performance Optimisation

### Frontend Optimisation

**React Performance**:

```typescript
// Use React.memo for expensive components
export const GraphVisualization = React.memo<GraphVisualizationProps>(
  ({ nodes, edges, layout }) => {
    // Expensive rendering logic
    return <Canvas>...</Canvas>;
  },
  // Custom comparison function
  (prevProps, nextProps) => {
    return (
      prevProps.nodes.length === nextProps.nodes.length &&
      prevProps.edges.length === nextProps.edges.length &&
      prevProps.layout === nextProps.layout
    );
  }
);

// Use useMemo for expensive calculations
const processedNodes = useMemo(() => {
  return nodes.map(node => ({
    ...node,
    position: calculatePosition(node, layout)
  }));
}, [nodes, layout]);

// Use useCallback for stable function references
const handleNodeClick = useCallback((nodeId: string) => {
  dispatch({ type: 'SELECT_NODE', payload: nodeId });
}, [dispatch]);
```

**Bundle Optimisation**:

```javascript
// vite.config.js
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom', 'zustand'],
          'three': ['three', '@react-three/fiber', '@react-three/drei'],
          'ui': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
        }
      }
    },
    chunkSizeWarningLimit: 1000
  }
});
```

### Backend Optimisation

**Async Performance**:

```rust
use tokio::task::JoinSet;

pub async fn process_batch(items: Vec<Item>) -> Vec<Result<Processed>> {
    let mut set = JoinSet::new();

    // Process in batches of 10
    for batch in items.chunks(10) {
        for item in batch {
            let item = item.clone();
            set.spawn(async move {
                process_item(item).await
            });
        }
    }

    let mut results = Vec::new();
    while let Some(result) = set.join_next().await {
        results.push(result.unwrap());
    }

    results
}
```

**Database Optimisation**:

```rust
// Use connection pooling
let pool = sqlx::postgres::PgPoolOptions::new()
    .max_connections(5)
    .connect(&database_url)
    .await?;

// Use prepared statements
let agents = sqlx::query_as!(
    Agent,
    r#"
    SELECT id, name, status, created_at
    FROM agents
    WHERE status = $1 AND created_at > $2
    "#,
    status,
    cutoff_date
)
.fetch_all(&pool)
.await?;

// Batch operations
let mut tx = pool.begin().await?;

for agent in agents {
    sqlx::query!(
        "UPDATE agents SET last_seen = $1 WHERE id = $2",
        Utc::now(),
        agent.id
    )
    .execute(&mut *tx)
    .await?;
}

tx.commit().await?;
```

### Monitoring Performance

```bash
# Frontend bundle analysis
cd client
npm run build -- --analyze

# Backend profiling
cd server
cargo build --release
perf record --call-graph=dwarf ./target/release/visionflow
perf report

# Docker resource monitoring
docker stats --no-stream

# Database query analysis
EXPLAIN ANALYZE SELECT * FROM agents WHERE status = 'active';
```

## Next Steps

- **Start Contributing**: See [Contributing Guide](../contributing.md)
- **Extend the System**: Read [Extending the System](./extending-the-system.md)
- **Understand Architecture**: Review [ADRs](../concepts/decisions/)
- **Troubleshooting**: Consult [Troubleshooting Guide](./troubleshooting.md)

---

*Need help? Check [Troubleshooting](./troubleshooting.md) or open an issue on GitHub.*
