---
skill: agentic-qe
version: 2.5.9
description: Agentic QE Fleet - AI-powered quality engineering with 20 specialized agents, 46 QE skills, and self-learning test automation
author: Multi-Agent Docker Team
tags: [testing, qa, qe, tdd, test-automation, coverage, security, accessibility, playwright, jest, cypress]
mcp_server: true
---

# Agentic QE Fleet Skill

AI-powered quality engineering platform with 20 specialized agents and 46 QE skills. Supports test generation, coverage analysis, security scanning, performance testing, and continuous learning.

## Quick Start

```bash
# Install globally
npm install -g agentic-qe

# Initialize in project
aqe init

# Start MCP server
python server.py

# Or via npx
npx aqe-mcp
```

## Architecture

### Agent Fleet (20 Specialized Agents)

| Agent | Domain | Capabilities |
|-------|--------|--------------|
| `qe-test-generator` | Test Creation | Generate tests with 95%+ coverage |
| `qe-coverage-analyzer` | Coverage | Gap analysis, dead code detection |
| `qe-security-scanner` | Security | SAST/DAST, OWASP Top 10 |
| `qe-performance-tester` | Performance | Load testing, profiling |
| `qe-accessibility-auditor` | A11y | WCAG compliance |
| `qe-api-validator` | API | Contract testing, OpenAPI |
| `qe-visual-regressor` | Visual | Screenshot comparison |
| `qe-chaos-engineer` | Resilience | Failure injection |
| `qe-data-generator` | Test Data | Synthetic data, GDPR compliant |
| `qe-flaky-detector` | Stability | ML-powered detection |

### TDD Subagents (11)

RED/GREEN/REFACTOR workflow specialists for London and Chicago School TDD.

### Skills Library (46)

Complete coverage of modern QE practices:
- **Core**: TDD, BDD, exploratory testing, risk-based testing
- **Modern**: Shift-left/right, chaos engineering, accessibility
- **Specialized**: Database, API contract, visual regression
- **Advanced**: Six thinking hats, compliance, CI/CD orchestration

## Available MCP Tools (25)

### Fleet Management
- `qe_fleet_status` - Get fleet health and agent availability
- `qe_spawn_agent` - Spawn specific QE agent
- `qe_orchestrate_pipeline` - Run multi-agent testing pipeline

### Test Generation
- `qe_generate_tests` - AI-powered test generation
- `qe_generate_tdd` - TDD workflow (RED/GREEN/REFACTOR)
- `qe_generate_contract_tests` - API contract tests
- `qe_generate_e2e` - End-to-end test scenarios

### Coverage Analysis
- `qe_analyze_coverage` - Comprehensive coverage report
- `qe_find_gaps` - Identify untested code paths
- `qe_dead_code` - Detect unreachable code

### Execution
- `qe_run_tests` - Execute tests with framework detection
- `qe_run_parallel` - Parallel test execution (10,000+ tests)
- `qe_run_selective` - Smart test selection

### Quality Gates
- `qe_quality_gate` - Enforce quality thresholds
- `qe_flaky_analysis` - ML-powered flaky detection
- `qe_regression_check` - Regression analysis

### Security
- `qe_security_scan` - SAST/DAST scanning
- `qe_dependency_audit` - Vulnerability scanning

### Performance
- `qe_performance_test` - Load/stress testing
- `qe_benchmark` - Performance benchmarking

### Accessibility
- `qe_a11y_audit` - WCAG compliance check
- `qe_visual_test` - Visual regression testing

### Learning
- `qe_learn_enable` - Enable RL learning
- `qe_pattern_search` - Search learned patterns
- `qe_metrics` - Learning metrics

## Integration with Claude-Flow

```javascript
// In claude-flow swarm
const pipeline = await qe_orchestrate_pipeline({
  files: ["src/services/*.ts"],
  stages: ["generate", "execute", "analyze", "gate"],
  coverage_threshold: 95,
  enable_learning: true
});

// Or invoke specific agent
await claude("Use qe-test-generator to create tests for src/services/auth.ts with TDD");
```

## Framework Support

| Framework | Test Gen | Execution | Coverage |
|-----------|----------|-----------|----------|
| Jest | Yes | Yes | Yes |
| Mocha | Yes | Yes | Yes |
| Cypress | Yes | Yes | Yes |
| Playwright | Yes | Yes | Yes |
| Vitest | Yes | Yes | Yes |
| Jasmine | Yes | Yes | Yes |
| AVA | Yes | Yes | Limited |

## Self-Learning Pipeline

The system learns from test executions:

1. **Q-Learning** - Optimal action-value functions from execution history
2. **Pattern Bank** - 85%+ accuracy across 6 frameworks
3. **Flaky Detection** - 90%+ accuracy with root cause analysis
4. **Experience Replay** - 10,000+ past executions in AgentDB

```bash
# Enable learning
aqe learn enable --all

# Check learning status
aqe learn status

# View patterns
aqe patterns list --min-confidence 0.8
```

## Multi-Model Cost Optimization

Intelligent routing across models for 70-81% cost savings:

| Model | Task Complexity | Cost |
|-------|-----------------|------|
| Claude Haiku | Simple assertions | $ |
| GPT-3.5 | Basic tests | $ |
| Claude Sonnet | Complex logic | $$ |
| GPT-4 | Architecture | $$$ |

## Real-Time Visualization

- **Mindmap**: Cytoscape.js, 1000+ nodes in <500ms
- **Metrics**: Recharts, 7-dimension radar
- **Timeline**: Virtual scroll, 1000+ events
- **WebSocket**: <50ms latency streaming

## CLI Commands

```bash
# Test operations
aqe test <file>              # Generate and run tests
aqe coverage --threshold 95   # Coverage analysis
aqe quality                   # Quality assessment

# Fleet operations
aqe fleet status              # Fleet health
aqe agent spawn <type>        # Spawn agent

# Learning
aqe learn enable --all        # Enable RL
aqe patterns list             # View patterns
aqe metrics                   # Performance metrics
```

## Environment Variables

```bash
AQE_COVERAGE_THRESHOLD=95     # Default coverage target
AQE_ENABLE_LEARNING=true      # Enable RL learning
AQE_PARALLEL_WORKERS=10       # Parallel test workers
AQE_MODEL_ROUTER=auto         # Model routing strategy
AGENTDB_PATH=./agentdb.db     # AgentDB path for learning
```

## Memory Namespaces

Agents coordinate through shared memory:
- `aqe/test-plan/*` - Test planning
- `aqe/coverage/*` - Coverage data
- `aqe/quality/*` - Quality metrics
- `aqe/patterns/*` - Learned patterns

## Dependencies

- agentic-qe>=2.5.9
- agentdb>=2.0.0 (for learning)
- Node.js>=18.0.0
