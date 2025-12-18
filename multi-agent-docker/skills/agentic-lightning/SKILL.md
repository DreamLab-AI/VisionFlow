---
skill: agentic-lightning
version: 1.0.0
description: Agent Lightning RL training integration leveraging AgentDB + RuVector for reinforcement learning, pattern discovery, and self-improving agents
author: Multi-Agent Docker Team
tags: [rl, reinforcement-learning, agent-training, agentdb, ruvector, gnn, q-learning, ppo]
mcp_server: true
---

# Agentic Lightning Skill

Train AI agents using reinforcement learning with AgentDB's 9 RL algorithms and RuVector's GNN-enhanced vector search. Implements Agent Lightning patterns for continuous agent improvement.

## Quick Start

```bash
# Start MCP server
python server.py

# Or via npx (uses agentdb directly)
npx agentdb mcp
```

## Architecture Integration

This skill bridges three systems:

1. **AgentDB LearningSystem** - 9 RL algorithms (Q-Learning, SARSA, DQN, PPO, Actor-Critic, Decision Transformer, MCTS, Model-Based, Policy Gradient)
2. **RuVector GNN** - Graph Neural Network enhanced search with self-learning
3. **ReasoningBank** - Pattern storage with semantic similarity and outcome tracking

## Available Tools (18)

### Session Management
- `lightning_start_session` - Start RL training session with algorithm selection
- `lightning_end_session` - End session and save policy
- `lightning_list_sessions` - List active/completed sessions

### Training & Prediction
- `lightning_predict` - Get action prediction with confidence scores
- `lightning_feedback` - Submit feedback for learning (state, action, reward)
- `lightning_train` - Batch training with configurable epochs/learning rate
- `lightning_train_gnn` - Train GNN model on collected samples

### Pattern Management
- `lightning_store_pattern` - Store reasoning pattern with success rate
- `lightning_search_patterns` - Semantic search with optional GNN enhancement
- `lightning_record_outcome` - Record pattern outcome for continuous learning

### Transfer Learning
- `lightning_transfer` - Transfer learning between sessions/tasks
- `lightning_explain` - XAI explanations for action recommendations

### Metrics & Analysis
- `lightning_metrics` - Performance metrics with time windows and trends
- `lightning_convergence` - Calculate policy convergence rate
- `lightning_nightly_learn` - Run nightly learner for causal discovery

### Integration
- `lightning_record_tool_execution` - Record tool executions as RL experiences
- `lightning_calculate_reward` - Shaped reward calculation with multiple factors
- `lightning_ruvector_status` - RuVector backend status and GNN state

## RL Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| Q-Learning | Value-based | Discrete actions, tabular |
| SARSA | Value-based | On-policy learning |
| DQN | Value-based | High-dimensional states |
| Policy Gradient | Policy-based | Continuous actions |
| Actor-Critic | Hybrid | Balance bias/variance |
| PPO | Policy-based | Stable training |
| Decision Transformer | Sequence | Trajectory modeling |
| MCTS | Tree search | Planning |
| Model-Based | Model | Sample efficiency |

## Integration with Claude-Flow

```javascript
// In claude-flow swarm coordination
const session = await lightning_start_session({
  userId: "swarm-agent-1",
  sessionType: "ppo",
  config: {
    learningRate: 0.001,
    discountFactor: 0.99,
    explorationRate: 0.1
  }
});

// During task execution
const action = await lightning_predict({
  sessionId: session,
  state: "code_review:complex_pr"
});

// After task completion
await lightning_feedback({
  sessionId: session,
  state: "code_review:complex_pr",
  action: action.action,
  reward: 0.85,
  success: true,
  nextState: "code_review:approved"
});
```

## ReasoningBank Patterns

Store successful reasoning patterns for semantic retrieval:

```python
# Store pattern
pattern_id = await lightning_store_pattern({
  "taskType": "code_generation",
  "approach": "TDD with London School mocking",
  "successRate": 0.92,
  "tags": ["tdd", "mocking", "unit-tests"]
})

# Search with GNN enhancement
patterns = await lightning_search_patterns({
  "task": "generate tests for authentication service",
  "k": 5,
  "useGNN": True,  # Enable GNN-enhanced search
  "threshold": 0.7
})
```

## Reward Shaping

Multiple reward functions for different scenarios:

- **sparse** - Binary success/failure only
- **dense** - Partial rewards for progress
- **shaped** - Time efficiency and quality bonuses
- **standard** - Weighted combination (default)

```python
reward = lightning_calculate_reward({
  "success": True,
  "targetAchieved": True,
  "efficiencyScore": 0.8,
  "qualityScore": 0.9,
  "rewardFunction": "shaped"
})
```

## Environment Variables

```bash
AGENTDB_PATH=./agentdb.db  # Database path
RUVECTOR_GNN_ENABLED=true  # Enable GNN learning
LIGHTNING_DEFAULT_ALGO=ppo # Default RL algorithm
```

## Dependencies

- agentdb>=2.0.0
- ruvector>=0.1.24
- @ruvector/gnn>=0.1.19 (optional, for GNN features)
