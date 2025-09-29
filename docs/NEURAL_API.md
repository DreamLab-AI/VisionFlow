# Neural API Documentation

## Overview

The Neural API provides comprehensive access to the neural-enhanced swarm controller's capabilities. This RESTful API exposes cognitive patterns, swarm intelligence, neural memory, and real-time coordination features through a unified interface.

## Base URL

```
http://localhost:8080/api/v1/neural
```

## Authentication

All API endpoints require authentication via JWT tokens or API keys.

```http
Authorization: Bearer <jwt_token>
# OR
X-API-Key: <api_key>
```

## Core Endpoints

### Swarm Management

#### Create Neural Swarm

```http
POST /swarms
Content-Type: application/json

{
  "config": {
    "max_agents": 100,
    "topology": {
      "type": "mesh",
      "connectivity": 0.7,
      "redundancy": 3
    },
    "swarm_pattern": {
      "type": "emergent",
      "emergence_threshold": 0.8,
      "pattern_stability": 0.9,
      "collective_memory": true
    },
    "cognitive_diversity": 0.8,
    "neural_plasticity": 0.7,
    "learning_rate": 0.01,
    "gpu_acceleration": true
  }
}
```

**Response:**
```json
{
  "swarm_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "created",
  "agent_count": 0,
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Get Swarm Status

```http
GET /swarms/{swarm_id}
```

**Response:**
```json
{
  "swarm_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_count": 25,
  "active_tasks": 12,
  "topology": {
    "type": "mesh",
    "connectivity": 0.75,
    "current_connections": 187
  },
  "metrics": {
    "collective_intelligence": 0.82,
    "swarm_coherence": 0.78,
    "task_throughput": 15.3,
    "adaptation_rate": 0.65,
    "energy_efficiency": 0.71,
    "fault_tolerance": 0.89,
    "learning_velocity": 0.58,
    "emergence_index": 0.74
  },
  "uptime": 1642248600
}
```

#### Update Swarm Configuration

```http
PUT /swarms/{swarm_id}/config
Content-Type: application/json

{
  "topology": {
    "type": "adaptive",
    "base_topology": {
      "type": "hierarchical",
      "levels": 4,
      "branching_factor": 3
    },
    "adaptation_rate": 0.1,
    "performance_threshold": 0.75
  },
  "learning_rate": 0.02
}
```

#### Delete Swarm

```http
DELETE /swarms/{swarm_id}
```

### Agent Management

#### Add Neural Agent

```http
POST /swarms/{swarm_id}/agents
Content-Type: application/json

{
  "role": "researcher",
  "cognitive_pattern": "divergent",
  "capabilities": [
    "data_analysis",
    "pattern_recognition",
    "hypothesis_generation"
  ],
  "initial_position": {
    "x": 50.0,
    "y": 30.0,
    "z": 10.0
  }
}
```

**Response:**
```json
{
  "agent_id": "660e8400-e29b-41d4-a716-446655440001",
  "role": "researcher",
  "cognitive_pattern": "divergent",
  "neural_state": {
    "activation_level": 0.5,
    "cognitive_load": 0.0,
    "learning_rate": 0.01,
    "synaptic_strength": 0.5
  },
  "status": "active"
}
```

#### Get Agent Details

```http
GET /swarms/{swarm_id}/agents/{agent_id}
```

**Response:**
```json
{
  "agent_id": "660e8400-e29b-41d4-a716-446655440001",
  "role": "researcher",
  "cognitive_pattern": "divergent",
  "position": {
    "x": 52.3,
    "y": 31.7,
    "z": 11.2
  },
  "velocity": {
    "x": 0.8,
    "y": 0.5,
    "z": 0.3
  },
  "connections": [
    "770e8400-e29b-41d4-a716-446655440002",
    "880e8400-e29b-41d4-a716-446655440003"
  ],
  "neural_state": {
    "activation_level": 0.75,
    "cognitive_load": 0.45,
    "learning_rate": 0.012,
    "attention_weights": {
      "primary_task": 0.6,
      "collaboration": 0.3,
      "monitoring": 0.1
    },
    "memory_utilization": 0.32,
    "neural_connections": 15,
    "synaptic_strength": 0.78
  },
  "performance_metrics": {
    "task_completion_rate": 0.89,
    "response_time": 0.23,
    "accuracy_score": 0.91,
    "collaboration_score": 0.85,
    "innovation_index": 0.77,
    "energy_efficiency": 0.68,
    "adaptation_speed": 0.72
  },
  "workload": 0.45,
  "trust_score": 0.83,
  "last_activity": "2024-01-15T10:45:30Z"
}
```

#### Update Agent Configuration

```http
PUT /swarms/{swarm_id}/agents/{agent_id}
Content-Type: application/json

{
  "cognitive_pattern": "convergent",
  "learning_rate": 0.015,
  "capabilities": [
    "data_analysis",
    "pattern_recognition",
    "hypothesis_generation",
    "statistical_modeling"
  ]
}
```

#### Remove Agent

```http
DELETE /swarms/{swarm_id}/agents/{agent_id}
```

### Task Management

#### Submit Neural Task

```http
POST /swarms/{swarm_id}/tasks
Content-Type: application/json

{
  "description": "Analyze customer behavior patterns in e-commerce data",
  "cognitive_requirements": ["divergent", "critical_analysis"],
  "priority": "high",
  "complexity": 0.75,
  "neural_constraints": {
    "min_activation_level": 0.7,
    "max_cognitive_load": 0.8,
    "required_trust_score": 0.6,
    "neural_synchronization": true,
    "collective_intelligence": true
  },
  "collaboration_type": "mesh",
  "dependencies": [],
  "deadline": "2024-01-20T18:00:00Z"
}
```

**Response:**
```json
{
  "task_id": "990e8400-e29b-41d4-a716-446655440004",
  "status": "assigned",
  "assigned_agents": [
    "660e8400-e29b-41d4-a716-446655440001",
    "770e8400-e29b-41d4-a716-446655440002"
  ],
  "estimated_duration": "PT45M",
  "created_at": "2024-01-15T11:00:00Z"
}
```

#### Get Task Status

```http
GET /swarms/{swarm_id}/tasks/{task_id}
```

**Response:**
```json
{
  "task_id": "990e8400-e29b-41d4-a716-446655440004",
  "description": "Analyze customer behavior patterns in e-commerce data",
  "status": "in_progress",
  "progress": 0.65,
  "assigned_agents": [
    {
      "agent_id": "660e8400-e29b-41d4-a716-446655440001",
      "role": "researcher",
      "contribution": 0.4
    },
    {
      "agent_id": "770e8400-e29b-41d4-a716-446655440002",
      "role": "analyzer",
      "contribution": 0.6
    }
  ],
  "cognitive_patterns_active": ["divergent", "critical_analysis"],
  "neural_metrics": {
    "collective_activation": 0.78,
    "synchronization_level": 0.85,
    "cognitive_load_average": 0.62
  },
  "started_at": "2024-01-15T11:05:00Z",
  "estimated_completion": "2024-01-15T11:50:00Z"
}
```

#### List Tasks

```http
GET /swarms/{swarm_id}/tasks?status=active&priority=high
```

### Cognitive Patterns

#### Get Available Patterns

```http
GET /cognitive-patterns
```

**Response:**
```json
{
  "patterns": [
    {
      "name": "convergent",
      "description": "Focused problem-solving with logical reasoning",
      "characteristics": [
        "systematic_analysis",
        "single_solution_focus",
        "logical_reasoning"
      ],
      "use_cases": ["debugging", "optimization", "validation"]
    },
    {
      "name": "divergent",
      "description": "Creative exploration generating multiple solutions",
      "characteristics": [
        "creative_exploration",
        "multiple_solutions",
        "innovative_thinking"
      ],
      "use_cases": ["brainstorming", "design", "research"]
    },
    {
      "name": "critical_analysis",
      "description": "Evaluation and assessment with quality focus",
      "characteristics": [
        "quality_evaluation",
        "risk_assessment",
        "decision_verification"
      ],
      "use_cases": ["code_review", "security_audit", "qa"]
    },
    {
      "name": "systems_thinking",
      "description": "Holistic understanding of complex systems",
      "characteristics": [
        "holistic_view",
        "interconnection_awareness",
        "emergent_behavior"
      ],
      "use_cases": ["architecture", "integration", "ecosystem_design"]
    }
  ]
}
```

#### Analyze Cognitive Compatibility

```http
POST /cognitive-patterns/compatibility
Content-Type: application/json

{
  "agent_patterns": ["divergent", "convergent"],
  "task_requirements": ["creative_exploration", "logical_validation"],
  "collaboration_type": "sequential"
}
```

**Response:**
```json
{
  "compatibility_score": 0.85,
  "recommendations": [
    {
      "pattern": "divergent",
      "phase": "exploration",
      "agents_needed": 2
    },
    {
      "pattern": "convergent",
      "phase": "validation",
      "agents_needed": 1
    }
  ],
  "potential_conflicts": [],
  "optimization_suggestions": [
    "Sequential execution recommended",
    "Add critical_analysis agent for final review"
  ]
}
```

### Neural Memory

#### Store Experience

```http
POST /memory/experiences
Content-Type: application/json

{
  "memory_type": "task",
  "key": "task_990e8400_analysis",
  "experience_data": {
    "type": "task_completion",
    "task_id": "990e8400-e29b-41d4-a716-446655440004",
    "agents_involved": [
      "660e8400-e29b-41d4-a716-446655440001",
      "770e8400-e29b-41d4-a716-446655440002"
    ],
    "cognitive_patterns_used": ["divergent", "critical_analysis"],
    "performance_metrics": {
      "completion_time": "PT43M",
      "accuracy": 0.94,
      "efficiency": 0.87
    },
    "lessons_learned": [
      "Divergent-convergent sequence effective for analysis tasks",
      "High-trust agents improve collaboration efficiency"
    ],
    "timestamp": "2024-01-15T11:48:00Z"
  }
}
```

#### Retrieve Memories

```http
GET /memory/experiences?memory_type=task&pattern=analysis&limit=10
```

**Response:**
```json
{
  "experiences": [
    {
      "key": "task_990e8400_analysis",
      "memory_type": "task",
      "relevance_score": 0.92,
      "experience_data": {
        "type": "task_completion",
        "performance_metrics": {
          "completion_time": "PT43M",
          "accuracy": 0.94,
          "efficiency": 0.87
        }
      },
      "stored_at": "2024-01-15T11:48:00Z"
    }
  ],
  "total_count": 7,
  "query_time_ms": 12
}
```

#### Search Patterns

```http
POST /memory/search
Content-Type: application/json

{
  "query": "successful task completion with divergent thinking",
  "memory_types": ["task", "agent"],
  "cognitive_filters": ["divergent"],
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  },
  "similarity_threshold": 0.7
}
```

### Consensus Management

#### Initiate Consensus

```http
POST /swarms/{swarm_id}/consensus
Content-Type: application/json

{
  "proposal": "Change swarm topology from mesh to hierarchical for better performance",
  "participating_agents": [
    "660e8400-e29b-41d4-a716-446655440001",
    "770e8400-e29b-41d4-a716-446655440002",
    "880e8400-e29b-41d4-a716-446655440003"
  ],
  "consensus_threshold": 0.75,
  "timeout_seconds": 300
}
```

**Response:**
```json
{
  "consensus_id": "aa0e8400-e29b-41d4-a716-446655440005",
  "status": "initiated",
  "proposal": "Change swarm topology from mesh to hierarchical for better performance",
  "participating_agents": 3,
  "required_votes": 3,
  "threshold": 0.75,
  "deadline": "2024-01-15T11:35:00Z"
}
```

#### Get Consensus Status

```http
GET /swarms/{swarm_id}/consensus/{consensus_id}
```

**Response:**
```json
{
  "consensus_id": "aa0e8400-e29b-41d4-a716-446655440005",
  "status": "completed",
  "result": "accepted",
  "final_score": 0.83,
  "votes": [
    {
      "agent_id": "660e8400-e29b-41d4-a716-446655440001",
      "vote": "agree",
      "confidence": 0.85,
      "reasoning": "Hierarchical topology would improve task delegation efficiency"
    },
    {
      "agent_id": "770e8400-e29b-41d4-a716-446655440002",
      "vote": "agree",
      "confidence": 0.78,
      "reasoning": "Performance metrics support this change"
    },
    {
      "agent_id": "880e8400-e29b-41d4-a716-446655440003",
      "vote": "agree",
      "confidence": 0.92,
      "reasoning": "Aligns with optimal coordination patterns"
    }
  ],
  "completed_at": "2024-01-15T11:33:45Z"
}
```

### Swarm Intelligence Patterns

#### Execute Swarm Pattern

```http
POST /swarms/{swarm_id}/patterns
Content-Type: application/json

{
  "pattern": {
    "type": "flocking",
    "separation_weight": 0.3,
    "alignment_weight": 0.4,
    "cohesion_weight": 0.3
  },
  "duration_seconds": 120,
  "target_agents": "all"
}
```

#### Get Active Patterns

```http
GET /swarms/{swarm_id}/patterns
```

**Response:**
```json
{
  "active_patterns": [
    {
      "pattern_id": "bb0e8400-e29b-41d4-a716-446655440006",
      "type": "flocking",
      "parameters": {
        "separation_weight": 0.3,
        "alignment_weight": 0.4,
        "cohesion_weight": 0.3
      },
      "participating_agents": 15,
      "effectiveness_score": 0.78,
      "started_at": "2024-01-15T11:40:00Z"
    }
  ]
}
```

### GPU Acceleration

#### Get GPU Status

```http
GET /gpu/status
```

**Response:**
```json
{
  "available": true,
  "devices": [
    {
      "device_id": 0,
      "name": "NVIDIA RTX 4090",
      "memory_total": 24576,
      "memory_used": 8192,
      "utilization": 0.65,
      "temperature": 72
    }
  ],
  "current_tasks": [
    {
      "task_type": "neural_inference",
      "agent_count": 25,
      "memory_usage": 4096
    }
  ]
}
```

#### Submit GPU Task

```http
POST /gpu/tasks
Content-Type: application/json

{
  "task_type": "cognitive_pattern_analysis",
  "input_data": {
    "agents": [
      {
        "agent_id": "660e8400-e29b-41d4-a716-446655440001",
        "neural_state": {
          "activation_level": 0.75,
          "cognitive_load": 0.45
        }
      }
    ],
    "pattern_requirements": ["divergent", "critical_analysis"]
  },
  "priority": "high"
}
```

## WebSocket API

### Real-time Neural Communication

**Connection:**
```
ws://localhost:8080/ws/neural/{swarm_id}
```

**Message Format:**
```json
{
  "type": "neural_message",
  "source_agent": "660e8400-e29b-41d4-a716-446655440001",
  "target_agent": "770e8400-e29b-41d4-a716-446655440002",
  "cognitive_pattern": "convergent",
  "neural_state": {
    "activation_level": 0.85,
    "cognitive_load": 0.6,
    "synaptic_strength": 0.9
  },
  "payload": {
    "message_type": "task_result",
    "data": {
      "analysis_complete": true,
      "findings": ["pattern_detected", "anomaly_found"],
      "confidence": 0.92
    }
  },
  "timestamp": "2024-01-15T11:45:30Z"
}
```

### Event Subscriptions

**Subscribe to Events:**
```json
{
  "type": "subscribe",
  "events": [
    "agent_status_change",
    "task_assignment",
    "cognitive_pattern_change",
    "swarm_topology_update",
    "consensus_initiated"
  ],
  "filters": {
    "agent_roles": ["researcher", "analyzer"],
    "cognitive_patterns": ["divergent"]
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "NEURAL_AGENT_NOT_FOUND",
    "message": "Neural agent with ID 660e8400-e29b-41d4-a716-446655440001 not found in swarm",
    "details": {
      "swarm_id": "550e8400-e29b-41d4-a716-446655440000",
      "agent_id": "660e8400-e29b-41d4-a716-446655440001"
    },
    "timestamp": "2024-01-15T11:50:00Z"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `SWARM_NOT_FOUND` | Swarm ID does not exist |
| `NEURAL_AGENT_NOT_FOUND` | Agent ID not found in swarm |
| `COGNITIVE_PATTERN_INVALID` | Invalid cognitive pattern specified |
| `TASK_ASSIGNMENT_FAILED` | Unable to assign task to agents |
| `INSUFFICIENT_AGENTS` | Not enough agents for task requirements |
| `NEURAL_CONSTRAINTS_VIOLATED` | Task constraints cannot be satisfied |
| `CONSENSUS_TIMEOUT` | Consensus decision timeout |
| `GPU_UNAVAILABLE` | GPU acceleration not available |
| `MEMORY_LIMIT_EXCEEDED` | Neural memory capacity exceeded |
| `TOPOLOGY_INVALID` | Invalid swarm topology configuration |

## Rate Limiting

- Standard endpoints: 100 requests/minute
- GPU-intensive endpoints: 10 requests/minute
- WebSocket connections: 5 connections per client
- Real-time streaming: 1000 messages/minute

## SDK Examples

### Python SDK

```python
from neural_swarm_sdk import NeuralSwarmClient

client = NeuralSwarmClient(
    base_url="http://localhost:8080/api/v1/neural",
    api_key="your_api_key_here"
)

# Create swarm
swarm = client.create_swarm({
    "max_agents": 50,
    "topology": {"type": "mesh", "connectivity": 0.8},
    "gpu_acceleration": True
})

# Add cognitive agent
agent = swarm.add_agent(
    role="researcher",
    cognitive_pattern="divergent",
    capabilities=["data_analysis", "pattern_recognition"]
)

# Submit neural task
task = swarm.submit_task(
    description="Analyze customer behavior patterns",
    cognitive_requirements=["divergent", "critical_analysis"],
    complexity=0.75
)

# Monitor progress
while task.status != "completed":
    task.refresh()
    print(f"Progress: {task.progress:.2%}")
    time.sleep(5)
```

### JavaScript SDK

```javascript
import { NeuralSwarmClient } from 'neural-swarm-sdk';

const client = new NeuralSwarmClient({
  baseUrl: 'http://localhost:8080/api/v1/neural',
  apiKey: 'your_api_key_here'
});

// Create swarm with adaptive topology
const swarm = await client.createSwarm({
  maxAgents: 75,
  topology: {
    type: 'adaptive',
    baseTopology: { type: 'hierarchical', levels: 3 },
    adaptationRate: 0.1
  },
  cognitivePattern: 'emergent'
});

// Real-time monitoring
swarm.on('agent_added', (agent) => {
  console.log(`New agent: ${agent.id} (${agent.role})`);
});

swarm.on('task_completed', (task) => {
  console.log(`Task completed: ${task.description}`);
});

// WebSocket connection
const ws = swarm.connectWebSocket();
ws.on('neural_message', (message) => {
  console.log('Neural communication:', message);
});
```

This Neural API provides comprehensive access to all cognitive and swarm intelligence capabilities, enabling developers to build sophisticated applications that leverage the full power of the neural-enhanced architecture.