# Multi-Model Router Guide

## Overview

The Multi-Model Router is an intelligent routing system that enables the Agentic Flow workstation to dynamically select and switch between multiple LLM providers based on task requirements, cost optimization, and performance criteria. It provides unified access to Google Gemini, OpenAI, Anthropic Claude, OpenRouter, Xinference, and local ONNX models through a single configuration-driven interface.

**Key Benefits:**
- **Cost Optimization**: Automatically route to the most cost-effective model for each task (up to 99% cost savings)
- **Intelligent Fallback**: Automatic failover across provider chains ensures reliability
- **Privacy Control**: Route sensitive tasks to local-only inference (Xinference/ONNX)
- **Quality Optimization**: Select highest quality models for critical tasks (code review, architecture)
- **Performance Tuning**: Configure speed, quality, and cost weights for different routing modes

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Agentic Flow Workstation             │
│        (Multi-Agent Container System)           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          Multi-Model Router Layer               │
│  ┌──────────────────────────────────────────┐  │
│  │  Router Mode Selection                   │  │
│  │  - performance │ cost │ quality          │  │
│  │  - balanced    │ offline                 │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Rule-Based Routing Engine               │  │
│  │  - Task type matching                    │  │
│  │  - Privacy requirements                  │  │
│  │  - Agent preferences                     │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Fallback Chain Management               │  │
│  │  - Circuit breaker                       │  │
│  │  - Retry logic                           │  │
│  │  - Health monitoring                     │  │
│  └──────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐          ┌──────────────┐
│  Cloud APIs  │          │ Local Models │
└──────────────┘          └──────────────┘
   │   │   │                  │      │
   ▼   ▼   ▼                  ▼      ▼
┌────┐┌────┐┌────┐      ┌────────┐┌────┐
│Gem││Ope││Cla│      │Xinfer││ONNX│
│ini││nAI││ude│      │ence  ││Phi4│
└────┘└────┘└────┘      └────────┘└────┘
  │      │      │            │        │
┌────┐┌────┐┌────┐      ┌────────┐┌────┐
│OR  ││Vert││Bedr│      │RAGFlow││GPU │
│    ││ex  ││ock │      │Network││Acc │
└────┘└────┘└────┘      └────────┘└────┘
```

## Router Modes

The router supports five distinct operating modes, each optimized for different use cases:

### 1. Performance Mode

**Optimize for speed and quality**

```json
"mode": "performance"
```

**Characteristics:**
- **Weights**: Speed 50%, Quality 40%, Cost 10%
- **Default Chain**: Gemini → OpenAI → Claude
- **Fallback Chain**: Gemini → OpenAI → Claude → OpenRouter → Xinference → ONNX
- **Best For**: Time-sensitive applications, interactive workflows, real-time responses

**Example Use Cases:**
- Live coding sessions with instant feedback
- Real-time code completion and suggestions
- Interactive debugging and troubleshooting
- Quick prototyping and experimentation

### 2. Cost Mode

**Optimize for lowest cost**

```json
"mode": "cost"
```

**Characteristics:**
- **Weights**: Speed 20%, Quality 30%, Cost 50%
- **Default Chain**: Xinference → OpenRouter → Gemini
- **Fallback Chain**: Xinference → ONNX → OpenRouter → Gemini → OpenAI
- **Best For**: Budget-conscious projects, high-volume tasks, development/testing

**Example Use Cases:**
- Bulk documentation generation
- Large-scale refactoring tasks
- Test case generation
- Development and experimentation
- Learning and prototyping

### 3. Quality Mode

**Optimize for highest quality**

```json
"mode": "quality"
```

**Characteristics:**
- **Weights**: Speed 10%, Quality 70%, Cost 20%
- **Default Chain**: Claude → OpenAI → Gemini
- **Fallback Chain**: Claude → OpenAI → Gemini → OpenRouter
- **Best For**: Critical production code, architecture decisions, security analysis

**Example Use Cases:**
- Code review for production releases
- System architecture design
- Security vulnerability analysis
- Complex refactoring with safety requirements
- API design and documentation

### 4. Balanced Mode

**Balance all factors equally**

```json
"mode": "balanced"
```

**Characteristics:**
- **Weights**: Speed 33%, Quality 34%, Cost 33%
- **Default Chain**: Gemini → OpenAI → OpenRouter
- **Fallback Chain**: Gemini → OpenAI → OpenRouter → Claude → Xinference
- **Best For**: General-purpose development, most common use cases

**Example Use Cases:**
- General application development
- Routine code generation
- Standard refactoring tasks
- Documentation writing
- Everyday development workflows

### 5. Offline Mode

**Use only local inference (no internet)**

```json
"mode": "offline"
```

**Characteristics:**
- **Weights**: Speed 50%, Quality 50%, Cost 0%
- **Default Chain**: Xinference → ONNX
- **Fallback Chain**: Xinference → ONNX (local only)
- **Best For**: Privacy-sensitive tasks, air-gapped environments, no network access

**Example Use Cases:**
- Processing confidential code or data
- Secure development environments
- Air-gapped systems
- Development without internet access
- GDPR-compliant workflows

## Configuring router.config.json

The router configuration file located at `/config/router.config.json` controls all routing behavior.

### Configuration Structure

```json
{
  "version": "1.0.0",
  "mode": "performance",
  "description": "Intelligent model router for Agentic Flow workstation",

  "providers": { ... },
  "routing": { ... },
  "costTracking": { ... },
  "performanceMonitoring": { ... },
  "fallbackBehavior": { ... }
}
```

### Provider Configuration

Each provider has specific configuration options:

#### Gemini Configuration

```json
"gemini": {
  "enabled": true,
  "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
  "apiKey": "${GOOGLE_GEMINI_API_KEY}",
  "models": {
    "default": "gemini-2.5-flash",
    "pro": "gemini-2.5-pro",
    "exp": "gemini-2.0-flash-exp"
  },
  "metrics": {
    "speed": 95,      // Performance rating (0-100)
    "quality": 85,    // Quality rating (0-100)
    "cost": 98,       // Cost efficiency (0-100, higher = cheaper)
    "reliability": 92 // Reliability score (0-100)
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 1000000,  // 1M token context
    "maxTokens": 8192
  },
  "priority": 1  // Lower = higher priority
}
```

**Gemini Strengths:**
- Extremely fast inference (95 speed rating)
- Massive 1M token context window
- Cost-effective (98 cost rating = very cheap)
- Excellent for large codebases and long contexts

#### OpenAI Configuration

```json
"openai": {
  "enabled": true,
  "baseUrl": "https://api.openai.com/v1",
  "apiKey": "${OPENAI_API_KEY}",
  "models": {
    "default": "gpt-4o",
    "mini": "gpt-4o-mini",
    "legacy": "gpt-4-turbo"
  },
  "metrics": {
    "speed": 85,
    "quality": 90,
    "cost": 70,
    "reliability": 95
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 128000,
    "maxTokens": 4096
  },
  "priority": 2
}
```

**OpenAI Strengths:**
- Excellent code quality (90 quality rating)
- Strong reasoning capabilities
- Highly reliable (95 reliability)
- Good for complex logic and problem-solving

#### Anthropic Claude Configuration

```json
"anthropic": {
  "enabled": true,
  "baseUrl": "https://api.anthropic.com/v1",
  "apiKey": "${ANTHROPIC_API_KEY}",
  "models": {
    "default": "claude-3-5-sonnet-20241022",
    "haiku": "claude-3-5-haiku-20241022",
    "opus": "claude-3-opus-20240229"
  },
  "metrics": {
    "speed": 80,
    "quality": 95,
    "cost": 40,
    "reliability": 98
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 200000,
    "maxTokens": 8192
  },
  "priority": 3,
  "useFor": [
    "code-review",
    "architecture",
    "complex-reasoning",
    "refactoring",
    "security-analysis"
  ]
}
```

**Claude Strengths:**
- Highest quality rating (95)
- Best for code review and architecture
- Excellent at complex reasoning
- Most reliable provider (98 reliability)
- Ideal for security-sensitive tasks

#### OpenRouter Configuration

```json
"openrouter": {
  "enabled": true,
  "baseUrl": "https://openrouter.ai/api/v1",
  "apiKey": "${OPENROUTER_API_KEY}",
  "models": {
    "default": "meta-llama/llama-3.1-8b-instruct",
    "code": "deepseek/deepseek-coder-v2",
    "chat": "deepseek/deepseek-chat",
    "reasoning": "deepseek/deepseek-r1",
    "claude": "anthropic/claude-3.5-sonnet"
  },
  "metrics": {
    "speed": 75,
    "quality": 75,
    "cost": 99,
    "reliability": 88
  },
  "priority": 4
}
```

**OpenRouter Strengths:**
- 99% cost savings (99 cost rating)
- Access to 100+ models through one API
- Excellent for budget-constrained scenarios
- Good variety of specialized models

#### Xinference Configuration

```json
"xinference": {
  "enabled": true,
  "baseUrl": "http://172.18.0.11:9997/v1",
  "apiKey": "none",
  "models": {
    "default": "auto",
    "code": "deepseek-coder",
    "chat": "qwen2.5"
  },
  "metrics": {
    "speed": 70,
    "quality": 75,
    "cost": 100,
    "reliability": 85
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 32768,
    "maxTokens": 2048,
    "localInference": true,
    "networkRequired": true  // Requires RAGFlow network
  },
  "priority": 5
}
```

**Xinference Strengths:**
- 100% free (100 cost rating = no API costs)
- Local inference via RAGFlow network
- Good for development and testing
- No API rate limits

#### ONNX Configuration

```json
"onnx": {
  "enabled": true,
  "modelPath": "/home/devuser/models/phi-4.onnx",
  "executionProviders": ["cuda", "cpu"],
  "models": {
    "default": "phi-4-mini-instruct"
  },
  "metrics": {
    "speed": 60,
    "quality": 70,
    "cost": 100,
    "reliability": 90
  },
  "features": {
    "streaming": false,
    "toolCalling": false,
    "contextWindow": 4096,
    "maxTokens": 2048,
    "localInference": true,
    "offline": true,          // Works without internet
    "gpuAccelerated": true
  },
  "priority": 6
}
```

**ONNX Strengths:**
- 100% free and offline (no internet required)
- GPU-accelerated local inference
- Complete privacy (data never leaves machine)
- Ideal for air-gapped environments

## Provider Selection Strategies

The router uses multiple strategies to select the optimal provider:

### 1. Explicit Selection

Directly specify provider in configuration or environment:

```bash
# Via environment variable
export AGENTIC_ROUTER_PROVIDER="claude"
export AGENTIC_ROUTER_MODEL="claude-3-5-sonnet-20241022"
```

### 2. Mode-Based Selection

Router mode determines default provider chain:

```json
{
  "mode": "performance",  // Uses: gemini → openai → claude
  "mode": "cost",         // Uses: xinference → openrouter → gemini
  "mode": "quality"       // Uses: claude → openai → gemini
}
```

### 3. Rule-Based Selection

Define custom routing rules based on task characteristics:

```json
"routing": {
  "rules": [
    {
      "condition": {
        "privacy": "high",
        "localOnly": true
      },
      "action": {
        "provider": "onnx"
      },
      "reason": "Privacy-sensitive tasks use offline ONNX"
    },
    {
      "condition": {
        "task": "code-review",
        "quality": "required"
      },
      "action": {
        "provider": "claude"
      },
      "reason": "Code reviews need highest quality"
    },
    {
      "condition": {
        "speed": "critical",
        "latency": "low"
      },
      "action": {
        "provider": "gemini",
        "model": "gemini-2.5-flash"
      },
      "reason": "Speed-critical tasks use fastest Gemini"
    }
  ]
}
```

### 4. Agent-Based Preferences

Different agents have preferred providers:

```json
"agentPreferences": {
  "coder": {
    "preferredProviders": ["claude", "openai", "gemini"],
    "minQuality": 85,
    "description": "Code generation needs high quality"
  },
  "reviewer": {
    "preferredProviders": ["claude", "openai"],
    "minQuality": 90,
    "description": "Code review requires highest quality"
  },
  "researcher": {
    "preferredProviders": ["gemini", "openai", "openrouter"],
    "minQuality": 70,
    "description": "Research can use cost-effective models"
  },
  "tester": {
    "preferredProviders": ["openrouter", "gemini", "xinference"],
    "minQuality": 70,
    "description": "Test generation works with budget models"
  }
}
```

### 5. Metric-Based Scoring

Router calculates scores based on mode weights:

```javascript
// Performance mode: speed=0.5, quality=0.4, cost=0.1
Score = (speed * 0.5) + (quality * 0.4) + (cost * 0.1)

// Example scores in performance mode:
// Gemini:  (95*0.5) + (85*0.4) + (98*0.1) = 91.3 ✓ Winner
// OpenAI:  (85*0.5) + (90*0.4) + (70*0.1) = 85.5
// Claude:  (80*0.5) + (95*0.4) + (40*0.1) = 82.0
```

## Fallback Chain Configuration

The fallback chain ensures reliability by automatically trying alternative providers when failures occur.

### Fallback Chain Structure

```json
"routing": {
  "modes": {
    "performance": {
      "defaultChain": ["gemini", "openai", "claude"],
      "fallbackChain": ["gemini", "openai", "claude", "openrouter", "xinference", "onnx"]
    }
  }
}
```

### Fallback Behavior Configuration

```json
"fallbackBehavior": {
  "maxRetries": 3,              // Retry same provider up to 3 times
  "retryDelay": 1000,           // Wait 1 second between retries
  "circuitBreaker": {
    "enabled": true,
    "failureThreshold": 5,      // Open circuit after 5 failures
    "timeout": 60000            // Wait 60s before retry
  }
}
```

### Fallback Flow Example

```
Request → Try Gemini (fail)
       → Retry Gemini (fail)
       → Retry Gemini (fail) [maxRetries reached]
       → Try OpenAI (fail)
       → Retry OpenAI (fail)
       → Try Claude (success) ✓
```

### Circuit Breaker Pattern

When a provider consistently fails, the circuit breaker prevents wasted attempts:

```
Provider State Machine:
CLOSED → [5 failures] → OPEN → [60s timeout] → HALF_OPEN → [1 success] → CLOSED
                                             → [1 failure] → OPEN
```

## Cost Tracking and Optimization

### Cost Tracking Configuration

```json
"costTracking": {
  "enabled": true,
  "budgetLimits": {
    "daily": 10.0,    // $10/day limit
    "weekly": 50.0,   // $50/week limit
    "monthly": 200.0  // $200/month limit
  },
  "alerts": {
    "thresholds": [0.5, 0.8, 0.95],  // Alert at 50%, 80%, 95% of budget
    "notificationMethod": "log"       // Future: email, webhook, etc.
  }
}
```

### Cost Optimization Strategies

#### 1. Provider Cost Comparison

Approximate costs per 1M tokens (input/output):

| Provider | Input | Output | Notes |
|----------|-------|--------|-------|
| ONNX | $0 | $0 | Free local inference |
| Xinference | $0 | $0 | Free via RAGFlow network |
| OpenRouter | $0.10 | $0.30 | 99% savings vs direct |
| Gemini Flash | $0.15 | $0.60 | Very cost-effective |
| OpenAI GPT-4o | $2.50 | $10.00 | Premium pricing |
| Claude Sonnet | $3.00 | $15.00 | Highest quality |

#### 2. Task-Based Optimization

Route different task types to appropriate cost tiers:

```json
{
  "condition": { "task": "simple", "cost": "minimal" },
  "action": { "provider": "openrouter", "model": "llama-3.1-8b" }
},
{
  "condition": { "task": "code-generation", "cost": "free" },
  "action": { "provider": "xinference", "model": "deepseek-coder" }
},
{
  "condition": { "task": "code-review", "quality": "required" },
  "action": { "provider": "claude" }  // Worth the cost
}
```

#### 3. Budget Enforcement

Router enforces budget limits automatically:

```javascript
// Before routing request
if (dailyCost + estimatedCost > dailyBudget) {
  // Fallback to free providers
  return routeToProvider(["xinference", "onnx"]);
}
```

#### 4. Cost Monitoring

Monitor costs in real-time:

```bash
# View cost logs
tail -f /var/log/agentic-flow/cost-tracking.log

# Example output:
# 2025-10-12 14:23:15 [INFO] gemini request: $0.023 (daily: $2.45/10.00)
# 2025-10-12 14:25:30 [INFO] claude request: $0.156 (daily: $2.61/10.00)
# 2025-10-12 14:30:00 [WARN] Daily budget 50% threshold reached: $5.12/10.00
```

## Performance Tuning

### Performance Monitoring Configuration

```json
"performanceMonitoring": {
  "enabled": true,
  "metrics": [
    "latency",              // Request response time
    "tokens_per_second",    // Throughput
    "cost_per_request",     // Per-request cost
    "success_rate",         // Request success %
    "error_rate"            // Request failure %
  ],
  "loggingLevel": "info"
}
```

### Optimizing Latency

#### 1. Timeout Configuration

Set appropriate timeouts based on expected response times:

```json
"providers": {
  "gemini": {
    "timeout": 30000      // 30s for fast provider
  },
  "claude": {
    "timeout": 60000      // 60s for quality provider
  },
  "onnx": {
    "timeout": 120000     // 120s for local inference
  }
}
```

#### 2. Circuit Breaker Tuning

Adjust circuit breaker for faster failover:

```json
"circuitBreaker": {
  "enabled": true,
  "failureThreshold": 3,    // Open circuit faster
  "timeout": 30000          // Shorter timeout
}
```

#### 3. Retry Strategy

Optimize retry behavior:

```json
"fallbackBehavior": {
  "maxRetries": 2,          // Fewer retries = faster failover
  "retryDelay": 500         // Shorter delay between retries
}
```

### Optimizing Throughput

#### 1. Concurrent Requests

Enable concurrent request handling (implementation-dependent):

```json
"performanceMonitoring": {
  "concurrentRequests": 5   // Process up to 5 requests simultaneously
}
```

#### 2. Connection Pooling

Reuse HTTP connections for better performance:

```json
"providers": {
  "gemini": {
    "connectionPool": {
      "maxConnections": 10,
      "keepAlive": true,
      "timeout": 30000
    }
  }
}
```

### Provider Performance Characteristics

| Provider | Avg Latency | Throughput (tokens/s) | Best For |
|----------|-------------|----------------------|----------|
| Gemini Flash | 1-2s | ~500 | Interactive development |
| OpenAI GPT-4o | 2-4s | ~300 | Balanced performance |
| Claude Sonnet | 3-5s | ~250 | Quality over speed |
| OpenRouter | 2-6s | ~200 | Variable by model |
| Xinference | 5-10s | ~100 | Local development |
| ONNX Phi-4 | 10-20s | ~50 | Offline/privacy tasks |

## Examples for Different Use Cases

### Use Case 1: Cost-Optimized Development

**Scenario**: Building a new application on a budget

**Configuration**:
```json
{
  "mode": "cost",
  "routing": {
    "rules": [
      {
        "condition": { "task": "code-generation" },
        "action": { "provider": "xinference", "model": "deepseek-coder" }
      },
      {
        "condition": { "task": "documentation" },
        "action": { "provider": "openrouter", "model": "llama-3.1-8b" }
      },
      {
        "condition": { "task": "testing" },
        "action": { "provider": "openrouter", "model": "llama-3.1-8b" }
      }
    ]
  },
  "costTracking": {
    "budgetLimits": {
      "daily": 5.0,
      "monthly": 100.0
    }
  }
}
```

**Expected Costs**: ~$0-2/day using free and budget providers

### Use Case 2: Production Code Quality

**Scenario**: Code review and production releases

**Configuration**:
```json
{
  "mode": "quality",
  "routing": {
    "rules": [
      {
        "condition": { "task": "code-review" },
        "action": { "provider": "claude", "model": "claude-3-5-sonnet-20241022" }
      },
      {
        "condition": { "task": "architecture" },
        "action": { "provider": "claude" }
      },
      {
        "condition": { "task": "security-analysis" },
        "action": { "provider": "claude" }
      }
    ]
  },
  "agentPreferences": {
    "reviewer": {
      "preferredProviders": ["claude"],
      "minQuality": 95
    }
  }
}
```

**Expected Costs**: ~$5-15/day for high-quality reviews

### Use Case 3: Interactive Development

**Scenario**: Real-time coding with fast feedback

**Configuration**:
```json
{
  "mode": "performance",
  "routing": {
    "rules": [
      {
        "condition": { "speed": "critical" },
        "action": { "provider": "gemini", "model": "gemini-2.5-flash" }
      }
    ]
  },
  "fallbackBehavior": {
    "maxRetries": 1,
    "retryDelay": 500
  },
  "circuitBreaker": {
    "failureThreshold": 3,
    "timeout": 30000
  }
}
```

**Expected Costs**: ~$2-5/day with fast Gemini model

### Use Case 4: Privacy-Sensitive Development

**Scenario**: Working with confidential code

**Configuration**:
```json
{
  "mode": "offline",
  "routing": {
    "rules": [
      {
        "condition": { "privacy": "high" },
        "action": { "provider": "onnx" }
      },
      {
        "condition": { "privacy": "medium" },
        "action": { "provider": "xinference" }
      }
    ]
  },
  "providers": {
    "gemini": { "enabled": false },
    "openai": { "enabled": false },
    "anthropic": { "enabled": false },
    "openrouter": { "enabled": false }
  }
}
```

**Expected Costs**: $0 (fully local inference)

### Use Case 5: Balanced General Development

**Scenario**: Day-to-day application development

**Configuration**:
```json
{
  "mode": "balanced",
  "routing": {
    "rules": [
      {
        "condition": { "agentType": "coder" },
        "action": { "provider": "gemini", "model": "gemini-2.5-flash" }
      },
      {
        "condition": { "agentType": "reviewer" },
        "action": { "provider": "openai", "model": "gpt-4o" }
      },
      {
        "condition": { "agentType": "tester" },
        "action": { "provider": "openrouter", "model": "llama-3.1-8b" }
      }
    ]
  },
  "costTracking": {
    "budgetLimits": {
      "daily": 10.0,
      "monthly": 200.0
    }
  }
}
```

**Expected Costs**: ~$3-8/day with mixed provider usage

### Use Case 6: Multi-Agent Collaboration

**Scenario**: Multiple agents working on complex project

**Configuration**:
```json
{
  "mode": "balanced",
  "agentPreferences": {
    "architect": {
      "preferredProviders": ["claude", "openai"],
      "minQuality": 90
    },
    "coder": {
      "preferredProviders": ["gemini", "openai"],
      "minQuality": 85
    },
    "reviewer": {
      "preferredProviders": ["claude"],
      "minQuality": 95
    },
    "tester": {
      "preferredProviders": ["openrouter", "gemini"],
      "minQuality": 70
    },
    "researcher": {
      "preferredProviders": ["gemini", "openrouter"],
      "minQuality": 70
    }
  },
  "costTracking": {
    "budgetLimits": {
      "daily": 15.0
    },
    "alerts": {
      "thresholds": [0.8, 0.95]
    }
  }
}
```

**Expected Costs**: ~$5-12/day with intelligent agent routing

## Environment Variables

Required environment variables for router operation:

```bash
# Provider API Keys
export GOOGLE_GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Router Configuration (optional overrides)
export AGENTIC_ROUTER_MODE="performance"
export AGENTIC_ROUTER_PROVIDER="gemini"
export AGENTIC_ROUTER_MODEL="gemini-2.5-flash"

# Local Provider Configuration
export XINFERENCE_BASE_URL="http://172.18.0.11:9997/v1"
export ONNX_MODEL_PATH="/home/devuser/models/phi-4.onnx"
export ONNX_EXECUTION_PROVIDER="cuda"

# Cost Tracking
export AGENTIC_COST_TRACKING_ENABLED="true"
export AGENTIC_DAILY_BUDGET="10.0"
export AGENTIC_MONTHLY_BUDGET="200.0"

# Performance Tuning
export AGENTIC_REQUEST_TIMEOUT="60000"
export AGENTIC_MAX_RETRIES="3"
export AGENTIC_CIRCUIT_BREAKER_ENABLED="true"
```

## Monitoring and Debugging

### Log Files

```bash
# Router decision logs
/var/log/agentic-flow/router.log

# Cost tracking logs
/var/log/agentic-flow/cost-tracking.log

# Performance metrics
/var/log/agentic-flow/performance.log

# Error logs
/var/log/agentic-flow/error.log
```

### Log Format Examples

```
# Router decisions
2025-10-12 14:23:15 [INFO] Router mode: performance
2025-10-12 14:23:15 [INFO] Task: code-generation, Agent: coder
2025-10-12 14:23:15 [INFO] Selected provider: gemini (score: 91.3)
2025-10-12 14:23:15 [INFO] Fallback chain: gemini → openai → claude

# Cost tracking
2025-10-12 14:23:16 [INFO] Request: gemini-2.5-flash
2025-10-12 14:23:16 [INFO] Tokens: 1250 input, 450 output
2025-10-12 14:23:16 [INFO] Cost: $0.023 (daily: $2.45/$10.00)

# Performance metrics
2025-10-12 14:23:17 [INFO] Latency: 1.8s
2025-10-12 14:23:17 [INFO] Throughput: 487 tokens/s
2025-10-12 14:23:17 [INFO] Success rate: 98.5%
```

### Troubleshooting Common Issues

#### Issue 1: Provider Always Failing

**Symptoms**: Circuit breaker constantly open for a provider

**Solution**:
```bash
# Check provider connectivity
curl -X POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: $GOOGLE_GEMINI_API_KEY" \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}'

# Check API key validity
echo $GOOGLE_GEMINI_API_KEY

# Reset circuit breaker state (if needed)
rm -f /tmp/agentic-flow-circuit-breaker-state.json
```

#### Issue 2: High Costs

**Symptoms**: Budget exceeded, unexpected costs

**Solution**:
```bash
# Check cost logs
grep "WARN.*budget" /var/log/agentic-flow/cost-tracking.log

# Switch to cost mode
export AGENTIC_ROUTER_MODE="cost"

# Enable stricter budget limits
export AGENTIC_DAILY_BUDGET="5.0"

# Force free providers only
export AGENTIC_ROUTER_PROVIDER="xinference"
```

#### Issue 3: Slow Performance

**Symptoms**: High latency, timeouts

**Solution**:
```bash
# Switch to performance mode
export AGENTIC_ROUTER_MODE="performance"

# Reduce timeout for faster failover
export AGENTIC_REQUEST_TIMEOUT="30000"

# Enable faster providers only
# Edit router.config.json, disable slow providers
```

#### Issue 4: Quality Issues

**Symptoms**: Poor code quality, incorrect results

**Solution**:
```bash
# Switch to quality mode
export AGENTIC_ROUTER_MODE="quality"

# Force high-quality provider
export AGENTIC_ROUTER_PROVIDER="claude"
export AGENTIC_ROUTER_MODEL="claude-3-5-sonnet-20241022"

# Increase quality threshold for agents
# Edit router.config.json agentPreferences.minQuality
```

## Best Practices

1. **Start with Balanced Mode**: Use balanced mode for general development, then optimize
2. **Monitor Costs Daily**: Check cost logs regularly to avoid budget surprises
3. **Configure Fallback Chains**: Always have at least 3 providers in fallback chain
4. **Use Agent Preferences**: Configure agent-specific provider preferences for best results
5. **Enable Circuit Breaker**: Prevent wasted attempts on failing providers
6. **Test Provider Connectivity**: Verify all providers work before production use
7. **Set Realistic Budgets**: Budget for expected usage patterns plus 20% buffer
8. **Use Local Providers for Development**: Xinference/ONNX for cost-free development
9. **Reserve Premium Providers for Critical Tasks**: Claude for code review, architecture
10. **Log and Analyze**: Enable detailed logging to understand routing patterns

## Next Steps

- Review `/config/router.config.json` and customize for your use case
- Set environment variables for provider API keys
- Test router with different modes: `export AGENTIC_ROUTER_MODE="performance"`
- Monitor costs and performance in log files
- Adjust routing rules based on your workflow patterns
- Consider setting up local providers (Xinference/ONNX) for cost savings

## Related Documentation

- `/docs/GETTING_STARTED.md` - Initial setup and configuration
- `/docs/CONFIGURATION.md` - Detailed configuration reference
- `/docs/DEPLOYMENT.md` - Production deployment guide
- `/docs/router/ROUTER_USER_GUIDE.md` - Additional router examples
- `/docs/router/ROUTER_CONFIG_REFERENCE.md` - Configuration schema reference
- `/docs/router/TOP20_MODELS_MATRIX.md` - Model comparison matrix
