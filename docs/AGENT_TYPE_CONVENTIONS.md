# Agent Type Conventions and Mapping

## VisionFlow Agent Types (Rust Backend)

The VisionFlow system recognizes the following agent types defined in `/src/types/claude_flow.rs`:

```rust
enum AgentType {
    Coordinator,
    Researcher,
    Coder,
    Analyst,
    Architect,
    Tester,
    Reviewer,
    Optimizer,
    Documenter,
}
```

## MCP Server Agent Types

The MCP server (Claude Flow) uses various agent type strings that need to be mapped to VisionFlow types:

### Direct Mappings
- `"coordinator"` → `AgentType::Coordinator`
- `"researcher"` → `AgentType::Researcher`
- `"coder"` → `AgentType::Coder`
- `"analyst"` → `AgentType::Analyst`
- `"architect"` → `AgentType::Architect`
- `"tester"` → `AgentType::Tester`
- `"reviewer"` → `AgentType::Reviewer`
- `"optimizer"` → `AgentType::Optimizer`
- `"documenter"` → `AgentType::Documenter`

### Alias Mappings
- `"task-orchestrator"` → `AgentType::Coordinator`
- `"analyzer"` → `AgentType::Analyst`
- `"code-analyzer"` → `AgentType::Analyst`

### Fallback Mappings
- `"worker"` → `AgentType::Coder` (workers implement features)
- `"specialist"` → `AgentType::Analyst` (specialists analyze specific domains)
- Any unknown type → `AgentType::Coordinator` (default)

## Colour Coding in Graph Visualisation

Each agent type has a specific colour for graph visualisation:

- **Coordinator**: `#FF6B6B` (Red) - Central orchestration role
- **Researcher**: `#4ECDC4` (Teal) - Information gathering
- **Coder**: `#45B7D1` (Blue) - Implementation
- **Analyst**: `#FFA07A` (Light Salmon) - Analysis and evaluation
- **Architect**: `#98D8C8` (Mint) - System design
- **Tester**: `#F7DC6F` (Yellow) - Testing and validation
- **Reviewer**: `#95E77E` (Light Green) - Code review
- **Optimizer**: `#FFB6D9` (Pink) - Performance optimisation
- **Documenter**: `#D4A5A5` (Dusty Rose) - Documentation

## Connection Affinity Rules

Agents have different connection strengths based on their types:

### High Affinity (0.8-0.9)
- Coordinator ↔ Any (0.9) - Coordinators connect strongly with all types
- Coder ↔ Tester (0.8) - Implementation and testing go hand-in-hand

### Medium Affinity (0.6-0.7)
- Researcher ↔ Analyst (0.7) - Research feeds into analysis
- Architect ↔ Coder (0.7) - Design guides implementation
- Architect ↔ Analyst (0.6) - Architecture requires analysis
- Reviewer ↔ Coder (0.6) - Review provides feedback to implementation
- Optimizer ↔ Analyst (0.6) - Optimisation needs performance analysis

### Default Affinity (0.5)
- All other combinations

## Implementation Notes

### TCP Message Processing
The MCP server sends agent data in the following format:
```json
{
  "jsonrpc": "2.0",
  "id": "...",
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"agents\": [...]}"
    }]
  }
}
```

The VisionFlow backend extracts and parses this nested structure to convert MCP agents to internal `AgentStatus` objects.

### Agent Status Fields
When converting from MCP to VisionFlow format, the following fields are populated:
- `agent_id`: Direct from MCP
- `profile.name`: From MCP name field
- `profile.agent_type`: Mapped using conventions above
- `status`: Direct from MCP (default: "active")
- `swarm_id`: From MCP swarmId field
- Default metrics are initialized for visualisation

## Adding New Agent Types

To add a new agent type:

1. Add to `AgentType` enum in `/src/types/claude_flow.rs`
2. Add ToString implementation
3. Add mapping in `mcp_agent_to_status()` function
4. Define colour in graph visualisation
5. Define connection affinities if needed
6. Update this documentation

## UI Considerations

The UI should display agent types consistently:
- Use the string representation from ToString trait
- Apply the defined colors for visual consistency
- Show agent type in tooltips and info panels
- Group agents by type in listings when appropriate