# Managing the Claude-Flow System

*[Architecture](../index.md)*

This document outlines the correct procedure for managing the `claude-flow` multi-agent system within the project's Docker-based architecture.

## Architectural Separation of Concerns

The system is designed with a clear separation between the main application and the agent orchestration service:

1.  **Application Container (`ar-ai-knowledge-graph`)**: This container runs the Rust backend and the React frontend. Its primary role is to serve the user interface and act as a *client* to the MCP. It initiates TCP connections to the `multi-agent-container` but does not host the orchestration logic itself.

2.  **Orchestrator Container (`multi-agent-container`)**: This container runs the `claude-flow` orchestrator as a dedicated service. It is responsible for managing the lifecycle of agents, tasks, and memory. It exposes a TCP server on port `9500` for clients like the main application to connect to.

This separation is crucial for scalability, stability, and maintainability. The `claude-flow` CLI tools are installed and intended to be run *inside* the `multi-agent-container`.

## Primary Management Method: `docker exec`

To manage the `claude-flow` system, you must execute commands within the context of the `multi-agent-container`. The standard way to achieve this is with the `docker exec` command.

The general syntax is:

```bash
docker exec -it multi-agent-container <command>
```

Where `<command>` is the `claude-flow` CLI command you wish to run.

### Example: Listing Active Agents

To list all active agents managed by the orchestrator, run the following command from your host machine's terminal:

```bash
docker exec -it multi-agent-container claude-flow agent list
```

This command instructs Docker to:
1.  `exec`: Execute a command.
2.  `-it`: Run it in interactive TTY mode, which provides a proper terminal interface.
3.  `multi-agent-container`: Target the container named `multi-agent-container`.
4.  `claude-flow agent list`: The command to run inside the container.

## Common Management Commands

Here are some practical examples of how to use the `claude-flow` CLI to manage the system via `docker exec`.

### Agent Management

*   **List all agents:**
    ```bash
    docker exec -it multi-agent-container claude-flow agent list
    ```

*   **Get detailed information about a specific agent:**
    ```bash
    docker exec -it powerdev claude-flow agent info <agent-id> --detailed
    ```

*   **Spawn a new researcher agent:**
    ```bash
    docker exec -it powerdev claude-flow agent spawn researcher --name "Market-Analyst"
    ```

*   **Terminate an agent:**
    ```bash
    docker exec -it powerdev claude-flow agent terminate <agent-id> --reason "Completed analysis task"
    ```

### Task Management

*   **List all tasks:**
    ```bash
    docker exec -it powerdev claude-flow task list
    ```

*   **Create a new research task and assign it:**
    ```bash
    docker exec -it powerdev claude-flow task create research "Analyze Q4 2024 AI hardware trends" --priority high --assign-to <agent-id>
    ```

*   **Check the status of a task:**
    ```bash
    docker exec -it powerdev claude-flow task status <task-id> --logs
    ```

### System Status

*   **View the live monitoring dashboard:**
    ```bash
    docker exec -it powerdev claude-flow monitor
    ```

*   **Get a detailed system health check:**
    ```bash
    docker exec -it powerdev claude-flow status --health-check --detailed
    ```

By using `docker exec`, you are directly interacting with the `claude-flow` instance that is managing your application's backend agents, ensuring that you have full control and visibility over the entire system.



## See Also

- [Configuration Architecture](../server/config.md)
- [Feature Access Control](../server/feature-access.md)
- [GPU Compute Architecture](../server/gpu-compute.md)

## Related Topics

- [Agent Visualisation Architecture](../agent-visualization-architecture.md)
- [Architecture Documentation](../architecture/README.md)
- [Architecture Migration Guide](../architecture/migration-guide.md)
- [Bots Visualisation Architecture](../architecture/bots-visualization.md)
- [Bots/VisionFlow System Architecture](../architecture/bots-visionflow-system.md)
- [Case Conversion Architecture](../architecture/CASE_CONVERSION.md)
- [ClaudeFlowActor Architecture](../architecture/claude-flow-actor.md)
- [Client Architecture](../client/architecture.md)
- [Decoupled Graph Architecture](../technical/decoupled-graph-architecture.md)
- [Dynamic Agent Architecture (DAA) Setup Guide](../architecture/daa-setup-guide.md)
- [GPU Compute Improvements & Troubleshooting Guide](../architecture/gpu-compute-improvements.md)
- [MCP Connection Architecture](../architecture/mcp_connection.md)
- [MCP Integration Architecture](../architecture/mcp-integration.md)
- [MCP WebSocket Relay Architecture](../architecture/mcp-websocket-relay.md)
- [Parallel Graph Architecture](../architecture/parallel-graphs.md)
- [Server Architecture](../server/architecture.md)
- [Settings Architecture Analysis Report](../architecture_analysis_report.md)
- [VisionFlow Component Architecture](../architecture/components.md)
- [VisionFlow Data Flow Architecture](../architecture/data-flow.md)
- [VisionFlow GPU Compute Integration](../architecture/gpu-compute.md)
- [VisionFlow GPU Migration Architecture](../architecture/visionflow-gpu-migration.md)
- [VisionFlow System Architecture Overview](../architecture/index.md)
- [VisionFlow System Architecture](../architecture/system-overview.md)
- [arch-system-design](../reference/agents/architecture/system-design/arch-system-design.md)
- [architecture](../reference/agents/sparc/architecture.md)
