# Server Documentation

The VisionFlow backend is a high-performance data processing and visualization engine built in Rust. It leverages an actor-based architecture (via Actix) to handle complex, concurrent operations with high throughput. Key architectural features include a unified GPU compute kernel for accelerated analytics, a dual-graph model for managing knowledge and agent states, and a resilient, TCP-only MCP integration for external agent control.

## Core Architecture

- **[System Architecture](architecture.md)** - High-level overview of the server design, including the actor system and data flow.
- **[Actor System](actors.md)** - Detailed descriptions of the core actors (`GraphServiceActor`, `GPUComputeActor`, `ClaudeFlowActorTcp`, `OntologyActor`).
- **[Unified GPU Compute](gpu-compute.md)** - Documentation for the unified CUDA kernel (`visionflow_unified.cu`) and its compute modes.
- **[Physics Engine](physics-engine.md)** - Details on the force-directed layout, semantic constraints, and Stress Majorization.
- **[MCP Integration](mcp-integration.md)** - The TCP-only communication protocol for controlling agents.

## Features

- **[Ontology Validation](features/ontology.md)** - Formal validation and logical inference for the knowledge graph.
- **[Graph Clustering](features/clustering.md)** - GPU-accelerated algorithms for identifying communities and semantic groups.
- **[Semantic Analysis](features/semantic-analysis.md)** - A multi-stage pipeline to enrich the graph with deeper meaning and generate dynamic layout constraints.

## Getting Started

### Running the Server
```bash
# Development mode (with GPU support)
cargo run --features gpu

# Production build
cargo build --release --features gpu
./target/release/agent-server

# With Docker
docker-compose up webxr
```

### Environment Variables
```bash
# Core settings
RUST_LOG=info
BIND_ADDRESS=0.0.0.0:3001

# Agent control (MCP)
AGENT_CONTROL_URL=multi-agent-container:9500

# GPU features
ENABLE_GPU_PHYSICS=true
```

## Project Structure
```
src/
├── actors/          # Core actor implementations
├── services/        # Business logic (e.g., OwlValidatorService, SemanticAnalyzer)
├── handlers/        # API endpoint handlers
├── gpu/             # Unified GPU kernel and related code
├── physics/         # Physics engine, including constraints
├── models/          # Data structures and types
├── config/          # Configuration management
├── utils/           # Shared utilities, including network resilience
└── main.rs          # Server entry point
```

## Testing
```bash
# Run all tests
cargo test

# Run tests with GPU features enabled
cargo test --features gpu