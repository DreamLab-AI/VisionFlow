---
layout: default
title: Getting Started
nav_order: 2
description: "Quick start guide for VisionFlow - installation, configuration, and first steps"
permalink: /getting-started/
---

# Getting Started with VisionFlow
{: .fs-9 }

Get up and running with VisionFlow in minutes.
{: .fs-6 .fw-300 }

---

## Prerequisites

Before installing VisionFlow, ensure you have the following:

| Requirement | Version | Purpose |
|:------------|:--------|:--------|
| **Rust** | 1.75+ | Server components |
| **Node.js** | 18+ | Client and tooling |
| **Neo4j** | 5.x | Graph database |
| **Docker** | 24+ | Container runtime (optional) |
| **CUDA** | 12.0+ | GPU acceleration (optional) |

## Quick Installation

### Option 1: Docker (Recommended)

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/dreamlab-ai/VisionFlow.git
cd VisionFlow

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps
```

Services will be available at:
- **Client**: http://localhost:3000
- **Server API**: http://localhost:8080
- **Neo4j Browser**: http://localhost:7474

### Option 2: Manual Installation

For development or customisation:

```bash
# Clone repository
git clone https://github.com/dreamlab-ai/VisionFlow.git
cd VisionFlow

# Install server dependencies
cargo build --release

# Install client dependencies
cd client && npm install && cd ..

# Start Neo4j (if not using Docker)
# Follow Neo4j installation guide for your platform

# Start the server
cargo run --release --bin visionflow-server

# In another terminal, start the client
cd client && npm run dev
```

## First Steps

### 1. Create Your First Graph

Once VisionFlow is running, create a simple knowledge graph:

```bash
# Using the CLI
visionflow graph create --name "My First Graph"

# Or via the API
curl -X POST http://localhost:8080/api/graphs \
  -H "Content-Type: application/json" \
  -d '{"name": "My First Graph"}'
```

### 2. Add Some Nodes

```bash
# Add nodes via CLI
visionflow node create --graph "My First Graph" \
  --type "Person" \
  --properties '{"name": "Alice", "role": "Developer"}'

visionflow node create --graph "My First Graph" \
  --type "Project" \
  --properties '{"name": "VisionFlow", "status": "Active"}'
```

### 3. Create Relationships

```bash
# Connect nodes
visionflow edge create --graph "My First Graph" \
  --from "Alice" \
  --to "VisionFlow" \
  --type "WORKS_ON"
```

### 4. View in the UI

Open http://localhost:3000 and select your graph. You should see:

- Two nodes (Alice and VisionFlow)
- One relationship connecting them
- Interactive 3D visualisation

## Configuration

### Basic Configuration

Create or edit `config/visionflow.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8080

[database]
uri = "bolt://localhost:7687"
username = "neo4j"
password = "your-password"

[features]
gpu_acceleration = true
semantic_search = true
xr_support = false
```

### Environment Variables

You can also configure via environment:

```bash
export VISIONFLOW_SERVER_PORT=8080
export VISIONFLOW_NEO4J_URI=bolt://localhost:7687
export VISIONFLOW_GPU_ENABLED=true
```

## Next Steps

Now that you have VisionFlow running:

| Goal | Resource |
|:-----|:---------|
| Learn the basics | [First Graph Tutorial](/tutorials/first-graph/) |
| Understand the architecture | [Architecture Overview](/architecture-overview/) |
| Configure features | [Configuration Guide](/guides/configuration/) |
| Set up for development | [Developer Setup](/guides/developer/setup/) |
| Deploy to production | [Deployment Guide](/guides/deployment/) |

## Troubleshooting

### Common Issues

**Neo4j connection failed**
```bash
# Check Neo4j is running
docker-compose logs neo4j

# Verify credentials
curl -u neo4j:password http://localhost:7474/db/neo4j/tx
```

**GPU not detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify CUDA toolkit
nvcc --version
```

**Port already in use**
```bash
# Find and kill process on port 8080
lsof -i :8080
kill -9 <PID>
```

For more issues, see the [Troubleshooting Guide](/guides/troubleshooting/).

---

{: .note }
Need help? Open an issue on [GitHub](https://github.com/dreamlab-ai/VisionFlow/issues) or join our [Discussions](https://github.com/dreamlab-ai/VisionFlow/discussions).
