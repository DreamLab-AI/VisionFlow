---
title: Development Guide
description: Setting up a local development environment for VisionFlow
category: how-to
diataxis: how-to
tags:
  - development
  - setup
  - testing
updated-date: 2025-01-29
---

# Development Guide

Setting up a local development environment for VisionFlow.

## Contents

- [Environment Setup](setup.md) - Initial development setup
- [Testing](testing.md) - Running and writing tests
- [Contributing](contributing.md) - Contribution guidelines
- [Code Style](code-style.md) - Coding standards

## Quick Start

### Prerequisites
- Rust 1.75+ (with cargo)
- Node.js 18+ (for frontend)
- Docker (for Neo4j)
- Git

### Setup Steps

```bash
# Clone repository
git clone https://github.com/your-org/visionflow.git
cd visionflow

# Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Configure environment
cp .env.example .env

# Build and run
cargo build
cargo run

# Run tests
cargo test
```

## Related

- [Testing Guide](testing.md)
- [Architecture](../../architecture/README.md)
