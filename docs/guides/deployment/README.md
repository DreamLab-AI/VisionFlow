---
title: Deployment Guide
description: How to deploy VisionFlow to various environments
category: how-to
diataxis: how-to
tags:
  - deployment
  - docker
  - production
updated-date: 2025-01-29
---

# Deployment Guide

How to deploy VisionFlow to various environments.

## Contents

- [Docker Deployment](docker.md) - Container-based deployment
- [Kubernetes](kubernetes.md) - K8s deployment patterns
- [Cloud Providers](cloud.md) - AWS, GCP, Azure deployments

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Neo4j instance (or use provided container)
- GitHub personal access token (for sync)

### Basic Docker Deployment

```bash
# Clone repository
git clone https://github.com/your-org/visionflow.git
cd visionflow

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d

# Verify health
curl http://localhost:8080/api/health
```

## Related

- [Configuration Reference](../../reference/configuration/README.md)
- [Operations Guide](../operations/README.md)
