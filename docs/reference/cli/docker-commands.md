---
title: Docker Commands Reference
description: Docker and docker-compose commands for VisionFlow deployment
category: reference
difficulty-level: intermediate
tags:
  - cli
  - docker
  - deployment
updated-date: 2025-01-29
---

# Docker Commands Reference

Docker and docker-compose commands for VisionFlow deployment and management.

---

## Docker Compose Commands

### Start Services

```bash
# Start all services in background
docker-compose up -d

# Start specific service
docker-compose up -d visionflow

# Start with build
docker-compose up -d --build

# Start with Solid sidecar
docker-compose -f docker-compose.yml -f docker-compose.solid.yml up -d
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific service
docker-compose stop visionflow
```

### Logs

```bash
# View all logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Follow specific service logs
docker-compose logs -f visionflow

# Show last N lines
docker-compose logs --tail=100 visionflow
```

### Status

```bash
# List running containers
docker-compose ps

# List all containers (including stopped)
docker-compose ps -a
```

---

## Build Commands

### Build Images

```bash
# Build all images
docker-compose build

# Build with no cache
docker-compose build --no-cache

# Build specific service
docker-compose build visionflow

# Build with build args
docker-compose build --build-arg VERSION=1.0.0
```

### Push Images

```bash
# Push to registry
docker-compose push

# Push specific service
docker-compose push visionflow
```

---

## Container Management

### Execute Commands

```bash
# Execute command in running container
docker-compose exec visionflow /bin/bash

# Execute as root
docker-compose exec -u root visionflow /bin/bash

# Run one-off command
docker-compose run --rm visionflow cargo test
```

### Resource Usage

```bash
# View resource usage
docker stats

# View specific container
docker stats visionflow_visionflow_1
```

### Inspect

```bash
# Inspect container
docker inspect visionflow_visionflow_1

# View container IP
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' visionflow_visionflow_1
```

---

## Volume Management

### List Volumes

```bash
# List all volumes
docker volume ls

# List project volumes
docker volume ls | grep visionflow
```

### Inspect Volumes

```bash
# Inspect volume
docker volume inspect visionflow_postgres_data
```

### Remove Volumes

```bash
# Remove unused volumes
docker volume prune

# Remove specific volume
docker volume rm visionflow_postgres_data
```

---

## Network Management

### List Networks

```bash
# List networks
docker network ls

# Inspect network
docker network inspect visionflow_default
```

### Debug Connectivity

```bash
# Test connectivity from container
docker-compose exec visionflow nc -zv postgres 5432

# Check DNS resolution
docker-compose exec visionflow nslookup postgres
```

---

## Cleanup Commands

### Remove Containers

```bash
# Remove stopped containers
docker container prune

# Remove all project containers
docker-compose rm -f
```

### Remove Images

```bash
# Remove unused images
docker image prune

# Remove all unused images
docker image prune -a
```

### Full Cleanup

```bash
# Remove everything unused
docker system prune

# Remove everything including volumes
docker system prune --volumes -a
```

---

## Health Checks

### Check Container Health

```bash
# View health status
docker-compose ps

# Inspect health
docker inspect --format='{{.State.Health.Status}}' visionflow_visionflow_1

# View health logs
docker inspect --format='{{json .State.Health}}' visionflow_visionflow_1 | jq
```

### Manual Health Check

```bash
# Check API health
docker-compose exec visionflow curl http://localhost:9090/api/health

# Check database
docker-compose exec postgres pg_isready

# Check Redis
docker-compose exec redis redis-cli ping
```

---

## Environment Configuration

### Environment Variables

```bash
# Start with env file
docker-compose --env-file .env.production up -d

# Override environment variable
MEMORY_LIMIT=32g docker-compose up -d
```

### Profiles

```bash
# Start with profile
docker-compose --profile gpu up -d

# Start with multiple profiles
docker-compose --profile gpu --profile monitoring up -d
```

---

## Troubleshooting

### View Logs

```bash
# All service logs since container start
docker-compose logs

# Last 100 lines with timestamps
docker-compose logs -f --tail=100 --timestamps

# Filter by time
docker-compose logs --since 1h
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart visionflow

# Force recreate
docker-compose up -d --force-recreate
```

### Debug Mode

```bash
# Run with debug logging
RUST_LOG=debug docker-compose up

# Run interactively
docker-compose run --rm visionflow /bin/bash
```

---

## Related Documentation

- [Cargo Commands](./cargo-commands.md)
- [Docker Compose Options](../configuration/docker-compose-options.md)
- [Deployment Guide](../../how-to/deployment/deployment.md)
