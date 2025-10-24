# Docker Deployment

## Quick Start

```bash
# Clone repository
git clone https://github.com/org/visionflow.git
cd visionflow

# Configure environment
cp .env.example .env
nano .env

# Start with Docker Compose
docker-compose up -d
```

## Docker Compose Configuration

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  api:
    build: 
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "9090:9090"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/visionflow
    depends_on:
      - db
      - redis

  web:
    build:
      context: .
      dockerfile: docker/Dockerfile.web
    ports:
      - "8080:80"
    depends_on:
      - api

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - NODE_ENV=production
    depends_on:
      - db
      - redis
      - queue

  db:
    image: postgres:14-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=visionflow
      - POSTGRES_USER=visionflow
      - POSTGRES_PASSWORD=${DB_PASSWORD}

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

  queue:
    image: rabbitmq:3-management-alpine
    ports:
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
```

## Environment Variables

Create `.env` file:

```bash
# Application
NODE_ENV=production
PORT=9090

# Database
DB_HOST=db
DB_PORT=5432
DB_NAME=visionflow
DB_USER=visionflow
DB_PASSWORD=change_this_password

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Security
JWT_SECRET=change_this_secret
API_KEY_SECRET=change_this_secret

# Storage
STORAGE_TYPE=s3
S3_BUCKET=visionflow-data
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

## Health Checks

```bash
# Check all services
docker-compose ps

# Check API health
curl http://localhost:9090/health

# View logs
docker-compose logs -f api
```

## Production Deployment

```bash
# Build images
docker-compose build

# Start in detached mode
docker-compose up -d

# Scale workers
docker-compose up -d --scale worker=3

# Update deployment
docker-compose pull
docker-compose up -d
```

## Monitoring

Access service dashboards:
- **API**: http://localhost:9090
- **RabbitMQ**: http://localhost:15672 (guest/guest)

## Backup

```bash
# Backup database
docker-compose exec db pg_dump -U visionflow visionflow > backup.sql

# Restore database
docker-compose exec -T db psql -U visionflow visionflow < backup.sql
```
