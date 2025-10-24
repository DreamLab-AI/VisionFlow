# Configuration Guide

## Configuration Management

VisionFlow uses hierarchical configuration with environment-specific overrides.

## Configuration Files

```
config/
├── default.yml          # Base configuration
├── development.yml      # Development overrides
├── test.yml            # Test overrides
├── staging.yml         # Staging overrides
├── production.yml      # Production overrides
└── custom-environment-variables.yml
```

## Base Configuration

**config/default.yml**:

```yaml
server:
  host: 0.0.0.0
  port: 8080
  timeout: 30000

api:
  host: 0.0.0.0
  port: 9090
  cors:
    enabled: true
    origin: "*"
  rateLimit:
    windowMs: 900000
    max: 100

database:
  host: localhost
  port: 5432
  name: visionflow
  pool:
    min: 2
    max: 10

redis:
  host: localhost
  port: 6379
  password: null

storage:
  type: local
  local:
    path: /var/lib/visionflow/storage
  s3:
    bucket: null
    region: us-east-1

processing:
  maxConcurrent: 5
  timeout: 3600000
  retryAttempts: 3

logging:
  level: info
  format: json
  file:
    enabled: true
    path: /var/log/visionflow

security:
  jwt:
    secret: null
    expiresIn: 86400
  apiKeys:
    enabled: true
```

## Environment Variables

**config/custom-environment-variables.yml**:

```yaml
database:
  host: DB_HOST
  port: DB_PORT
  name: DB_NAME
  username: DB_USER
  password: DB_PASSWORD

redis:
  host: REDIS_HOST
  port: REDIS_PORT
  password: REDIS_PASSWORD

storage:
  s3:
    bucket: S3_BUCKET
    region: AWS_REGION
    accessKeyId: AWS_ACCESS_KEY_ID
    secretAccessKey: AWS_SECRET_ACCESS_KEY

security:
  jwt:
    secret: JWT_SECRET
```

## Production Configuration

**config/production.yml**:

```yaml
server:
  host: 0.0.0.0
  port: 8080

api:
  cors:
    origin: https://visionflow.example.com
  rateLimit:
    max: 1000

database:
  pool:
    min: 5
    max: 20
  ssl: true

logging:
  level: warn
  format: json

security:
  https:
    enabled: true
    cert: /etc/ssl/certs/cert.pem
    key: /etc/ssl/private/key.pem
```

## Accessing Configuration

```javascript
const config = require('config');

const dbHost = config.get('database.host');
const port = config.get('server.port');
```

## Best Practices

1. **Never commit secrets**: Use environment variables
2. **Use defaults**: Provide sensible defaults
3. **Document all options**: Comment configuration files
4. **Validate configuration**: Check on startup
5. **Environment-specific**: Override per environment
