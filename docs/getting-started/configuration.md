# Configuration Guide

This comprehensive guide covers all configuration options for VisionFlow, from basic environment variables to advanced performance tuning and feature customisation.

## Configuration Overview

VisionFlow uses a layered configuration system:

1. **Environment Variables** - Primary configuration via `.env` file
2. **Docker Compose** - Service orchestration and resource allocation
3. **Runtime Settings** - Dynamic configuration via web interface
4. **API Configuration** - Programmatic settings management

## Environment Variables

### Core Configuration

#### Basic Settings
```bash
# Application Identification
APP_NAME=VisionFlow                    # Application name
APP_VERSION=0.1.0                     # Version identifier
ENVIRONMENT=production                 # Environment: development, staging, production

# Network Configuration
HOST_PORT=3001                        # External access port
INTERNAL_API_PORT=4000               # Internal API port
WEBSOCKET_PORT=4001                  # WebSocket communication port
HOST_IP=0.0.0.0                      # Bind address (0.0.0.0 for all interfaces)

# Service Discovery
CLAUDE_FLOW_HOST=multi-agent-container # Claude Flow MCP hostname
MCP_TCP_PORT=9500                     # Claude Flow MCP TCP port
MCP_TRANSPORT=tcp                     # Transport protocol: tcp, websocket
MCP_ENABLE_TCP=true                   # Enable TCP server
MCP_ENABLE_UNIX=false                 # Enable Unix socket
```

#### Security and Authentication
```bash
# JWT Configuration
JWT_SECRET=your_256_bit_secret_key_here           # JWT signing secret (256-bit)
JWT_EXPIRATION=3600                               # Token expiration (seconds)
JWT_REFRESH_EXPIRATION=604800                     # Refresh token expiration (7 days)

# CORS Configuration
CORS_ORIGINS=http://localhost:3001,https://yourdomain.com  # Allowed origins
CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS                    # Allowed HTTP methods
CORS_HEADERS=Content-Type,Authorization,X-Requested-With    # Allowed headers

# API Rate Limiting
RATE_LIMIT_WINDOW=900                             # Rate limit window (15 minutes)
RATE_LIMIT_MAX=1000                              # Max requests per window
RATE_LIMIT_BURST=50                              # Burst allowance

# Security Headers
HSTS_MAX_AGE=31536000                            # HTTPS Strict Transport Security
CONTENT_SECURITY_POLICY=default-src 'self'       # CSP header
X_FRAME_OPTIONS=DENY                             # X-Frame-Options header
```

### Performance Configuration

#### System Resources
```bash
# Memory Management
MEMORY_LIMIT=16g                      # Container memory limit
SWAP_LIMIT=4g                        # Swap memory limit
SHARED_MEMORY_SIZE=2g                # Shared memory allocation

# CPU Configuration
CPU_LIMIT=8.0                        # CPU core limit (floating point)
CPU_RESERVATION=4.0                  # Reserved CPU cores
CPU_AFFINITY=0-7                     # CPU core affinity mask

# I/O Configuration
IO_WEIGHT=1000                       # I/O scheduling weight (100-10000)
DISK_READ_BPS=1048576000            # Disk read bandwidth limit (bytes/sec)
DISK_WRITE_BPS=1048576000           # Disk write bandwidth limit (bytes/sec)
```

#### Application Performance
```bash
# Multi-Agent System
MAX_AGENTS=20                        # Maximum concurrent agents
AGENT_TIMEOUT=300                    # Agent operation timeout (seconds)
TASK_QUEUE_SIZE=1000                # Maximum queued tasks
WORKER_THREADS=8                     # Background worker threads

# Graph Processing
MAX_NODES=10000                      # Maximum nodes per graph
MAX_EDGES=50000                      # Maximum edges per graph
PHYSICS_FPS=60                       # Physics simulation frame rate
RENDER_FPS=60                        # Rendering frame rate

# Caching
CACHE_SIZE=512                       # Cache size (MB)
CACHE_TTL=3600                       # Cache time-to-live (seconds)
ENABLE_REDIS_CACHE=true             # Enable Redis caching
REDIS_URL=redis://redis:6379        # Redis connection URL
```

### GPU and Compute Configuration

#### NVIDIA GPU Settings
```bash
# GPU Basic Configuration
ENABLE_GPU=true                      # Enable GPU acceleration
NVIDIA_VISIBLE_DEVICES=0             # GPU device IDs (0,1,2 or all)
NVIDIA_DRIVER_CAPABILITIES=compute,utility # Driver capabilities

# CUDA Configuration
CUDA_ARCH=89                         # CUDA architecture (86=RTX30xx, 89=RTX40xx)
CUDA_VERSION=12.4                    # CUDA runtime version
CUDA_MEMORY_LIMIT=8g                 # GPU memory limit

# GPU Compute Features
ENABLE_GPU_PHYSICS=true              # GPU-accelerated physics
ENABLE_GPU_RENDERING=true            # GPU-accelerated rendering
GPU_BATCH_SIZE=1000                  # GPU computation batch size
GPU_COMPUTE_THREADS=1024             # GPU thread block size
```

#### Advanced GPU Settings
```bash
# Performance Tuning
GPU_CLOCK_SPEED=auto                 # GPU clock speed (auto, max, or Hz)
GPU_MEMORY_CLOCK=auto                # Memory clock speed
GPU_POWER_LIMIT=300                  # Power limit (watts)
GPU_TEMPERATURE_LIMIT=83             # Temperature limit (celsius)

# Compute Configuration
WARP_SIZE=32                         # GPU warp size
SHARED_MEMORY_PER_BLOCK=49152       # Shared memory per thread block
MAX_THREADS_PER_BLOCK=1024          # Maximum threads per block
MAX_BLOCKS_PER_SM=16                # Maximum blocks per streaming multiprocessor

# Debugging
CUDA_DEBUG=false                     # Enable CUDA debugging
CUDA_PROFILE=false                   # Enable CUDA profiling
GPU_ERROR_CHECKING=true              # Enable GPU error checking
```

### Feature Configuration

#### AI and Machine Learning
```bash
# API Keys
OPENAI_API_KEY=sk-your_openai_key_here           # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key     # Anthropic Claude API key
PERPLEXITY_API_KEY=pplx-your_perplexity_key     # Perplexity API key
GOOGLE_AI_API_KEY=your_google_ai_key            # Google AI API key

# Model Configuration
DEFAULT_LLM_MODEL=gpt-4                         # Default language model
LLM_TEMPERATURE=0.7                             # Model creativity (0.0-1.0)
LLM_MAX_TOKENS=4096                             # Maximum tokens per request
LLM_TIMEOUT=30                                  # API request timeout (seconds)

# Neural Enhancement
ENABLE_NEURAL_ENHANCEMENT=true                   # Enable neural processing
NEURAL_BATCH_SIZE=64                            # Neural network batch size
NEURAL_LEARNING_RATE=0.001                     # Learning rate for training
ENABLE_WASM_ACCELERATION=true                   # WebAssembly acceleration
```

#### Extended Reality (XR)
```bash
# XR Configuration
ENABLE_XR=false                      # Enable XR/VR features
XR_DEFAULT_MODE=ar                   # Default mode: ar, vr, mixed
QUEST3_SUPPORT=true                  # Meta Quest 3 support
HAND_TRACKING=true                   # Hand tracking support
EYE_TRACKING=false                   # Eye tracking support

# XR Performance
XR_RENDER_SCALE=1.0                  # Render scale factor (0.5-2.0)
XR_REFRESH_RATE=90                   # Display refresh rate (Hz)
XR_IPD=63.5                          # Interpupillary distance (mm)
SPATIAL_ANCHORS=true                 # Spatial anchor support
```

#### Voice and Audio
```bash
# Voice Configuration
ENABLE_VOICE=false                   # Enable voice interaction
VOICE_LANGUAGE=en-GB                 # Voice recognition language
VOICE_MODEL=whisper-1                # Speech recognition model
TTS_VOICE=neural                     # Text-to-speech voice type

# Audio Settings
AUDIO_SAMPLE_RATE=48000             # Audio sample rate (Hz)
AUDIO_BUFFER_SIZE=1024              # Audio buffer size
MICROPHONE_GAIN=1.0                 # Microphone input gain
SPEAKER_VOLUME=0.8                  # Speaker output volume
```

### Database Configuration

#### PostgreSQL Settings
```bash
# Database Connection
POSTGRES_HOST=postgres               # PostgreSQL hostname
POSTGRES_PORT=5432                  # PostgreSQL port
POSTGRES_USER=visionflow           # Database username
POSTGRES_PASSWORD=secure_password_here # Database password
POSTGRES_DB=visionflow             # Database name

# Connection Pool
POSTGRES_MAX_CONNECTIONS=20         # Maximum database connections
POSTGRES_MIN_CONNECTIONS=5          # Minimum database connections
POSTGRES_CONNECTION_TIMEOUT=30      # Connection timeout (seconds)
POSTGRES_IDLE_TIMEOUT=600          # Idle connection timeout (seconds)

# Performance
POSTGRES_SHARED_BUFFERS=256MB       # Shared buffer size
POSTGRES_WORK_MEM=4MB              # Work memory per operation
POSTGRES_MAINTENANCE_WORK_MEM=64MB  # Maintenance work memory
POSTGRES_WAL_BUFFERS=16MB          # Write-ahead log buffers
```

#### Redis Configuration
```bash
# Redis Connection
REDIS_HOST=redis                    # Redis hostname
REDIS_PORT=6379                     # Redis port
REDIS_PASSWORD=                     # Redis password (empty for no auth)
REDIS_DB=0                         # Redis database number

# Redis Performance
REDIS_MAX_CONNECTIONS=100           # Maximum Redis connections
REDIS_CONNECTION_TIMEOUT=5          # Connection timeout (seconds)
REDIS_MAX_MEMORY=512mb             # Maximum memory usage
REDIS_EVICTION_POLICY=allkeys-lru   # Memory eviction policy
```

### Integration Configuration

#### GitHub Integration
```bash
# GitHub API
GITHUB_TOKEN=ghp_your_github_token_here         # GitHub personal access token
GITHUB_API_URL=https://api.github.com           # GitHub API base URL
GITHUB_WEBHOOK_SECRET=your_webhook_secret       # Webhook verification secret

# Repository Settings
GITHUB_DEFAULT_BRANCH=main                      # Default branch name
GITHUB_SYNC_INTERVAL=300                        # Sync interval (seconds)
GITHUB_MAX_FILE_SIZE=10485760                   # Max file size (10MB)
ENABLE_GITHUB_WEBHOOKS=true                     # Enable webhook support
```

#### Logseq Integration
```bash
# Logseq Configuration
LOGSEQ_GRAPH_PATH=/data/logseq      # Path to Logseq graph data
LOGSEQ_SYNC_MODE=auto               # Sync mode: auto, manual, webhook
LOGSEQ_FILE_EXTENSIONS=.md,.org     # Supported file extensions
LOGSEQ_EXCLUDE_PATTERNS=.git,.DS_Store # Excluded patterns

# Processing Options
ENABLE_BLOCK_REFERENCES=true        # Process block references
ENABLE_PAGE_PROPERTIES=true         # Process page properties
ENABLE_ALIAS_RESOLUTION=true        # Resolve page aliases
MAX_BLOCK_DEPTH=10                  # Maximum block nesting depth
```

### Development Configuration

#### Debug and Logging
```bash
# Logging Configuration
LOG_LEVEL=info                      # Log level: trace, debug, info, warn, error
LOG_FORMAT=json                     # Log format: json, pretty, compact
LOG_FILE=/app/logs/visionflow.log   # Log file path
LOG_ROTATION=daily                  # Log rotation: daily, weekly, size

# Debug Features
DEBUG_MODE=false                    # Enable debug mode
RUST_LOG=info                       # Rust logging configuration
RUST_BACKTRACE=1                    # Enable Rust backtraces
ENABLE_PROFILING=false              # Enable performance profiling

# Development Tools
HOT_RELOAD=false                    # Enable hot reload (development only)
MOCK_SERVICES=false                 # Use mock services for testing
DISABLE_HTTPS_REDIRECT=true         # Disable HTTPS redirect in development
CORS_ALLOW_ALL=false               # Allow all CORS origins (development only)
```

#### Testing Configuration
```bash
# Test Environment
TEST_DATABASE_URL=postgres://test:test@postgres:5432/visionflow_test
TEST_REDIS_URL=redis://redis:6379/1
TEST_CLAUDE_FLOW_HOST=localhost
TEST_MCP_TCP_PORT=9501

# Test Settings
RUN_INTEGRATION_TESTS=false        # Run integration tests
TEST_TIMEOUT=60                    # Test timeout (seconds)
PARALLEL_TESTS=4                   # Parallel test processes
GENERATE_COVERAGE=false            # Generate code coverage reports
```

## Docker Compose Configuration

### Service Profiles

#### Development Profile
```yaml
# docker-compose.dev.yml
services:
  webxr-dev:
    profiles: ["dev"]
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - ./client:/app/client:cached
      - ./src:/app/src:cached
      - ./data:/app/data:cached
    environment:
      - NODE_ENV=development
      - RUST_LOG=debug
      - HOT_RELOAD=true
```

#### Production Profile
```yaml
# docker-compose.production.yml
services:
  webxr-prod:
    profiles: ["production"]
    build:
      context: .
      dockerfile: Dockerfile.production
    deploy:
      resources:
        limits:
          memory: 16g
          cpus: '8.0'
        reservations:
          memory: 8g
          cpus: '4.0'
```

#### GPU Profile
```yaml
# docker-compose.gpu.yml
services:
  webxr-gpu:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - ENABLE_GPU=true
```

### Volume Configuration

#### Data Persistence
```yaml
volumes:
  # Application Data
  app_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/visionflow/data
  
  # Database Storage
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/visionflow/postgres
  
  # Cache Storage
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/visionflow/redis
```

#### Performance Volumes
```yaml
volumes:
  # SSD Storage for High Performance
  fast_storage:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/nvme/visionflow
  
  # Shared Memory for GPU
  gpu_shared_memory:
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: size=2g,uid=1000,gid=1000
```

### Network Configuration

#### Custom Networks
```yaml
networks:
  # Internal Communication
  internal:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.0.0/16
  
  # External Access
  external:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/16
  
  # GPU Communication
  gpu_network:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 9000
```

## Runtime Configuration

### Web Interface Settings

#### Visualisation Settings
```javascript
// Accessible via Settings Panel in Web Interface
{
  "physics": {
    "gravity": 0.1,              // Gravitational force
    "repulsion": 1000,           // Node repulsion strength
    "attraction": 0.01,          // Edge attraction strength
    "damping": 0.9,             // Movement damping
    "timeStep": 0.016,          // Physics time step (60fps)
    "maxVelocity": 50,          // Maximum node velocity
    "restLength": 100           // Default edge rest length
  },
  
  "rendering": {
    "quality": "high",           // Rendering quality: low, medium, high, ultra
    "antialias": true,          // Enable anti-aliasing
    "shadows": true,            // Enable shadows
    "bloom": false,             // Enable bloom effect
    "postProcessing": true,     // Enable post-processing
    "maxFPS": 60,              // Target frame rate
    "vsync": true              // Enable vertical sync
  },
  
  "camera": {
    "fov": 75,                  // Field of view (degrees)
    "near": 0.1,               // Near clipping plane
    "far": 10000,              // Far clipping plane
    "sensitivity": 1.0,         // Mouse sensitivity
    "autoRotate": false,        // Auto-rotate camera
    "rotateSpeed": 0.5         // Auto-rotation speed
  }
}
```

#### Interaction Settings
```javascript
{
  "controls": {
    "enableRotate": true,       // Enable camera rotation
    "enableZoom": true,         // Enable camera zoom
    "enablePan": true,          // Enable camera panning
    "zoomSpeed": 1.0,          // Zoom sensitivity
    "panSpeed": 1.0,           // Pan sensitivity
    "rotateSpeed": 1.0,        // Rotation sensitivity
    "maxDistance": 5000,       // Maximum zoom out distance
    "minDistance": 10          // Minimum zoom in distance
  },
  
  "selection": {
    "multiSelect": true,        // Enable multi-node selection
    "highlightConnected": true, // Highlight connected nodes
    "fadeUnselected": false,    // Fade unselected nodes
    "selectionColor": "#ff6b35", // Selection highlight color
    "hoverColor": "#4ecdc4"     // Hover highlight color
  },
  
  "keyboard": {
    "enableShortcuts": true,    // Enable keyboard shortcuts
    "customBindings": {},       // Custom key bindings
    "modifierKeys": ["ctrl", "shift", "alt"] // Recognised modifiers
  }
}
```

### API Configuration

#### REST API Settings
```javascript
// Configurable via /api/config endpoint
{
  "api": {
    "version": "v1",            // API version
    "rateLimit": {
      "enabled": true,          // Enable rate limiting
      "windowMs": 900000,       // 15 minutes
      "maxRequests": 1000,      // Max requests per window
      "skipSuccessfulRequests": false
    },
    
    "pagination": {
      "defaultLimit": 50,       // Default page size
      "maxLimit": 1000,         // Maximum page size
      "allowAll": false         // Allow unlimited results
    },
    
    "authentication": {
      "required": false,        // Require authentication
      "providers": ["jwt", "oauth"], // Auth providers
      "tokenExpiry": 3600       // Token expiry (seconds)
    }
  }
}
```

#### WebSocket Configuration
```javascript
{
  "websocket": {
    "protocol": "binary",       // Protocol: binary, json
    "compression": true,        // Enable compression
    "heartbeat": 30,           // Heartbeat interval (seconds)
    "maxConnections": 1000,    // Maximum concurrent connections
    "messageQueue": 10000,     // Maximum queued messages
    "binaryFormat": {
      "nodeUpdate": 28,        // Bytes per node update
      "edgeUpdate": 16,        // Bytes per edge update
      "metadata": 64           // Bytes for metadata
    }
  }
}
```

## Advanced Configuration

### Performance Tuning

#### CPU Optimization
```bash
# System-level optimisations
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Container CPU settings
--cpuset-cpus="0-7"                    # Bind to specific CPU cores
--cpu-quota=800000                     # CPU quota (80% of 8 cores)
--cpu-period=100000                    # CPU period
--cpu-shares=1024                      # CPU weight
```

#### Memory Optimization
```bash
# Kernel parameters
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=15' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' >> /etc/sysctl.conf

# Container memory settings
--memory=16g                           # Memory limit
--memory-swap=20g                      # Memory + swap limit
--oom-kill-disable=false               # Allow OOM killer
--memory-swappiness=10                 # Container swappiness
```

#### I/O Optimization
```bash
# Storage optimisations
echo 'mq-deadline' > /sys/block/nvme0n1/queue/scheduler

# Container I/O settings
--device-read-bps=/dev/nvme0n1:1048576000   # 1GB/s read limit
--device-write-bps=/dev/nvme0n1:1048576000  # 1GB/s write limit
--blkio-weight=1000                         # I/O scheduling weight
```

### Security Hardening

#### Container Security
```bash
# Run as non-root user
USER=1000:1000

# Drop capabilities
--cap-drop=ALL
--cap-add=NET_BIND_SERVICE

# Read-only filesystem
--read-only
--tmpfs /tmp
--tmpfs /var/run

# Security options
--security-opt=no-new-privileges
--security-opt=apparmor:unconfined
```

#### Network Security
```bash
# Firewall rules
sudo ufw allow 3001/tcp               # VisionFlow web interface
sudo ufw allow 9500/tcp               # Claude Flow MCP
sudo ufw deny 5432/tcp                # Block external database access
sudo ufw deny 6379/tcp                # Block external Redis access

# SSL/TLS Configuration
SSL_CERT_PATH=/etc/ssl/certs/visionflow.crt
SSL_KEY_PATH=/etc/ssl/private/visionflow.key
SSL_PROTOCOLS=TLSv1.2,TLSv1.3
SSL_CIPHERS=ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS
```

### Monitoring Configuration

#### Metrics Collection
```bash
# Prometheus metrics
ENABLE_METRICS=true
METRICS_PORT=9090
METRICS_PATH=/metrics
METRICS_INTERVAL=15                    # Collection interval (seconds)

# Application metrics
TRACK_PERFORMANCE=true
TRACK_USAGE=true
TRACK_ERRORS=true
TRACK_SECURITY_EVENTS=true
```

#### Health Checks
```bash
# Health check configuration
HEALTH_CHECK_INTERVAL=30               # Health check interval (seconds)
HEALTH_CHECK_TIMEOUT=10                # Health check timeout (seconds)
HEALTH_CHECK_RETRIES=3                 # Health check retries
HEALTH_CHECK_START_PERIOD=60           # Initial start period (seconds)
```

### Backup Configuration

#### Data Backup
```bash
# Backup settings
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"           # Daily at 2 AM (cron format)
BACKUP_RETENTION_DAYS=30              # Keep backups for 30 days
BACKUP_LOCATION=/opt/backups/visionflow
BACKUP_COMPRESSION=gzip               # Compression method

# What to backup
BACKUP_DATABASE=true                  # PostgreSQL database
BACKUP_REDIS=true                     # Redis data
BACKUP_USER_DATA=true                 # User-uploaded data
BACKUP_CONFIGURATION=true             # Configuration files
BACKUP_LOGS=false                     # Log files
```

### Multi-Instance Configuration

#### Load Balancing
```bash
# Load balancer settings
ENABLE_LOAD_BALANCING=true
LB_ALGORITHM=round_robin              # round_robin, least_conn, ip_hash
LB_HEALTH_CHECK=true
LB_SESSION_AFFINITY=false

# Instance configuration
INSTANCE_ID=visionflow-01
CLUSTER_NAME=visionflow-cluster
CLUSTER_SECRET=your_cluster_secret
```

#### High Availability
```bash
# HA Configuration
ENABLE_HA=true
HA_MODE=active_passive                # active_passive, active_active
FAILOVER_TIMEOUT=30                   # Failover timeout (seconds)
ENABLE_AUTO_FAILBACK=true             # Automatic failback
CONSENSUS_ALGORITHM=raft              # Consensus algorithm
```

## Configuration Validation

### Environment Validation Script
```bash
#!/bin/bash
# validate-config.sh

echo "Validating VisionFlow configuration..."

# Check required environment variables
required_vars=(
    "CLAUDE_FLOW_HOST"
    "MCP_TCP_PORT"
    "JWT_SECRET"
    "POSTGRES_PASSWORD"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set"
        exit 1
    fi
done

# Validate port ranges
if [ "$HOST_PORT" -lt 1024 ] || [ "$HOST_PORT" -gt 65535 ]; then
    echo "ERROR: HOST_PORT must be between 1024 and 65535"
    exit 1
fi

# Validate memory limits
if ! [[ "$MEMORY_LIMIT" =~ ^[0-9]+[gG]$ ]]; then
    echo "ERROR: MEMORY_LIMIT must be in format like '16g'"
    exit 1
fi

# Check GPU configuration
if [ "$ENABLE_GPU" = "true" ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: GPU enabled but nvidia-smi not found"
    fi
fi

echo "Configuration validation completed successfully!"
```

### Configuration Testing
```bash
# Test configuration changes
docker-compose config                  # Validate compose file
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml config

# Test environment file
set -a && source .env && set +a       # Load environment
./scripts/validate-config.sh          # Run validation

# Test API endpoints
curl -f http://localhost:3001/api/health || echo "Health check failed"
curl -f http://localhost:3001/api/config || echo "Config endpoint failed"
```

## Troubleshooting Configuration

### Common Configuration Issues

#### Port Conflicts
```bash
# Check for port conflicts
sudo netstat -tulpn | grep :3001
sudo lsof -i :3001

# Solution: Change HOST_PORT in .env
HOST_PORT=3002
```

#### Memory Issues
```bash
# Check available memory
free -h
docker system df

# Solution: Adjust memory limits
MEMORY_LIMIT=8g                       # Reduce if insufficient memory
```

#### GPU Configuration Problems
```bash
# Check GPU availability
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Common solutions:
ENABLE_GPU=false                      # Disable GPU if not available
NVIDIA_VISIBLE_DEVICES=all            # Use all GPUs
CUDA_ARCH=86                         # Match your GPU architecture
```

#### Permission Issues
```bash
# Check file permissions
ls -la .env
chmod 600 .env                       # Secure environment file

# Check Docker permissions
groups | grep docker                  # Check user in docker group
sudo usermod -aG docker $USER        # Add user to docker group
```

## Best Practices

### Security Best Practices
1. **Use strong secrets**: Generate random JWT secrets and passwords
2. **Limit network exposure**: Only expose necessary ports
3. **Regular updates**: Keep Docker images and dependencies updated
4. **Monitor access**: Log and monitor API access patterns
5. **Backup configuration**: Regularly backup configuration files

### Performance Best Practices
1. **Monitor resource usage**: Use monitoring tools to track performance
2. **Optimise for workload**: Adjust settings based on actual usage patterns
3. **Use SSD storage**: Store data on fast storage for better performance
4. **Enable caching**: Use Redis caching for frequently accessed data
5. **Load test**: Regularly test system under expected load

### Operational Best Practices
1. **Version control**: Keep configuration in version control
2. **Environment parity**: Use similar settings across environments
3. **Automated testing**: Test configuration changes automatically
4. **Documentation**: Document all custom configuration choices
5. **Rollback plan**: Have a plan to rollback configuration changes

---

This comprehensive configuration guide should help you optimise VisionFlow for your specific needs, from development through to production deployment.