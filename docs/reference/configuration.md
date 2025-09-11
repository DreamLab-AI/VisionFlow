# VisionFlow Configuration Reference

*[Reference Documentation](index.md)*

This document provides comprehensive technical reference for VisionFlow's configuration system, covering all configuration files, environment variables, and runtime settings with exhaustive technical details.

## Configuration Architecture

VisionFlow implements a hierarchical configuration system with multiple layers and clear precedence rules:

```
Environment Variables (Highest Precedence)
    ↓
Command-line Arguments
    ↓ 
User Settings (Per-user via Nostr authentication)
    ↓
Application Settings (settings.yaml)
    ↓
Default Values (Lowest Precedence)
```

## File Structure Overview

```
ext/
├── .env                     # Environment variables & secrets (highest precedence)
├── .env_template            # Template for environment configuration
├── data/
│   └── settings.yaml        # Primary application configuration
├── config.yml               # Cloudflare tunnel configuration
├── docker-compose.yml       # Container orchestration (development)
├── docker-compose.gpu.yml   # GPU-accelerated deployment
├── docker-compose.prod.yml  # Production deployment
└── supervisord.*.conf       # Process management
```

## Environment Variables Reference

### Core System Configuration

#### Application Identity
```bash
APP_NAME=VisionFlow                    # Application identifier
APP_VERSION=0.1.0                     # Semantic version
ENVIRONMENT=production                 # Environment: development, staging, production
```

#### Network Configuration
```bash
# External Access
HOST_IP=0.0.0.0                      # Bind address (0.0.0.0 for all interfaces)
HOST_PORT=3001                       # External HTTP port
INTERNAL_API_PORT=4000              # Internal REST API port
WEBSOCKET_PORT=4001                 # WebSocket communication port

# Service Discovery
CLAUDE_FLOW_HOST=multi-agent-container # Claude Flow MCP hostname
MCP_TCP_PORT=9500                      # Claude Flow MCP TCP port
MCP_TRANSPORT=tcp                      # Transport: tcp, websocket, unix
MCP_ENABLE_TCP=true                    # Enable TCP MCP server
MCP_ENABLE_UNIX=false                  # Enable Unix socket MCP
MCP_ENABLE_WEBSOCKET=false             # Enable WebSocket MCP bridge
```

#### Security and Authentication
```bash
# JWT Configuration
JWT_SECRET=your_256_bit_secret_here    # MUST be 256-bit (32+ chars)
JWT_EXPIRATION=3600                    # Token expiry (seconds)
JWT_REFRESH_EXPIRATION=604800          # Refresh token expiry (7 days)
JWT_ALGORITHM=HS256                    # Signing algorithm

# CORS Configuration
CORS_ORIGINS=http://localhost:3001,https://visionflow.info
CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS,PATCH
CORS_HEADERS=Content-Type,Authorization,X-Requested-With,X-Nostr-Pubkey
CORS_CREDENTIALS=true                  # Allow credentials
CORS_MAX_AGE=86400                     # Preflight cache duration

# Content Security Policy
CSP_DEFAULT_SRC="'self'"               # Default source directive
CSP_SCRIPT_SRC="'self' 'unsafe-inline' 'unsafe-eval'"
CSP_STYLE_SRC="'self' 'unsafe-inline'"
CSP_CONNECT_SRC="'self' ws: wss:"
CSP_IMG_SRC="'self' data: blob:"
CSP_FONT_SRC="'self'"

# Security Headers
HSTS_MAX_AGE=31536000                  # HTTPS Strict Transport Security
HSTS_INCLUDE_SUBDOMAINS=true           # Apply to subdomains
X_FRAME_OPTIONS=DENY                   # Prevent clickjacking
X_CONTENT_TYPE_OPTIONS=nosniff         # MIME type sniffing protection
REFERRER_POLICY=strict-origin-when-cross-origin
```

#### Rate Limiting
```bash
# API Rate Limiting
RATE_LIMIT_ENABLED=true                # Enable rate limiting
RATE_LIMIT_WINDOW=900                  # Window size (15 minutes)
RATE_LIMIT_MAX=1000                    # Max requests per window
RATE_LIMIT_BURST=50                    # Burst allowance
RATE_LIMIT_SKIP_SUCCESSFUL=false       # Don't count successful requests

# WebSocket Rate Limiting
WS_RATE_LIMIT_ENABLED=true             # Enable WebSocket rate limiting
WS_RATE_LIMIT_WINDOW=60                # WebSocket window (1 minute)
WS_RATE_LIMIT_MAX=1000                 # Max WebSocket messages
WS_CONNECTION_LIMIT=100                # Max concurrent WebSocket connections
```

### Performance Configuration

#### Memory Management
```bash
MEMORY_LIMIT=16g                       # Container memory hard limit
MEMORY_RESERVATION=8g                  # Guaranteed memory reservation
SWAP_LIMIT=4g                          # Swap memory limit
SHARED_MEMORY_SIZE=2g                  # Shared memory allocation
MEMORY_SWAPPINESS=10                   # VM swappiness (0-100)

# Rust Memory Configuration
RUST_MIN_STACK=8388608                 # Minimum stack size (8MB)
RUST_MAX_STACK=134217728               # Maximum stack size (128MB)
RUST_BACKTRACE=1                       # Enable backtraces
MALLOC_ARENA_MAX=4                     # Limit malloc arenas
```

#### CPU Configuration
```bash
CPU_LIMIT=8.0                          # CPU core limit (floating point)
CPU_RESERVATION=4.0                    # Reserved CPU cores
CPU_SHARES=1024                        # CPU shares (relative weight)
CPU_PERIOD=100000                      # CPU period (microseconds)
CPU_QUOTA=800000                       # CPU quota (80% of 8 cores)
CPU_AFFINITY=0-7                       # CPU affinity mask
CPU_SET="0,2,4,6"                      # Specific CPU cores

# CPU Governor Settings (requires privileged mode)
CPU_GOVERNOR=performance               # Performance governor
CPU_SCALING_MIN_FREQ=800000            # Minimum frequency (Hz)
CPU_SCALING_MAX_FREQ=3800000           # Maximum frequency (Hz)
```

#### I/O Configuration
```bash
# Block I/O Settings
BLKIO_WEIGHT=1000                      # I/O weight (100-1000)
BLKIO_READ_BPS=1073741824             # Read bandwidth limit (1GB/s)
BLKIO_WRITE_BPS=1073741824            # Write bandwidth limit (1GB/s)
BLKIO_READ_IOPS=10000                 # Read IOPS limit
BLKIO_WRITE_IOPS=10000                # Write IOPS limit

# Disk I/O Scheduler
IO_SCHEDULER=mq-deadline               # I/O scheduler (mq-deadline, kyber, bfq)
READ_AHEAD_KB=256                      # Read-ahead size (KB)
QUEUE_DEPTH=32                         # Device queue depth
```

#### Application Performance
```bash
# Multi-Agent System
MAX_AGENTS=20                          # Maximum concurrent agents
AGENT_TIMEOUT=300                      # Agent operation timeout (seconds)
TASK_QUEUE_SIZE=1000                   # Maximum queued tasks
WORKER_THREADS=8                       # Background worker threads
AGENT_POOL_SIZE=50                     # Agent pool size

# Graph Processing
MAX_NODES=10000                        # Maximum nodes per graph
MAX_EDGES=50000                        # Maximum edges per graph
PHYSICS_FPS=60                         # Physics simulation frame rate
RENDER_FPS=60                          # Target rendering frame rate
GRAPH_BATCH_SIZE=1000                  # Graph processing batch size

# Caching Configuration
CACHE_ENABLED=true                     # Enable application caching
CACHE_SIZE_MB=512                      # Cache size in megabytes
CACHE_TTL=3600                         # Default cache TTL (seconds)
CACHE_MAX_ENTRIES=10000                # Maximum cache entries
```

### GPU and Compute Configuration

#### NVIDIA GPU Settings
```bash
# GPU Basic Configuration
ENABLE_GPU=true                        # Enable GPU acceleration
ENABLE_GPU_PHYSICS=true                # GPU-accelerated physics
ENABLE_GPU_RENDERING=true              # GPU-accelerated rendering
NVIDIA_VISIBLE_DEVICES=0               # GPU device IDs (0,1,2 or all)
NVIDIA_DRIVER_CAPABILITIES=compute,utility # Required capabilities

# CUDA Runtime
CUDA_VERSION=12.4                      # CUDA runtime version
CUDA_ARCH=89                           # Target architecture (86=RTX30xx, 89=RTX40xx)
CUDA_HOME=/usr/local/cuda              # CUDA installation path
CUDA_PATH=/usr/local/cuda              # Alternative CUDA path

# GPU Memory Management
GPU_MEMORY_LIMIT=8g                    # GPU memory limit
CUDA_MEMORY_LIMIT=8192                 # CUDA memory limit (MB)
GPU_MEMORY_FRACTION=0.8                # Memory fraction to use
CUDA_CACHE_DISABLE=0                   # Disable CUDA cache (0/1)
```

#### Advanced GPU Configuration
```bash
# Compute Configuration
GPU_BATCH_SIZE=1000                    # GPU computation batch size
GPU_COMPUTE_THREADS=1024               # GPU thread block size
WARP_SIZE=32                           # GPU warp size
SHARED_MEMORY_PER_BLOCK=49152         # Shared memory per block (bytes)
MAX_THREADS_PER_BLOCK=1024            # Maximum threads per block
MAX_BLOCKS_PER_SM=16                  # Maximum blocks per SM

# Performance Tuning
GPU_CLOCK_SPEED=auto                   # GPU clock (auto, max, or Hz)
GPU_MEMORY_CLOCK=auto                  # Memory clock speed
GPU_POWER_LIMIT=300                    # Power limit (watts)
GPU_TEMPERATURE_LIMIT=83               # Temperature limit (°C)
GPU_FAN_SPEED=auto                     # Fan speed (auto, 0-100%)

# Error Handling and Debugging
CUDA_LAUNCH_BLOCKING=0                 # Synchronous kernel launches (0/1)
CUDA_DEBUG=0                           # Enable CUDA debugging (0/1)
CUDA_PROFILE=0                         # Enable CUDA profiling (0/1)
GPU_ERROR_CHECKING=true                # Enable error checking
GPU_SYNC_MODE=false                    # Synchronous GPU operations
```

### AI and Machine Learning

#### API Keys and Authentication
```bash
# Large Language Models
OPENAI_API_KEY=sk-proj-your_key_here   # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-your_key      # Anthropic Claude API key
GOOGLE_AI_API_KEY=your_google_key      # Google AI Studio key
PERPLEXITY_API_KEY=pplx-your_key       # Perplexity API key
GROQ_API_KEY=your_groq_key             # Groq API key
COHERE_API_KEY=your_cohere_key         # Cohere API key

# Specialised AI Services
HUGGINGFACE_API_TOKEN=hf_your_token    # Hugging Face token
REPLICATE_API_TOKEN=your_replicate_token # Replicate API token
STABILITY_API_KEY=sk-your_stability_key # Stability AI key

# Model Configuration
DEFAULT_LLM_MODEL=gpt-4o               # Default language model
LLM_TEMPERATURE=0.7                    # Model creativity (0.0-1.0)
LLM_MAX_TOKENS=4096                    # Maximum tokens per request
LLM_TIMEOUT=30                         # API request timeout (seconds)
LLM_MAX_RETRIES=3                      # Maximum retry attempts
LLM_RETRY_DELAY=1                      # Retry delay (seconds)
```

#### Neural Enhancement
```bash
# Neural Processing
ENABLE_NEURAL_ENHANCEMENT=true         # Enable neural processing
NEURAL_BATCH_SIZE=64                   # Neural network batch size
NEURAL_LEARNING_RATE=0.001             # Learning rate
NEURAL_EPOCHS=100                      # Training epochs
NEURAL_DROPOUT=0.2                     # Dropout rate

# WebAssembly Acceleration
ENABLE_WASM_ACCELERATION=true          # WebAssembly acceleration
WASM_MEMORY_PAGES=1024                 # WASM memory pages (64KB each)
WASM_MAX_MEMORY=67108864               # Maximum WASM memory (64MB)
WASM_THREAD_COUNT=4                    # WASM thread count

# Model Caching
MODEL_CACHE_ENABLED=true               # Enable model caching
MODEL_CACHE_SIZE=2048                  # Model cache size (MB)
MODEL_CACHE_TTL=86400                  # Model cache TTL (24 hours)
```

### Extended Reality (XR)

#### XR Core Configuration
```bash
# XR Features
ENABLE_XR=false                        # Enable XR/VR features
XR_DEFAULT_MODE=ar                     # Default mode: ar, vr, mixed
XR_SESSION_MODE=immersive-vr           # WebXR session mode
XR_REFERENCE_SPACE=local-floor         # Reference space type

# Device Support
QUEST3_SUPPORT=true                    # Meta Quest 3 support
PICO4_SUPPORT=false                    # Pico 4 support
VIVE_SUPPORT=false                     # HTC Vive support
HAND_TRACKING=true                     # Hand tracking support
EYE_TRACKING=false                     # Eye tracking support
PASSTHROUGH_SUPPORT=true               # Passthrough support
```

#### XR Performance
```bash
# Rendering Configuration
XR_RENDER_SCALE=1.0                    # Render scale factor (0.5-2.0)
XR_REFRESH_RATE=90                     # Display refresh rate (Hz)
XR_FOVEATED_RENDERING=true             # Enable foveated rendering
XR_MULTIVIEW_RENDERING=true            # Enable multiview rendering

# Comfort Settings
XR_IPD=63.5                            # Interpupillary distance (mm)
XR_COMFORT_SETTINGS=moderate           # Comfort level: aggressive, moderate, conservative
TELEPORT_ENABLED=true                  # Enable teleport locomotion
SMOOTH_LOCOMOTION=false                # Enable smooth locomotion
SNAP_TURNING=true                      # Enable snap turning
```

#### XR Interaction
```bash
# Spatial Computing
SPATIAL_ANCHORS=true                   # Spatial anchor support
PLANE_DETECTION=true                   # Plane detection
MESH_DETECTION=false                   # Mesh detection
LIGHT_ESTIMATION=true                  # Light estimation

# Hand Tracking
HAND_MESH_ENABLED=true                 # Display hand mesh
HAND_JOINT_TRACKING=true               # Joint position tracking
GESTURE_RECOGNITION=true               # Gesture recognition
PINCH_THRESHOLD=0.8                    # Pinch detection threshold
```

### Database Configuration

#### PostgreSQL Settings
```bash
# Connection Configuration
POSTGRES_HOST=postgres                 # Database hostname
POSTGRES_PORT=5432                     # Database port
POSTGRES_USER=visionflow              # Database username
POSTGRES_PASSWORD=secure_password      # Database password
POSTGRES_DB=visionflow                # Database name
POSTGRES_SCHEMA=public                # Default schema

# SSL Configuration
POSTGRES_SSL_MODE=prefer               # SSL mode: disable, allow, prefer, require
POSTGRES_SSL_CERT=/certs/client.crt    # Client certificate
POSTGRES_SSL_KEY=/certs/client.key     # Client private key
POSTGRES_SSL_ROOT_CERT=/certs/ca.crt   # Root certificate

# Connection Pool
POSTGRES_MAX_CONNECTIONS=20            # Maximum connections
POSTGRES_MIN_CONNECTIONS=5             # Minimum connections
POSTGRES_CONNECTION_TIMEOUT=30         # Connection timeout (seconds)
POSTGRES_IDLE_TIMEOUT=600             # Idle timeout (10 minutes)
POSTGRES_MAX_LIFETIME=3600            # Connection max lifetime (1 hour)

# Performance Tuning
POSTGRES_STATEMENT_TIMEOUT=30000      # Statement timeout (ms)
POSTGRES_LOCK_TIMEOUT=10000           # Lock timeout (ms)
POSTGRES_IDLE_IN_TRANSACTION_TIMEOUT=60000 # Idle in transaction timeout (ms)
```

#### PostgreSQL Advanced Configuration
```bash
# Memory Configuration
POSTGRES_SHARED_BUFFERS=256MB          # Shared buffer cache
POSTGRES_WORK_MEM=4MB                  # Work memory per operation
POSTGRES_MAINTENANCE_WORK_MEM=64MB     # Maintenance work memory
POSTGRES_EFFECTIVE_CACHE_SIZE=1GB      # Effective cache size
POSTGRES_RANDOM_PAGE_COST=1.1          # Random page access cost

# WAL Configuration
POSTGRES_WAL_BUFFERS=16MB              # WAL buffer size
POSTGRES_WAL_LEVEL=replica             # WAL level: minimal, replica, logical
POSTGRES_MAX_WAL_SENDERS=10            # Maximum WAL senders
POSTGRES_WAL_KEEP_SIZE=1GB             # WAL retention size

# Checkpoint Configuration
POSTGRES_CHECKPOINT_COMPLETION_TARGET=0.9 # Checkpoint completion target
POSTGRES_CHECKPOINT_TIMEOUT=5min       # Checkpoint timeout
POSTGRES_MAX_WAL_SIZE=1GB              # Maximum WAL size
POSTGRES_MIN_WAL_SIZE=80MB             # Minimum WAL size
```

#### Redis Configuration
```bash
# Connection Settings
REDIS_HOST=redis                       # Redis hostname
REDIS_PORT=6379                        # Redis port
REDIS_PASSWORD=                        # Redis password (empty = no auth)
REDIS_DB=0                            # Redis database number
REDIS_USERNAME=default                # Redis username (Redis 6+)

# Connection Pool
REDIS_MAX_CONNECTIONS=100             # Maximum connections
REDIS_MIN_CONNECTIONS=10              # Minimum connections
REDIS_CONNECTION_TIMEOUT=5            # Connection timeout (seconds)
REDIS_COMMAND_TIMEOUT=10              # Command timeout (seconds)
REDIS_RETRY_ATTEMPTS=3                # Connection retry attempts
REDIS_RETRY_DELAY=1000                # Retry delay (ms)

# Memory and Performance
REDIS_MAX_MEMORY=512mb                # Maximum memory usage
REDIS_MAX_MEMORY_POLICY=allkeys-lru   # Eviction policy
REDIS_TCP_KEEPALIVE=300               # TCP keep-alive (seconds)
REDIS_TCP_USER_TIMEOUT=30000          # TCP user timeout (ms)

# Persistence
REDIS_SAVE_ENABLED=true               # Enable RDB snapshots
REDIS_SAVE_INTERVAL=900 1             # Save every 900s if 1+ keys changed
REDIS_AOF_ENABLED=false               # Enable AOF persistence
REDIS_AOF_FSYNC=everysec              # AOF fsync policy
```

### Integration Configuration

#### GitHub Integration
```bash
# Authentication
GITHUB_TOKEN=ghp_your_token_here       # Personal access token
GITHUB_API_URL=https://api.github.com  # GitHub API base URL
GITHUB_WEBHOOK_SECRET=your_secret      # Webhook verification secret
GITHUB_CLIENT_ID=your_oauth_client_id  # OAuth application ID
GITHUB_CLIENT_SECRET=your_oauth_secret # OAuth application secret

# Repository Settings
GITHUB_DEFAULT_BRANCH=main             # Default branch name
GITHUB_MAX_FILE_SIZE=10485760         # Max file size (10MB)
GITHUB_SYNC_INTERVAL=300              # Sync interval (5 minutes)
GITHUB_RATE_LIMIT=5000                # API rate limit per hour
GITHUB_TIMEOUT=30                     # API timeout (seconds)

# Webhook Configuration
ENABLE_GITHUB_WEBHOOKS=true           # Enable webhook support
GITHUB_WEBHOOK_PATH=/webhook/github   # Webhook endpoint path
GITHUB_WEBHOOK_EVENTS=push,pull_request,issues # Monitored events

# Enterprise Configuration
GITHUB_ENTERPRISE_URL=                # GitHub Enterprise URL
GITHUB_ENTERPRISE_API_URL=            # GitHub Enterprise API URL
GITHUB_SKIP_VERIFY=false              # Skip SSL verification
```

#### Logseq Integration
```bash
# Core Configuration
LOGSEQ_GRAPH_PATH=/data/logseq        # Path to Logseq graph data
LOGSEQ_SYNC_MODE=auto                 # Sync mode: auto, manual, webhook
LOGSEQ_WATCH_ENABLED=true             # Enable file system watching
LOGSEQ_POLL_INTERVAL=5                # Polling interval (seconds)

# File Processing
LOGSEQ_FILE_EXTENSIONS=.md,.org,.markdown # Supported extensions
LOGSEQ_EXCLUDE_PATTERNS=.git,.DS_Store,node_modules # Excluded patterns
LOGSEQ_MAX_FILE_SIZE=5242880          # Max file size (5MB)
LOGSEQ_ENCODING=utf-8                 # File encoding

# Content Processing
ENABLE_BLOCK_REFERENCES=true          # Process block references
ENABLE_PAGE_PROPERTIES=true           # Process page properties
ENABLE_ALIAS_RESOLUTION=true          # Resolve page aliases
MAX_BLOCK_DEPTH=10                    # Maximum block nesting depth
ENABLE_NAMESPACE_PARSING=true         # Parse namespaced pages
```

#### RAGFlow Integration
```bash
# RAGFlow Configuration
RAGFLOW_API_URL=http://ragflow:9380    # RAGFlow API endpoint
RAGFLOW_API_KEY=your_ragflow_key       # RAGFlow API key
RAGFLOW_TIMEOUT=30                     # Request timeout (seconds)
RAGFLOW_MAX_RETRIES=3                  # Maximum retry attempts
RAGFLOW_RETRY_DELAY=1                  # Retry delay (seconds)

# Agent Configuration
RAGFLOW_AGENT_ID=default_agent_id      # Default agent ID
RAGFLOW_CONVERSATION_ID=               # Conversation ID (optional)
RAGFLOW_SESSION_TIMEOUT=3600           # Session timeout (1 hour)

# Processing Options
RAGFLOW_CHUNK_SIZE=512                 # Text chunk size
RAGFLOW_CHUNK_OVERLAP=50               # Chunk overlap tokens
RAGFLOW_MAX_CHUNKS=100                 # Maximum chunks per document
```

### Voice and Audio

#### Voice Configuration
```bash
# Core Voice Settings
ENABLE_VOICE=false                     # Enable voice interaction
VOICE_LANGUAGE=en-GB                   # Voice recognition language
VOICE_ACCENT=british                   # Voice accent preference
DEFAULT_VOICE=neural-british           # Default TTS voice

# Speech Recognition
STT_PROVIDER=whisper                   # Speech-to-text provider
STT_MODEL=whisper-1                    # STT model version
STT_LANGUAGE=en                        # STT language
STT_TEMPERATURE=0.0                    # STT temperature
STT_RESPONSE_FORMAT=json               # Response format

# Text-to-Speech
TTS_PROVIDER=openai                    # TTS provider
TTS_VOICE=alloy                        # TTS voice selection
TTS_MODEL=tts-1                        # TTS model version
TTS_SPEED=1.0                          # Speech speed (0.25-4.0)
TTS_FORMAT=mp3                         # Audio format
```

#### Audio Settings
```bash
# Audio Configuration
AUDIO_SAMPLE_RATE=48000               # Sample rate (Hz)
AUDIO_BIT_DEPTH=16                    # Bit depth
AUDIO_CHANNELS=2                      # Channel count (1=mono, 2=stereo)
AUDIO_BUFFER_SIZE=1024                # Buffer size (samples)
AUDIO_LATENCY=low                     # Latency: low, normal, high

# Input/Output Settings
MICROPHONE_ENABLED=true               # Enable microphone input
MICROPHONE_GAIN=1.0                   # Microphone gain (0.0-2.0)
SPEAKER_ENABLED=true                  # Enable speaker output
SPEAKER_VOLUME=0.8                    # Speaker volume (0.0-1.0)
AUDIO_ECHO_CANCELLATION=true          # Enable echo cancellation
AUDIO_NOISE_SUPPRESSION=true          # Enable noise suppression

# Advanced Audio
AUDIO_CODEC=opus                      # Audio codec: opus, aac, mp3
AUDIO_BITRATE=128000                  # Audio bitrate (bits/second)
AUDIO_VAD_ENABLED=true                # Voice activity detection
AUDIO_AGC_ENABLED=true                # Automatic gain control
```

#### Kokoro TTS Integration
```bash
# Kokoro TTS Configuration
KOKORO_API_URL=http://kokoro:8880      # Kokoro TTS API endpoint
KOKORO_DEFAULT_VOICE=af_heart          # Default voice selection
KOKORO_DEFAULT_FORMAT=mp3              # Output format: mp3, wav, ogg
KOKORO_DEFAULT_SPEED=1.0               # Speech speed (0.1-3.0)
KOKORO_TIMEOUT=30                      # Request timeout (seconds)

# Audio Processing
KOKORO_SAMPLE_RATE=24000               # Output sample rate
KOKORO_RETURN_TIMESTAMPS=true          # Include timing information
KOKORO_STREAM=true                     # Enable streaming output
KOKORO_BATCH_SIZE=1                    # Processing batch size

# Voice Selection
KOKORO_VOICE_EMOTION=neutral           # Emotional tone
KOKORO_VOICE_STABILITY=0.75            # Voice stability (0.0-1.0)
KOKORO_VOICE_SIMILARITY=0.75           # Voice similarity boost
```

#### Whisper Integration
```bash
# Whisper Configuration
WHISPER_API_URL=http://whisper:8000    # Whisper API endpoint
WHISPER_MODEL=base                     # Model size: tiny, base, small, medium, large
WHISPER_LANGUAGE=en                    # Language code
WHISPER_TIMEOUT=30                     # Processing timeout (seconds)

# Processing Options
WHISPER_TEMPERATURE=0.0                # Sampling temperature
WHISPER_VAD_FILTER=false               # Voice activity detection
WHISPER_WORD_TIMESTAMPS=false          # Word-level timestamps
WHISPER_RETURN_TIMESTAMPS=false        # Return timing information
WHISPER_INITIAL_PROMPT=""              # Initial prompt for context

# Advanced Settings
WHISPER_BEAM_SIZE=5                    # Beam search size
WHISPER_PATIENCE=1.0                   # Beam search patience
WHISPER_LENGTH_PENALTY=1.0             # Length penalty
WHISPER_SUPPRESS_TOKENS="-1"           # Tokens to suppress
```

### Development and Debugging

#### Logging Configuration
```bash
# Log Levels
RUST_LOG=info                          # Rust logging: error, warn, info, debug, trace
LOG_LEVEL=info                         # Application log level
LOG_FORMAT=json                        # Format: json, pretty, compact
LOG_TIMESTAMP=true                     # Include timestamps
LOG_TARGET=true                        # Include log targets

# Log Output
LOG_FILE=/app/logs/visionflow.log      # Log file path
LOG_ROTATION=daily                     # Rotation: daily, weekly, size-based
LOG_MAX_SIZE=100MB                     # Maximum log file size
LOG_MAX_FILES=10                       # Maximum log file count
LOG_COMPRESS=true                      # Compress rotated logs

# Structured Logging
LOG_JSON_PRETTY=false                  # Pretty-print JSON logs
LOG_INCLUDE_LOCATION=false             # Include file/line information
LOG_INCLUDE_THREAD=true                # Include thread information
LOG_INCLUDE_MODULE=true                # Include module information
```

#### Debug Features
```bash
# Debug Mode
DEBUG_MODE=false                       # Enable debug mode
RUST_BACKTRACE=1                       # Rust backtrace: 0, 1, full
ENABLE_PROFILING=false                 # Enable performance profiling
PROFILE_SAMPLE_RATE=1000               # Profiling sample rate (Hz)

# Development Features
HOT_RELOAD=false                       # Enable hot reload (dev only)
MOCK_SERVICES=false                    # Use mock services
AUTO_RELOAD=false                      # Auto-reload on file changes
DISABLE_HTTPS_REDIRECT=true            # Disable HTTPS redirect (dev)

# API Debugging
ENABLE_REQUEST_LOGGING=false           # Log all HTTP requests
ENABLE_SQL_LOGGING=false               # Log SQL queries
ENABLE_WEBSOCKET_LOGGING=false         # Log WebSocket messages
LOG_SENSITIVE_DATA=false               # Log sensitive information (dev only)
```

#### Testing Configuration
```bash
# Test Environment
TEST_MODE=false                        # Enable test mode
TEST_DATABASE_URL=postgres://test:test@postgres:5432/visionflow_test
TEST_REDIS_URL=redis://redis:6379/1    # Test Redis database
TEST_CLAUDE_FLOW_HOST=localhost        # Test MCP host
TEST_MCP_TCP_PORT=9501                 # Test MCP port

# Test Execution
RUN_INTEGRATION_TESTS=false            # Run integration tests
TEST_TIMEOUT=60                        # Test timeout (seconds)
PARALLEL_TESTS=4                       # Parallel test processes
GENERATE_COVERAGE=false                # Generate coverage reports
COVERAGE_FORMAT=html                   # Coverage format: html, lcov, json

# Test Data
TEST_DATA_SEED=12345                   # Random seed for test data
TEST_CLEANUP_ENABLED=true              # Clean up test data
MOCK_EXTERNAL_APIS=true                # Mock external API calls
TEST_FIXTURES_PATH=/tests/fixtures     # Test fixtures directory
```

## settings.yaml Configuration Reference

The `data/settings.yaml` file contains the primary application configuration with snake_case naming. All values can be overridden by environment variables using the pattern `SECTION_SUBSECTION_KEY`.

### System Configuration

```yaml
system:
  # Network Configuration
  network:
    bind_address: "0.0.0.0"            # Server bind address
    port: 4000                         # Internal API port
    domain: "visionflow.info"          # Primary domain
    enable_http2: false                # HTTP/2 support
    enable_tls: false                  # TLS encryption
    min_tls_version: "1.2"             # Minimum TLS version
    
    # Performance Settings
    max_request_size: 10485760         # Max request size (10MB)
    api_client_timeout: 30             # API client timeout (seconds)
    max_concurrent_requests: 1000      # Concurrent request limit
    max_retries: 3                     # Maximum retry attempts
    retry_delay: 5                     # Retry delay (seconds)
    
    # Rate Limiting
    enable_rate_limiting: true         # Enable rate limiting
    rate_limit_requests: 10000         # Requests per window
    rate_limit_window: 600             # Rate limit window (10 minutes)
    
    # Metrics and Monitoring
    enable_metrics: true               # Enable Prometheus metrics
    metrics_port: 9090                 # Metrics endpoint port
    tunnel_id: "production"            # Cloudflare tunnel ID

  # WebSocket Configuration
  websocket:
    # Connection Settings
    max_connections: 1000              # Maximum concurrent connections
    max_message_size: 10485760         # Maximum message size (10MB)
    heartbeat_interval: 10000          # Heartbeat interval (ms)
    heartbeat_timeout: 60000           # Heartbeat timeout (ms)
    reconnect_attempts: 5              # Auto-reconnect attempts
    reconnect_delay: 1000              # Reconnect delay (ms)
    
    # Binary Protocol Settings
    binary_message_version: 1          # Binary protocol version
    binary_chunk_size: 2048            # Binary data chunk size
    binary_update_rate: 30             # Updates per second
    min_update_rate: 5                 # Minimum update rate
    max_update_rate: 60                # Maximum update rate
    
    # Compression
    compression_enabled: true          # Enable message compression
    compression_threshold: 512         # Compression threshold (bytes)
    compression_level: 6               # Compression level (1-9)
    
    # Motion Tracking
    motion_threshold: 0.05             # Motion detection threshold
    motion_damping: 0.9                # Motion damping factor
    update_rate: 60                    # Update rate (Hz)

  # Security Configuration
  security:
    # CORS Settings
    allowed_origins:
      - "https://www.visionflow.info"
      - "https://visionflow.info"
      - "http://localhost:3001"
    
    # Cookie Security
    cookie_secure: true                # Secure cookies (HTTPS only)
    cookie_httponly: true              # HTTP-only cookies
    cookie_samesite: "Strict"          # SameSite policy
    session_timeout: 3600              # Session timeout (1 hour)
    csrf_token_timeout: 3600           # CSRF token timeout
    
    # Audit and Validation
    enable_audit_logging: true         # Enable audit logging
    audit_log_path: "/app/logs/audit.log" # Audit log file
    enable_request_validation: true    # Validate all requests
    max_login_attempts: 5              # Maximum login attempts
    login_lockout_duration: 900        # Lockout duration (15 minutes)

  # Debug and Development
  debug:
    enabled: false                     # Enable debug mode
    log_level: "info"                  # Debug log level
    enable_profiling: false            # Enable profiling
    profiling_sample_rate: 1000        # Profiling sample rate
    memory_tracking: false             # Track memory usage
    
  # Settings Persistence
  persist_settings: true               # Persist settings changes
  settings_backup_enabled: true       # Backup settings
  settings_backup_interval: 86400     # Backup interval (24 hours)
```

### Visualisation Configuration

```yaml
visualisation:
  # Rendering Settings
  rendering:
    # Lighting Configuration
    ambient_light_intensity: 1.5       # Ambient light strength
    directional_light_intensity: 1.3   # Directional light strength
    environment_intensity: 0.7         # Environment light strength
    background_color: '#0a0e1a'        # Background colour
    
    # Quality Settings
    enable_antialiasing: true          # Anti-aliasing
    enable_shadows: true               # Shadow rendering
    enable_ambient_occlusion: true     # Ambient occlusion
    shadow_map_size: 2048              # Shadow map resolution
    shadow_bias: 0.0001                # Shadow bias
    context: desktop                   # Rendering context: desktop, mobile, vr

  # Animation Settings
  animations:
    enable_node_animations: true       # Node animations
    enable_motion_blur: false          # Motion blur effect
    motion_blur_strength: 0.2          # Motion blur intensity
    
    # Selection Effects
    selection_wave_enabled: true       # Selection wave effect
    pulse_enabled: true                # Node pulse effect
    pulse_speed: 1.2                   # Pulse animation speed
    pulse_strength: 0.8                # Pulse intensity
    wave_speed: 0.5                    # Wave animation speed

  # Bloom Effect Configuration
  bloom:
    enabled: false                     # Enable bloom effect
    strength: 0.34                     # Overall bloom strength
    radius: 0.30                       # Bloom radius
    threshold: 0.41                    # Bloom threshold
    
    # Component-Specific Bloom
    node_bloom_strength: 0.31          # Node bloom intensity
    edge_bloom_strength: 0.37          # Edge bloom intensity
    environment_bloom_strength: 0.37   # Environment bloom intensity

  # Hologram Effects
  hologram:
    # Ring Configuration
    ring_count: 3                      # Number of rings
    ring_color: '#e83211'              # Ring colour
    ring_opacity: 0.23                 # Ring transparency
    ring_rotation_speed: 24.78         # Ring rotation speed
    sphere_sizes: [40.0, 80.0]         # Ring sphere sizes
    
    # Geometric Elements
    enable_buckminster: true           # Buckminster fuller sphere
    buckminster_size: 50.0             # Buckminster sphere size
    buckminster_opacity: 0.3           # Buckminster transparency
    
    enable_geodesic: true              # Geodesic sphere
    geodesic_size: 60.0                # Geodesic sphere size
    geodesic_opacity: 0.25             # Geodesic transparency
    
    enable_triangle_sphere: true       # Triangle sphere
    triangle_sphere_size: 70.0         # Triangle sphere size
    triangle_sphere_opacity: 0.4       # Triangle transparency
    
    global_rotation_speed: 0.5         # Global rotation speed

  # Graph-Specific Configuration
  graphs:
    # Logseq Knowledge Graph
    logseq:
      nodes:
        base_color: '#a06522'          # Default node colour
        metalness: 0.44                # Material metalness (0-1)
        roughness: 0.56                # Material roughness (0-1)
        opacity: 0.51                  # Node transparency
        node_size: 1.87                # Base node size
        quality: high                  # Rendering quality: low, medium, high
        
        # Advanced Features
        enable_instancing: false       # GPU instancing for performance
        enable_hologram: true          # Hologram effects
        enable_metadata_shape: false   # Shape based on metadata
        enable_metadata_visualisation: false # Visualise metadata

      edges:
        colour: '#e8edee'              # Edge colour
        base_width: 1.90               # Base edge width
        opacity: 0.95                  # Edge transparency
        width_range: [0.2, 1.0]        # Width variation range
        quality: medium                # Rendering quality
        
        # Arrow Configuration
        enable_arrows: false           # Show direction arrows
        arrow_size: 0.34               # Arrow size multiplier

      labels:
        enable_labels: true            # Show node labels
        desktop_font_size: 1.01        # Font size for desktop
        text_color: '#5d5d09'          # Text colour
        text_outline_color: '#4561b5'  # Text outline colour
        text_outline_width: 0.003      # Outline width
        text_resolution: 32            # Text texture resolution
        text_padding: 0.6              # Text padding
        billboard_mode: camera         # Billboard mode: camera, screen
        show_metadata: true            # Show metadata in labels
        max_label_width: 5.0           # Maximum label width

      physics:
        # Core Physics Parameters
        enabled: true                  # Enable physics simulation
        iterations: 50                 # Simulation iterations per frame
        dt: 0.016                      # Time step (16ms for 60fps)
        damping: 0.85                  # Velocity damping factor
        max_velocity: 5.0              # Maximum node velocity
        temperature: 0.01              # Simulated annealing temperature
        gravity: 0.0001                # Gravitational force
        
        # Force Parameters
        attraction_k: 8.378            # Edge attraction strength
        repel_k: 2.0                   # Node repulsion strength
        spring_k: 0.1                  # Spring stiffness
        mass_scale: 1.0                # Node mass scaling
        
        # Boundary Configuration
        enable_bounds: true            # Enable boundary constraints
        bounds_size: 5000.0            # Boundary size
        boundary_damping: 0.23         # Boundary damping
        boundary_limit: 44.64          # Boundary force limit
        boundary_margin: 0.85          # Boundary margin factor
        boundary_force_strength: 2.0   # Boundary force strength
        
        # Distance and Collision
        separation_radius: 2.0         # Minimum node separation
        min_distance: 0.45             # Minimum enforced distance
        max_repulsion_dist: 300.0      # Maximum repulsion distance
        update_threshold: 0.01         # Update threshold for optimisation
        
        # GPU-Specific Parameters
        stress_weight: 0.58            # Stress majorization weight
        stress_alpha: 0.55             # Stress alpha parameter
        alignment_strength: 0.0        # Node alignment force
        cluster_strength: 0.0          # Clustering force
        compute_mode: 0                # Compute mode (0=basic)
        
        # Warmup and Cooling
        warmup_iterations: 219         # Initial warmup iterations
        warmup_curve: quadratic        # Warmup curve: linear, quadratic, cubic
        zero_velocity_iterations: 5    # Iterations with zero velocity
        cooling_rate: 0.004            # Temperature cooling rate
        
        # Auto-Balance System
        auto_balance: false            # Enable automatic parameter balancing
        auto_balance_interval_ms: 500  # Balance check interval
        auto_balance_config:
          stability_variance_threshold: 100.0    # Variance threshold for stability
          stability_frame_count: 180             # Frames to check for stability
          clustering_distance_threshold: 20.0    # Distance threshold for clustering
          bouncing_node_percentage: 0.33         # Percentage of bouncing nodes
          boundary_min_distance: 90.0            # Minimum boundary distance
          boundary_max_distance: 100.0           # Maximum boundary distance
          extreme_distance_threshold: 1000.0     # Extreme distance threshold
          explosion_distance_threshold: 10000.0  # Explosion detection threshold
          spreading_distance_threshold: 500.0    # Spreading detection threshold
          oscillation_detection_frames: 10       # Frames for oscillation detection
          oscillation_change_threshold: 5.0      # Change threshold for oscillation
          min_oscillation_changes: 5             # Minimum changes for oscillation
        
        # Clustering Parameters
        clustering_algorithm: none     # Clustering algorithm: none, kmeans, hierarchical
        cluster_count: 5               # Number of clusters
        clustering_resolution: 1.0     # Clustering resolution
        clustering_iterations: 30      # Clustering iterations

    # VisionFlow Agent Graph
    visionflow:
      nodes:
        base_color: '#ff8800'          # Orange colour for agents
        metalness: 0.7                 # High metalness for agents
        opacity: 0.9                   # High opacity
        roughness: 0.3                 # Low roughness for shine
        node_size: 1.5                 # Agent node size
        quality: high                  # High rendering quality
        
        # Agent-Specific Features
        enable_instancing: true        # Use instancing for performance
        enable_hologram: true          # Agent hologram effects
        enable_metadata_shape: true    # Shape indicates agent type
        enable_metadata_visualisation: true # Visualise agent state

      edges:
        colour: '#ffaa00'              # Agent communication colour
        base_width: 0.15               # Thinner base width
        opacity: 0.6                   # Semi-transparent
        width_range: [0.15, 1.5]       # Communication intensity range
        quality: high                  # High quality for agents
        enable_arrows: true            # Show communication direction
        arrow_size: 0.7                # Larger arrows for visibility

      labels:
        enable_labels: true            # Show agent labels
        desktop_font_size: 16.0        # Larger font for agents
        text_color: '#ffaa00'          # Orange text colour
        text_outline_color: '#000000'  # Black outline
        text_outline_width: 2.5        # Thick outline
        text_resolution: 256           # High resolution text
        text_padding: 5.0              # Generous padding
        billboard_mode: screen         # Screen-aligned billboards
        show_metadata: true            # Show agent metadata
        max_label_width: 8.0           # Wide labels for agent names

      physics:
        # Agent Physics (More Dynamic)
        enabled: true                  # Enable physics
        iterations: 100                # More iterations for stability
        dt: 0.016                      # Standard time step
        damping: 0.95                  # High damping for smooth movement
        max_velocity: 1.0              # Conservative velocity limit
        temperature: 0.01              # Low temperature for stability
        gravity: 0.0001                # Minimal gravity
        
        # Agent Force Parameters
        attraction_k: 0.0001           # Weak attraction between agents
        repel_k: 5.0                   # Strong repulsion to prevent overlap
        spring_k: 0.5                  # Strong springs for clear communication
        mass_scale: 1.0                # Standard mass
        
        # Agent Boundary Configuration
        enable_bounds: true            # Keep agents in view
        bounds_size: 5000.0            # Large boundary for agent movement
        boundary_damping: 0.95         # High boundary damping
        boundary_limit: 490.0          # Boundary limit
        boundary_margin: 0.85          # Boundary margin
        boundary_force_strength: 2.0   # Boundary force
        
        # Agent Spacing
        separation_radius: 2.0         # Agent separation
        min_distance: 0.15             # Minimum agent distance
        max_repulsion_dist: 300.0      # Repulsion range
        update_threshold: 0.01         # Update threshold
        
        # Agent GPU Parameters
        stress_weight: 0.1             # Lower stress weight for agents
        stress_alpha: 0.1              # Lower stress alpha
        alignment_strength: 0.0        # No alignment for agents
        cluster_strength: 0.0          # No artificial clustering
        compute_mode: 0                # Basic compute mode
        
        # Agent Warmup
        warmup_iterations: 200         # Extended warmup for agents
        warmup_curve: quadratic        # Quadratic warmup
        zero_velocity_iterations: 5    # Zero velocity iterations
        cooling_rate: 0.0001           # Slow cooling for stability
        
        # Agent Auto-Balance (Disabled)
        auto_balance: false            # Disable auto-balance for agents
        auto_balance_interval_ms: 500  # Balance interval
        auto_balance_config:
          stability_variance_threshold: 100.0
          stability_frame_count: 180
          clustering_distance_threshold: 20.0
          bouncing_node_percentage: 0.33
          boundary_min_distance: 90.0
          boundary_max_distance: 100.0
          extreme_distance_threshold: 1000.0
          explosion_distance_threshold: 10000.0
          spreading_distance_threshold: 500.0
          oscillation_detection_frames: 10
          oscillation_change_threshold: 5.0
          min_oscillation_changes: 5
        
        # No Clustering for Agents
        clustering_algorithm: none     # No clustering
        cluster_count: 5               # Default cluster count
        clustering_resolution: 1.0     # Default resolution
        clustering_iterations: 30      # Default iterations
```

### XR Configuration

```yaml
xr:
  # Core XR Settings
  enabled: false                       # Enable XR features
  client_side_enable_xr: false         # Client-side XR toggle
  mode: "immersive-vr"                 # XR mode: immersive-vr, immersive-ar, inline
  space_type: "local-floor"            # Reference space: viewer, local, local-floor, bounded-floor
  room_scale: 1.0                      # Room scale factor
  quality: medium                      # Rendering quality: low, medium, high
  render_scale: 0.92                   # Render scale factor (0.1-2.0)

  # Interaction Configuration
  interaction_distance: 1.5            # Maximum interaction distance (metres)
  locomotion_method: teleport          # Locomotion: teleport, smooth, room-scale
  teleport_ray_color: '#ffffff'        # Teleport ray colour
  controller_ray_color: '#ffffff'      # Controller ray colour
  controller_model: default            # Controller model: default, custom

  # Hand Tracking
  enable_hand_tracking: true           # Enable hand tracking
  hand_mesh_enabled: true              # Show hand mesh
  hand_mesh_color: '#4287f5'           # Hand mesh colour
  hand_mesh_opacity: 0.3               # Hand mesh transparency
  hand_point_size: 0.006               # Hand point size
  hand_ray_enabled: true               # Enable hand rays
  hand_ray_color: '#4287f5'            # Hand ray colour
  hand_ray_width: 0.003                # Hand ray width
  gesture_smoothing: 0.7               # Gesture smoothing factor

  # Haptic Feedback
  enable_haptics: true                 # Enable haptic feedback
  haptic_intensity: 0.3                # Haptic intensity (0.0-1.0)
  haptic_duration: 100                 # Haptic duration (ms)

  # Interaction Thresholds
  drag_threshold: 0.08                 # Drag gesture threshold
  pinch_threshold: 0.3                 # Pinch gesture threshold
  rotation_threshold: 0.08             # Rotation gesture threshold
  interaction_radius: 0.15             # Interaction sphere radius

  # Movement Configuration
  movement_speed: 1.0                  # Movement speed multiplier
  dead_zone: 0.12                      # Controller dead zone
  movement_axes:
    horizontal: 2                      # Horizontal axis index
    vertical: 3                        # Vertical axis index

  # Environment Understanding
  enable_light_estimation: false       # Estimate environment lighting
  enable_plane_detection: false        # Detect real-world planes
  enable_scene_understanding: false    # Advanced scene understanding
  plane_color: '#4287f5'               # Detected plane colour
  plane_opacity: 0.001                 # Plane overlay opacity
  plane_detection_distance: 3.0        # Plane detection distance (metres)
  show_plane_overlay: false            # Show plane overlay
  snap_to_floor: false                 # Snap objects to floor

  # Passthrough Portal
  enable_passthrough_portal: false     # Enable passthrough view
  passthrough_opacity: 0.8             # Passthrough transparency
  passthrough_brightness: 1.1          # Brightness adjustment
  passthrough_contrast: 1.2            # Contrast adjustment
  portal_size: 2.5                     # Portal size (metres)
  portal_edge_color: '#4287f5'         # Portal edge colour
  portal_edge_width: 0.02              # Portal edge width
```

### Authentication Configuration

```yaml
auth:
  # Authentication System
  enabled: true                        # Enable authentication
  provider: nostr                      # Auth provider: nostr, oauth, jwt
  required: true                       # Require authentication for access
  
  # Session Management
  session_duration: 86400              # Session duration (24 hours)
  refresh_token_duration: 604800       # Refresh token duration (7 days)
  
  # Nostr Configuration
  nostr:
    relay_urls:                        # Nostr relay servers
      - "wss://relay.damus.io"
      - "wss://nos.lol"
      - "wss://relay.snort.social"
    event_kinds: [1, 30023]            # Supported event kinds
    max_event_size: 65536              # Maximum event size (64KB)
    
  # OAuth Configuration (if enabled)
  oauth:
    client_id: your_oauth_client_id
    client_secret: your_oauth_secret
    redirect_uri: http://localhost:3001/auth/callback
    scope: "read write"
```

### AI Service Configuration

```yaml
# RAGFlow Integration
ragflow:
  agent_id: aa2e328812ef11f083dc0a0d6226f61b  # Default agent ID
  timeout: 30                          # Request timeout (seconds)
  max_retries: 3                       # Maximum retry attempts
  retry_delay: 1                       # Retry delay (seconds)
  
  # Processing Configuration
  chunk_size: 512                      # Text chunk size
  chunk_overlap: 50                    # Chunk overlap tokens
  max_chunks: 100                      # Maximum chunks per document

# Perplexity Configuration
perplexity:
  model: "llama-3.1-sonar-small-128k-online"  # Default model
  max_tokens: 4096                     # Maximum response tokens
  temperature: 0.5                     # Response creativity
  top_p: 0.9                          # Nucleus sampling
  presence_penalty: 0.0                # Presence penalty
  frequency_penalty: 0.0               # Frequency penalty
  timeout: 30                          # Request timeout
  rate_limit: 100                      # Requests per hour

# OpenAI Configuration
openai:
  model: "gpt-4o"                      # Default model
  max_tokens: 4096                     # Maximum tokens
  temperature: 0.7                     # Temperature setting
  timeout: 30                          # Request timeout
  rate_limit: 100                      # Requests per hour

# Kokoro TTS Configuration
kokoro:
  api_url: "http://kokoro:8880"        # Kokoro API endpoint
  default_voice: af_heart              # Default voice
  default_format: mp3                  # Audio format
  default_speed: 1.0                   # Speech speed
  timeout: 30                          # Request timeout
  stream: true                         # Enable streaming
  return_timestamps: true              # Return timing data
  sample_rate: 24000                   # Audio sample rate

# Whisper Configuration
whisper:
  api_url: "http://whisper:8000"       # Whisper API endpoint
  default_model: base                  # Model size
  default_language: en                 # Language code
  timeout: 30                          # Processing timeout
  temperature: 0.0                     # Sampling temperature
  return_timestamps: false             # Return timestamps
  vad_filter: false                    # Voice activity detection
  word_timestamps: false               # Word-level timestamps
```

## Docker Compose Configuration Reference

### Service Definitions

#### Core Services

```yaml
services:
  # Main VisionFlow Application
  webxr:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        RUST_VERSION: "1.75"
        NODE_VERSION: "20"
    ports:
      - "${HOST_PORT:-3001}:4000"
    environment:
      # Pass all environment variables
      - RUST_LOG=${RUST_LOG:-info}
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    volumes:
      - ./data:/app/data:cached
      - ./logs:/app/logs:cached
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Claude Flow Multi-Agent System
  claude-flow:
    build:
      context: ./claude-flow
      dockerfile: Dockerfile
    ports:
      - "${MCP_TCP_PORT:-9500}:9500"
    environment:
      - MCP_TRANSPORT=${MCP_TRANSPORT:-tcp}
      - MAX_AGENTS=${MAX_AGENTS:-20}
      - ENABLE_GPU=${ENABLE_GPU:-false}
    volumes:
      - ./claude-flow/data:/app/data
      - ./claude-flow/logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import socket; socket.create_connection(('localhost', 9500))"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL Database
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-visionflow}
      POSTGRES_USER: ${POSTGRES_USER:-visionflow}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secure_password}
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=C"
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init:/docker-entrypoint-initdb.d:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-visionflow}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory ${REDIS_MAX_MEMORY:-512mb} --maxmemory-policy allkeys-lru
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
```

#### GPU-Accelerated Services

```yaml
# docker-compose.gpu.yml
services:
  webxr-gpu:
    extends:
      file: docker-compose.yml
      service: webxr
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: ${NVIDIA_GPU_COUNT:-1}
              capabilities: [gpu]
    environment:
      - ENABLE_GPU=true
      - CUDA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-0}
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Volume Configuration

```yaml
volumes:
  # Application Data
  app_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}

  # Database Storage
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${POSTGRES_DATA_PATH:-./data/postgres}

  # Cache Storage
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${REDIS_DATA_PATH:-./data/redis}

  # Log Storage
  log_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${LOG_PATH:-./logs}

  # High-Performance SSD Volume
  nvme_data:
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
      o: size=${GPU_SHARED_MEMORY_SIZE:-2g},uid=1000,gid=1000
```

### Network Configuration

```yaml
networks:
  # Default Network
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1

  # Internal Services Network
  internal:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.21.0.0/16

  # External Access Network
  external:
    driver: bridge
    ipam:
      config:
        - subnet: 172.22.0.0/16

  # High-Performance Network
  gpu_network:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 9000
    ipam:
      config:
        - subnet: 172.23.0.0/16
```

### Production Configuration

```yaml
# docker-compose.prod.yml
services:
  webxr-prod:
    extends:
      file: docker-compose.yml
      service: webxr
    build:
      dockerfile: Dockerfile.production
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '${CPU_LIMIT:-8.0}'
          memory: ${MEMORY_LIMIT:-16g}
        reservations:
          cpus: '${CPU_RESERVATION:-4.0}'
          memory: ${MEMORY_RESERVATION:-8g}
      restart_policy:
        condition: any
        delay: 5s
        max_attempts: 3
        window: 120s
      update_config:
        parallelism: 1
        delay: 30s
        failure_action: rollback
        order: start-first

  # Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - webxr-prod
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
```

## Cloudflare Configuration Reference

### Tunnel Configuration (`config.yml`)

```yaml
# Cloudflare Tunnel Configuration
tunnel: ${CLOUDFLARE_TUNNEL_ID}
credentials-file: /etc/cloudflared/credentials.json

# Ingress Rules
ingress:
  # Main Application
  - hostname: ${DOMAIN:-visionflow.info}
    service: http://webxr:4000
    originRequest:
      connectTimeout: 30s
      tlsTimeout: 10s
      tcpKeepAlive: 30s
      noHappyEyeballs: false
      keepAliveTimeout: 90s
      httpHostHeader: ${DOMAIN:-visionflow.info}

  # WebSocket Endpoint
  - hostname: ${DOMAIN:-visionflow.info}
    path: /ws/*
    service: http://webxr:4000
    originRequest:
      noTLSVerify: false

  # API Subdomain
  - hostname: api.${DOMAIN:-visionflow.info}
    service: http://webxr:4000
    originRequest:
      httpHostHeader: api.${DOMAIN:-visionflow.info}

  # Claude Flow MCP
  - hostname: mcp.${DOMAIN:-visionflow.info}
    service: tcp://claude-flow:9500

  # Monitoring
  - hostname: metrics.${DOMAIN:-visionflow.info}
    service: http://prometheus:9090
    originRequest:
      httpHostHeader: metrics.${DOMAIN:-visionflow.info}

  # Grafana Dashboard
  - hostname: dashboard.${DOMAIN:-visionflow.info}
    service: http://grafana:3000

  # Catch-all rule
  - service: http_status:404

# Logging
loglevel: info
logfile: /var/log/cloudflared/cloudflared.log

# Retry Configuration
retries: 3
grace-period: 30s

# Tunnel Management
management:
  enable: true
  diagnostics: true
```

## Supervisor Configuration Reference

### Development Configuration (`supervisord.dev.conf`)

```ini
[supervisord]
nodaemon=true
logfile=/dev/stdout
logfile_maxbytes=0
loglevel=info

[program:rust-watch]
command=cargo watch -x run
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=RUST_LOG="%(ENV_RUST_LOG)s",DEBUG_MODE="true"

[program:vite-dev]
command=npm run dev
directory=/app/client
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:claude-flow]
command=python -m claude_flow.server
directory=/app/claude-flow
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=MCP_TRANSPORT="%(ENV_MCP_TRANSPORT)s",MAX_AGENTS="%(ENV_MAX_AGENTS)s"

[group:dev-services]
programs=rust-watch,vite-dev,claude-flow
priority=999
```

### Production Configuration (`supervisord.conf`)

```ini
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
childlogdir=/var/log/supervisor/
loglevel=warn

[program:nginx]
command=nginx -g "daemon off;"
autostart=true
autorestart=true
startretries=3
stdout_logfile=/var/log/nginx/access.log
stderr_logfile=/var/log/nginx/error.log

[program:webxr]
command=/app/target/release/visionflow
directory=/app
user=visionflow
autostart=true
autorestart=true
startretries=3
stdout_logfile=/var/log/visionflow/app.log
stderr_logfile=/var/log/visionflow/error.log
environment=RUST_LOG="%(ENV_RUST_LOG)s"

[program:claude-flow]
command=/app/claude-flow/venv/bin/python -m claude_flow.server
directory=/app/claude-flow
user=claude-flow
autostart=true
autorestart=true
startretries=3
stdout_logfile=/var/log/claude-flow/app.log
stderr_logfile=/var/log/claude-flow/error.log

[group:production-services]
programs=nginx,webxr,claude-flow
priority=999
```

## Configuration Validation

### Environment Variable Validation

```bash
#!/bin/bash
# validate-env.sh

set -euo pipefail

# Required variables
REQUIRED_VARS=(
    "CLAUDE_FLOW_HOST"
    "MCP_TCP_PORT"
    "JWT_SECRET"
    "POSTGRES_PASSWORD"
)

# Optional variables with defaults
declare -A DEFAULT_VARS=(
    ["HOST_PORT"]="3001"
    ["INTERNAL_API_PORT"]="4000"
    ["MEMORY_LIMIT"]="16g"
    ["CPU_LIMIT"]="8.0"
    ["MAX_AGENTS"]="20"
)

echo "🔍 Validating VisionFlow configuration..."

# Check required variables
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "❌ ERROR: Required variable $var is not set"
        exit 1
    fi
    echo "✅ $var is set"
done

# Validate JWT secret length
if [[ ${#JWT_SECRET} -lt 32 ]]; then
    echo "❌ ERROR: JWT_SECRET must be at least 32 characters long"
    exit 1
fi

# Validate port ranges
if [[ "${HOST_PORT:-3001}" -lt 1024 ]] || [[ "${HOST_PORT:-3001}" -gt 65535 ]]; then
    echo "❌ ERROR: HOST_PORT must be between 1024 and 65535"
    exit 1
fi

# Validate memory format
if [[ ! "${MEMORY_LIMIT:-16g}" =~ ^[0-9]+[gGmM]$ ]]; then
    echo "❌ ERROR: MEMORY_LIMIT must be in format like '16g' or '1024m'"
    exit 1
fi

# Validate CPU limit format
if [[ ! "${CPU_LIMIT:-8.0}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "❌ ERROR: CPU_LIMIT must be a number like '8.0'"
    exit 1
fi

# Check GPU configuration
if [[ "${ENABLE_GPU:-false}" == "true" ]]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️  WARNING: GPU enabled but nvidia-smi not found"
    else
        echo "✅ GPU configuration validated"
    fi
fi

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker is not installed"
    exit 1
fi

# Check Docker Compose availability
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ ERROR: Docker Compose is not available"
    exit 1
fi

echo "✅ Configuration validation completed successfully!"
```

### YAML Configuration Validation

```bash
#!/bin/bash
# validate-yaml.sh

set -euo pipefail

SETTINGS_FILE="${1:-./data/settings.yaml}"

echo "🔍 Validating YAML configuration..."

# Check if file exists
if [[ ! -f "$SETTINGS_FILE" ]]; then
    echo "❌ ERROR: Settings file $SETTINGS_FILE not found"
    exit 1
fi

# Validate YAML syntax
if ! python3 -c "import yaml; yaml.safe_load(open('$SETTINGS_FILE'))" 2>/dev/null; then
    echo "❌ ERROR: Invalid YAML syntax in $SETTINGS_FILE"
    exit 1
fi

# Validate required sections
REQUIRED_SECTIONS=(
    "system"
    "visualisation"
    "auth"
)

for section in "${REQUIRED_SECTIONS[@]}"; do
    if ! python3 -c "import yaml; config = yaml.safe_load(open('$SETTINGS_FILE')); assert '$section' in config" 2>/dev/null; then
        echo "❌ ERROR: Required section '$section' missing from $SETTINGS_FILE"
        exit 1
    fi
    echo "✅ Section '$section' found"
done

echo "✅ YAML validation completed successfully!"
```

### Docker Configuration Testing

```bash
#!/bin/bash
# test-docker-config.sh

set -euo pipefail

echo "🐳 Testing Docker configuration..."

# Test standard configuration
echo "Testing standard Docker Compose configuration..."
if docker-compose config > /dev/null 2>&1; then
    echo "✅ Standard configuration is valid"
else
    echo "❌ ERROR: Standard configuration is invalid"
    exit 1
fi

# Test GPU configuration
if [[ -f "docker-compose.gpu.yml" ]]; then
    echo "Testing GPU configuration..."
    if docker-compose -f docker-compose.yml -f docker-compose.gpu.yml config > /dev/null 2>&1; then
        echo "✅ GPU configuration is valid"
    else
        echo "❌ ERROR: GPU configuration is invalid"
        exit 1
    fi
fi

# Test production configuration
if [[ -f "docker-compose.prod.yml" ]]; then
    echo "Testing production configuration..."
    if docker-compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null 2>&1; then
        echo "✅ Production configuration is valid"
    else
        echo "❌ ERROR: Production configuration is invalid"
        exit 1
    fi
fi

echo "✅ Docker configuration testing completed successfully!"
```

## Migration Guide

### Configuration Version Migration

```bash
#!/bin/bash
# migrate-config.sh

set -euo pipefail

BACKUP_DIR="./config-backups/$(date +%Y%m%d_%H%M%S)"
SETTINGS_FILE="./data/settings.yaml"

echo "🔄 Migrating VisionFlow configuration..."

# Create backup
mkdir -p "$BACKUP_DIR"
cp "$SETTINGS_FILE" "$BACKUP_DIR/settings.yaml.backup"
cp .env "$BACKUP_DIR/.env.backup" 2>/dev/null || true

echo "📦 Configuration backed up to $BACKUP_DIR"

# Migration functions
migrate_v1_to_v2() {
    echo "🔄 Migrating from v1.0 to v2.0..."
    
    # Rename old physics parameters
    python3 << 'EOF'
import yaml
import sys

with open('./data/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Migrate physics parameters
if 'physics' in config.get('visualisation', {}):
    old_physics = config['visualisation']['physics']
    
    # Create new structure
    if 'graphs' not in config['visualisation']:
        config['visualisation']['graphs'] = {}
    
    if 'logseq' not in config['visualisation']['graphs']:
        config['visualisation']['graphs']['logseq'] = {}
    
    config['visualisation']['graphs']['logseq']['physics'] = old_physics
    
    # Remove old structure
    del config['visualisation']['physics']
    
    print("✅ Migrated physics configuration to new structure")

# Add new required fields
config['version'] = '2.0.0'

with open('./data/settings.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print("✅ Migration completed successfully")
EOF
}

migrate_v2_to_v3() {
    echo "🔄 Migrating from v2.0 to v3.0..."
    
    # Add XR configuration
    python3 << 'EOF'
import yaml

with open('./data/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Add XR section if missing
if 'xr' not in config:
    config['xr'] = {
        'enabled': False,
        'mode': 'immersive-vr',
        'quality': 'medium',
        'render_scale': 0.92
    }
    print("✅ Added XR configuration section")

config['version'] = '3.0.0'

with open('./data/settings.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print("✅ Migration completed successfully")
EOF
}

# Detect current version and migrate
CURRENT_VERSION=$(python3 -c "import yaml; config = yaml.safe_load(open('./data/settings.yaml')); print(config.get('version', '1.0.0'))" 2>/dev/null || echo "1.0.0")

echo "📋 Current configuration version: $CURRENT_VERSION"

case "$CURRENT_VERSION" in
    "1.0.0")
        migrate_v1_to_v2
        migrate_v2_to_v3
        ;;
    "2.0.0")
        migrate_v2_to_v3
        ;;
    "3.0.0")
        echo "✅ Configuration is already up to date"
        ;;
    *)
        echo "❌ ERROR: Unknown configuration version $CURRENT_VERSION"
        exit 1
        ;;
esac

echo "✅ Configuration migration completed successfully!"
echo "💾 Backup available at: $BACKUP_DIR"
```

---

This comprehensive configuration reference provides complete technical documentation for all aspects of VisionFlow's configuration system. Use this reference for advanced configuration scenarios, troubleshooting, and system administration.

## Related Topics

- [Configuration Guide](../guides/configuration.md)
- [System Architecture](../architecture/system-overview.md)
- [API Reference](../api/index.md)
- [Deployment Guide](../deployment/index.md)
- [Troubleshooting](../troubleshooting/configuration.md)