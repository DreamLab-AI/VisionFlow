# Scripts Reference

Comprehensive documentation for all utility scripts in the multi-agent-docker system.

## Table of Contents

- [Main Startup Scripts](#main-startup-scripts)
- [Initialization Scripts](#initialization-scripts)
- [MCP Management](#mcp-management)
- [Testing Scripts](#testing-scripts)
- [Health Monitoring](#health-monitoring)
- [Session Management](#session-management)
- [Security and Validation](#security-and-validation)
- [Utility Scripts](#utility-scripts)
- [Helper Scripts](#helper-scripts)

---

## Main Startup Scripts

### start-agentic-flow.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/start-agentic-flow.sh`

**Purpose:** Main entry point for starting the Agentic Flow + Z.AI combined Docker system with RAGFlow network integration.

**Usage:**
```bash
./start-agentic-flow.sh [options]
```

**Options:**
- `--build` - Force rebuild containers (no-cache)
- `--no-ragflow` - Skip RAGFlow network connection
- `--stop` - Stop all services
- `--restart` - Restart all services
- `--logs` - Show container logs (tail 100, follow)
- `--status` - Show service status and health checks
- `--shell` - Open interactive shell in container
- `--clean` - Clean up Docker resources (containers, images, volumes)
- `--test` - Run validation test suite
- `-h, --help` - Show help message

**Services Managed:**
- `agentic-flow-cachyos` - Main orchestration container (port 9090)
- `claude-zai-service` - Z.AI semantic processing (port 9600)

**Networks:**
- `agentic-network` - Internal bridge network
- `docker_ragflow` - RAGFlow integration (optional)

**Examples:**
```bash
# First time setup
./start-agentic-flow.sh --build

# Regular startup
./start-agentic-flow.sh

# Check status
./start-agentic-flow.sh --status

# Open shell
./start-agentic-flow.sh --shell

# Clean everything
./start-agentic-flow.sh --clean

# View logs
./start-agentic-flow.sh --logs

# Restart services
./start-agentic-flow.sh --restart
```

**Environment Requirements:**
- Docker and docker-compose installed
- `.env` file configured (created from `.env.example` if missing)
- API keys: ZAI_API_KEY, GOOGLE_API_KEY (optional but recommended)

**Health Checks:**
- Management API: http://localhost:9090/health
- Claude-ZAI API: http://localhost:9600/health

**Return Codes:**
- `0` - Success
- `1` - Error occurred

---

## Initialization Scripts

### init-workstation.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/init-workstation.sh`

**Purpose:** Initializes the Agentic Flow CachyOS workstation environment on container startup. Sets up APIs, MCP servers, workspace directories, and performs connectivity tests.

**Usage:**
```bash
# Run automatically on container startup
./init-workstation.sh

# Also invoked as entrypoint
exec /path/to/init-workstation.sh
```

**Initialization Steps:**

1. **Environment Check**
   - Validates API keys (ANTHROPIC, OPENAI, GOOGLE_GEMINI, OPENROUTER, E2B)
   - Displays which keys are configured

2. **Management API**
   - Starts Management API on port 9090 using pm2
   - Creates log directory structure
   - Configures pm2 for auto-restart

3. **MCP Servers** (if `MCP_AUTO_START=true`)
   - Starts MCP servers in background after 5s delay
   - Logs to `/tmp/mcp-startup.log`

4. **Network Connectivity**
   - Tests Xinference at 172.18.0.11:9997
   - Validates external API endpoints (Anthropic, OpenAI, Google)
   - Reports available models

5. **GPU Check**
   - Detects NVIDIA GPUs (nvidia-smi)
   - Detects AMD GPUs (rocm-smi)
   - Reports GPU name, memory, CUDA version
   - Falls back to CPU-only mode if no GPU

6. **Workspace Setup**
   - Creates workspace directories: projects, temp, agents
   - Sets up model storage
   - Creates memory and metrics directories
   - Configures git if not present

7. **Verification**
   - Verifies agentic-flow installation
   - Counts available agents
   - Creates helpful symlinks

**Environment Variables:**
- `MCP_AUTO_START` - Enable automatic MCP server startup (default: false)
- `ENABLE_XINFERENCE` - Enable Xinference connectivity tests (default: true)
- `GPU_ACCELERATION` - Enable GPU detection (default: true)
- `WORKSPACE` - Workspace root path (default: /workspace)

**Output:**
- Displays colorized status for each step
- Lists quick start commands
- Shows documentation locations

**Return Codes:**
- Exec's into `/usr/bin/zsh` to keep container running

---

## MCP Management

### mcp-cli.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/mcp-cli.sh`

**Purpose:** Command-line interface for managing MCP (Model Context Protocol) tools. Simplifies adding, removing, and configuring MCP servers.

**Configuration File:** `/home/devuser/.config/claude/mcp.json`

**Usage:**
```bash
mcp <command> [arguments]
```

**Commands:**

#### list
List all configured MCP tools with details.
```bash
mcp list
```
Output: Tool name, command, type, category

#### show
Show detailed information about a specific tool.
```bash
mcp show <tool_name>
```

#### add
Add a new MCP tool to configuration.
```bash
mcp add <name> <command> [args] [description] [env_vars] [category]
```

Examples:
```bash
# Simple NPM package
mcp add weather npx "-y @modelcontextprotocol/server-weather" "Weather data"

# With environment variable
mcp add github npx "-y @modelcontextprotocol/server-github" "GitHub API" '{"GITHUB_TOKEN":"${GITHUB_TOKEN}"}'

# Custom Python script
mcp add custom-tool /opt/venv/bin/python3 "-u /app/tools/my-tool.py" "Custom tool"
```

#### remove
Remove an MCP tool from configuration.
```bash
mcp remove <tool_name>
# Alias: mcp rm <tool_name>
```

#### update
Update tool configuration properties.
```bash
mcp update <tool_name> [--command <cmd>] [--args <args>] [--description <desc>]
```

Examples:
```bash
mcp update github --description "GitHub API integration"
mcp update weather --command "node" --args "/path/to/weather-server.js"
```

#### validate
Validate MCP configuration JSON syntax and required fields.
```bash
mcp validate
```

Checks:
- JSON syntax validity
- Required fields: command, type
- Reports all validation errors

#### backup
Create timestamped backup of configuration.
```bash
mcp backup
```
Backup location: `/home/devuser/.config/claude/backups/mcp-YYYYMMDD_HHMMSS.json`

#### restore
Restore configuration from backup file.
```bash
mcp restore <backup_file>
```
Creates safety backup before restoring.

**Tool Categories:**
- `core` - Essential MCP servers
- `development` - Development tools
- `data` - Data processing tools
- `other` - Uncategorized tools

**Return Codes:**
- `0` - Success
- `1` - Error (tool not found, invalid syntax, etc.)

---

## Testing Scripts

### test-all-providers.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/test-all-providers.sh`

**Purpose:** Comprehensive testing of all AI model providers for connectivity and basic functionality.

**Usage:**
```bash
./test-all-providers.sh
```

**Test Configuration:**
- Test task: "Write a Python function that prints 'Hello World'"
- Max tokens: 50 (quick validation)

**Providers Tested:**

1. **Google Gemini** - gemini-2.5-flash
   - Requires: GOOGLE_GEMINI_API_KEY
2. **OpenAI** - gpt-4o
   - Requires: OPENAI_API_KEY
3. **Anthropic Claude** - claude-3-5-sonnet-20241022
   - Requires: ANTHROPIC_API_KEY
4. **OpenRouter** - meta-llama/llama-3.1-8b-instruct
   - Requires: OPENROUTER_API_KEY
5. **Xinference (Local)** - auto model selection
   - Requires: Network connectivity to 172.18.0.11:9997
6. **ONNX (Offline)** - phi-4.onnx
   - Requires: Model at `/home/devuser/models/phi-4.onnx`
7. **Intelligent Router** - Auto-selects best model
   - Tests routing with performance priority

**Pre-Tests:**
- GPU availability check (NVIDIA/AMD/CPU-only)
- Xinference connectivity test
- Available models enumeration

**Output:**
- Colorized test results (pass/fail/skip)
- Test summary with counts
- Performance metrics

**Return Codes:**
- `0` - All available providers passed
- `1` - One or more tests failed

**Notes:**
- Skips tests when required API keys are missing
- Shows first 20 lines of output per test
- Recommends `npx agentic-flow --provider onnx --download-model` for ONNX setup

---

### test-gemini-flow.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/test-gemini-flow.sh`

**Purpose:** Validates Gemini-Flow production AI orchestration system with A2A (Agent-to-Agent) + MCP protocol support.

**Usage:**
```bash
./test-gemini-flow.sh
```

**Test Suites:**

#### 1. Installation Check
- Verifies `gemini-flow` command availability
- Reports installed version

#### 2. Configuration Check
- Validates `~/.gemini-flow/production.config.ts`
- Checks environment variables:
  - GOOGLE_API_KEY
  - GOOGLE_CLOUD_PROJECT
  - GEMINI_FLOW_ENABLED
  - GEMINI_FLOW_PROTOCOLS

#### 3. Protocol Support
- Tests A2A (Agent-to-Agent) protocol availability
- Tests MCP (Model Context Protocol) support
- Validates protocol flags in help output

#### 4. Agent Spawn Test
- Spawns 3-agent test swarm
- Uses protocols: a2a, mcp
- Runs with --dry-run flag
- Timeout: 30 seconds

#### 5. Google AI Services Integration
Available services:
- Veo3 - Video Generation
- Imagen4 - Image Creation
- Lyria - Music Composition
- Chirp - Speech Synthesis
- Co-Scientist - Research Automation
- Project Mariner - Browser Automation
- AgentSpace - Agent Coordination
- Multi-modal Streaming

#### 6. Performance Targets
Core System:
- SQLite Operations: 396,610 ops/sec (target: 300K)
- Agent Spawn Time: <100ms (target: <180ms)
- Routing Latency: <75ms (target: <100ms)
- Memory per Agent: 4.2MB (target: 7.1MB)
- Parallel Tasks: 10,000 concurrent

A2A Protocol:
- Agent-to-Agent Latency: <25ms (target: <50ms)
- Consensus Speed: 2.4s for 1000 nodes
- Message Throughput: 50,000 msgs/sec
- Fault Recovery: <500ms

#### 7. Integration Test
- Verifies both agentic-flow and gemini-flow installed
- Confirms simultaneous framework operation

#### 8. 66-Agent Swarm Capability
Agent specializations:
- System Architects: 5
- Master Coders: 12
- Research Scientists: 8
- Data Analysts: 10
- Strategic Planners: 6
- Security Experts: 5
- Performance Optimizers: 8
- Documentation Writers: 4
- QA Specialists: 4
- DevOps Engineers: 4

**Next Steps (on success):**
```bash
gf-init                          # Initialize with protocols
gf-swarm                         # Deploy 66-agent swarm
gf-deploy 'your-objective' 20    # Custom swarm deployment
gf-monitor                       # Monitor A2A protocols
```

**Return Codes:**
- `0` - Ready for production
- `1` - Configuration issues detected

---

## Health Monitoring

### healthcheck.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/healthcheck.sh`

**Purpose:** Simple Docker container health check for monitoring.

**Usage:**
```bash
./healthcheck.sh
```

**Checks Performed:**
1. API endpoint at http://localhost:8080/health
2. Fallback: agentic-flow command availability

**Return Codes:**
- `0` - Container healthy
- `1` - Container unhealthy

**Docker Integration:**
```yaml
healthcheck:
  test: ["/scripts/healthcheck.sh"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

### healthcheck-db.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/healthcheck-db.sh`

**Purpose:** Database health monitoring for SQLite locks, contention, and MCP process status.

**Usage:**
```bash
./healthcheck-db.sh
```

**Health Levels:**
- `0` - DB_HEALTH_OK (all checks passed)
- `1` - DB_HEALTH_WARN (warnings present)
- `2` - DB_HEALTH_FAIL (critical errors)

**Checks Performed:**

#### Database Locks
- **Stale WAL files**: Checks for *.db-wal files older than 30 minutes
  - Warning: >5 stale WAL files
- **Running sessions**: Reads `/workspace/.swarm/sessions/index.json`
  - Warning: >10 running sessions
  - Error: >20 running sessions
- **Database locks**: Tests quick_check on `/workspace/.swarm/memory.db`
  - Error: Lock timeout or corruption

#### MCP Processes
- **TCP server**: Verifies `mcp-tcp-server.js` process running
  - Error: Process not found
- **Port connectivity**: Tests TCP connection to localhost:9500
  - Warning: Port not responding

**Output:**
```
Database health: OK|WARNING|FAILED
```

**Return Codes:**
- `0` - Healthy
- `1` - Warning level
- `2` - Failed

**Integration:**
Run via cron or supervisord for continuous monitoring.

---

## Session Management

### hive-session-manager.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/hive-session-manager.sh`

**Purpose:** Manages isolated hive-mind sessions with UUID tracking for external coordination. Prevents SQLite lock conflicts between concurrent spawns.

**Session Infrastructure:**
- Sessions root: `/workspace/.swarm/sessions/`
- Sessions index: `/workspace/.swarm/sessions/index.json`
- Output directory: `/workspace/ext/hive-sessions/<UUID>/`

**Commands:**

#### create
Create new session and return UUID.
```bash
hive-session-manager.sh create <task> [priority] [metadata_json]
```

Priority levels: low, medium, high
Returns: Session UUID
Output: Session metadata JSON to stderr

Session structure:
```
/workspace/.swarm/sessions/<UUID>/
├── session.json          # Metadata
├── .swarm/memory.db      # Isolated database
└── output/               # Symlink to ext output
```

#### start
Start hive-mind spawn in isolated session.
```bash
hive-session-manager.sh start <uuid> [hive-mind args...]
```

Process:
1. Updates status to "starting"
2. Reads session metadata
3. Creates log file
4. Executes as dev user in session directory
5. Updates status to "completed" or "failed"
6. Closes database connections

#### status
Get current session status.
```bash
hive-session-manager.sh status <uuid>
```

Status values: created, starting, running, completed, failed, not_found

#### get
Get full session metadata JSON.
```bash
hive-session-manager.sh get <uuid>
```

Returns session.json contents

#### list
List all sessions from index.
```bash
hive-session-manager.sh list
```

Returns sessions index JSON

#### output-dir
Get session output directory path.
```bash
hive-session-manager.sh output-dir <uuid>
```

#### log
Get session log file path.
```bash
hive-session-manager.sh log <uuid>
```

#### update-status
Manually update session status.
```bash
hive-session-manager.sh update-status <uuid> <status>
```

#### cleanup
Remove old completed/failed sessions.
```bash
hive-session-manager.sh cleanup [max_age_hours]
```

Default: 24 hours
Preserves output in ext/ directory

#### init
Initialize sessions infrastructure.
```bash
hive-session-manager.sh init
```

Creates directories and index.json

**Examples:**
```bash
# Create and start session
UUID=$(docker exec multi-agent-container /app/scripts/hive-session-manager.sh create "build rust app" high)
docker exec multi-agent-container /app/scripts/hive-session-manager.sh start $UUID

# Check status
docker exec multi-agent-container /app/scripts/hive-session-manager.sh status $UUID

# Get output directory
docker exec multi-agent-container /app/scripts/hive-session-manager.sh output-dir $UUID

# Read logs
docker exec multi-agent-container cat $(docker exec multi-agent-container /app/scripts/hive-session-manager.sh log $UUID)
```

**Database Isolation:**
Each session creates its own `.swarm/memory.db` in the session working directory. This prevents lock conflicts when multiple hive-mind processes run concurrently.

**File Locking:**
Uses flock on `/workspace/.swarm/sessions/.lock` for atomic index updates.

---

### session-cleanup.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/session-cleanup.sh`

**Purpose:** Periodic cleanup of stale sessions, WAL files, and database locks. Intended for cron or supervisord execution.

**Usage:**
```bash
./session-cleanup.sh
```

**Environment Variables:**
- `STALE_THRESHOLD_MINUTES` - Minutes before session is stale (default: 30)
- `PRUNE_AGE_HOURS` - Hours before pruning old sessions (default: 24)
- `VACUUM_DBS` - Enable database vacuuming (default: false)
- `ARCHIVE_DIR` - Directory for archived sessions (optional)

**Cleanup Operations:**

#### 1. Clean WAL/SHM Files
- Finds all *.db files under sessions root
- Removes -wal and -shm files if database is not in use
- Uses `fuser` to check for active connections
- Reports count of cleaned files

#### 2. Mark Abandoned Sessions
- Identifies running sessions from index.json
- Checks session activity:
  - Recent file modifications (within threshold)
  - Active database connections
- Marks inactive sessions as "completed"
- Updates both index.json and session.json
- Uses flock for atomic updates

#### 3. Prune Old Sessions
- Finds sessions older than PRUNE_AGE_HOURS
- Only prunes completed/failed sessions
- Archives to ARCHIVE_DIR if configured (tar.gz)
- Removes session directory
- Updates index atomically

#### 4. Vacuum Databases (optional)
- Runs VACUUM on all *.db files
- Skips databases in use
- Reclaims unused space
- Can be resource-intensive

**Activity Detection:**
Session is active if:
- Files modified within STALE_THRESHOLD_MINUTES
- Database has active fuser connections

**Output:**
```
[SESSION-CLEANUP YYYY-MM-DD HH:MM:SS] Starting session cleanup...
[SESSION-CLEANUP] Config: stale_threshold=30min, prune_age=24h
[SESSION-CLEANUP] Cleaning stale WAL/SHM files...
[SESSION-CLEANUP] Cleaned 3 WAL/SHM files
[SESSION-CLEANUP] Checking for abandoned sessions...
[SESSION-CLEANUP] Marked 1 abandoned sessions as completed
[SESSION-CLEANUP] Pruning sessions older than 24h...
[SESSION-CLEANUP] Pruned 2 old sessions
[SESSION-CLEANUP] Cleanup complete
[SESSION-CLEANUP] Status: 5 active sessions, 12 total sessions
```

**Scheduling:**
```bash
# Cron (every hour)
0 * * * * /path/to/session-cleanup.sh

# Supervisord
[program:session-cleanup]
command=/bin/bash -c "while true; do /path/to/session-cleanup.sh; sleep 3600; done"
```

**Return Codes:**
- `0` - Cleanup completed successfully

---

## Security and Validation

### validate-security.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/validate-security.sh`

**Purpose:** Comprehensive security validation for multi-agent Docker environment. Tests security features, configurations, and best practices.

**Usage:**
```bash
./validate-security.sh
```

**Test Categories:**

#### 1. Service Status
- Verifies multi-agent container is running
- Checks Docker availability

#### 2. Environment Configuration
- Validates .env file exists
- Checks for default tokens:
  - WebSocket token (your-secure-websocket-token-change-me)
  - TCP token (your-secure-tcp-token-change-me)
  - JWT secret (your-super-secret-jwt-key-minimum-32-chars)
- Warns if defaults detected

#### 3. Network Connectivity
- Tests WebSocket service (port 3002)
- Tests TCP service (port 9500)
- Tests health check endpoint (port 9501)

#### 4. Authentication
- Tests WebSocket authentication
- Verifies unauthenticated connections are rejected
- Uses timeout to prevent hanging

#### 5. Rate Limiting
- Creates 10 rapid connections
- Checks if rate limiting blocks excess connections
- Validates protection against connection flooding

#### 6. Security Scripts
- Verifies auth-middleware.js exists
- Verifies secure-client-example.js exists
- Checks script permissions (should be rwxr-x---)

#### 7. Logging and Monitoring
- Validates /app/mcp-logs directory
- Checks for security log directory
- Verifies recent log files exist

#### 8. Docker Security
- Verifies container runs as non-root user
- Checks user ID (should not be 0)
- Validates seccomp profile configuration

#### 9. SSL/TLS Configuration
- Checks SSL_ENABLED environment variable
- Validates certificate files if SSL enabled:
  - /app/certs/server.crt
  - /app/certs/server.key

#### 10. Performance and Resources
- Monitors memory usage (warns if >80%)
- Monitors CPU usage (warns if >80%)
- Uses docker stats for metrics

#### 11. Backup and Recovery
- Checks DB_BACKUP_ENABLED setting
- Validates workspace directory accessibility

**Output:**
```
🔒 Multi-Agent Docker Security Validation
==========================================

✅ Multi-agent container is running
✅ .env file exists
⚠️  Default WebSocket token detected in .env
✅ TCP token has been changed
...

==========================================
🔒 Security Validation Summary
==========================================

✅ Tests Passed: 15
❌ Tests Failed: 0
⚠️  Warnings: 3

🎉 All security tests passed! Your environment is secure.
```

**Return Codes:**
- `0` - All tests passed (warnings allowed)
- `1` - One or more tests failed

**Requirements:**
- Docker installed
- bc command for numeric calculations (optional)

---

## Utility Scripts

### gpu-detect.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/gpu-detect.sh`

**Purpose:** Automatic GPU detection and configuration for both NVIDIA and AMD hardware.

**Usage:**
```bash
# Run detection
./gpu-detect.sh --detect

# Or source to export variables
source ./gpu-detect.sh --detect
```

**Detection Process:**

#### NVIDIA GPU
Checks:
- nvidia-smi command availability
- GPU detection via nvidia-smi
- Extracts GPU name, memory, driver version

Exports:
```bash
export CUDA_VISIBLE_DEVICES=0
export USE_GPU=true
export GPU_VENDOR=nvidia
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```

#### AMD GPU
Checks:
- rocm-smi command availability
- GPU detection via rocm-smi

Exports:
```bash
export USE_GPU=true
export GPU_VENDOR=amd
```

#### CPU Fallback
Exports:
```bash
export USE_GPU=false
export GPU_VENDOR=none
```

**GPU Tool Configuration:**

Tests PyTorch:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Tests TensorFlow:
```bash
python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
```

**Resource Limits:**
Sets GPU memory growth settings:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Configuration File:**
Saves to `/workspace/.gpu_config`:
```
# GPU Configuration
# Generated: 2025-01-15 10:30:00
USE_GPU=true
GPU_VENDOR=nvidia
CUDA_VISIBLE_DEVICES=0
GPU_NAME=NVIDIA GeForce RTX 3090
GPU_MEMORY=24576
CUDA_VERSION=535.104.05
```

**Output:**
```
[GPU] Detecting GPU hardware...
[GPU] ✅ NVIDIA GPU detected: NVIDIA GeForce RTX 3090
[GPU] GPU Memory: 24576MB
[GPU] CUDA Driver: 535.104.05
[GPU] Configuring GPU-accelerated tools...
[GPU] ✅ PyTorch CUDA support enabled
[GPU] ✅ TensorFlow GPU support enabled
[GPU] ✅ GPU configuration saved to /workspace/.gpu_config
[GPU] ✅ GPU configuration complete
```

**Return Codes:**
- `0` - Success (GPU or CPU mode)

---

### validate-mounts.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/validate-mounts.sh`

**Purpose:** Validates Docker volume mounts and permissions.

**Usage:**
```bash
./validate-mounts.sh
```

Expected mounts:
- /workspace
- /app
- /home/devuser
- Volume mounts for persistence

---

### supervisor-recovery.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/supervisor-recovery.sh`

**Purpose:** Handles automatic recovery of supervisord-managed processes.

**Usage:**
```bash
./supervisor-recovery.sh
```

Recovery actions:
- Detects failed processes
- Attempts restart with backoff
- Logs recovery attempts
- Notifies on repeated failures

---

## Helper Scripts

### claude-flow-wrapper.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/claude-flow-wrapper.sh`

**Purpose:** Wrapper script for claude-flow commands with environment setup.

**Usage:**
```bash
./claude-flow-wrapper.sh <claude-flow-args>
```

---

### configure-claude-mcp.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/configure-claude-mcp.sh`

**Purpose:** Configures Claude MCP settings and tools.

**Usage:**
```bash
./configure-claude-mcp.sh
```

Configuration:
- Sets up MCP server connections
- Configures authentication
- Validates tool availability

---

### install-chrome-extensions.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/install-chrome-extensions.sh`

**Purpose:** Installs Chrome/Chromium browser extensions for agent automation.

**Usage:**
```bash
./install-chrome-extensions.sh
```

Extensions installed:
- Development tools
- Automation helpers
- Security extensions

---

### launch-chromium.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/launch-chromium.sh`

**Purpose:** Launches Chromium browser with proper configuration for container environment.

**Usage:**
```bash
./launch-chromium.sh [url]
```

Chromium flags:
- --no-sandbox (required in Docker)
- --disable-dev-shm-usage
- --disable-gpu (optional)

---

### send-client-message.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/send-client-message.sh`

**Purpose:** Sends test messages to MCP clients for debugging.

**Usage:**
```bash
./send-client-message.sh <message> [client_id]
```

---

### sqlite-db-setup.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/sqlite-db-setup.sh`

**Purpose:** Initializes SQLite databases with proper schema and permissions.

**Usage:**
```bash
./sqlite-db-setup.sh <database_path>
```

Setup:
- Creates database file
- Applies schema
- Sets WAL mode
- Configures permissions

---

### test-docker-exec-spawn.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/test-docker-exec-spawn.sh`

**Purpose:** Tests docker exec spawn capabilities for session isolation.

**Usage:**
```bash
./test-docker-exec-spawn.sh
```

---

### hive-mind-spawn-wrapper.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/hive-mind-spawn-wrapper.sh`

**Purpose:** Wrapper for hive-mind spawn with session context.

**Usage:**
```bash
./hive-mind-spawn-wrapper.sh <task>
```

---

### monitor-hive.sh

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/monitor-hive.sh`

**Purpose:** Monitors hive-mind sessions and reports status.

**Usage:**
```bash
./monitor-hive.sh
```

Monitoring:
- Active sessions count
- Resource usage
- Task completion status

---

## Python Scripts

### generate-topics-from-markdown.py

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/generate-topics-from-markdown.py`

**Purpose:** Generates topic indexes from markdown documentation.

**Usage:**
```bash
python3 generate-topics-from-markdown.py <input_dir> [output_file]
```

---

### log-analyzer.py

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/log-analyzer.py`

**Purpose:** Analyzes log files for patterns, errors, and performance metrics.

**Usage:**
```bash
python3 log-analyzer.py <log_file> [options]
```

Options:
- `--errors` - Show only errors
- `--warnings` - Include warnings
- `--stats` - Display statistics
- `--json` - Output as JSON

---

### supervisord-healthcheck.py

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/supervisord-healthcheck.py`

**Purpose:** Health check for supervisord-managed processes.

**Usage:**
```bash
python3 supervisord-healthcheck.py
```

Returns:
- 0: All processes healthy
- 1: One or more processes failed

---

## JavaScript Scripts

### stdio-to-tcp-bridge.js

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/stdio-to-tcp-bridge.js`

**Purpose:** Bridges stdio-based MCP servers to TCP connections.

**Usage:**
```bash
node stdio-to-tcp-bridge.js <port> <command> [args...]
```

Example:
```bash
node stdio-to-tcp-bridge.js 9500 python3 /app/mcp-server.py
```

---

### test-tcp-mcp.js

**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/test-tcp-mcp.js`

**Purpose:** Tests TCP-based MCP server connectivity.

**Usage:**
```bash
node test-tcp-mcp.js <host> <port>
```

Tests:
- Connection establishment
- Protocol handshake
- Tool listing
- Basic tool execution

---

## Script Dependencies

### Common Dependencies
- bash >= 4.0
- jq (JSON processing)
- curl (HTTP requests)
- docker (container management)
- sqlite3 (database operations)
- fuser (process checking)

### Optional Dependencies
- bc (numeric calculations)
- nvidia-smi (GPU detection)
- rocm-smi (AMD GPU)
- pm2 (process management)
- nodejs (JavaScript scripts)
- python3 (Python scripts)

---

## Environment Variables Reference

### API Keys
- `ANTHROPIC_API_KEY` - Anthropic Claude API
- `OPENAI_API_KEY` - OpenAI API
- `GOOGLE_GEMINI_API_KEY` - Google Gemini API
- `GOOGLE_API_KEY` - Google services
- `OPENROUTER_API_KEY` - OpenRouter API
- `E2B_API_KEY` - E2B sandbox
- `ZAI_API_KEY` - Z.AI semantic processing

### Configuration
- `WORKSPACE` - Workspace root (default: /workspace)
- `MCP_AUTO_START` - Auto-start MCP servers
- `GPU_ACCELERATION` - Enable GPU detection
- `ENABLE_XINFERENCE` - Enable Xinference tests
- `MANAGEMENT_API_KEY` - Management API authentication

### Session Management
- `STALE_THRESHOLD_MINUTES` - Session staleness threshold
- `PRUNE_AGE_HOURS` - Session pruning age
- `VACUUM_DBS` - Enable DB vacuuming
- `ARCHIVE_DIR` - Session archive directory

---

## Common Patterns

### Error Handling
All scripts follow consistent error handling:
```bash
set -e  # Exit on error
log_error "message" && exit 1
```

### Logging
Consistent logging functions:
```bash
log_info "Information message"
log_success "Success message"
log_warning "Warning message"
log_error "Error message"
```

### File Locking
Atomic operations use flock:
```bash
(
    flock -x 200
    # Critical section
) 200>"$LOCK_FILE"
```

### JSON Processing
All JSON operations use jq:
```bash
jq '.field = "value"' input.json > output.json
```

---

## Troubleshooting

### Script Not Found
```bash
# Ensure scripts are executable
chmod +x /path/to/script.sh
```

### Permission Denied
```bash
# Run as correct user
su - devuser -c "./script.sh"
```

### Command Not Found
```bash
# Check dependencies
which jq curl sqlite3 docker
```

### Database Locked
```bash
# Run cleanup
./session-cleanup.sh
# Or check locks
./healthcheck-db.sh
```

---

## Quick Reference

### Start System
```bash
./start-agentic-flow.sh --build
```

### Check Health
```bash
./healthcheck-db.sh
./validate-security.sh
```

### Test Providers
```bash
./test-all-providers.sh
./test-gemini-flow.sh
```

### Manage Sessions
```bash
UUID=$(hive-session-manager.sh create "task")
hive-session-manager.sh start $UUID
hive-session-manager.sh status $UUID
```

### Clean Up
```bash
./session-cleanup.sh
./start-agentic-flow.sh --clean
```

---

## Contributing

When adding new scripts:
1. Follow naming conventions (lowercase-with-hyphens.sh)
2. Add usage documentation in header
3. Use consistent logging functions
4. Include error handling
5. Document in this reference
6. Add to appropriate category

---

## License

All scripts are part of the AR-AI-Knowledge-Graph multi-agent-docker system.
