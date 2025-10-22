# Agent Container Settings Audit

**Date**: 2025-10-22
**Container**: Turbo Flow Unified (multi-agent-docker)
**Total Parameters Discovered**: 287
**Configuration Files Analyzed**: 15

---

## Executive Summary

This comprehensive audit documents all configurable parameters across the multi-agent-docker unified container system. The container implements a sophisticated multi-user, multi-service architecture with 4 isolated users, 18 supervisord-managed services, 9 Claude Code skills, and extensive GPU/CUDA support.

**Key Statistics**:
- **Users**: 4 isolated Linux users (devuser, gemini-user, openai-user, zai-user)
- **Services**: 18 supervisord-managed services
- **Skills**: 9 Claude Code skills with MCP integration
- **Ports**: 5 exposed ports (22, 5901, 8080, 9090, 9600)
- **MCP Servers**: 11 configured MCP servers
- **Agent Templates**: 610+ available agents

---

## Table of Contents

1. [Docker Container Configuration](#docker-container-configuration)
2. [Supervisord Services](#supervisord-services)
3. [MCP Server Settings](#mcp-server-settings)
4. [Z.AI Service Configuration](#zai-service-configuration)
5. [Management API Settings](#management-api-settings)
6. [Skills System Configuration](#skills-system-configuration)
7. [User Management](#user-management)
8. [Resource Limits](#resource-limits)
9. [Network Configuration](#network-configuration)
10. [Environment Variables](#environment-variables)
11. [tmux Workspace Configuration](#tmux-workspace-configuration)
12. [VNC Desktop Settings](#vnc-desktop-settings)
13. [SSH Configuration](#ssh-configuration)
14. [Logging Configuration](#logging-configuration)
15. [Summary Tables](#summary-tables)

---

## Docker Container Configuration

### Build Configuration (Dockerfile.unified)

| Parameter | Current Value | Type | Options | Location | Priority | Category |
|-----------|--------------|------|---------|----------|----------|----------|
| **Base Image** | archlinux:latest | string | Any Docker image | Dockerfile.unified:6 | Critical | Container/Base |
| **Node.js Version** | v22.11.0 | string | Any Node.js version | Dockerfile.unified:48-51 | High | Container/Runtime |
| **CUDA Home** | /opt/cuda | path | Any valid path | Dockerfile.unified:65 | Critical | Container/GPU |
| **Virtual Environment Path** | /opt/venv | path | Any valid path | Dockerfile.unified:136 | High | Container/Python |
| **Code-Server Version** | v4.96.2 | string | Any version | Dockerfile.unified:313 | Medium | Container/IDE |

### Docker Compose Settings (docker-compose.unified.yml)

| Parameter | Current Value | Type | Range/Options | Location | Priority | Category |
|-----------|--------------|------|---------------|----------|----------|----------|
| **container_name** | agentic-workstation | string | Any valid name | docker-compose.unified.yml:9 | High | Container/Identity |
| **hostname** | agentic-workstation | string | Any valid hostname | docker-compose.unified.yml:10 | High | Container/Identity |
| **runtime** | nvidia | string | nvidia, runc | docker-compose.unified.yml:13 | Critical | Container/GPU |
| **shm_size** | 32gb | size | Any size (e.g., 8gb, 64gb) | docker-compose.unified.yml:14 | High | Container/Memory |
| **memory (limits)** | 64G | size | Any size | docker-compose.unified.yml:20 | Critical | Container/Resources |
| **memory (reservations)** | 16G | size | Any size | docker-compose.unified.yml:23 | High | Container/Resources |
| **cpus (limits)** | 32 | number | 1-∞ | docker-compose.unified.yml:21 | Critical | Container/Resources |
| **cpus (reservations)** | 8 | number | 1-∞ | docker-compose.unified.yml:24 | High | Container/Resources |
| **GPU count** | all | string/number | all, 0, 1, 2, etc. | docker-compose.unified.yml:27 | Critical | Container/GPU |
| **GPU capabilities** | gpu,compute,utility,graphics,display | list | Any GPU capability | docker-compose.unified.yml:28 | High | Container/GPU |
| **restart** | unless-stopped | string | no, always, on-failure, unless-stopped | docker-compose.unified.yml:140 | High | Container/Lifecycle |

### Volume Configuration

| Parameter | Current Value | Type | Options | Location | Priority | Category |
|-----------|--------------|------|---------|----------|----------|----------|
| **workspace** | Named volume | volume | local, nfs, etc. | docker-compose.unified.yml:48 | Critical | Container/Storage |
| **agents** | Named volume | volume | local, nfs, etc. | docker-compose.unified.yml:49 | High | Container/Storage |
| **claude-config** | Named volume | volume | local, nfs, etc. | docker-compose.unified.yml:50 | High | Container/Storage |
| **gemini-workspace** | Named volume | volume | local, nfs, etc. | docker-compose.unified.yml:53 | Medium | Container/Storage |
| **openai-workspace** | Named volume | volume | local, nfs, etc. | docker-compose.unified.yml:54 | Medium | Container/Storage |
| **model-cache** | Named volume | volume | local, nfs, etc. | docker-compose.unified.yml:57 | High | Container/Storage |
| **logs** | Named volume | volume | local, nfs, etc. | docker-compose.unified.yml:58 | Medium | Container/Storage |
| **HOST_CLAUDE_DIR** | ${HOME}/.claude | path | Any path | docker-compose.unified.yml:61 | Low | Container/Storage |
| **PROJECT_DIR** | /tmp/empty | path | Any path | docker-compose.unified.yml:64 | High | Container/Storage |

### Security Settings

| Parameter | Current Value | Type | Options | Location | Priority | Category |
|-----------|--------------|------|---------|----------|----------|----------|
| **cap_add (SYS_ADMIN)** | true | boolean | true/false | docker-compose.unified.yml:115 | High | Container/Security |
| **cap_add (NET_ADMIN)** | true | boolean | true/false | docker-compose.unified.yml:116 | Medium | Container/Security |
| **cap_add (SYS_PTRACE)** | true | boolean | true/false | docker-compose.unified.yml:117 | Low | Container/Security |
| **apparmor** | unconfined | string | unconfined, default, custom | docker-compose.unified.yml:120 | Medium | Container/Security |
| **seccomp** | unconfined | string | unconfined, default, custom | docker-compose.unified.yml:121 | Medium | Container/Security |

### Health Check

| Parameter | Current Value | Type | Options | Location | Priority | Category |
|-----------|--------------|------|---------|----------|----------|----------|
| **test** | curl -f http://localhost:9090/health | string | Any health check command | docker-compose.unified.yml:133 | High | Container/Health |
| **interval** | 30s | duration | Any duration | docker-compose.unified.yml:134 | Medium | Container/Health |
| **timeout** | 10s | duration | Any duration | docker-compose.unified.yml:135 | Medium | Container/Health |
| **retries** | 3 | number | 1-∞ | docker-compose.unified.yml:136 | Medium | Container/Health |
| **start_period** | 60s | duration | Any duration | docker-compose.unified.yml:137 | Medium | Container/Health |

---

## Supervisord Services

**Configuration File**: `unified-config/supervisord.unified.conf`
**Total Services**: 18

### Core System Services

| Service | Parameter | Current Value | Type | Options | Location | Priority |
|---------|-----------|--------------|------|---------|----------|----------|
| **supervisord** | nodaemon | true | boolean | true/false | supervisord.unified.conf:2 | Critical |
| **supervisord** | logfile | /var/log/supervisord.log | path | Any path | supervisord.unified.conf:3 | High |
| **supervisord** | pidfile | /var/run/supervisord.pid | path | Any path | supervisord.unified.conf:4 | High |
| **supervisord** | childlogdir | /var/log/supervisor | path | Any path | supervisord.unified.conf:5 | Medium |
| **supervisord** | user | root | string | Any user | supervisord.unified.conf:6 | Critical |

### Service Definitions

#### 1. DBus System Daemon

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/bin/dbus-daemon --system --nofork | string | Any command | supervisord.unified.conf:231 | Critical |
| user | root | string | Any user | supervisord.unified.conf:232 | Critical |
| autostart | true | boolean | true/false | supervisord.unified.conf:233 | High |
| autorestart | true | boolean | true/false | supervisord.unified.conf:234 | High |
| priority | 10 | number | 0-999 | supervisord.unified.conf:235 | Critical |
| stdout_logfile | /var/log/dbus.log | path | Any path | supervisord.unified.conf:236 | Medium |
| stderr_logfile | /var/log/dbus.error.log | path | Any path | supervisord.unified.conf:237 | Medium |

#### 2. DBus User Session

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/bin/dbus-daemon --session --nofork | string | Any command | supervisord.unified.conf:240 | High |
| user | devuser | string | Any user | supervisord.unified.conf:241 | High |
| environment | HOME="/home/devuser",USER="devuser",XDG_RUNTIME_DIR="/run/user/1000" | env vars | Any env vars | supervisord.unified.conf:242 | High |
| autostart | true | boolean | true/false | supervisord.unified.conf:243 | High |
| autorestart | true | boolean | true/false | supervisord.unified.conf:244 | High |
| priority | 15 | number | 0-999 | supervisord.unified.conf:245 | High |

#### 3. SSH Server

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/sbin/sshd -D -e | string | Any command | supervisord.unified.conf:51 | Critical |
| user | root | string | Any user | supervisord.unified.conf:52 | Critical |
| autostart | true | boolean | true/false | supervisord.unified.conf:53 | High |
| autorestart | true | boolean | true/false | supervisord.unified.conf:54 | High |
| priority | 50 | number | 0-999 | supervisord.unified.conf:55 | High |

#### 4. VNC Server (Xvnc)

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/bin/Xvnc :1 -geometry 1920x1080 -depth 24 -rfbport 5901 | string | Any command | supervisord.unified.conf:23 | Critical |
| **VNC Display** | :1 | string | :0, :1, :2, etc. | supervisord.unified.conf:23 | Critical |
| **VNC Geometry** | 1920x1080 | resolution | Any resolution (e.g., 1280x720, 2560x1440) | supervisord.unified.conf:23 | Medium |
| **VNC Depth** | 24 | number | 8, 16, 24, 32 | supervisord.unified.conf:23 | Low |
| **VNC Port** | 5901 | number | 5900-5999 | supervisord.unified.conf:23 | Critical |
| **SecurityTypes** | None | string | None, VncAuth, Plain, etc. | supervisord.unified.conf:23 | High |
| **AlwaysShared** | true | boolean | true/false | supervisord.unified.conf:23 | Medium |
| **rendernode** | /dev/dri/renderD128 | device | Any DRI device | supervisord.unified.conf:23 | Medium |
| user | devuser | string | Any user | supervisord.unified.conf:24 | Critical |
| environment | __GLX_VENDOR_LIBRARY_NAME="nvidia",__NV_PRIME_RENDER_OFFLOAD="1" | env vars | Any GPU env | supervisord.unified.conf:25 | High |
| autostart | true | boolean | true/false | supervisord.unified.conf:26 | High |
| autorestart | true | boolean | true/false | supervisord.unified.conf:27 | High |
| priority | 100 | number | 0-999 | supervisord.unified.conf:28 | High |

#### 5. XFCE4 Desktop

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/bin/startxfce4 | string | Any command | supervisord.unified.conf:37 | High |
| user | devuser | string | Any user | supervisord.unified.conf:38 | High |
| environment | DISPLAY=":1",DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus" | env vars | Any env vars | supervisord.unified.conf:39 | High |
| autostart | true | boolean | true/false | supervisord.unified.conf:40 | High |
| autorestart | true | boolean | true/false | supervisord.unified.conf:41 | High |
| priority | 200 | number | 0-999 | supervisord.unified.conf:42 | High |

#### 6. Management API

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/bin/node /opt/management-api/server.js | string | Any command | supervisord.unified.conf:64 | Critical |
| directory | /opt/management-api | path | Any path | supervisord.unified.conf:65 | High |
| user | devuser | string | Any user | supervisord.unified.conf:66 | High |
| environment | PORT="9090",NODE_ENV="production" | env vars | Any env vars | supervisord.unified.conf:67 | High |
| autostart | true | boolean | true/false | supervisord.unified.conf:68 | High |
| autorestart | true | boolean | true/false | supervisord.unified.conf:69 | High |
| priority | 300 | number | 0-999 | supervisord.unified.conf:70 | High |

#### 7. code-server (Web IDE)

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/bin/code-server --bind-addr 0.0.0.0:8080 --auth none | string | Any command | supervisord.unified.conf:79 | High |
| **bind-addr** | 0.0.0.0:8080 | address | Any IP:PORT | supervisord.unified.conf:79 | High |
| **auth** | none | string | none, password | supervisord.unified.conf:79 | Medium |
| **workspace** | /home/devuser/workspace | path | Any path | supervisord.unified.conf:79 | High |
| user | devuser | string | Any user | supervisord.unified.conf:80 | High |
| autostart | true | boolean | true/false | supervisord.unified.conf:82 | Medium |
| autorestart | true | boolean | true/false | supervisord.unified.conf:83 | Medium |
| priority | 400 | number | 0-999 | supervisord.unified.conf:84 | Medium |

#### 8. Claude Z.AI Service

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/bin/node /opt/claude-zai/server.js | string | Any command | supervisord.unified.conf:94 | Critical |
| directory | /opt/claude-zai | path | Any path | supervisord.unified.conf:95 | High |
| user | zai-user | string | Any user | supervisord.unified.conf:96 | Critical |
| environment | PORT="9600",NODE_ENV="production" | env vars | Any env vars | supervisord.unified.conf:97 | High |
| autostart | true | boolean | true/false | supervisord.unified.conf:98 | High |
| autorestart | true | boolean | true/false | supervisord.unified.conf:99 | High |
| priority | 500 | number | 0-999 | supervisord.unified.conf:100 | High |

#### 9. Gemini Flow Daemon

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /usr/bin/gemini-flow mcp start | string | Any command | supervisord.unified.conf:201 | High |
| directory | /home/gemini-user/workspace | path | Any path | supervisord.unified.conf:202 | Medium |
| user | gemini-user | string | Any user | supervisord.unified.conf:203 | Critical |
| environment | MCP_SOCKET="/var/run/agentic-services/gemini-mcp.sock" | env vars | Any env vars | supervisord.unified.conf:204 | High |
| autostart | true | boolean | true/false | supervisord.unified.conf:205 | Medium |
| autorestart | true | boolean | true/false | supervisord.unified.conf:206 | Medium |
| priority | 600 | number | 0-999 | supervisord.unified.conf:207 | Medium |

#### 10-17. MCP Skill Servers

| Service | Port | Autostart | User | Priority | Location |
|---------|------|-----------|------|----------|----------|
| **web-summary-mcp** | - | true | devuser | 510 | supervisord.unified.conf:108 |
| **qgis-mcp** | 9877 | false | devuser | 511 | supervisord.unified.conf:119 |
| **blender-mcp** | 9876 | false | devuser | 512 | supervisord.unified.conf:130 |
| **imagemagick-mcp** | - | true | devuser | 513 | supervisord.unified.conf:141 |
| **kicad-mcp** | - | false | devuser | 514 | supervisord.unified.conf:152 |
| **ngspice-mcp** | - | false | devuser | 515 | supervisord.unified.conf:163 |
| **pbr-mcp** | 9878 | false | devuser | 516 | supervisord.unified.conf:174 |
| **playwright-mcp** | - | true | devuser | 517 | supervisord.unified.conf:185 |

#### 18. tmux Workspace Auto-Start

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| command | /bin/bash -c "sleep 5 && /home/devuser/.config/tmux-autostart.sh" | string | Any command | supervisord.unified.conf:217 | Medium |
| **startup delay** | 5 seconds | number | 0-∞ | supervisord.unified.conf:217 | Low |
| user | devuser | string | Any user | supervisord.unified.conf:218 | High |
| autostart | true | boolean | true/false | supervisord.unified.conf:220 | Medium |
| autorestart | false | boolean | true/false | supervisord.unified.conf:221 | Low |
| priority | 900 | number | 0-999 | supervisord.unified.conf:222 | Low |

---

## MCP Server Settings

**Configuration File**: `mcp-infrastructure/mcp.json`
**Total MCP Servers**: 11

### MCP Server Definitions

| Server Name | Command | Type | Environment Variables | Location | Priority |
|-------------|---------|------|----------------------|----------|----------|
| **claude-flow** | node | stdio | MCP_HOST=127.0.0.1, MCP_PORT=9500 | mcp.json:3-11 | Critical |
| **ruv-swarm** | npx | stdio | (none) | mcp.json:12-16 | High |
| **blender-mcp** | node | stdio | BLENDER_HOST=localhost, BLENDER_PORT=9876 | mcp.json:17-25 | Medium |
| **qgis-mcp** | python3 | stdio | QGIS_HOST=localhost, QGIS_PORT=9877 | mcp.json:26-34 | Medium |
| **kicad-mcp** | python3 | stdio | (none) | mcp.json:35-39 | Medium |
| **ngspice-mcp** | python3 | stdio | (none) | mcp.json:40-44 | Low |
| **imagemagick-mcp** | python3 | stdio | (none) | mcp.json:45-49 | Medium |
| **pbr-generator-mcp** | python3 | stdio | PBR_HOST=localhost, PBR_PORT=9878 | mcp.json:50-58 | Medium |
| **playwright-visual** | node | stdio | PLAYWRIGHT_PROXY_HOST=localhost, PLAYWRIGHT_PROXY_PORT=9879 | mcp.json:59-67 | Medium |
| **playwright** | node | stdio | (none) | mcp.json:68-72 | High |
| **web-summary** | /opt/venv312/bin/python3 | stdio | GOOGLE_API_KEY=${GOOGLE_API_KEY} | mcp.json:73-80 | High |

### MCP Environment Variables

| Variable | Server | Current Value | Type | Options | Priority |
|----------|--------|--------------|------|---------|----------|
| **MCP_HOST** | claude-flow | 127.0.0.1 | IP address | Any valid IP | High |
| **MCP_PORT** | claude-flow | 9500 | number | 1024-65535 | High |
| **BLENDER_HOST** | blender-mcp | localhost | hostname | Any hostname/IP | Medium |
| **BLENDER_PORT** | blender-mcp | 9876 | number | 1024-65535 | Medium |
| **QGIS_HOST** | qgis-mcp | localhost | hostname | Any hostname/IP | Medium |
| **QGIS_PORT** | qgis-mcp | 9877 | number | 1024-65535 | Medium |
| **PBR_HOST** | pbr-generator-mcp | localhost | hostname | Any hostname/IP | Medium |
| **PBR_PORT** | pbr-generator-mcp | 9878 | number | 1024-65535 | Medium |
| **PLAYWRIGHT_PROXY_HOST** | playwright-visual | localhost | hostname | Any hostname/IP | Medium |
| **PLAYWRIGHT_PROXY_PORT** | playwright-visual | 9879 | number | 1024-65535 | Medium |
| **GOOGLE_API_KEY** | web-summary | ${GOOGLE_API_KEY} | string | API key from env | High |

---

## Z.AI Service Configuration

**Service File**: `multi-agent-docker/claude-zai/wrapper/server.js`
**Port**: 9600
**User**: zai-user

### Core Settings

| Parameter | Current Value | Type | Range | Location | Priority | Category |
|-----------|--------------|------|-------|----------|----------|----------|
| **PORT** | 9600 | number | 1024-65535 | server.js:6 | Critical | Service/Network |
| **WORKER_POOL_SIZE** | 4 | number | 1-16 | server.js:7 | High | Service/Performance |
| **MAX_QUEUE_SIZE** | 50 | number | 1-∞ | server.js:8 | High | Service/Performance |
| **Body Parser Limit** | 10mb | size | Any size | server.js:10 | Medium | Service/HTTP |
| **MAX_RETRIES** | 3 | number | 0-∞ | server.js:59 | Medium | Service/Reliability |
| **BASE_DELAY** | 1000ms | number | 0-∞ | server.js:60 | Low | Service/Reliability |
| **Default Timeout** | 30000ms | number | 1000-∞ | server.js:27 | High | Service/Performance |

### Environment Variables

| Variable | Current Value | Type | Options | Location | Priority |
|----------|--------------|------|---------|----------|----------|
| **CLAUDE_WORKER_POOL_SIZE** | 4 | number | 1-16 | server.js:7 | High |
| **CLAUDE_MAX_QUEUE_SIZE** | 50 | number | 1-∞ | server.js:8 | High |
| **ZAI_ANTHROPIC_API_KEY** | (from env) | string | API key | server.js:70 | Critical |
| **ANTHROPIC_API_KEY** | (fallback) | string | API key | server.js:70 | Critical |
| **ZAI_BASE_URL** | https://api.z.ai/api/anthropic | URL | Any URL | server.js:71 | Critical |
| **ANTHROPIC_BASE_URL** | (fallback) | URL | Any URL | server.js:71 | High |
| **ANTHROPIC_AUTH_TOKEN** | (from env) | string | Auth token | server.js:72 | Medium |

### Endpoints

| Endpoint | Method | Purpose | Location | Priority |
|----------|--------|---------|----------|----------|
| **/health** | GET | Health check with stats | server.js:175 | High |
| **/prompt** | POST | Execute Claude prompt | server.js:184 | Critical |

### Worker Pool Configuration

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **Pool Size** | 4 | number | 1-16 | server.js:14 | High |
| **Worker Busy State** | boolean | boolean | true/false | server.js:23 | Internal |
| **Queue Processing** | Automatic | mode | Auto/Manual | server.js:40 | Internal |

---

## Management API Settings

**Service File**: `multi-agent-docker/management-api/server.js`
**Port**: 9090
**Framework**: Fastify

### Core Configuration

| Parameter | Current Value | Type | Range | Location | Priority | Category |
|-----------|--------------|------|-------|----------|----------|----------|
| **PORT** | 9090 | number | 1024-65535 | server.js:17 | Critical | API/Network |
| **HOST** | 0.0.0.0 | IP address | Any IP | server.js:18 | Critical | API/Network |
| **API_KEY** | change-this-secret-key | string | Any string | server.js:19 | Critical | API/Security |
| **requestIdLogLabel** | reqId | string | Any label | server.js:24 | Low | API/Logging |
| **disableRequestLogging** | false | boolean | true/false | server.js:25 | Low | API/Logging |
| **trustProxy** | true | boolean | true/false | server.js:26 | Medium | API/Network |

### CORS Settings

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **origin** | true | boolean/string | true, false, domain | server.js:35 | Medium |
| **credentials** | true | boolean | true/false | server.js:36 | Medium |

### Rate Limiting

| Parameter | Current Value | Type | Range | Location | Priority |
|-----------|--------------|------|-------|----------|----------|
| **max** | 100 | number | 1-∞ | server.js:41 | High |
| **timeWindow** | 1 minute | duration | Any duration | server.js:42 | High |
| **cache** | 10000 | number | 1-∞ | server.js:43 | Medium |
| **allowList** | 127.0.0.1 | IP array | Any IPs | server.js:44 | Medium |
| **continueExceeding** | true | boolean | true/false | server.js:45 | Low |
| **skipOnError** | false | boolean | true/false | server.js:46 | Low |

### OpenAPI/Swagger Configuration

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **openapi version** | 3.0.0 | string | 3.0.0, 3.1.0 | server.js:79 | Low |
| **API title** | Agentic Flow Management API | string | Any string | server.js:81 | Low |
| **API version** | 2.1.0 | string | Any version | server.js:83 | Medium |
| **routePrefix** | /docs | path | Any path | server.js:115 | Low |
| **docExpansion** | list | string | list, full, none | server.js:117 | Low |
| **deepLinking** | true | boolean | true/false | server.js:118 | Low |

### Endpoints

| Endpoint | Method | Auth Required | Purpose | Location |
|----------|--------|--------------|---------|----------|
| **/** | GET | Yes | API information | server.js:158 |
| **/health** | GET | No | Health check | routes/status |
| **/ready** | GET | No | Readiness check | routes/status |
| **/metrics** | GET | No | Prometheus metrics | server.js:141 |
| **/docs** | GET | No | Swagger UI | server.js:114 |
| **/v1/tasks** | POST | Yes | Create task | routes/tasks |
| **/v1/tasks/:taskId** | GET | Yes | Get task status | routes/tasks |
| **/v1/tasks** | GET | Yes | List tasks | routes/tasks |
| **/v1/status** | GET | Yes | System status | routes/status |

### Cleanup Configuration

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **Cleanup Interval** | 10 minutes (600000ms) | duration | Any duration | server.js:230 | Medium |
| **Task Max Age** | 1 hour (3600000ms) | duration | Any duration | server.js:230 | Medium |

---

## Skills System Configuration

**Skills Directory**: `/home/devuser/.claude/skills/`
**Total Skills**: 9

### Available Skills

| Skill | Communication | Port | Autostart | User | MCP Server | Location | Priority |
|-------|--------------|------|-----------|------|------------|----------|----------|
| **web-summary** | stdio | - | true | devuser | Yes | skills/web-summary | High |
| **blender** | socket | 9876 | false | devuser | Yes | skills/blender | Medium |
| **qgis** | socket | 9877 | false | devuser | Yes | skills/qgis | Medium |
| **kicad** | stdio | - | false | devuser | Yes | skills/kicad | Medium |
| **imagemagick** | stdio | - | true | devuser | Yes | skills/imagemagick | High |
| **ngspice** | stdio | - | false | devuser | Yes | skills/ngspice | Low |
| **pbr-rendering** | socket | 9878 | false | devuser | Yes | skills/pbr-rendering | Medium |
| **playwright** | stdio | - | true | devuser | Yes | skills/playwright | High |
| **logseq-formatted** | - | - | - | devuser | No | skills/logseq-formatted | Low |

### Skill-Specific Settings

#### web-summary

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **ZAI_CONTAINER_URL** | http://localhost:9600 | URL | Any URL | supervisord.unified.conf:112 | High |
| **WEB_SUMMARY_TOOL_PATH** | /home/devuser/.claude/skills/web-summary/tools/web_summary_tool.py | path | Any path | supervisord.unified.conf:112 | High |
| **GOOGLE_API_KEY** | ${GOOGLE_API_KEY} | string | API key | mcp.json:78 | High |

#### blender

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **BLENDER_HOST** | localhost | hostname | Any hostname/IP | supervisord.unified.conf:134 | Medium |
| **BLENDER_PORT** | 9876 | number | 1024-65535 | supervisord.unified.conf:134 | Medium |

#### qgis

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **QGIS_HOST** | localhost | hostname | Any hostname/IP | supervisord.unified.conf:123 | Medium |
| **QGIS_PORT** | 9877 | number | 1024-65535 | supervisord.unified.conf:123 | Medium |

#### pbr-rendering

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **PBR_HOST** | localhost | hostname | Any hostname/IP | supervisord.unified.conf:178 | Medium |
| **PBR_PORT** | 9878 | number | 1024-65535 | supervisord.unified.conf:178 | Medium |

---

## User Management

**Total Users**: 4 (isolated Linux users)

### User Configuration

| User | UID:GID | Home | Shell | Groups | Purpose | Sudo Access | Location |
|------|---------|------|-------|--------|---------|------------|----------|
| **devuser** | 1000:1000 | /home/devuser | /usr/bin/zsh | wheel,video,audio,docker | Primary Claude Code development | Full (NOPASSWD:ALL) | Dockerfile.unified:93-95 |
| **gemini-user** | 1001:1001 | /home/gemini-user | /usr/bin/zsh | (default) | Google Gemini tools | No | Dockerfile.unified:98-99 |
| **openai-user** | 1002:1002 | /home/openai-user | /usr/bin/zsh | (default) | OpenAI Codex tools | No | Dockerfile.unified:102-103 |
| **zai-user** | 1003:1003 | /home/zai-user | /usr/bin/zsh | (default) | Z.AI service (port 9600) | No | Dockerfile.unified:106-107 |

### User Switching Configuration

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **devuser sudo access to other users** | NOPASSWD: ALL | sudoers | Standard sudo | Dockerfile.unified:110 | High |
| **as-gemini command** | sudo -u gemini-user -i | alias | Any command | Dockerfile.unified:352 | Medium |
| **as-openai command** | sudo -u openai-user -i | alias | Any command | Dockerfile.unified:353 | Medium |
| **as-zai command** | sudo -u zai-user -i | alias | Any command | Dockerfile.unified:354 | Medium |

### User Credentials Distribution

**Entrypoint Script**: `unified-config/entrypoint-unified.sh`

| User | Credential Type | Config File | Environment Variable Source | Location |
|------|----------------|-------------|----------------------------|----------|
| **devuser** | Claude API | ~/.config/claude/config.json | ANTHROPIC_API_KEY | entrypoint-unified.sh:43-51 |
| **devuser** | Z.AI API | ~/.config/zai/api.json | ZAI_API_KEY | entrypoint-unified.sh:54-61 |
| **gemini-user** | Gemini API | ~/.config/gemini/config.json | GOOGLE_GEMINI_API_KEY | entrypoint-unified.sh:64-73 |
| **openai-user** | OpenAI API | ~/.config/openai/config.json | OPENAI_API_KEY | entrypoint-unified.sh:76-84 |
| **zai-user** | Z.AI Config | ~/.config/zai/config.json | ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL | entrypoint-unified.sh:87-98 |
| **All users** | GitHub Token | ~/.config/gh/config.yml | GITHUB_TOKEN | entrypoint-unified.sh:101-112 |

---

## Resource Limits

### CPU Configuration

| Parameter | Current Value | Type | Range | Location | Priority | Category |
|-----------|--------------|------|-------|----------|----------|----------|
| **CPU Limit** | 32 cores | number | 1-∞ | docker-compose.unified.yml:21 | Critical | Resources/CPU |
| **CPU Reservation** | 8 cores | number | 1-∞ | docker-compose.unified.yml:24 | High | Resources/CPU |

### Memory Configuration

| Parameter | Current Value | Type | Range | Location | Priority | Category |
|-----------|--------------|------|-------|----------|----------|----------|
| **Memory Limit** | 64G | size | 1G-∞ | docker-compose.unified.yml:20 | Critical | Resources/Memory |
| **Memory Reservation** | 16G | size | 1G-∞ | docker-compose.unified.yml:23 | High | Resources/Memory |
| **Shared Memory** | 32gb | size | 1gb-∞ | docker-compose.unified.yml:14 | High | Resources/Memory |

### GPU Configuration

| Parameter | Current Value | Type | Options | Location | Priority | Category |
|-----------|--------------|------|---------|----------|----------|----------|
| **GPU Count** | all | string/number | all, 0, 1, 2, etc. | docker-compose.unified.yml:27 | Critical | Resources/GPU |
| **GPU Capabilities** | gpu,compute,utility,graphics,display | list | Any capability | docker-compose.unified.yml:28 | High | Resources/GPU |
| **CUDA_VISIBLE_DEVICES** | all | string | all, 0,1, etc. | docker-compose.unified.yml:102 | Critical | Resources/GPU |
| **CUDA_HOME** | /opt/cuda | path | Any path | Dockerfile.unified:65 | Critical | Resources/GPU |
| **__GLX_VENDOR_LIBRARY_NAME** | nvidia | string | nvidia, mesa | Dockerfile.unified:70 | High | Resources/GPU |
| **__NV_PRIME_RENDER_OFFLOAD** | 1 | number | 0, 1 | Dockerfile.unified:71 | Medium | Resources/GPU |
| **__VK_LAYER_NV_optimus** | NVIDIA_only | string | NVIDIA_only, auto | Dockerfile.unified:72 | Medium | Resources/GPU |
| **LIBGL_ALWAYS_INDIRECT** | 0 | number | 0, 1 | Dockerfile.unified:73 | Low | Resources/GPU |

### Device Access

| Device | Path | Purpose | Location | Priority |
|--------|------|---------|----------|----------|
| **/dev/dri** | /dev/dri | Intel/AMD GPU | docker-compose.unified.yml:125 | High |
| **/dev/nvidia0** | /dev/nvidia0 | NVIDIA GPU 0 | docker-compose.unified.yml:126 | Critical |
| **/dev/nvidiactl** | /dev/nvidiactl | NVIDIA control | docker-compose.unified.yml:127 | Critical |
| **/dev/nvidia-uvm** | /dev/nvidia-uvm | NVIDIA unified memory | docker-compose.unified.yml:128 | Critical |
| **/dev/nvidia-modeset** | /dev/nvidia-modeset | NVIDIA modesetting | docker-compose.unified.yml:129 | High |

---

## Network Configuration

### Docker Network

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **Network Name** | docker_ragflow | string | Any network | docker-compose.unified.yml:156-158 | High |
| **Network External** | true | boolean | true/false | docker-compose.unified.yml:157 | High |
| **Hostname** | agentic-workstation | string | Any hostname | docker-compose.unified.yml:10 | Medium |
| **Network Aliases** | agentic-workstation.ragflow, agentic-workstation.local | list | Any aliases | docker-compose.unified.yml:33-34 | Medium |

### Port Mappings

| Host Port | Container Port | Service | Access Level | Location | Priority |
|-----------|---------------|---------|--------------|----------|----------|
| **2222** | 22 | SSH | Public (LAN) | docker-compose.unified.yml:39 | High |
| **5901** | 5901 | VNC | Public (LAN) | docker-compose.unified.yml:40 | High |
| **8080** | 8080 | code-server | Public (LAN) | docker-compose.unified.yml:41 | High |
| **9090** | 9090 | Management API | Public (LAN) | docker-compose.unified.yml:42 | High |
| **-** | 9600 | Z.AI | Internal Only | docker-compose.unified.yml:43 (commented) | Critical |

**Note**: Port 9600 (Z.AI) is NOT exposed to host - accessible only within docker_ragflow network.

### DNS Configuration

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **DNS Servers** | 8.8.8.8, 8.8.4.4 | IP array | Any DNS IPs | docker-compose.unified.yml:143-145 | Medium |

### Extra Hosts

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **host.docker.internal** | host-gateway | special | host-gateway, IP | docker-compose.unified.yml:149 | Medium |

---

## Environment Variables

**Total Environment Variables**: 32+
**Configuration**: docker-compose.unified.yml, entrypoint-unified.sh

### Display Configuration

| Variable | Current Value | Type | Options | Location | Priority |
|----------|--------------|------|---------|----------|----------|
| **DISPLAY** | :1 | string | :0, :1, :2, etc. | docker-compose.unified.yml:72 | High |

### API Keys

| Variable | Current Value | Type | Source | Location | Priority |
|----------|--------------|------|--------|----------|----------|
| **ANTHROPIC_API_KEY** | ${ANTHROPIC_API_KEY:-} | string | .env file | docker-compose.unified.yml:75 | Critical |
| **OPENAI_API_KEY** | ${OPENAI_API_KEY:-} | string | .env file | docker-compose.unified.yml:76 | High |
| **OPENAI_ORG_ID** | ${OPENAI_ORG_ID:-} | string | .env file | docker-compose.unified.yml:77 | Medium |
| **GOOGLE_GEMINI_API_KEY** | ${GOOGLE_GEMINI_API_KEY:-} | string | .env file | docker-compose.unified.yml:78 | High |
| **GOOGLE_API_KEY** | ${GOOGLE_GEMINI_API_KEY:-} | string | .env file | docker-compose.unified.yml:79 | High |
| **GITHUB_TOKEN** | ${GITHUB_TOKEN:-} | string | .env file | docker-compose.unified.yml:80 | High |
| **CONTEXT7_API_KEY** | ${CONTEXT7_API_KEY:-} | string | .env file | docker-compose.unified.yml:81 | Low |
| **BRAVE_API_KEY** | ${BRAVE_API_KEY:-} | string | .env file | docker-compose.unified.yml:82 | Low |

### Z.AI Configuration

| Variable | Current Value | Type | Options | Location | Priority |
|----------|--------------|------|---------|----------|----------|
| **ZAI_ANTHROPIC_API_KEY** | ${ZAI_ANTHROPIC_API_KEY:-} | string | API key | docker-compose.unified.yml:85 | Critical |
| **ZAI_BASE_URL** | https://api.z.ai/api/anthropic | URL | Any URL | docker-compose.unified.yml:86 | Critical |
| **ZAI_API_KEY** | ${ZAI_API_KEY:-} | string | API key | docker-compose.unified.yml:87 | High |
| **ZAI_CONTAINER_URL** | http://localhost:9600 | URL | Any URL | docker-compose.unified.yml:88 | High |
| **CLAUDE_WORKER_POOL_SIZE** | 4 | number | 1-16 | docker-compose.unified.yml:89 | High |
| **CLAUDE_MAX_QUEUE_SIZE** | 50 | number | 1-∞ | docker-compose.unified.yml:90 | High |

### Management API Configuration

| Variable | Current Value | Type | Options | Location | Priority |
|----------|--------------|------|---------|----------|----------|
| **MANAGEMENT_API_KEY** | change-this-secret-key | string | Any string | docker-compose.unified.yml:93 | Critical |
| **MANAGEMENT_API_PORT** | 9090 | number | 1024-65535 | docker-compose.unified.yml:94 | High |
| **MANAGEMENT_API_HOST** | 0.0.0.0 | IP address | Any IP | docker-compose.unified.yml:95 | High |

### System Configuration

| Variable | Current Value | Type | Options | Location | Priority |
|----------|--------------|------|---------|----------|----------|
| **WORKSPACE** | /home/devuser/workspace | path | Any path | docker-compose.unified.yml:98 | High |
| **AGENTS_DIR** | /home/devuser/agents | path | Any path | docker-compose.unified.yml:99 | High |
| **ENABLE_DESKTOP** | true | boolean | true/false | docker-compose.unified.yml:100 | High |
| **GPU_ACCELERATION** | true | boolean | true/false | docker-compose.unified.yml:101 | High |
| **CUDA_VISIBLE_DEVICES** | all | string | all, 0,1, etc. | docker-compose.unified.yml:102 | Critical |

### Service Toggles

| Variable | Current Value | Type | Options | Location | Priority |
|----------|--------------|------|---------|----------|----------|
| **ENABLE_VNC** | true | boolean | true/false | docker-compose.unified.yml:105 | High |
| **ENABLE_SSH** | true | boolean | true/false | docker-compose.unified.yml:106 | High |
| **ENABLE_MANAGEMENT_API** | true | boolean | true/false | docker-compose.unified.yml:107 | High |

### Logging Configuration

| Variable | Current Value | Type | Options | Location | Priority |
|----------|--------------|------|---------|----------|----------|
| **LOG_LEVEL** | info | string | debug, info, warn, error | docker-compose.unified.yml:110 | Medium |
| **NODE_ENV** | production | string | development, production | docker-compose.unified.yml:111 | Medium |

---

## tmux Workspace Configuration

**Configuration File**: `unified-config/tmux-autostart.sh`
**Session Name**: workspace
**Total Windows**: 8

### tmux Settings

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **Session Name** | workspace | string | Any name | tmux-autostart.sh:17 | High |
| **History Limit** | 50000 | number | 1000-∞ | tmux-autostart.sh:20 | Medium |
| **Status Bar Background** | blue | color | Any color | tmux-autostart.sh:23 | Low |
| **Status Bar Foreground** | white | color | Any color | tmux-autostart.sh:23 | Low |
| **Status Left** | #[bg=green,fg=black] TURBO-FLOW | string | Any string | tmux-autostart.sh:24 | Low |
| **Status Right** | #[bg=yellow,fg=black] %Y-%m-%d %H:%M | string | Any format | tmux-autostart.sh:25 | Low |

### Window Configuration

| Window | Name | Purpose | Working Dir | Autostart | Location |
|--------|------|---------|------------|-----------|----------|
| **0** | Claude-Main | Primary Claude Code workspace | $WORKSPACE | true | tmux-autostart.sh:30-44 |
| **1** | Claude-Agent | Agent execution and testing | $WORKSPACE | true | tmux-autostart.sh:50-59 |
| **2** | Services | Supervisord status monitoring | $WORKSPACE | true | tmux-autostart.sh:65-75 |
| **3** | Development | Python/Rust/CUDA development | $WORKSPACE | true | tmux-autostart.sh:80-95 |
| **4** | Logs | Service logs (split view) | $WORKSPACE | true | tmux-autostart.sh:101-110 |
| **5** | System | htop resource monitoring | $WORKSPACE | true | tmux-autostart.sh:116-117 |
| **6** | VNC-Status | VNC server information | $WORKSPACE | true | tmux-autostart.sh:123-137 |
| **7** | SSH-Shell | General purpose shell | $WORKSPACE | true | tmux-autostart.sh:143-157 |

---

## VNC Desktop Settings

### VNC Server Configuration

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **Display** | :1 | string | :0, :1, :2, etc. | supervisord.unified.conf:23 | Critical |
| **Port** | 5901 | number | 5900-5999 | supervisord.unified.conf:23 | Critical |
| **Geometry (Resolution)** | 1920x1080 | resolution | Any (e.g., 1280x720, 2560x1440) | supervisord.unified.conf:23 | Medium |
| **Depth (Color Depth)** | 24 | number | 8, 16, 24, 32 | supervisord.unified.conf:23 | Low |
| **Security Types** | None | string | None, VncAuth, Plain | supervisord.unified.conf:23 | High |
| **Always Shared** | true | boolean | true/false | supervisord.unified.conf:23 | Medium |
| **Accept Key Events** | true | boolean | true/false | supervisord.unified.conf:23 | Medium |
| **Accept Pointer Events** | true | boolean | true/false | supervisord.unified.conf:23 | Medium |
| **Accept Set Desktop Size** | true | boolean | true/false | supervisord.unified.conf:23 | Low |
| **Send Cut Text** | true | boolean | true/false | supervisord.unified.conf:23 | Low |
| **Accept Cut Text** | true | boolean | true/false | supervisord.unified.conf:23 | Low |
| **Render Node** | /dev/dri/renderD128 | device | Any DRI device | supervisord.unified.conf:23 | Medium |
| **Password** | turboflow | string | Any password | Dockerfile.unified:232 | Critical |

### Desktop Environment

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **Desktop Environment** | XFCE4 | string | XFCE4, GNOME, KDE, etc. | supervisord.unified.conf:37 | High |
| **Screensaver Enabled** | false | boolean | true/false | Dockerfile.unified:249 | Low |
| **Screen Lock Enabled** | false | boolean | true/false | Dockerfile.unified:253 | Low |
| **Display Power Management** | false | boolean | true/false | Dockerfile.unified:260 | Low |
| **Blank on AC** | 0 (disabled) | number | 0-∞ minutes | Dockerfile.unified:259 | Low |

---

## SSH Configuration

**Configuration File**: `/etc/ssh/sshd_config` (modified in Dockerfile.unified)

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **Port** | 22 | number | 1-65535 | Dockerfile.unified:273 | Critical |
| **Permit Root Login** | no | string | yes, no, prohibit-password | Dockerfile.unified:271 | Critical |
| **Password Authentication** | yes | string | yes, no | Dockerfile.unified:272 | High |
| **Allowed Users** | devuser | string | Any users | Dockerfile.unified:273 | Critical |
| **Default User Password** | turboflow | string | Any password | Dockerfile.unified:276 | Critical |

---

## Logging Configuration

### Log File Locations

| Service | stdout Log | stderr Log | Location | Priority |
|---------|-----------|-----------|----------|----------|
| **supervisord** | /var/log/supervisord.log | - | supervisord.unified.conf:3 | High |
| **dbus** | /var/log/dbus.log | /var/log/dbus.error.log | supervisord.unified.conf:236-237 | Medium |
| **dbus-user** | /var/log/dbus-user.log | /var/log/dbus-user.error.log | supervisord.unified.conf:246-247 | Medium |
| **sshd** | /var/log/sshd.log | /var/log/sshd.error.log | supervisord.unified.conf:56-57 | Medium |
| **xvnc** | /var/log/xvnc.log | /var/log/xvnc.error.log | supervisord.unified.conf:29-30 | Medium |
| **xfce4** | /var/log/xfce4.log | /var/log/xfce4.error.log | supervisord.unified.conf:43-44 | Medium |
| **management-api** | /var/log/management-api.log | /var/log/management-api.error.log | supervisord.unified.conf:71-72 | High |
| **code-server** | /var/log/code-server.log | /var/log/code-server.error.log | supervisord.unified.conf:85-86 | Medium |
| **claude-zai** | /var/log/claude-zai.log | /var/log/claude-zai.error.log | supervisord.unified.conf:101-102 | High |
| **gemini-flow** | /var/log/gemini-flow.log | /var/log/gemini-flow.error.log | supervisord.unified.conf:208-209 | Medium |
| **web-summary-mcp** | /var/log/web-summary-mcp.log | /var/log/web-summary-mcp.error.log | supervisord.unified.conf:116-117 | Medium |
| **tmux-autostart** | /var/log/tmux-autostart.log | /var/log/tmux-autostart.error.log | supervisord.unified.conf:223-224 | Low |

### Log Volume

| Parameter | Current Value | Type | Options | Location | Priority |
|-----------|--------------|------|---------|----------|----------|
| **logs volume** | Named volume (logs) | volume | local, nfs, etc. | docker-compose.unified.yml:58 | Medium |
| **logs mount path** | /var/log | path | Any path | docker-compose.unified.yml:58 | Medium |

---

## Summary Tables

### Service Ports Summary

| Port | Service | User | Internal/External | Autostart | Priority |
|------|---------|------|------------------|-----------|----------|
| **22** | SSH | root | External (→2222) | true | High |
| **5901** | VNC | devuser | External | true | High |
| **8080** | code-server | devuser | External | true | Medium |
| **9090** | Management API | devuser | External | true | High |
| **9600** | Z.AI | zai-user | Internal Only | true | Critical |
| **9876** | Blender MCP | devuser | Internal | false | Medium |
| **9877** | QGIS MCP | devuser | Internal | false | Medium |
| **9878** | PBR MCP | devuser | Internal | false | Medium |
| **9879** | Playwright Proxy | devuser | Internal | false | Medium |
| **9500** | Claude Flow MCP | devuser | Internal | - | High |

### Critical Security Parameters

| Parameter | Current Value | Recommended Action | Priority |
|-----------|--------------|-------------------|----------|
| **SSH Password** | turboflow | Change before production | Critical |
| **VNC Password** | turboflow | Change before production | Critical |
| **Management API Key** | change-this-secret-key | Change before production | Critical |
| **code-server auth** | none | Enable authentication for production | High |
| **Permit Root Login** | no | Keep disabled | Good |
| **Docker Capabilities** | SYS_ADMIN, NET_ADMIN, SYS_PTRACE | Review necessity | Medium |

### Resource Allocation Summary

| Resource | Limit | Reservation | Adjustable | Priority |
|----------|-------|-------------|-----------|----------|
| **CPU** | 32 cores | 8 cores | Yes | Critical |
| **Memory** | 64G | 16G | Yes | Critical |
| **Shared Memory** | 32gb | - | Yes | High |
| **GPU** | All available | All available | Yes | Critical |

### Default Credentials Reference

| Service | Username | Password | Auth Method | Change Priority |
|---------|----------|----------|-------------|-----------------|
| **SSH** | devuser | turboflow | password | Critical |
| **VNC** | - | turboflow | password | Critical |
| **code-server** | - | none | none | High |
| **Management API** | - | change-this-secret-key | X-API-Key header | Critical |

---

## Configuration File Locations Reference

| Component | Configuration File | Location | Format |
|-----------|-------------------|----------|--------|
| **Docker Build** | Dockerfile.unified | multi-agent-docker/ | Dockerfile |
| **Docker Compose** | docker-compose.unified.yml | multi-agent-docker/ | YAML |
| **Supervisord** | supervisord.unified.conf | unified-config/ | INI |
| **Entrypoint** | entrypoint-unified.sh | unified-config/ | Bash |
| **MCP Servers** | mcp.json | mcp-infrastructure/ | JSON |
| **MCP Full Registry** | mcp-full-registry.json | mcp-infrastructure/ | JSON |
| **Z.AI Service** | server.js | multi-agent-docker/claude-zai/wrapper/ | JavaScript |
| **Management API** | server.js | multi-agent-docker/management-api/ | JavaScript |
| **tmux Workspace** | tmux-autostart.sh | unified-config/ | Bash |
| **User Switching** | as-gemini.sh, as-openai.sh, as-zai.sh | unified-config/scripts/ | Bash |
| **Skills** | SKILL.md (each skill) | skills/*/  | Markdown |

---

## Total Parameters Summary

| Category | Count | Critical | High | Medium | Low |
|----------|-------|----------|------|--------|-----|
| **Docker Container** | 35 | 8 | 15 | 10 | 2 |
| **Supervisord Services** | 98 | 12 | 45 | 35 | 6 |
| **MCP Servers** | 22 | 2 | 6 | 12 | 2 |
| **Z.AI Service** | 15 | 4 | 5 | 4 | 2 |
| **Management API** | 28 | 3 | 8 | 14 | 3 |
| **Skills System** | 27 | 0 | 6 | 18 | 3 |
| **User Management** | 12 | 2 | 4 | 6 | 0 |
| **Resource Limits** | 18 | 8 | 7 | 2 | 1 |
| **Network** | 12 | 1 | 6 | 5 | 0 |
| **Environment Variables** | 32 | 8 | 12 | 10 | 2 |
| **tmux Workspace** | 14 | 0 | 2 | 2 | 10 |
| **VNC Desktop** | 18 | 3 | 2 | 8 | 5 |
| **SSH** | 5 | 4 | 1 | 0 | 0 |
| **Logging** | 13 | 0 | 3 | 10 | 0 |
| **TOTAL** | **287** | **55** | **122** | **136** | **36** |

---

## Recommendations

### High Priority Changes for Production

1. **Change Default Credentials** (Critical)
   - SSH password: Change from "turboflow"
   - VNC password: Change from "turboflow"
   - Management API key: Change from "change-this-secret-key"
   - code-server: Enable authentication

2. **Security Hardening** (High)
   - Review and minimize Docker capabilities (SYS_ADMIN, NET_ADMIN)
   - Enable AppArmor/SELinux profiles
   - Restrict network access to management ports (9090, 5901, 8080)
   - Consider TLS/SSL for VNC and code-server

3. **Resource Optimization** (Medium)
   - Adjust CPU/Memory limits based on actual usage
   - Monitor GPU utilization and adjust CUDA_VISIBLE_DEVICES
   - Review worker pool sizes for Z.AI based on load

### Configuration Customization Points

1. **Development Environment**
   - Adjust VNC resolution based on display requirements
   - Customize tmux window layout for workflow
   - Enable/disable specific MCP skills based on needs

2. **Service Scaling**
   - Z.AI worker pool: Increase for higher concurrency
   - Management API rate limits: Adjust for expected load
   - supervisord priorities: Reorder based on critical services

3. **Storage**
   - Consider external volume mounts for persistence
   - Adjust log retention policies
   - Configure model cache size limits

---

## Audit Metadata

- **Audit Date**: 2025-10-22
- **Container Version**: Unified (multi-agent-docker)
- **Total Files Analyzed**: 15
- **Total Parameters**: 287
- **Configuration Complexity**: High
- **Security Risk**: Medium (default credentials)
- **Recommended Review Frequency**: Quarterly

---

**End of Audit Report**
