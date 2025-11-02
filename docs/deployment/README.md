# VisionFlow Deployment Guide

## Overview

VisionFlow is deployed as a Rust backend + React frontend system. This guide covers both development and production deployment scenarios.

**Current Technology Stack:**
- **Backend**: Rust (Actix-web framework)
- **Frontend**: React + Vite
- **Database**: SQLite (unified.db with 8 core tables)
- **Default API Port**: 3030 (configurable via SYSTEM_NETWORK_PORT)

## Deployment Approaches

### Production Deployment (Recommended)

**Architecture:**
- Backend: Rust binary (compiled release build)
- Frontend: Static Vite build served by Nginx
- Database: SQLite (file-based, no server required)
- Configuration: Environment variables + TOML files

**Deployment Steps:**

```bash
# Step 1: Build backend
cd /path/to/visionflow
cargo build --release --features gpu  # or without gpu flag

# Step 2: Build frontend
cd client/
npm install
npm run build  # produces dist/ folder

# Step 3: Setup directories
mkdir -p /opt/visionflow
mkdir -p /opt/visionflow/data
cp target/release/webxr /opt/visionflow/
cp -r client/dist /opt/visionflow/www

# Step 4: Copy configuration
cp data/settings.yaml /opt/visionflow/data/
cp data/dev_config.toml /opt/visionflow/data/

# Step 5: Start backend
cd /opt/visionflow
export SYSTEM_NETWORK_PORT=3030
./webxr
```

**Configuration:**

```bash
# Environment variables
export SYSTEM_NETWORK_PORT=3030        # API server port
export RUST_LOG=info                   # Logging level: debug, info, warn, error
export ENABLE_GPU=true                 # GPU acceleration (if available)
```

**Systemd Service (recommended for production):**

Create `/etc/systemd/system/visionflow.service`:

```ini
[Unit]
Description=VisionFlow Backend API
After=network.target
Documentation=https://docs.visionflow.local

[Service]
Type=simple
User=visionflow
Group=visionflow
WorkingDirectory=/opt/visionflow
Environment="SYSTEM_NETWORK_PORT=3030"
Environment="RUST_LOG=info"
Environment="ENABLE_GPU=true"
ExecStart=/opt/visionflow/webxr
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable visionflow
sudo systemctl start visionflow
```

**Nginx Configuration (reverse proxy):**

Create `/etc/nginx/sites-available/visionflow`:

```nginx
upstream visionflow_backend {
    server localhost:3030;
}

server {
    listen 80;
    server_name visionflow.local;
    client_max_body_size 100M;

    # Frontend static files
    location / {
        root /opt/visionflow/www;
        try_files $uri $uri/ /index.html;
    }

    # API endpoints
    location /api/ {
        proxy_pass http://visionflow_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # WebSocket endpoints
    location /ws/ {
        proxy_pass http://visionflow_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/visionflow /etc/nginx/sites-enabled/
sudo nginx -t  # test configuration
sudo systemctl reload nginx
```

### Development Deployment

**Quick Start (5 minutes):**

```bash
# Terminal 1: Backend
cargo run  # or: cargo run --release

# Terminal 2: Frontend
cd client
npm install
npm run dev  # Vite dev server on localhost:5173
```

**Access Points:**
- Backend API: `http://localhost:3030`
- Frontend Dev: `http://localhost:5173`
- Direct API testing: `curl http://localhost:3030/api/health`

**API Health Check:**
```bash
curl -i http://localhost:3030/api/health
# Should return 200 OK with {"status":"healthy"} or similar
```

### Docker Deployment

**Status**: ❌ NOT CURRENTLY IMPLEMENTED for VisionFlow main project

**Important**: The `/multi-agent-docker/` directory is a SEPARATE system (Turbo Flow Claude infrastructure) and is NOT used for VisionFlow deployment.

**If you need Docker deployment**, you would need to create:
- `Dockerfile` for Rust backend
- `Dockerfile` for frontend build
- `docker-compose.yml` for orchestration

See `/docs/deployment/docker-future.md` for planned Docker implementation details.

## Configuration

### Required Files

```
data/
├── settings.yaml          # Main application configuration (legacy, being phased out)
├── dev_config.toml        # Physics parameters and tuning
└── unified.db             # Unified SQLite database (created automatically)
                           # Contains 8 tables: graph_nodes, graph_edges,
                           # owl_classes, owl_class_hierarchy, owl_properties,
                           # owl_axioms, graph_statistics, file_metadata
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYSTEM_NETWORK_PORT` | `3030` | API server listening port |
| `RUST_LOG` | `info` | Logging level: debug, info, warn, error |
| `ENABLE_GPU` | `true` | Enable GPU acceleration (if hardware available) |
| `CUDA_VISIBLE_DEVICES` | all | CUDA device selection (for multi-GPU systems) |
| `FORCE_FULL_SYNC` | `false` | Force full database sync on startup (ignores file_metadata cache) |
| `DB_PATH` | `./data/unified.db` | Path to unified SQLite database |

### Example Configuration (settings.yaml)

```yaml
# VisionFlow Settings
server:
  port: 3030
  host: "0.0.0.0"
  workers: 4
  timeout_secs: 30

database:
  unified_db_path: "./data/unified.db"  # Single unified database
  enable_wal: true  # Write-ahead logging for concurrency
  force_full_sync: false  # Set to true to ignore file_metadata cache

physics:
  repulsion_strength: 150
  attraction_strength: 50
  damping: 0.95

gpu:
  enabled: true
  device: 0  # Primary CUDA device
  max_concurrent_streams: 4

logging:
  level: info
  file: "logs/visionflow.log"
```

## Monitoring & Health Checks

### Health Check Endpoint

```bash
curl http://localhost:3030/health
# Response: 200 OK
# Body: {"status":"healthy","timestamp":"2025-10-27T..."}
```

### Log Files

**Development**: Logs to stdout (console)

**Production with systemd**:
```bash
# View logs
sudo journalctl -u visionflow -f

# Last 100 lines
sudo journalctl -u visionflow -n 100

# Since last boot
sudo journalctl -u visionflow -b
```

### Performance Monitoring

**API Performance** (via curl):
```bash
time curl http://localhost:3030/api/graph/data
# Check response time and HTTP status
```

**System Monitoring**:
```bash
# Monitor memory usage
watch -n 1 'ps aux | grep webxr'

# Monitor open connections
netstat -an | grep 3030 | grep ESTABLISHED | wc -l
```

## Security Checklist

⚠️ **CRITICAL**: VisionFlow currently has NO authentication implemented.

**Before Production Deployment:**
- [ ] **Review security implications** - All API endpoints are PUBLIC
- [ ] **Implement authentication** - Add JWT or OAuth2 middleware
- [ ] **Enable HTTPS/TLS** - Use Nginx reverse proxy with SSL certificate
- [ ] **Configure firewall** - Only expose necessary ports (80, 443)
  ```bash
  sudo ufw allow 80/tcp  # HTTP
  sudo ufw allow 443/tcp # HTTPS
  sudo ufw allow 3030/tcp # Backend (internal only)
  ```
- [ ] **Restrict CORS** - Configure allowed origins in backend
- [ ] **Database security** - SQLite file permissions:
  ```bash
  chmod 600 data/unified.db
  chown visionflow:visionflow data/unified.db
  ```
- [ ] **Review settings.yaml** - Ensure no secrets hardcoded
- [ ] **Enable logging** - Set `RUST_LOG=info` or `warn` for production
- [ ] **Rate limiting** - Consider implementing API rate limiting
- [ ] **Regular backups** - Backup unified database regularly:
  ```bash
  # Binary backup (recommended)
  sqlite3 data/unified.db ".backup /backup/visionflow-$(date +%Y%m%d).db"

  # SQL export (for version control)
  sqlite3 data/unified.db ".dump" > /backup/visionflow-$(date +%Y%m%d).sql

  # Compressed backup
  tar -czf /backup/visionflow-$(date +%Y%m%d).tar.gz data/unified.db
  ```

## Troubleshooting

### Backend won't start

```bash
# Check if port is already in use
sudo lsof -i :3030

# Check logs
journalctl -u visionflow -n 50 --no-pager

# Try different port
export SYSTEM_NETWORK_PORT=3031
./webxr
```

### Frontend not loading

```bash
# Check Nginx is running
sudo systemctl status nginx

# Check Nginx configuration
sudo nginx -t

# Check frontend files exist
ls -la /opt/visionflow/www/index.html

# Check permissions
sudo chown -R www-data:www-data /opt/visionflow/www
```

### WebSocket connection fails

```bash
# Check WebSocket path
curl -i http://localhost:3030/ws/client-messages

# Check Nginx WebSocket config
grep -A5 "location /ws/" /etc/nginx/sites-available/visionflow

# Ensure Upgrade/Connection headers are set
curl -i -H "Upgrade: websocket" -H "Connection: Upgrade" \
  http://localhost:3030/ws/client-messages
```

### GPU not detected

```bash
# Check CUDA availability
nvidia-smi

# Check ENABLE_GPU flag
echo $ENABLE_GPU

# Try disabling GPU
export ENABLE_GPU=false
./webxr
```

## Related Documentation

- [Development Setup](../developer-guide/01-development-setup.md)
- [Architecture Overview](../architecture/overview.md)
- [Configuration Reference](../reference/configuration.md)
- [Security Best Practices](../guides/security.md)
- [API Reference](../reference/api/index.md)
