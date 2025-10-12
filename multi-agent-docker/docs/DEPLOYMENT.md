# Production Deployment Guide

Comprehensive guide for deploying the Agentic Flow multi-agent system in production environments.

## Table of Contents

1. [Production vs Development Modes](#production-vs-development-modes)
2. [Security Hardening](#security-hardening)
3. [Reverse Proxy Setup](#reverse-proxy-setup)
4. [Resource Planning](#resource-planning)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Backup and Restore](#backup-and-restore)
7. [Scaling Considerations](#scaling-considerations)
8. [High Availability Setup](#high-availability-setup)
9. [Docker Swarm/Kubernetes](#docker-swarmkubernetes)
10. [Cloud Deployment](#cloud-deployment)
11. [Disaster Recovery](#disaster-recovery)
12. [Performance Tuning](#performance-tuning)

---

## Production vs Development Modes

### Development Mode

```yaml
# docker-compose.dev.yml
services:
  agentic-flow-cachyos:
    environment:
      NODE_ENV: development
      LOG_LEVEL: debug
      ENABLE_DESKTOP: true
      ENABLE_CODE_SERVER: true
    ports:
      - "9090:9090"   # Management API
      - "5901:5901"   # VNC
      - "6901:6901"   # noVNC
      - "8080:8080"   # code-server
    security_opt:
      - seccomp:unconfined
      - apparmor:unconfined
    volumes:
      - ./workspace:/home/devuser/workspace:rw
```

### Production Mode

```yaml
# docker-compose.yml
services:
  agentic-flow-cachyos:
    environment:
      NODE_ENV: production
      LOG_LEVEL: info
      ENABLE_DESKTOP: false
      ENABLE_CODE_SERVER: false
    ports:
      - "127.0.0.1:9090:9090"  # Only expose to localhost
    security_opt: []  # Use default security
    volumes:
      - workspace:/home/devuser/workspace:rw  # Named volumes
    deploy:
      resources:
        limits:
          memory: 64G
          cpus: '32'
      restart_policy:
        condition: on-failure
        max_attempts: 3
```

### Key Differences

| Feature | Development | Production |
|---------|-------------|------------|
| Logging | `debug` | `info` or `warn` |
| Desktop/VNC | Enabled | Disabled |
| Port Exposure | All ports | Localhost only |
| Restart Policy | `unless-stopped` | `on-failure` |
| Security Options | Relaxed | Hardened |
| Volume Mounts | Bind mounts | Named volumes |
| Resource Limits | Minimal | Strict |

---

## Security Hardening

### 1. API Key Management

**Generate Strong API Keys:**

```bash
# Generate secure API key
export MANAGEMENT_API_KEY=$(openssl rand -hex 32)

# Store in secrets manager (example with AWS Secrets Manager)
aws secretsmanager create-secret \
  --name agentic-flow/management-api-key \
  --secret-string "$MANAGEMENT_API_KEY"
```

**Environment Variable Security:**

```bash
# NEVER commit .env to version control
echo ".env" >> .gitignore

# Use .env.example as template
cp .env.example .env
chmod 600 .env  # Restrict permissions

# Validate required keys are set
required_keys=(
  "MANAGEMENT_API_KEY"
  "ANTHROPIC_API_KEY"
  "GOOGLE_GEMINI_API_KEY"
)

for key in "${required_keys[@]}"; do
  if [ -z "${!key}" ]; then
    echo "ERROR: $key is not set"
    exit 1
  fi
done
```

### 2. Firewall Configuration

**UFW (Ubuntu/Debian):**

```bash
# Default deny
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (change port if customized)
ufw allow 22/tcp

# Allow only specific IPs to access Management API
ufw allow from 10.0.0.0/8 to any port 9090 proto tcp

# Enable firewall
ufw enable
```

**iptables:**

```bash
# Flush existing rules
iptables -F

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Restrict Management API to specific subnet
iptables -A INPUT -p tcp --dport 9090 -s 10.0.0.0/8 -j ACCEPT

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### 3. Network Isolation

**Docker Network Security:**

```yaml
networks:
  agentic-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.enable_icc: "false"  # Disable inter-container communication
    ipam:
      config:
        - subnet: 172.28.0.0/16
          gateway: 172.28.0.1
```

**Internal Services Configuration:**

```yaml
services:
  agentic-flow-cachyos:
    networks:
      agentic-network:
        ipv4_address: 172.28.0.10

  claude-zai:
    networks:
      agentic-network:
        ipv4_address: 172.28.0.20
    # Only expose to internal network
    expose:
      - "9600"
    # Do NOT use 'ports' for internal services
```

### 4. TLS/HTTPS Configuration

Production environments MUST use TLS. Never expose HTTP APIs directly to the internet.

**Option A: Managed Load Balancer (Recommended)**

Use cloud provider's managed load balancer with automatic certificate management:
- AWS ALB + ACM
- GCP Load Balancer + Managed Certificates
- Azure Application Gateway + Key Vault

**Option B: Self-Managed with Let's Encrypt**

See [Reverse Proxy Setup](#reverse-proxy-setup) section.

### 5. Container Security

**Security Best Practices:**

```yaml
services:
  agentic-flow-cachyos:
    # Read-only root filesystem where possible
    read_only: false  # Cannot use read-only due to temp files

    # Drop unnecessary capabilities
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
      - DAC_OVERRIDE

    # No privileged mode
    privileged: false

    # Security options
    security_opt:
      - no-new-privileges:true
      - seccomp=default.json

    # Non-root user
    user: devuser

    # Resource limits prevent DoS
    deploy:
      resources:
        limits:
          memory: 64G
          cpus: '32'
          pids: 1000  # Limit processes
```

**Docker Daemon Hardening:**

```json
{
  "icc": false,
  "userns-remap": "default",
  "no-new-privileges": true,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "5"
  },
  "live-restore": true,
  "userland-proxy": false
}
```

---

## Reverse Proxy Setup

### Nginx Configuration

**Installation:**

```bash
apt-get update
apt-get install -y nginx certbot python3-certbot-nginx
```

**Configuration: `/etc/nginx/sites-available/agentic-flow`**

```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=5r/m;

# Upstream backend
upstream agentic_api {
    server 127.0.0.1:9090;
    keepalive 32;
}

# HTTP -> HTTPS redirect
server {
    listen 80;
    listen [::]:80;
    server_name api.yourdomain.com;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.yourdomain.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_trusted_certificate /etc/letsencrypt/live/api.yourdomain.com/chain.pem;

    # SSL configuration (Mozilla Intermediate)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Logging
    access_log /var/log/nginx/agentic-flow-access.log;
    error_log /var/log/nginx/agentic-flow-error.log warn;

    # API endpoints with rate limiting
    location /v1/ {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://agentic_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running tasks
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # Buffering
        proxy_buffering off;
        proxy_request_buffering off;
    }

    # Health check (no rate limit)
    location /health {
        proxy_pass http://agentic_api;
        access_log off;
    }

    # Metrics endpoint (restrict to monitoring systems)
    location /metrics {
        allow 10.0.0.0/8;    # Internal network
        deny all;

        proxy_pass http://agentic_api;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    # API documentation (optional, consider restricting)
    location /docs {
        proxy_pass http://agentic_api;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
```

**Enable and Start:**

```bash
# Test configuration
nginx -t

# Enable site
ln -s /etc/nginx/sites-available/agentic-flow /etc/nginx/sites-enabled/

# Get SSL certificate
certbot --nginx -d api.yourdomain.com

# Reload nginx
systemctl reload nginx

# Auto-renewal
systemctl enable certbot.timer
```

### Traefik Configuration

**docker-compose.traefik.yml:**

```yaml
version: '3.8'

services:
  traefik:
    image: traefik:v2.10
    container_name: traefik
    command:
      # Enable dashboard (disable in production or restrict access)
      - "--api.dashboard=false"

      # Docker provider
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"

      # Entrypoints
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"

      # Automatic HTTPS redirect
      - "--entrypoints.web.http.redirections.entrypoint.to=websecure"
      - "--entrypoints.web.http.redirections.entrypoint.scheme=https"

      # Let's Encrypt
      - "--certificatesresolvers.letsencrypt.acme.email=admin@yourdomain.com"
      - "--certificatesresolvers.letsencrypt.acme.storage=/acme.json"
      - "--certificatesresolvers.letsencrypt.acme.tlschallenge=true"

      # Logging
      - "--log.level=INFO"
      - "--accesslog=true"

      # Metrics
      - "--metrics.prometheus=true"
      - "--metrics.prometheus.buckets=0.1,0.3,1.2,5.0"

    ports:
      - "80:80"
      - "443:443"

    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - traefik-acme:/acme.json

    networks:
      - agentic-network

    restart: unless-stopped

  agentic-flow-cachyos:
    # ... existing configuration ...
    labels:
      - "traefik.enable=true"

      # HTTP router
      - "traefik.http.routers.agentic-api.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.agentic-api.entrypoints=websecure"
      - "traefik.http.routers.agentic-api.tls=true"
      - "traefik.http.routers.agentic-api.tls.certresolver=letsencrypt"

      # Service
      - "traefik.http.services.agentic-api.loadbalancer.server.port=9090"

      # Middleware: Rate limiting
      - "traefik.http.middlewares.api-ratelimit.ratelimit.average=10"
      - "traefik.http.middlewares.api-ratelimit.ratelimit.burst=20"

      # Middleware: Security headers
      - "traefik.http.middlewares.security-headers.headers.stsSeconds=31536000"
      - "traefik.http.middlewares.security-headers.headers.stsIncludeSubdomains=true"
      - "traefik.http.middlewares.security-headers.headers.frameDeny=true"
      - "traefik.http.middlewares.security-headers.headers.contentTypeNosniff=true"

      # Apply middlewares
      - "traefik.http.routers.agentic-api.middlewares=api-ratelimit,security-headers"

    ports: []  # Don't expose ports directly

volumes:
  traefik-acme:

networks:
  agentic-network:
    external: true
```

---

## Resource Planning

### Hardware Requirements

#### Minimum Production Configuration

| Component | Specification | Notes |
|-----------|---------------|-------|
| CPU | 8 cores (16 threads) | x86_64, AVX2 support recommended |
| RAM | 32 GB | Minimum for 2-3 concurrent agents |
| GPU | NVIDIA GPU with 8GB VRAM | Optional but recommended for ML workloads |
| Storage | 500 GB SSD | NVMe preferred for model cache |
| Network | 1 Gbps | For API calls and model downloads |

#### Recommended Production Configuration

| Component | Specification | Notes |
|-----------|---------------|-------|
| CPU | 32 cores (64 threads) | AMD Ryzen/EPYC or Intel Xeon |
| RAM | 128 GB | Supports 10+ concurrent agents |
| GPU | NVIDIA RTX 4090 or A100 | For local model inference |
| Storage | 2 TB NVMe SSD | RAID 10 recommended |
| Network | 10 Gbps | For high-throughput scenarios |

### Storage Planning

**Volume Sizing:**

```yaml
volumes:
  workspace:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/fast-storage/workspace  # 500 GB minimum

  model-cache:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/fast-storage/models    # 200 GB minimum

  agent-memory:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/fast-storage/memory    # 50 GB minimum

  management-logs:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/standard-storage/logs  # 100 GB minimum
```

**Storage Growth Estimates:**

- **Workspace**: 10-50 GB per active project
- **Model Cache**: 5-20 GB per model
- **Agent Memory**: 1-5 GB per agent session
- **Logs**: 1-10 GB per day (depending on verbosity)

**Monitoring Storage Usage:**

```bash
# Create monitoring script
cat > /usr/local/bin/check-docker-volumes.sh << 'EOF'
#!/bin/bash
df -h /var/lib/docker/volumes | awk 'NR>1 {print $5 " " $6}' | while read usage mount; do
  usage_pct=$(echo $usage | sed 's/%//')
  if [ $usage_pct -gt 80 ]; then
    echo "WARNING: $mount is ${usage} full"
  fi
done
EOF

chmod +x /usr/local/bin/check-docker-volumes.sh

# Add to cron (daily)
echo "0 6 * * * /usr/local/bin/check-docker-volumes.sh" | crontab -
```

### CPU and Memory Allocation

**Resource Limits per Service:**

```yaml
services:
  agentic-flow-cachyos:
    deploy:
      resources:
        limits:
          cpus: '32'        # Maximum CPU cores
          memory: 64G       # Maximum memory
        reservations:
          cpus: '8'         # Minimum guaranteed
          memory: 16G       # Minimum guaranteed
          devices:
            - driver: nvidia
              count: all    # All GPUs
              capabilities: [gpu, compute, utility]

  claude-zai:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 1G
```

**Dynamic Resource Adjustment:**

```bash
# Update container resources without restart
docker update --cpus="16" --memory="32g" agentic-flow-cachyos
```

### GPU Configuration

**NVIDIA Docker Runtime:**

```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker
```

**GPU Resource Management:**

```yaml
services:
  agentic-flow-cachyos:
    runtime: nvidia
    environment:
      CUDA_VISIBLE_DEVICES: "0,1"  # Use GPU 0 and 1
      GPU_ACCELERATION: "true"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']  # Specific GPUs
              capabilities: [gpu]
```

**Multi-GPU Setup:**

```bash
# List available GPUs
nvidia-smi -L

# Monitor GPU usage
nvidia-smi dmon -s pucvmet

# Set GPU persistence mode (recommended for production)
nvidia-smi -pm 1
```

---

## Monitoring and Logging

### Prometheus Metrics

**Prometheus Configuration: `prometheus.yml`**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agentic-flow'
    static_configs:
      - targets: ['agentic-flow-cachyos:9090']
    metrics_path: '/metrics'
    bearer_token: 'YOUR_API_KEY'

  - job_name: 'docker'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

**Docker Compose with Monitoring Stack:**

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "127.0.0.1:9091:9090"
    networks:
      - agentic-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      GF_SECURITY_ADMIN_PASSWORD: "${GRAFANA_ADMIN_PASSWORD}"
      GF_USERS_ALLOW_SIGN_UP: "false"
    ports:
      - "127.0.0.1:3000:3000"
    networks:
      - agentic-network
    restart: unless-stopped

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: cadvisor
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - agentic-network
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    command:
      - '--path.rootfs=/host'
    volumes:
      - /:/host:ro,rslave
    networks:
      - agentic-network
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

### Grafana Dashboard

**Example Dashboard JSON** (`grafana/dashboards/agentic-flow.json`):

```json
{
  "dashboard": {
    "title": "Agentic Flow Metrics",
    "panels": [
      {
        "title": "Active Tasks",
        "targets": [{
          "expr": "agentic_flow_tasks_active"
        }]
      },
      {
        "title": "API Response Time",
        "targets": [{
          "expr": "rate(agentic_flow_http_request_duration_seconds_sum[5m]) / rate(agentic_flow_http_request_duration_seconds_count[5m])"
        }]
      },
      {
        "title": "Memory Usage",
        "targets": [{
          "expr": "container_memory_usage_bytes{name=\"agentic-flow-cachyos\"}"
        }]
      },
      {
        "title": "CPU Usage",
        "targets": [{
          "expr": "rate(container_cpu_usage_seconds_total{name=\"agentic-flow-cachyos\"}[5m]) * 100"
        }]
      }
    ]
  }
}
```

### Centralized Logging

**ELK Stack (Elasticsearch, Logstash, Kibana):**

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - agentic-network
    restart: unless-stopped

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    container_name: logstash
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline:ro
    networks:
      - agentic-network
    depends_on:
      - elasticsearch
    restart: unless-stopped

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: kibana
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    ports:
      - "127.0.0.1:5601:5601"
    networks:
      - agentic-network
    depends_on:
      - elasticsearch
    restart: unless-stopped

  agentic-flow-cachyos:
    # ... existing config ...
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
        labels: "service,environment"
        tag: "{{.Name}}/{{.ID}}"

volumes:
  elasticsearch-data:
```

**Logstash Pipeline** (`logstash/pipeline/logstash.conf`):

```conf
input {
  file {
    path => "/var/lib/docker/containers/*/*.log"
    codec => json
    type => "docker"
  }
}

filter {
  if [type] == "docker" {
    json {
      source => "message"
    }

    mutate {
      rename => { "log" => "message" }
    }

    date {
      match => [ "time", "ISO8601" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "agentic-flow-%{+YYYY.MM.dd}"
  }
}
```

### Alert Configuration

**Alertmanager Configuration:**

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'slack'

receivers:
  - name: 'slack'
    slack_configs:
      - channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

**Prometheus Alert Rules** (`alert.rules.yml`):

```yaml
groups:
  - name: agentic_flow
    interval: 30s
    rules:
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{name="agentic-flow-cachyos"} / container_spec_memory_limit_bytes{name="agentic-flow-cachyos"} > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.name }}"
          description: "Memory usage is above 90% for 5 minutes"

      - alert: HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total{name="agentic-flow-cachyos"}[5m]) > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.name }}"
          description: "CPU usage is above 80% for 10 minutes"

      - alert: ContainerDown
        expr: up{job="agentic-flow"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Agentic Flow container is down"
          description: "The container has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(agentic_flow_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for 5 minutes"
```

---

## Backup and Restore

### Backup Strategy

**Critical Data:**

1. **Docker Volumes**: workspace, model-cache, agent-memory, config-persist
2. **Configuration Files**: `.env`, `docker-compose.yml`, config files
3. **Management API State**: Task history, metrics
4. **SSL Certificates**: Let's Encrypt certificates

**Backup Script:**

```bash
#!/bin/bash
# /usr/local/bin/backup-agentic-flow.sh

set -e

BACKUP_DIR="/backups/agentic-flow"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"

echo "Starting backup at $(date)"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup Docker volumes
echo "Backing up Docker volumes..."
docker run --rm \
  -v workspace:/source/workspace:ro \
  -v model-cache:/source/model-cache:ro \
  -v agent-memory:/source/agent-memory:ro \
  -v config-persist:/source/config-persist:ro \
  -v management-logs:/source/management-logs:ro \
  -v "$BACKUP_PATH":/backup \
  alpine \
  tar czf /backup/volumes.tar.gz -C /source .

# Backup configuration files
echo "Backing up configuration..."
tar czf "$BACKUP_PATH/config.tar.gz" \
  -C /opt/agentic-flow \
  .env \
  docker-compose.yml \
  config/

# Backup SSL certificates (if using Let's Encrypt)
if [ -d /etc/letsencrypt ]; then
  echo "Backing up SSL certificates..."
  tar czf "$BACKUP_PATH/ssl-certs.tar.gz" -C /etc letsencrypt/
fi

# Create manifest
cat > "$BACKUP_PATH/manifest.txt" << EOF
Backup Date: $(date)
Hostname: $(hostname)
Docker Version: $(docker --version)
Backup Contents:
  - volumes.tar.gz: Docker volumes
  - config.tar.gz: Configuration files
  - ssl-certs.tar.gz: SSL certificates
EOF

# Calculate checksums
cd "$BACKUP_PATH"
sha256sum *.tar.gz > checksums.sha256

echo "Backup completed: $BACKUP_PATH"

# Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true

echo "Backup finished at $(date)"
```

**Automated Backups:**

```bash
# Install backup script
chmod +x /usr/local/bin/backup-agentic-flow.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-agentic-flow.sh >> /var/log/agentic-flow-backup.log 2>&1") | crontab -
```

### Offsite Backup

**AWS S3:**

```bash
#!/bin/bash
# Upload to S3
aws s3 sync /backups/agentic-flow s3://your-backup-bucket/agentic-flow/ \
  --storage-class STANDARD_IA \
  --exclude "*" \
  --include "*/volumes.tar.gz" \
  --include "*/config.tar.gz"
```

**Rsync to Remote Server:**

```bash
#!/bin/bash
# Rsync to backup server
rsync -avz --delete \
  -e "ssh -i /root/.ssh/backup_key" \
  /backups/agentic-flow/ \
  backup-server:/backups/agentic-flow/
```

### Restore Procedure

**Full Restore Script:**

```bash
#!/bin/bash
# /usr/local/bin/restore-agentic-flow.sh

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <backup-timestamp>"
  echo "Available backups:"
  ls -1 /backups/agentic-flow/
  exit 1
fi

BACKUP_PATH="/backups/agentic-flow/$1"

if [ ! -d "$BACKUP_PATH" ]; then
  echo "Backup not found: $BACKUP_PATH"
  exit 1
fi

echo "Restoring from backup: $BACKUP_PATH"

# Verify checksums
cd "$BACKUP_PATH"
sha256sum -c checksums.sha256 || {
  echo "Checksum verification failed!"
  exit 1
}

# Stop containers
echo "Stopping containers..."
cd /opt/agentic-flow
docker-compose down

# Restore configuration
echo "Restoring configuration..."
tar xzf "$BACKUP_PATH/config.tar.gz" -C /opt/agentic-flow/

# Restore volumes
echo "Restoring Docker volumes..."
docker run --rm \
  -v workspace:/target/workspace \
  -v model-cache:/target/model-cache \
  -v agent-memory:/target/agent-memory \
  -v config-persist:/target/config-persist \
  -v management-logs:/target/management-logs \
  -v "$BACKUP_PATH":/backup \
  alpine \
  sh -c "cd /target && tar xzf /backup/volumes.tar.gz"

# Restore SSL certificates
if [ -f "$BACKUP_PATH/ssl-certs.tar.gz" ]; then
  echo "Restoring SSL certificates..."
  tar xzf "$BACKUP_PATH/ssl-certs.tar.gz" -C /etc/
fi

# Start containers
echo "Starting containers..."
docker-compose up -d

echo "Restore completed successfully"
```

**Disaster Recovery Test:**

```bash
# Schedule monthly DR tests
cat > /usr/local/bin/test-restore.sh << 'EOF'
#!/bin/bash
# Test restore on isolated host

LATEST_BACKUP=$(ls -t /backups/agentic-flow/ | head -1)
echo "Testing restore of backup: $LATEST_BACKUP"

# Restore to test environment
/usr/local/bin/restore-agentic-flow.sh "$LATEST_BACKUP"

# Verify functionality
curl -f http://localhost:9090/health || {
  echo "Health check failed!"
  exit 1
}

echo "Restore test successful"
EOF

chmod +x /usr/local/bin/test-restore.sh
```

---

## Scaling Considerations

### Horizontal Scaling

**Load Balancer Configuration:**

```yaml
# docker-compose.scale.yml
services:
  agentic-flow-worker-1:
    extends:
      file: docker-compose.yml
      service: agentic-flow-cachyos
    container_name: agentic-worker-1
    hostname: agentic-worker-1

  agentic-flow-worker-2:
    extends:
      file: docker-compose.yml
      service: agentic-flow-cachyos
    container_name: agentic-worker-2
    hostname: agentic-worker-2

  agentic-flow-worker-3:
    extends:
      file: docker-compose.yml
      service: agentic-flow-cachyos
    container_name: agentic-worker-3
    hostname: agentic-worker-3

  haproxy:
    image: haproxy:2.8-alpine
    container_name: haproxy
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    ports:
      - "9090:9090"
    networks:
      - agentic-network
    depends_on:
      - agentic-flow-worker-1
      - agentic-flow-worker-2
      - agentic-flow-worker-3
    restart: unless-stopped
```

**HAProxy Configuration** (`haproxy.cfg`):

```haproxy
global
    log stdout format raw local0
    maxconn 4096

defaults
    log global
    mode http
    option httplog
    option dontlognull
    timeout connect 5000ms
    timeout client 300000ms
    timeout server 300000ms

frontend agentic_api
    bind *:9090
    default_backend agentic_workers

    # Health check endpoint (no load balancing)
    acl is_health path /health
    use_backend agentic_worker_1 if is_health

backend agentic_workers
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200

    server worker1 agentic-worker-1:9090 check inter 10s fall 3 rise 2
    server worker2 agentic-worker-2:9090 check inter 10s fall 3 rise 2
    server worker3 agentic-worker-3:9090 check inter 10s fall 3 rise 2

backend agentic_worker_1
    server worker1 agentic-worker-1:9090

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats auth admin:your-password-here
```

### Vertical Scaling

**Dynamic Resource Adjustment:**

```bash
# Scale up for heavy workload
docker update \
  --cpus="48" \
  --memory="128g" \
  agentic-flow-cachyos

# Scale down during low usage
docker update \
  --cpus="16" \
  --memory="32g" \
  agentic-flow-cachyos
```

### Auto-Scaling Script

```bash
#!/bin/bash
# /usr/local/bin/autoscale-agentic-flow.sh

CONTAINER="agentic-flow-cachyos"
CPU_THRESHOLD=80
MEMORY_THRESHOLD=85

# Get current resource usage
CPU_USAGE=$(docker stats --no-stream --format "{{.CPUPerc}}" $CONTAINER | sed 's/%//')
MEM_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" $CONTAINER | sed 's/%//')

# Get current limits
CURRENT_CPUS=$(docker inspect $CONTAINER --format='{{.HostConfig.NanoCpus}}' | awk '{print $1/1000000000}')
CURRENT_MEM=$(docker inspect $CONTAINER --format='{{.HostConfig.Memory}}' | awk '{print $1/1073741824}')

echo "Current usage: CPU=${CPU_USAGE}%, Memory=${MEM_USAGE}%"
echo "Current limits: CPUs=${CURRENT_CPUS}, Memory=${CURRENT_MEM}G"

# Scale up if thresholds exceeded
if (( $(echo "$CPU_USAGE > $CPU_THRESHOLD" | bc -l) )); then
  NEW_CPUS=$(echo "$CURRENT_CPUS * 1.5" | bc)
  echo "Scaling up CPUs to $NEW_CPUS"
  docker update --cpus="$NEW_CPUS" $CONTAINER
fi

if (( $(echo "$MEM_USAGE > $MEMORY_THRESHOLD" | bc -l) )); then
  NEW_MEM=$(echo "$CURRENT_MEM * 1.5" | bc)
  echo "Scaling up memory to ${NEW_MEM}G"
  docker update --memory="${NEW_MEM}g" $CONTAINER
fi
```

---

## High Availability Setup

### Active-Passive HA with Keepalived

**Node 1 (Primary):**

```bash
# /etc/keepalived/keepalived.conf
vrrp_script check_management_api {
    script "/usr/local/bin/check_api.sh"
    interval 5
    weight -20
}

vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1

    authentication {
        auth_type PASS
        auth_pass your_secure_password
    }

    virtual_ipaddress {
        10.0.0.100/24
    }

    track_script {
        check_management_api
    }
}
```

**Node 2 (Secondary):**

```bash
# /etc/keepalived/keepalived.conf
vrrp_instance VI_1 {
    state BACKUP
    interface eth0
    virtual_router_id 51
    priority 90
    advert_int 1

    authentication {
        auth_type PASS
        auth_pass your_secure_password
    }

    virtual_ipaddress {
        10.0.0.100/24
    }

    track_script {
        check_management_api
    }
}
```

**Health Check Script** (`/usr/local/bin/check_api.sh`):

```bash
#!/bin/bash
curl -f -m 5 http://localhost:9090/health > /dev/null 2>&1
exit $?
```

### Shared Storage for HA

**NFS Configuration:**

```yaml
services:
  agentic-flow-cachyos:
    volumes:
      - type: volume
        source: workspace
        target: /home/devuser/workspace
        volume:
          driver: local
          driver_opts:
            type: nfs
            o: addr=nfs-server.local,rw,nfsvers=4
            device: ":/exports/agentic-flow/workspace"

      - type: volume
        source: agent-memory
        target: /home/devuser/.agentic-flow
        volume:
          driver: local
          driver_opts:
            type: nfs
            o: addr=nfs-server.local,rw,nfsvers=4
            device: ":/exports/agentic-flow/memory"
```

### Database Replication (If Using External DB)

**PostgreSQL Replication:**

```yaml
services:
  postgres-primary:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: secure_password
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: repl_password
    volumes:
      - postgres-primary-data:/var/lib/postgresql/data
    networks:
      - agentic-network

  postgres-replica:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: secure_password
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_MASTER_HOST: postgres-primary
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: repl_password
    volumes:
      - postgres-replica-data:/var/lib/postgresql/data
    networks:
      - agentic-network
    depends_on:
      - postgres-primary
```

---

## Docker Swarm/Kubernetes

### Docker Swarm Deployment

**Initialize Swarm:**

```bash
# On manager node
docker swarm init --advertise-addr <MANAGER-IP>

# On worker nodes
docker swarm join --token <TOKEN> <MANAGER-IP>:2377
```

**Stack File** (`docker-stack.yml`):

```yaml
version: '3.8'

services:
  agentic-flow:
    image: your-registry/agentic-flow:latest
    environment:
      NODE_ENV: production
      MANAGEMENT_API_KEY_FILE: /run/secrets/api_key
      ANTHROPIC_API_KEY_FILE: /run/secrets/anthropic_key
    secrets:
      - api_key
      - anthropic_key
    volumes:
      - workspace:/home/devuser/workspace
      - model-cache:/home/devuser/models
    networks:
      - agentic-overlay
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
          - node.labels.gpu == true
      resources:
        limits:
          cpus: '16'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 8G
          generic_resources:
            - discrete_resource_spec:
                kind: 'gpu'
                value: 1
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 10s
        order: stop-first
      rollback_config:
        parallelism: 1
        delay: 5s

networks:
  agentic-overlay:
    driver: overlay
    attachable: true

volumes:
  workspace:
    driver: local
  model-cache:
    driver: local

secrets:
  api_key:
    external: true
  anthropic_key:
    external: true
```

**Deploy Stack:**

```bash
# Create secrets
echo "your-api-key" | docker secret create api_key -
echo "your-anthropic-key" | docker secret create anthropic_key -

# Deploy stack
docker stack deploy -c docker-stack.yml agentic-flow

# Check status
docker stack services agentic-flow
docker stack ps agentic-flow
```

### Kubernetes Deployment

**Namespace:**

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agentic-flow
```

**Secrets:**

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: agentic-secrets
  namespace: agentic-flow
type: Opaque
stringData:
  management-api-key: "your-api-key-here"
  anthropic-api-key: "your-anthropic-key-here"
  google-gemini-api-key: "your-gemini-key-here"
```

**ConfigMap:**

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agentic-config
  namespace: agentic-flow
data:
  NODE_ENV: "production"
  LOG_LEVEL: "info"
  ROUTER_MODE: "performance"
  PRIMARY_PROVIDER: "gemini"
  GPU_ACCELERATION: "true"
```

**Persistent Volumes:**

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: workspace-pvc
  namespace: agentic-flow
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: nfs-storage
  resources:
    requests:
      storage: 500Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
  namespace: agentic-flow
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: nfs-storage
  resources:
    requests:
      storage: 200Gi
```

**Deployment:**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-flow
  namespace: agentic-flow
  labels:
    app: agentic-flow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentic-flow
  template:
    metadata:
      labels:
        app: agentic-flow
    spec:
      nodeSelector:
        gpu: "true"
      containers:
        - name: agentic-flow
          image: your-registry/agentic-flow:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 9090
              name: api
          env:
            - name: MANAGEMENT_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agentic-secrets
                  key: management-api-key
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agentic-secrets
                  key: anthropic-api-key
          envFrom:
            - configMapRef:
                name: agentic-config
          resources:
            requests:
              memory: "16Gi"
              cpu: "8"
              nvidia.com/gpu: 1
            limits:
              memory: "64Gi"
              cpu: "32"
              nvidia.com/gpu: 1
          volumeMounts:
            - name: workspace
              mountPath: /home/devuser/workspace
            - name: model-cache
              mountPath: /home/devuser/models
          livenessProbe:
            httpGet:
              path: /health
              port: 9090
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 9090
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: workspace-pvc
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
```

**Service:**

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: agentic-flow-service
  namespace: agentic-flow
spec:
  type: LoadBalancer
  selector:
    app: agentic-flow
  ports:
    - protocol: TCP
      port: 9090
      targetPort: 9090
      name: api
```

**Horizontal Pod Autoscaler:**

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentic-flow-hpa
  namespace: agentic-flow
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentic-flow
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

**Deploy to Kubernetes:**

```bash
# Apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml

# Check status
kubectl get all -n agentic-flow
kubectl describe deployment agentic-flow -n agentic-flow

# View logs
kubectl logs -f -l app=agentic-flow -n agentic-flow
```

---

## Cloud Deployment

### AWS Deployment

**Architecture:**

- **Compute**: EC2 instances with GPU (p3.2xlarge or p3.8xlarge)
- **Storage**: EBS volumes + EFS for shared storage
- **Load Balancer**: Application Load Balancer with SSL
- **Database**: RDS PostgreSQL (optional)
- **Monitoring**: CloudWatch + Prometheus

**Terraform Configuration:**

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC and Networking
resource "aws_vpc" "agentic_flow" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "agentic-flow-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.agentic_flow.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "agentic-flow-public-subnet"
  }
}

# Security Group
resource "aws_security_group" "agentic_flow" {
  name        = "agentic-flow-sg"
  description = "Security group for Agentic Flow"
  vpc_id      = aws_vpc.agentic_flow.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 Instance with GPU
resource "aws_instance" "agentic_flow" {
  ami           = var.gpu_ami  # Deep Learning AMI
  instance_type = "p3.2xlarge"

  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.agentic_flow.id]

  iam_instance_profile = aws_iam_instance_profile.agentic_flow.name

  user_data = file("${path.module}/user-data.sh")

  root_block_device {
    volume_size = 500
    volume_type = "gp3"
    encrypted   = true
  }

  tags = {
    Name = "agentic-flow-server"
  }
}

# Application Load Balancer
resource "aws_lb" "agentic_flow" {
  name               = "agentic-flow-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.agentic_flow.id]
  subnets            = [aws_subnet.public.id]

  enable_deletion_protection = true
}

# ALB Target Group
resource "aws_lb_target_group" "agentic_flow" {
  name     = "agentic-flow-tg"
  port     = 9090
  protocol = "HTTP"
  vpc_id   = aws_vpc.agentic_flow.id

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

# ALB Listener
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.agentic_flow.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.agentic_flow.arn
  }
}

# EFS for shared storage
resource "aws_efs_file_system" "agentic_flow" {
  creation_token = "agentic-flow-efs"
  encrypted      = true

  tags = {
    Name = "agentic-flow-efs"
  }
}

# Secrets Manager
resource "aws_secretsmanager_secret" "api_keys" {
  name = "agentic-flow/api-keys"
}

resource "aws_secretsmanager_secret_version" "api_keys" {
  secret_id = aws_secretsmanager_secret.api_keys.id
  secret_string = jsonencode({
    management_api_key  = var.management_api_key
    anthropic_api_key   = var.anthropic_api_key
    google_gemini_api_key = var.google_gemini_api_key
  })
}
```

**User Data Script** (`user-data.sh`):

```bash
#!/bin/bash
set -e

# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker

# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  tee /etc/yum.repos.d/nvidia-docker.repo
yum install -y nvidia-docker2
systemctl restart docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Fetch secrets from AWS Secrets Manager
SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id agentic-flow/api-keys --query SecretString --output text)
export MANAGEMENT_API_KEY=$(echo $SECRET_JSON | jq -r .management_api_key)
export ANTHROPIC_API_KEY=$(echo $SECRET_JSON | jq -r .anthropic_api_key)
export GOOGLE_GEMINI_API_KEY=$(echo $SECRET_JSON | jq -r .google_gemini_api_key)

# Create .env file
cat > /opt/agentic-flow/.env << EOF
MANAGEMENT_API_KEY=$MANAGEMENT_API_KEY
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
GOOGLE_GEMINI_API_KEY=$GOOGLE_GEMINI_API_KEY
NODE_ENV=production
LOG_LEVEL=info
EOF

# Start services
cd /opt/agentic-flow
docker-compose up -d
```

### GCP Deployment

**Architecture:**

- **Compute**: Compute Engine with T4/V100 GPUs
- **Storage**: Persistent Disks + Filestore
- **Load Balancer**: HTTPS Load Balancer
- **Monitoring**: Cloud Monitoring + Logging

**Terraform Configuration:**

```hcl
# main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "agentic_flow" {
  name                    = "agentic-flow-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "agentic_flow" {
  name          = "agentic-flow-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.agentic_flow.id
}

# Firewall Rules
resource "google_compute_firewall" "agentic_flow" {
  name    = "agentic-flow-firewall"
  network = google_compute_network.agentic_flow.name

  allow {
    protocol = "tcp"
    ports    = ["22", "9090"]
  }

  source_ranges = [var.admin_cidr]
}

# Compute Instance with GPU
resource "google_compute_instance" "agentic_flow" {
  name         = "agentic-flow-server"
  machine_type = "n1-standard-8"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2004-lts"
      size  = 500
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  network_interface {
    subnetwork = google_compute_subnetwork.agentic_flow.name

    access_config {
      // Ephemeral IP
    }
  }

  metadata_startup_script = file("${path.module}/startup-script.sh")

  service_account {
    scopes = ["cloud-platform"]
  }
}

# Filestore for shared storage
resource "google_filestore_instance" "agentic_flow" {
  name     = "agentic-flow-filestore"
  location = "${var.region}-a"
  tier     = "STANDARD"

  file_shares {
    capacity_gb = 1024
    name        = "workspace"
  }

  networks {
    network = google_compute_network.agentic_flow.name
    modes   = ["MODE_IPV4"]
  }
}

# Secret Manager
resource "google_secret_manager_secret" "api_keys" {
  secret_id = "agentic-flow-api-keys"

  replication {
    automatic = true
  }
}
```

### Azure Deployment

**Architecture:**

- **Compute**: Virtual Machines with NC-series (GPU)
- **Storage**: Azure Files + Managed Disks
- **Load Balancer**: Application Gateway
- **Monitoring**: Azure Monitor

**Terraform Configuration:**

```hcl
# main.tf
provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "agentic_flow" {
  name     = "agentic-flow-rg"
  location = var.location
}

# Virtual Network
resource "azurerm_virtual_network" "agentic_flow" {
  name                = "agentic-flow-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.agentic_flow.location
  resource_group_name = azurerm_resource_group.agentic_flow.name
}

resource "azurerm_subnet" "agentic_flow" {
  name                 = "agentic-flow-subnet"
  resource_group_name  = azurerm_resource_group.agentic_flow.name
  virtual_network_name = azurerm_virtual_network.agentic_flow.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Network Security Group
resource "azurerm_network_security_group" "agentic_flow" {
  name                = "agentic-flow-nsg"
  location            = azurerm_resource_group.agentic_flow.location
  resource_group_name = azurerm_resource_group.agentic_flow.name

  security_rule {
    name                       = "HTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Virtual Machine with GPU
resource "azurerm_linux_virtual_machine" "agentic_flow" {
  name                = "agentic-flow-vm"
  resource_group_name = azurerm_resource_group.agentic_flow.name
  location            = azurerm_resource_group.agentic_flow.location
  size                = "Standard_NC6"  # 1x Tesla K80
  admin_username      = "azureuser"

  network_interface_ids = [
    azurerm_network_interface.agentic_flow.id,
  ]

  admin_ssh_key {
    username   = "azureuser"
    public_key = file("~/.ssh/id_rsa.pub")
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 500
  }

  source_image_reference {
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-1804"
    sku       = "1804-gen2"
    version   = "latest"
  }

  custom_data = base64encode(file("${path.module}/cloud-init.yaml"))
}

# Key Vault for secrets
resource "azurerm_key_vault" "agentic_flow" {
  name                = "agentic-flow-kv"
  location            = azurerm_resource_group.agentic_flow.location
  resource_group_name = azurerm_resource_group.agentic_flow.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
}
```

---

## Disaster Recovery

### Disaster Recovery Plan

**RPO (Recovery Point Objective):** Maximum acceptable data loss
- Production: 1 hour (automated backups every hour)
- Development: 24 hours (daily backups)

**RTO (Recovery Time Objective):** Maximum acceptable downtime
- Production: 1 hour
- Development: 4 hours

### DR Procedures

**1. Full Site Failure:**

```bash
#!/bin/bash
# Failover to DR site

# 1. Verify primary site is down
if curl -f --max-time 10 https://primary.example.com/health; then
  echo "Primary site is still responding. Aborting failover."
  exit 1
fi

# 2. Restore latest backup to DR site
LATEST_BACKUP=$(aws s3 ls s3://backups/agentic-flow/ | sort | tail -n 1 | awk '{print $2}')
aws s3 sync s3://backups/agentic-flow/$LATEST_BACKUP /opt/agentic-flow-restore/

# 3. Restore and start services
cd /opt/agentic-flow-restore
./restore.sh
docker-compose up -d

# 4. Update DNS to point to DR site
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456 \
  --change-batch file://dns-failover.json

# 5. Notify team
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK \
  -d '{"text":"DR Failover Complete: Now serving from DR site"}'
```

**2. Data Corruption:**

```bash
#!/bin/bash
# Restore from point-in-time backup

# Find backup before corruption
echo "Available backups:"
ls -lt /backups/agentic-flow/

read -p "Enter backup timestamp to restore: " BACKUP_TIME

# Stop services
docker-compose down

# Restore specific volumes
docker run --rm \
  -v workspace:/target/workspace \
  -v agent-memory:/target/agent-memory \
  -v /backups/agentic-flow/$BACKUP_TIME:/backup \
  alpine \
  sh -c "cd /target && tar xzf /backup/volumes.tar.gz workspace/ agent-memory/"

# Restart services
docker-compose up -d
```

### Testing DR Procedures

**Monthly DR Test Checklist:**

```markdown
# DR Test Procedure

## Pre-Test
- [ ] Notify team of scheduled DR test
- [ ] Verify latest backups are available
- [ ] Document current state of production

## Test Execution
- [ ] Simulate primary site failure
- [ ] Restore from backup to isolated environment
- [ ] Verify all services start successfully
- [ ] Test API functionality
- [ ] Test agent task execution
- [ ] Verify data integrity

## Post-Test
- [ ] Document test results
- [ ] Identify and fix any issues
- [ ] Update DR procedures
- [ ] Notify team of test completion

## Success Criteria
- [ ] RTO met (< 1 hour)
- [ ] RPO met (< 1 hour data loss)
- [ ] All critical services operational
- [ ] Data integrity verified
```

---

## Performance Tuning

### System Optimization

**Kernel Parameters** (`/etc/sysctl.conf`):

```bash
# Network optimization
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15

# Memory management
vm.swappiness = 10
vm.overcommit_memory = 1
vm.max_map_count = 262144

# File system
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288

# Apply changes
sysctl -p
```

**Docker Daemon Optimization** (`/etc/docker/daemon.json`):

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "5"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "live-restore": true,
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 10
}
```

### Container Optimization

**Resource Tuning:**

```yaml
services:
  agentic-flow-cachyos:
    # CPU affinity for better cache locality
    cpuset: "0-15"

    # Memory swappiness
    mem_swappiness: 0

    # Ulimits
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
      nproc:
        soft: 8192
        hard: 8192

    # Shared memory
    shm_size: 32gb

    environment:
      # Node.js optimization
      NODE_OPTIONS: "--max-old-space-size=49152"  # 48GB
      UV_THREADPOOL_SIZE: 128

      # CUDA optimization
      CUDA_CACHE_MAXSIZE: 2147483648  # 2GB
      CUDA_LAUNCH_BLOCKING: 0
```

### Database Tuning (If Using PostgreSQL)

**PostgreSQL Configuration:**

```conf
# postgresql.conf

# Memory
shared_buffers = 16GB
effective_cache_size = 48GB
work_mem = 256MB
maintenance_work_mem = 2GB

# Checkpoints
checkpoint_completion_target = 0.9
wal_buffers = 16MB
max_wal_size = 4GB
min_wal_size = 1GB

# Query Planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Connections
max_connections = 200

# Logging
log_destination = 'csvlog'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_min_duration_statement = 1000  # Log queries > 1s
```

### Network Optimization

**TCP Tuning:**

```bash
# /etc/sysctl.d/99-network-tune.conf

# TCP buffer sizes
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
net.core.rmem_max = 67108864
net.core.wmem_max = 67108864
net.core.netdev_max_backlog = 5000

# TCP congestion control
net.ipv4.tcp_congestion_control = bbr

# TCP Fast Open
net.ipv4.tcp_fastopen = 3
```

### GPU Performance

**NVIDIA Optimization:**

```bash
# Set GPU persistence mode
nvidia-smi -pm 1

# Set application clocks to maximum
nvidia-smi -ac 877,1530  # Memory,Graphics (adjust for your GPU)

# Disable auto-boost (for consistent performance)
nvidia-smi --auto-boost-default=0

# Monitor GPU performance
nvidia-smi dmon -s pucvmet -d 5
```

**CUDA Environment Variables:**

```bash
# In container environment
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Monitoring Performance

**Performance Monitoring Script:**

```bash
#!/bin/bash
# /usr/local/bin/monitor-performance.sh

while true; do
  echo "=== $(date) ==="

  # CPU usage
  echo "CPU Usage:"
  mpstat 1 1 | tail -1

  # Memory usage
  echo "Memory Usage:"
  free -h

  # Disk I/O
  echo "Disk I/O:"
  iostat -x 1 1 | grep -A1 "Device"

  # Docker stats
  echo "Container Stats:"
  docker stats --no-stream agentic-flow-cachyos

  # GPU stats
  echo "GPU Stats:"
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader

  echo ""
  sleep 60
done
```

---

## Summary

This deployment guide covers:

1. **Production vs Development** - Separate configurations for different environments
2. **Security** - API keys, firewalls, network isolation, TLS
3. **Reverse Proxy** - Nginx and Traefik configurations with SSL
4. **Resources** - CPU, memory, GPU, and storage planning
5. **Monitoring** - Prometheus, Grafana, ELK stack setup
6. **Backup/Restore** - Automated backup procedures and disaster recovery
7. **Scaling** - Horizontal and vertical scaling strategies
8. **High Availability** - Active-passive HA with keepalived
9. **Orchestration** - Docker Swarm and Kubernetes deployments
10. **Cloud Deployment** - AWS, GCP, and Azure configurations
11. **Disaster Recovery** - DR procedures and testing
12. **Performance** - System, container, and GPU optimization

### Quick Start Checklist

For a production deployment:

- [ ] Generate secure API keys
- [ ] Configure firewall rules
- [ ] Set up TLS/HTTPS with reverse proxy
- [ ] Configure resource limits
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure automated backups
- [ ] Test disaster recovery procedures
- [ ] Set up alerting
- [ ] Document runbooks
- [ ] Perform load testing
- [ ] Security audit
- [ ] Go live

### Support and Documentation

- Main Documentation: `/docs/README.md`
- Architecture: `/docs/ARCHITECTURE-SIMPLIFIED.md`
- Getting Started: `/docs/GETTING_STARTED.md`
- Management API: Management API server at port 9090

---

**Last Updated:** 2025-10-12
