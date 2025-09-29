# Neural Deployment and Operations Guide

## Overview

This guide covers deployment strategies, operational procedures, and best practices for running the Neural-Enhanced Swarm Controller in production environments. From single-node deployments to large-scale distributed clusters, this document provides comprehensive guidance for successful operations.

## Deployment Architectures

### Single Node Deployment

**Suitable for**: Development, testing, small-scale production
**Resources**: 8GB RAM, 4 CPU cores, optional GPU

```yaml
# docker-compose.neural-single.yml
version: '3.8'
services:
  neural-controller:
    image: neural-swarm:latest
    ports:
      - "8080:8080"
      - "9090:9090"  # Metrics
    environment:
      - NEURAL_MODE=single_node
      - NEURAL_MAX_AGENTS=50
      - NEURAL_GPU_ENABLED=true
      - NEURAL_MEMORY_SIZE=2048
      - RUST_LOG=info
    volumes:
      - neural_memory:/app/memory
      - neural_logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  neural-memory:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - neural_redis:/data
    command: redis-server --appendonly yes

  neural-monitoring:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

volumes:
  neural_memory:
  neural_logs:
  neural_redis:
  prometheus_data:
```

### Multi-Node Cluster

**Suitable for**: Production environments, high availability
**Resources**: 3+ nodes, 16GB RAM per node, 8 CPU cores, GPU recommended

```yaml
# docker-compose.neural-cluster.yml
version: '3.8'
services:
  neural-controller-1:
    image: neural-swarm:latest
    environment:
      - NEURAL_MODE=cluster
      - NEURAL_NODE_ID=1
      - NEURAL_CLUSTER_PEERS=neural-controller-2:8080,neural-controller-3:8080
      - NEURAL_MAX_AGENTS=100
    deploy:
      placement:
        constraints: [node.labels.neural.role == controller]
      replicas: 1
      resources:
        limits:
          memory: 8G
          cpus: '4'
    networks:
      - neural_mesh

  neural-controller-2:
    image: neural-swarm:latest
    environment:
      - NEURAL_MODE=cluster
      - NEURAL_NODE_ID=2
      - NEURAL_CLUSTER_PEERS=neural-controller-1:8080,neural-controller-3:8080
      - NEURAL_MAX_AGENTS=100
    deploy:
      placement:
        constraints: [node.labels.neural.role == controller]
      replicas: 1
    networks:
      - neural_mesh

  neural-controller-3:
    image: neural-swarm:latest
    environment:
      - NEURAL_MODE=cluster
      - NEURAL_NODE_ID=3
      - NEURAL_CLUSTER_PEERS=neural-controller-1:8080,neural-controller-2:8080
      - NEURAL_MAX_AGENTS=100
    deploy:
      placement:
        constraints: [node.labels.neural.role == controller]
      replicas: 1
    networks:
      - neural_mesh

  neural-gpu-service:
    image: neural-gpu:latest
    environment:
      - GPU_MEMORY_LIMIT=8192
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      placement:
        constraints: [node.labels.neural.gpu == true]
      replicas: 3
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - neural_mesh

  neural-memory-cluster:
    image: redis:7-alpine
    command: |
      redis-server 
      --cluster-enabled yes 
      --cluster-config-file nodes.conf 
      --cluster-node-timeout 5000 
      --appendonly yes
    deploy:
      replicas: 6
      placement:
        max_replicas_per_node: 2
    networks:
      - neural_mesh

networks:
  neural_mesh:
    driver: overlay
    attachable: true
```

### Kubernetes Deployment

**Suitable for**: Enterprise production, auto-scaling, cloud environments

```yaml
# neural-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: neural-swarm
  labels:
    name: neural-swarm
---
# neural-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: neural-config
  namespace: neural-swarm
data:
  neural.toml: |
    [neural]
    max_agents = 200
    cognitive_diversity = 0.8
    neural_plasticity = 0.7
    gpu_acceleration = true
    
    [topology]
    type = "adaptive"
    base_connectivity = 0.7
    adaptation_rate = 0.1
    
    [memory]
    backend = "distributed"
    retention_days = 30
    consolidation_interval = "1h"
    
    [gpu]
    enabled = true
    memory_limit = 8192
    batch_size = 32
---
# neural-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-controller
  namespace: neural-swarm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-controller
  template:
    metadata:
      labels:
        app: neural-controller
    spec:
      containers:
      - name: neural-controller
        image: neural-swarm:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: NEURAL_MODE
          value: "kubernetes"
        - name: NEURAL_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: NEURAL_POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: memory
          mountPath: /app/memory
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: neural-config
      - name: memory
        persistentVolumeClaim:
          claimName: neural-memory-pvc
---
# neural-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: neural-controller-service
  namespace: neural-swarm
spec:
  selector:
    app: neural-controller
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
---
# neural-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neural-controller-hpa
  namespace: neural-swarm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neural-controller
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
  - type: Pods
    pods:
      metric:
        name: neural_collective_intelligence
      target:
        type: AverageValue
        averageValue: "0.8"
```

## Infrastructure Requirements

### Minimum System Requirements

#### Development Environment
- **CPU**: 4 cores (x86_64)
- **Memory**: 8GB RAM
- **Storage**: 50GB SSD
- **GPU**: Optional (NVIDIA with CUDA 11.0+)
- **Network**: 1Gbps
- **OS**: Linux (Ubuntu 20.04+), macOS (12+), Windows 11

#### Production Environment
- **CPU**: 16 cores (x86_64)
- **Memory**: 32GB RAM
- **Storage**: 500GB NVMe SSD
- **GPU**: NVIDIA RTX 3080+ or Tesla V100+
- **Network**: 10Gbps with low latency
- **OS**: Linux (Ubuntu 22.04 LTS recommended)

#### High-Scale Production
- **CPU**: 32+ cores per node
- **Memory**: 128GB+ RAM per node
- **Storage**: 2TB+ NVMe SSD, distributed storage
- **GPU**: Multiple NVIDIA A100 or H100 GPUs
- **Network**: 25Gbps+ with RDMA support
- **OS**: Linux with optimized kernel

### GPU Requirements

#### Supported GPU Architectures
- **NVIDIA**: Pascal (GTX 10xx), Turing (RTX 20xx), Ampere (RTX 30xx/A100), Hopper (H100)
- **CUDA**: Version 11.0 or higher
- **Memory**: Minimum 8GB VRAM, 16GB+ recommended
- **Compute Capability**: 6.0+

#### GPU Configuration

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Network Configuration

#### Firewall Rules

```bash
# Neural Controller API
sudo ufw allow 8080/tcp

# Metrics endpoint
sudo ufw allow 9090/tcp

# Inter-node communication
sudo ufw allow 8081:8089/tcp

# WebSocket connections
sudo ufw allow 8080/tcp

# GPU service communication
sudo ufw allow 8090:8099/tcp

# Memory service (Redis)
sudo ufw allow 6379/tcp

# Container orchestration
sudo ufw allow 2376:2377/tcp
sudo ufw allow 7946/tcp
sudo ufw allow 7946/udp
sudo ufw allow 4789/udp
```

#### Network Optimization

```bash
# Optimize network stack for neural communication
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf
echo 'net.core.netdev_max_backlog = 5000' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_congestion_control = bbr' >> /etc/sysctl.conf

sysctl -p
```

## Installation Procedures

### Automated Installation

```bash
#!/bin/bash
# install-neural-swarm.sh

set -e

# Configuration
NEURAL_VERSION="latest"
INSTALL_DIR="/opt/neural-swarm"
CONFIG_DIR="/etc/neural-swarm"
LOG_DIR="/var/log/neural-swarm"
DATA_DIR="/var/lib/neural-swarm"

# Check requirements
check_requirements() {
    echo "Checking system requirements..."
    
    # Check CPU
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -lt 4 ]; then
        echo "Error: Minimum 4 CPU cores required, found $CPU_CORES"
        exit 1
    fi
    
    # Check memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 8 ]; then
        echo "Error: Minimum 8GB RAM required, found ${MEMORY_GB}GB"
        exit 1
    fi
    
    # Check storage
    AVAILABLE_GB=$(df / | awk 'NR==2{print int($4/1024/1024)}')
    if [ "$AVAILABLE_GB" -lt 50 ]; then
        echo "Error: Minimum 50GB storage required, found ${AVAILABLE_GB}GB"
        exit 1
    fi
    
    echo "System requirements satisfied"
}

# Install Docker
install_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
}

# Install NVIDIA Docker (if GPU present)
install_nvidia_docker() {
    if lspci | grep -i nvidia &> /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA Docker..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
    fi
}

# Create directories
create_directories() {
    echo "Creating directories..."
    sudo mkdir -p $INSTALL_DIR $CONFIG_DIR $LOG_DIR $DATA_DIR
    sudo chown -R $USER:$USER $INSTALL_DIR $CONFIG_DIR $LOG_DIR $DATA_DIR
}

# Download and configure
download_configure() {
    echo "Downloading Neural Swarm Controller..."
    cd $INSTALL_DIR
    
    # Download release
    curl -L "https://github.com/your-org/neural-swarm/releases/latest/download/neural-swarm-${NEURAL_VERSION}.tar.gz" -o neural-swarm.tar.gz
    tar -xzf neural-swarm.tar.gz
    rm neural-swarm.tar.gz
    
    # Generate configuration
    cat > $CONFIG_DIR/neural.toml << EOF
[neural]
max_agents = 50
cognitive_diversity = 0.8
neural_plasticity = 0.7
gpu_acceleration = true

[topology]
type = "mesh"
connectivity = 0.7
redundancy = 3

[memory]
backend = "redis"
retention_days = 30
consolidation_interval = "1h"

[logging]
level = "info"
output = "$LOG_DIR/neural.log"
max_size = "100MB"
max_files = 10
EOF
    
    # Generate Docker Compose
    cp docker-compose.neural-single.yml $INSTALL_DIR/docker-compose.yml
}

# Create systemd service
create_service() {
    echo "Creating systemd service..."
    sudo cat > /etc/systemd/system/neural-swarm.service << EOF
[Unit]
Description=Neural Swarm Controller
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable neural-swarm
}

# Main installation
main() {
    echo "Neural Swarm Controller Installation"
    echo "===================================="
    
    check_requirements
    install_docker
    install_nvidia_docker
    create_directories
    download_configure
    create_service
    
    echo ""
    echo "Installation completed successfully!"
    echo ""
    echo "To start the service:"
    echo "  sudo systemctl start neural-swarm"
    echo ""
    echo "To view logs:"
    echo "  docker-compose -f $INSTALL_DIR/docker-compose.yml logs -f"
    echo ""
    echo "To access the API:"
    echo "  curl http://localhost:8080/health"
    echo ""
    echo "Configuration file: $CONFIG_DIR/neural.toml"
    echo "Log directory: $LOG_DIR"
    echo "Data directory: $DATA_DIR"
}

main "$@"
```

### Manual Installation

#### Step 1: Environment Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y curl wget git build-essential pkg-config libssl-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### Step 2: Build from Source

```bash
# Clone repository
git clone https://github.com/your-org/neural-swarm-controller.git
cd neural-swarm-controller

# Build release version
cargo build --release

# Run tests
cargo test

# Build Docker image
docker build -t neural-swarm:latest .
```

#### Step 3: Configuration

```bash
# Create configuration directory
sudo mkdir -p /etc/neural-swarm

# Create configuration file
sudo tee /etc/neural-swarm/neural.toml << EOF
[neural]
max_agents = 100
cognitive_diversity = 0.8
neural_plasticity = 0.7
gpu_acceleration = true
learning_rate = 0.01

[topology]
type = "adaptive"
base_topology = "mesh"
connectivity = 0.7
adaptation_rate = 0.1

[memory]
backend = "distributed"
retention_days = 30
consolidation_interval = "1h"
max_memory_size = "8GB"

[gpu]
enabled = true
memory_limit = 8192
batch_size = 32
optimization_level = "high"

[api]
host = "0.0.0.0"
port = 8080
max_connections = 1000
timeout = 30

[logging]
level = "info"
format = "json"
output = "/var/log/neural-swarm/neural.log"
max_size = "100MB"
max_files = 10
EOF
```

## Configuration Management

### Environment Variables

```bash
# Core Configuration
export NEURAL_MODE="production"               # deployment mode
export NEURAL_MAX_AGENTS=200                  # maximum agent count
export NEURAL_GPU_ENABLED=true                # enable GPU acceleration
export NEURAL_MEMORY_SIZE=4096                # memory size in MB

# Topology Configuration
export NEURAL_TOPOLOGY_TYPE="adaptive"        # topology type
export NEURAL_CONNECTIVITY=0.8               # network connectivity
export NEURAL_ADAPTATION_RATE=0.1            # adaptation rate

# Performance Configuration
export NEURAL_COGNITIVE_DIVERSITY=0.8        # cognitive diversity
export NEURAL_NEURAL_PLASTICITY=0.7          # neural plasticity
export NEURAL_LEARNING_RATE=0.01             # learning rate

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1              # visible GPU devices
export NEURAL_GPU_MEMORY_LIMIT=8192          # GPU memory limit
export NEURAL_GPU_BATCH_SIZE=32              # GPU batch size

# Network Configuration
export NEURAL_API_HOST="0.0.0.0"             # API host
export NEURAL_API_PORT=8080                  # API port
export NEURAL_WS_PORT=8081                   # WebSocket port

# Security Configuration
export NEURAL_API_KEY="your-secure-api-key"   # API authentication
export NEURAL_JWT_SECRET="your-jwt-secret"    # JWT secret
export NEURAL_TLS_ENABLED=true               # enable TLS

# Monitoring Configuration
export NEURAL_METRICS_ENABLED=true           # enable metrics
export NEURAL_METRICS_PORT=9090              # metrics port
export NEURAL_TRACING_ENABLED=true           # enable tracing

# Logging Configuration
export RUST_LOG="neural_swarm=info,tokio=warn" # log levels
export NEURAL_LOG_FORMAT="json"               # log format
export NEURAL_LOG_FILE="/var/log/neural.log"  # log file
```

### Configuration Templates

#### Development Configuration

```toml
# neural-dev.toml
[neural]
max_agents = 10
cognitive_diversity = 0.7
neural_plasticity = 0.8
gpu_acceleration = false
learning_rate = 0.02

[topology]
type = "mesh"
connectivity = 0.6
redundancy = 2

[memory]
backend = "local"
retention_days = 7
consolidation_interval = "30m"
max_memory_size = "1GB"

[api]
host = "127.0.0.1"
port = 8080
cors_enabled = true
debug_mode = true

[logging]
level = "debug"
format = "pretty"
output = "stdout"
```

#### Production Configuration

```toml
# neural-prod.toml
[neural]
max_agents = 500
cognitive_diversity = 0.85
neural_plasticity = 0.7
gpu_acceleration = true
learning_rate = 0.005
fault_tolerance = true

[topology]
type = "adaptive"
base_topology = "hierarchical"
connectivity = 0.8
adaptation_rate = 0.05
performance_threshold = 0.9

[memory]
backend = "distributed"
retention_days = 90
consolidation_interval = "2h"
max_memory_size = "32GB"
replication_factor = 3
compression_enabled = true

[gpu]
enabled = true
memory_limit = 16384
batch_size = 64
optimization_level = "maximum"
device_ids = [0, 1, 2, 3]

[api]
host = "0.0.0.0"
port = 8080
max_connections = 5000
timeout = 60
rate_limiting = true
auth_required = true

[security]
tls_enabled = true
cert_file = "/etc/ssl/neural/cert.pem"
key_file = "/etc/ssl/neural/key.pem"
api_key_required = true
jwt_expiration = "24h"

[monitoring]
metrics_enabled = true
metrics_port = 9090
tracing_enabled = true
tracing_endpoint = "http://jaeger:14268/api/traces"
health_check_interval = "30s"

[logging]
level = "info"
format = "json"
output = "/var/log/neural-swarm/neural.log"
max_size = "500MB"
max_files = 20
compress_old_files = true
```

## Scaling Strategies

### Horizontal Scaling

#### Auto-scaling with Kubernetes

```yaml
# neural-vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: neural-controller-vpa
  namespace: neural-swarm
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neural-controller
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: neural-controller
      maxAllowed:
        cpu: 8
        memory: 32Gi
        nvidia.com/gpu: 2
      minAllowed:
        cpu: 2
        memory: 4Gi
        nvidia.com/gpu: 1
---
# neural-custom-metrics.yaml
apiVersion: v1
kind: Service
metadata:
  name: neural-metrics-service
  namespace: neural-swarm
  labels:
    app: neural-controller
spec:
  ports:
  - name: metrics
    port: 9090
  selector:
    app: neural-controller
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: neural-metrics
  namespace: neural-swarm
spec:
  selector:
    matchLabels:
      app: neural-controller
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

#### Custom Metrics for HPA

```bash
# Register custom metrics
kubectl apply -f - << EOF
apiVersion: v1
kind: Service
metadata:
  name: neural-custom-metrics-api
  namespace: neural-swarm
spec:
  ports:
  - port: 443
    targetPort: 8443
  selector:
    app: neural-custom-metrics-adapter
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-custom-metrics-adapter
  namespace: neural-swarm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: neural-custom-metrics-adapter
  template:
    metadata:
      labels:
        app: neural-custom-metrics-adapter
    spec:
      containers:
      - name: custom-metrics-adapter
        image: neural-metrics-adapter:latest
        ports:
        - containerPort: 8443
        env:
        - name: NEURAL_METRICS_ENDPOINT
          value: "http://neural-controller-service:9090/metrics"
EOF
```

### Vertical Scaling

#### Memory Scaling

```bash
# Monitor memory usage
kubectl top pods -n neural-swarm

# Scale memory for deployment
kubectl patch deployment neural-controller -n neural-swarm -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "neural-controller",
          "resources": {
            "requests": {
              "memory": "8Gi"
            },
            "limits": {
              "memory": "16Gi"
            }
          }
        }]
      }
    }
  }
}'
```

#### GPU Scaling

```bash
# Scale GPU resources
kubectl patch deployment neural-controller -n neural-swarm -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "neural-controller",
          "resources": {
            "requests": {
              "nvidia.com/gpu": 2
            },
            "limits": {
              "nvidia.com/gpu": 2
            }
          }
        }]
      }
    }
  }
}'
```

## High Availability

### Multi-Region Deployment

```yaml
# neural-ha-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-controller-region-1
  namespace: neural-swarm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-controller
      region: region-1
  template:
    metadata:
      labels:
        app: neural-controller
        region: region-1
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values: ["neural-controller"]
            topologyKey: kubernetes.io/hostname
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west-1"]
      containers:
      - name: neural-controller
        image: neural-swarm:latest
        env:
        - name: NEURAL_REGION
          value: "region-1"
        - name: NEURAL_PEER_DISCOVERY
          value: "kubernetes"
        - name: NEURAL_CLUSTER_PEERS
          value: "neural-controller-region-2.neural-swarm.svc.cluster.local,neural-controller-region-3.neural-swarm.svc.cluster.local"
```

### Disaster Recovery

```bash
#!/bin/bash
# neural-backup.sh

BACKUP_DIR="/backup/neural-swarm/$(date +%Y%m%d-%H%M%S)"
S3_BUCKET="neural-swarm-backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup neural memory
echo "Backing up neural memory..."
kubectl exec -n neural-swarm deployment/neural-memory -- redis-cli --rdb /tmp/neural-memory.rdb
kubectl cp neural-swarm/neural-memory-pod:/tmp/neural-memory.rdb $BACKUP_DIR/neural-memory.rdb

# Backup configuration
echo "Backing up configuration..."
kubectl get configmap neural-config -n neural-swarm -o yaml > $BACKUP_DIR/neural-config.yaml
kubectl get secret neural-secrets -n neural-swarm -o yaml > $BACKUP_DIR/neural-secrets.yaml

# Backup persistent volumes
echo "Backing up persistent volumes..."
kubectl get pv -o yaml > $BACKUP_DIR/persistent-volumes.yaml
kubectl get pvc -n neural-swarm -o yaml > $BACKUP_DIR/persistent-volume-claims.yaml

# Create deployment snapshot
echo "Creating deployment snapshot..."
kubectl get all -n neural-swarm -o yaml > $BACKUP_DIR/deployment-snapshot.yaml

# Compress backup
tar -czf $BACKUP_DIR.tar.gz -C $(dirname $BACKUP_DIR) $(basename $BACKUP_DIR)
rm -rf $BACKUP_DIR

# Upload to S3
echo "Uploading backup to S3..."
aws s3 cp $BACKUP_DIR.tar.gz s3://$S3_BUCKET/

# Cleanup old backups (keep last 30 days)
find /backup/neural-swarm -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

## Security Hardening

### Network Security

```bash
# Configure iptables for neural swarm
#!/bin/bash
# neural-firewall.sh

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (change port as needed)
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Neural API (restrict to known IPs)
iptables -A INPUT -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -s 172.16.0.0/12 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -s 192.168.0.0/16 -j ACCEPT

# Inter-node communication
iptables -A INPUT -p tcp --dport 8081:8089 -s 10.0.0.0/8 -j ACCEPT

# Metrics (restrict to monitoring network)
iptables -A INPUT -p tcp --dport 9090 -s 10.100.0.0/16 -j ACCEPT

# GPU service communication
iptables -A INPUT -p tcp --dport 8090:8099 -s 10.0.0.0/8 -j ACCEPT

# Redis cluster
iptables -A INPUT -p tcp --dport 6379 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 16379 -s 10.0.0.0/8 -j ACCEPT

# Docker swarm
iptables -A INPUT -p tcp --dport 2377 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 7946 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p udp --dport 7946 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p udp --dport 4789 -s 10.0.0.0/8 -j ACCEPT

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### TLS Configuration

```bash
# Generate TLS certificates
#!/bin/bash
# generate-certs.sh

CERT_DIR="/etc/ssl/neural"
CA_KEY="$CERT_DIR/ca-key.pem"
CA_CERT="$CERT_DIR/ca-cert.pem"
SERVER_KEY="$CERT_DIR/server-key.pem"
SERVER_CERT="$CERT_DIR/server-cert.pem"

mkdir -p $CERT_DIR
cd $CERT_DIR

# Generate CA key
openssl genrsa -out $CA_KEY 4096

# Generate CA certificate
openssl req -new -x509 -days 365 -key $CA_KEY -out $CA_CERT -subj "/C=US/ST=CA/L=San Francisco/O=Neural Swarm/CN=Neural Swarm CA"

# Generate server key
openssl genrsa -out $SERVER_KEY 4096

# Generate server certificate signing request
openssl req -new -key $SERVER_KEY -out server.csr -subj "/C=US/ST=CA/L=San Francisco/O=Neural Swarm/CN=neural-controller"

# Generate server certificate
openssl x509 -req -days 365 -in server.csr -CA $CA_CERT -CAkey $CA_KEY -CAcreateserial -out $SERVER_CERT

# Set permissions
chown -R neural:neural $CERT_DIR
chmod 600 $CERT_DIR/*.pem

# Create Kubernetes secret
kubectl create secret tls neural-tls-secret \
  --cert=$SERVER_CERT \
  --key=$SERVER_KEY \
  -n neural-swarm

echo "TLS certificates generated and installed"
```

### RBAC Configuration

```yaml
# neural-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: neural-controller
  namespace: neural-swarm
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: neural-controller
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: neural-controller
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: neural-controller
subjects:
- kind: ServiceAccount
  name: neural-controller
  namespace: neural-swarm
```

This comprehensive deployment guide ensures successful production deployment of the Neural-Enhanced Swarm Controller across various environments and scales.