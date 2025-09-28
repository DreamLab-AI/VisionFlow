# Deployment Guide

This guide covers deployment strategies, containerization, orchestration, and production setup for AutoSchema KG Rust.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring and Observability](#monitoring-and-observability)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)
- [Scaling Strategies](#scaling-strategies)

## Deployment Overview

AutoSchema KG Rust supports multiple deployment patterns:

- **Single Node**: All components on one machine
- **Multi-Node**: Distributed across multiple machines
- **Containerized**: Docker-based deployment
- **Orchestrated**: Kubernetes or Docker Swarm
- **Cloud Native**: AWS, GCP, Azure managed services

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Application   │    │   Vector Store  │
│   (nginx/HAProxy)│───▶│   (AutoSchema)  │───▶│   (In-memory)   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Graph DB      │    │   Monitoring    │
                       │   (Neo4j)       │    │   (Prometheus)  │
                       │                 │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## Docker Deployment

### Dockerfile

Create an optimized production Dockerfile:

```dockerfile
# Build stage
FROM rust:1.70-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests and build dependencies first (for caching)
COPY Cargo.toml Cargo.lock ./
COPY kg_construction/Cargo.toml ./kg_construction/
COPY llm_generator/Cargo.toml ./llm_generator/
COPY retriever/Cargo.toml ./retriever/
COPY vectorstore/Cargo.toml ./vectorstore/
COPY utils/Cargo.toml ./utils/

# Create dummy source files to build dependencies
RUN mkdir -p kg_construction/src llm_generator/src retriever/src vectorstore/src utils/src && \
    echo "fn main() {}" > kg_construction/src/main.rs && \
    echo "fn main() {}" > llm_generator/src/main.rs && \
    echo "fn main() {}" > retriever/src/main.rs && \
    echo "fn main() {}" > vectorstore/src/main.rs && \
    echo "fn main() {}" > utils/src/main.rs && \
    echo "fn main() {}" > src/main.rs

# Build dependencies (this will be cached)
RUN cargo build --release --bins
RUN rm -rf src/ kg_construction/src/ llm_generator/src/ retriever/src/ vectorstore/src/ utils/src/

# Copy source code
COPY . .

# Build application with optimizations
ENV RUSTFLAGS="-C target-cpu=native"
RUN cargo build --release --bin autoschema_kg_rust

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false -d /app autoschema

# Create directories
RUN mkdir -p /app/{data,logs,config} && \
    chown -R autoschema:autoschema /app

# Copy binary and configuration
COPY --from=builder /app/target/release/autoschema_kg_rust /usr/local/bin/
COPY --from=builder /app/config/ /app/config/

# Set up environment
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto"

USER autoschema
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080 9090

CMD ["autoschema_kg_rust"]
```

### Docker Compose

#### Development Setup (`docker-compose.dev.yml`)

```yaml
version: '3.8'

services:
  autoschema-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - cargo_cache:/usr/local/cargo/registry
      - target_cache:/app/target
    environment:
      - RUST_LOG=debug
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=devpassword
    ports:
      - "8080:8080"
      - "9090:9090"
    depends_on:
      neo4j:
        condition: service_healthy
    command: cargo watch -x run

  neo4j:
    image: neo4j:5.11
    environment:
      NEO4J_AUTH: neo4j/devpassword
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_dev_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "devpassword", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  cargo_cache:
  target_cache:
  neo4j_dev_data:
```

#### Production Setup (`docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  autoschema:
    image: autoschema-kg-rust:${VERSION:-latest}
    restart: unless-stopped

    environment:
      - APP_ENV=production
      - RUST_LOG=info
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD_FILE=/run/secrets/neo4j_password
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
      - VECTOR_STORE_PATH=/app/data/vectors
      - METRICS_ENABLED=true

    volumes:
      - app_data:/app/data
      - app_logs:/app/logs
      - ./config/production.toml:/app/config/config.toml:ro

    ports:
      - "8080:8080"
      - "9090:9090"

    secrets:
      - neo4j_password
      - openai_api_key

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

    depends_on:
      neo4j:
        condition: service_healthy

    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 1G
          cpus: '0.5'

  neo4j:
    image: neo4j:5.11-enterprise
    restart: unless-stopped

    environment:
      NEO4J_AUTH: neo4j/neo4j
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"

      # Memory configuration
      NEO4J_dbms_memory_heap_initial__size: 2G
      NEO4J_dbms_memory_heap_max__size: 4G
      NEO4J_dbms_memory_pagecache_size: 2G

      # Performance tuning
      NEO4J_dbms_default__database: knowledge_graph
      NEO4J_dbms_security_procedures_unrestricted: apoc.*,gds.*
      NEO4J_dbms_checkpoint_interval_time: 15m
      NEO4J_dbms_checkpoint_interval_tx: 100000

    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_conf:/conf
      - ./neo4j/plugins:/plugins

    ports:
      - "7474:7474"
      - "7687:7687"

    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "neo4j", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5

    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'

  nginx:
    image: nginx:alpine
    restart: unless-stopped

    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx_cache:/var/cache/nginx

    ports:
      - "80:80"
      - "443:443"

    depends_on:
      - autoschema

    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.5'

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped

    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus

    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--storage.tsdb.retention.time=30d'

    ports:
      - "9000:9090"

    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped

    environment:
      GF_SECURITY_ADMIN_PASSWORD_FILE: /run/secrets/grafana_password
      GF_USERS_ALLOW_SIGN_UP: "false"

    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning

    ports:
      - "3000:3000"

    secrets:
      - grafana_password

    depends_on:
      - prometheus

secrets:
  neo4j_password:
    external: true
  openai_api_key:
    external: true
  grafana_password:
    external: true

volumes:
  app_data:
  app_logs:
  neo4j_data:
  neo4j_logs:
  neo4j_conf:
  nginx_cache:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Multi-Stage Build Optimization

```dockerfile
# Dockerfile.optimized
FROM rust:1.70-alpine as chef

RUN apk add --no-cache musl-dev pkgconfig openssl-dev
RUN cargo install cargo-chef

WORKDIR /app

# Chef prepare (dependency analysis)
FROM chef as planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Chef cook (dependency building)
FROM chef as builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Build application
COPY . .
ENV RUSTFLAGS="-C target-cpu=native -C link-arg=-s"
RUN cargo build --release --bin autoschema_kg_rust

# Runtime image
FROM alpine:latest

RUN apk add --no-cache ca-certificates libgcc

RUN adduser -D -s /bin/sh autoschema

COPY --from=builder /app/target/release/autoschema_kg_rust /usr/local/bin/
COPY config/ /app/config/

USER autoschema
WORKDIR /app

EXPOSE 8080 9090
CMD ["autoschema_kg_rust"]
```

## Kubernetes Deployment

### Namespace and Resources

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autoschema
  labels:
    name: autoschema
```

### ConfigMap and Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: autoschema-config
  namespace: autoschema
data:
  config.toml: |
    [general]
    app_name = "autoschema_kg_rust"
    log_level = "info"
    max_concurrent_tasks = 500

    [knowledge_graph]
    neo4j_uri = "bolt://neo4j-service:7687"
    username = "neo4j"
    max_connections = 20
    connection_timeout = 30

    [llm]
    provider = "openai"
    model = "gpt-4"
    temperature = 0.1
    max_tokens = 2048

    [monitoring]
    enable_metrics = true
    metrics_port = 9090

---
apiVersion: v1
kind: Secret
metadata:
  name: autoschema-secrets
  namespace: autoschema
type: Opaque
stringData:
  neo4j-password: "your-secure-password"
  openai-api-key: "sk-your-openai-api-key"
  anthropic-api-key: "your-anthropic-key"
```

### Application Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autoschema-kg-rust
  namespace: autoschema
  labels:
    app: autoschema-kg-rust
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: autoschema-kg-rust
  template:
    metadata:
      labels:
        app: autoschema-kg-rust
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: autoschema-service-account

      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000

      containers:
      - name: autoschema-kg-rust
        image: autoschema-kg-rust:latest
        imagePullPolicy: Always

        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        env:
        - name: APP_ENV
          value: "production"
        - name: RUST_LOG
          value: "info"
        - name: NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: autoschema-secrets
              key: neo4j-password
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: autoschema-secrets
              key: openai-api-key

        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: data
          mountPath: /app/data
        - name: logs
          mountPath: /app/logs

        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        startupProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 30

      volumes:
      - name: config
        configMap:
          name: autoschema-config
      - name: data
        persistentVolumeClaim:
          claimName: autoschema-data-pvc
      - name: logs
        emptyDir: {}

      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - autoschema-kg-rust
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: autoschema-service
  namespace: autoschema
  labels:
    app: autoschema-kg-rust
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
    name: http
  - port: 9090
    targetPort: metrics
    protocol: TCP
    name: metrics
  selector:
    app: autoschema-kg-rust

---
apiVersion: v1
kind: Service
metadata:
  name: autoschema-headless
  namespace: autoschema
  labels:
    app: autoschema-kg-rust
spec:
  type: ClusterIP
  clusterIP: None
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
    name: http
  selector:
    app: autoschema-kg-rust
```

### Neo4j StatefulSet

```yaml
# neo4j-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
  namespace: autoschema
spec:
  serviceName: neo4j-headless
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      securityContext:
        runAsUser: 7474
        runAsGroup: 7474
        fsGroup: 7474

      containers:
      - name: neo4j
        image: neo4j:5.11-enterprise

        env:
        - name: NEO4J_AUTH
          valueFrom:
            secretKeyRef:
              name: autoschema-secrets
              key: neo4j-auth
        - name: NEO4J_ACCEPT_LICENSE_AGREEMENT
          value: "yes"
        - name: NEO4J_PLUGINS
          value: '["apoc", "graph-data-science"]'
        - name: NEO4J_dbms_memory_heap_initial__size
          value: "2G"
        - name: NEO4J_dbms_memory_heap_max__size
          value: "4G"
        - name: NEO4J_dbms_memory_pagecache_size
          value: "2G"

        ports:
        - name: http
          containerPort: 7474
        - name: bolt
          containerPort: 7687

        volumeMounts:
        - name: neo4j-data
          mountPath: /data
        - name: neo4j-logs
          mountPath: /logs

        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"

        livenessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - cypher-shell -u neo4j -p $NEO4J_AUTH "RETURN 1"
          initialDelaySeconds: 60
          periodSeconds: 30

        readinessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - cypher-shell -u neo4j -p $NEO4J_AUTH "RETURN 1"
          initialDelaySeconds: 30
          periodSeconds: 10

  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd
  - metadata:
      name: neo4j-logs
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi

---
apiVersion: v1
kind: Service
metadata:
  name: neo4j-service
  namespace: autoschema
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 7474
    targetPort: 7474
  - name: bolt
    port: 7687
    targetPort: 7687
  selector:
    app: neo4j

---
apiVersion: v1
kind: Service
metadata:
  name: neo4j-headless
  namespace: autoschema
spec:
  type: ClusterIP
  clusterIP: None
  ports:
  - name: http
    port: 7474
    targetPort: 7474
  - name: bolt
    port: 7687
    targetPort: 7687
  selector:
    app: neo4j
```

### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autoschema-ingress
  namespace: autoschema
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.autoschema.com
    secretName: autoschema-tls
  rules:
  - host: api.autoschema.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: autoschema-service
            port:
              number: 80
      - path: /metrics
        pathType: Prefix
        backend:
          service:
            name: autoschema-service
            port:
              number: 9090
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: autoschema-hpa
  namespace: autoschema
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autoschema-kg-rust
  minReplicas: 3
  maxReplicas: 20
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
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

## Cloud Platform Deployment

### AWS EKS Deployment

#### EKS Cluster Setup

```yaml
# eks-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: autoschema-cluster
  region: us-west-2
  version: "1.24"

vpc:
  enableDnsHostnames: true
  enableDnsSupport: true

nodeGroups:
  - name: autoschema-workers
    instanceType: c5.2xlarge
    desiredCapacity: 3
    minSize: 1
    maxSize: 10
    volumeSize: 100
    volumeType: gp3

    ssh:
      enableSsm: true

    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

    labels:
      nodegroup-type: autoschema-workers

    taints:
      - key: autoschema.com/dedicated
        value: autoschema
        effect: NoSchedule

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]
```

#### RDS Configuration

```yaml
# rds-neo4j-alternative.yaml (if using managed graph database)
apiVersion: v1
kind: ConfigMap
metadata:
  name: rds-config
  namespace: autoschema
data:
  endpoint: "autoschema-cluster.cluster-xxxxx.us-west-2.rds.amazonaws.com"
  port: "5432"
  database: "knowledge_graph"

---
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: autoschema
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: rds-credentials
  namespace: autoschema
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: rds-secret
    creationPolicy: Owner
  data:
  - secretKey: username
    remoteRef:
      key: autoschema/rds
      property: username
  - secretKey: password
    remoteRef:
      key: autoschema/rds
      property: password
```

### Google Cloud (GKE) Deployment

```yaml
# gke-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gke-deployment-config
data:
  cluster-config: |
    gcloud container clusters create autoschema-cluster \
      --region=us-central1 \
      --machine-type=c2-standard-8 \
      --num-nodes=3 \
      --min-nodes=1 \
      --max-nodes=10 \
      --enable-autoscaling \
      --enable-autorepair \
      --enable-autoupgrade \
      --disk-type=pd-ssd \
      --disk-size=100GB \
      --enable-network-policy \
      --enable-logging \
      --enable-monitoring \
      --addons=HorizontalPodAutoscaling,HttpLoadBalancing,NetworkPolicy
```

### Azure AKS Deployment

```yaml
# aks-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aks-deployment-config
data:
  cluster-config: |
    az aks create \
      --resource-group autoschema-rg \
      --name autoschema-cluster \
      --location eastus \
      --node-count 3 \
      --min-count 1 \
      --max-count 10 \
      --enable-cluster-autoscaler \
      --node-vm-size Standard_D4s_v3 \
      --node-osdisk-size 100 \
      --node-osdisk-type Managed \
      --enable-addons monitoring \
      --generate-ssh-keys
```

## Production Considerations

### Security Hardening

#### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: autoschema-network-policy
  namespace: autoschema
spec:
  podSelector:
    matchLabels:
      app: autoschema-kg-rust
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: neo4j
    ports:
    - protocol: TCP
      port: 7687
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS for LLM APIs
```

#### Pod Security Standards

```yaml
# pod-security-policy.yaml
apiVersion: v1
kind: Pod
metadata:
  name: autoschema-kg-rust
  annotations:
    seccomp.security.alpha.kubernetes.io/pod: runtime/default
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: autoschema-kg-rust
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
      runAsNonRoot: true
      runAsUser: 1000
```

### Resource Management

#### Resource Quotas

```yaml
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: autoschema-quota
  namespace: autoschema
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "10"
    requests.storage: 1Ti
    pods: "50"
    services: "10"
    secrets: "20"
    configmaps: "20"
```

#### Limit Ranges

```yaml
# limit-range.yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: autoschema-limits
  namespace: autoschema
spec:
  limits:
  - default:
      cpu: "2"
      memory: "4Gi"
    defaultRequest:
      cpu: "500m"
      memory: "1Gi"
    max:
      cpu: "8"
      memory: "16Gi"
    min:
      cpu: "100m"
      memory: "256Mi"
    type: Container
  - default:
      storage: "10Gi"
    max:
      storage: "100Gi"
    min:
      storage: "1Gi"
    type: PersistentVolumeClaim
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    rule_files:
      - "/etc/prometheus/rules/*.yml"

    scrape_configs:
    - job_name: 'autoschema-kg-rust'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

    - job_name: 'neo4j'
      static_configs:
      - targets: ['neo4j-service:7474']
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "AutoSchema KG Rust",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"autoschema-kg-rust\"}[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"autoschema-kg-rust\"}[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes{job=\"autoschema-kg-rust\"}",
            "legendFormat": "RSS Memory"
          }
        ]
      }
    ]
  }
}
```

## Backup and Disaster Recovery

### Neo4j Backup Strategy

```yaml
# neo4j-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: neo4j-backup
  namespace: autoschema
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: neo4j-backup
            image: neo4j:5.11
            command:
            - /bin/bash
            - -c
            - |
              neo4j-admin database dump --database=knowledge_graph --to-path=/backup/$(date +%Y%m%d_%H%M%S)_knowledge_graph.dump
              # Upload to S3/GCS/Azure Blob
              aws s3 cp /backup/ s3://autoschema-backups/neo4j/ --recursive
            env:
            - name: NEO4J_AUTH
              valueFrom:
                secretKeyRef:
                  name: autoschema-secrets
                  key: neo4j-auth
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
            - name: neo4j-data
              mountPath: /data
              readOnly: true
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          - name: neo4j-data
            persistentVolumeClaim:
              claimName: neo4j-data-neo4j-0
          restartPolicy: OnFailure
```

### Application State Backup

```yaml
# application-backup.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: autoschema-backup
  namespace: autoschema
spec:
  schedule: "0 1 * * *"  # Daily at 1 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: autoschema-backup-utils:latest
            command:
            - /bin/bash
            - -c
            - |
              # Backup vector indices
              tar -czf /backup/vectors_$(date +%Y%m%d_%H%M%S).tar.gz /app/data/vectors/

              # Backup configuration
              tar -czf /backup/config_$(date +%Y%m%d_%H%M%S).tar.gz /app/config/

              # Upload to cloud storage
              aws s3 cp /backup/ s3://autoschema-backups/application/ --recursive

              # Cleanup old backups (keep 30 days)
              find /backup -name "*.tar.gz" -mtime +30 -delete
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
            - name: app-data
              mountPath: /app/data
              readOnly: true
            - name: app-config
              mountPath: /app/config
              readOnly: true
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          - name: app-data
            persistentVolumeClaim:
              claimName: autoschema-data-pvc
          - name: app-config
            configMap:
              name: autoschema-config
          restartPolicy: OnFailure
```

## Scaling Strategies

### Vertical Scaling

```yaml
# vertical-pod-autoscaler.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: autoschema-vpa
  namespace: autoschema
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autoschema-kg-rust
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: autoschema-kg-rust
      minAllowed:
        cpu: 500m
        memory: 1Gi
      maxAllowed:
        cpu: 8
        memory: 16Gi
      controlledResources: ["cpu", "memory"]
```

### Multi-Region Deployment

```yaml
# multi-region-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: autoschema-us-west
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/autoschema-k8s
    targetRevision: main
    path: manifests/
  destination:
    server: https://us-west-cluster.k8s.local
    namespace: autoschema
  syncPolicy:
    automated:
      prune: true
      selfHeal: true

---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: autoschema-eu-west
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/autoschema-k8s
    targetRevision: main
    path: manifests/
  destination:
    server: https://eu-west-cluster.k8s.local
    namespace: autoschema
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

This comprehensive deployment guide covers all major deployment scenarios from development to production, including containerization, orchestration, cloud platforms, security, monitoring, and disaster recovery strategies.