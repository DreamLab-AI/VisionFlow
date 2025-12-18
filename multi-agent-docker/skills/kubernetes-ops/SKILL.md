---
name: kubernetes-ops
version: 1.0.0
description: Kubernetes cluster operations using native Python client - pods, deployments, services
author: agentic-workstation
tags: [kubernetes, k8s, kubectl, pods, deployments, cluster]
mcp_server: true
---

# Kubernetes Operations Skill

Professional Kubernetes cluster management using the native Python client library. No kubectl wrapper - direct API interaction for pods, deployments, services, and cluster operations.

## Overview

This skill provides comprehensive Kubernetes cluster operations through the official Python client:

- **Context Management**: Switch between clusters and namespaces
- **Resource Inspection**: Pods, deployments, services, configmaps, secrets
- **Operations**: Scale, restart, exec, logs, events
- **Manifest Application**: Apply YAML with dry-run support
- **Node Management**: Cluster node status and health

Uses `~/.kube/config` by default with support for in-cluster configuration.

## Tools

### Context & Namespace Management

**get_contexts()**
- Lists all available kubeconfig contexts
- Returns: List of context names with current context marked

**use_context(context: str)**
- Switch to a different Kubernetes context
- Parameters:
  - `context`: Context name from kubeconfig
- Returns: Confirmation message

**get_namespaces()**
- Lists all namespaces in the current cluster
- Returns: List of namespace names with status

### Pod Operations

**get_pods(namespace: str = "default", labels: str = None)**
- Lists pods in a namespace
- Parameters:
  - `namespace`: Target namespace (default: "default")
  - `labels`: Label selector (e.g., "app=nginx,tier=frontend")
- Returns: Pod list with name, status, age, restarts

**get_pod(name: str, namespace: str = "default")**
- Detailed information about a specific pod
- Parameters:
  - `name`: Pod name
  - `namespace`: Target namespace
- Returns: Pod spec, status, containers, conditions

**describe_pod(name: str, namespace: str = "default")**
- kubectl describe equivalent with full pod details
- Parameters:
  - `name`: Pod name
  - `namespace`: Target namespace
- Returns: Comprehensive pod description including events

**pod_logs(name: str, namespace: str = "default", container: str = None, tail: int = 100, previous: bool = False)**
- Retrieve pod logs
- Parameters:
  - `name`: Pod name
  - `namespace`: Target namespace
  - `container`: Container name (if multi-container pod)
  - `tail`: Number of lines from end (default: 100)
  - `previous`: Get logs from previous container instance
- Returns: Container logs

**exec_command(pod: str, command: list, namespace: str = "default", container: str = None)**
- Execute command in pod container
- Parameters:
  - `pod`: Pod name
  - `command`: Command as list (e.g., ["ls", "-la"])
  - `namespace`: Target namespace
  - `container`: Container name (if multi-container)
- Returns: Command output

### Deployment Operations

**get_deployments(namespace: str = "default")**
- Lists deployments in namespace
- Parameters:
  - `namespace`: Target namespace
- Returns: Deployment list with replicas, available, updated

**get_deployment(name: str, namespace: str = "default")**
- Detailed deployment information
- Parameters:
  - `name`: Deployment name
  - `namespace`: Target namespace
- Returns: Deployment spec, status, replica details

**scale_deployment(name: str, replicas: int, namespace: str = "default")**
- Scale deployment replicas
- Parameters:
  - `name`: Deployment name
  - `replicas`: Desired replica count
  - `namespace`: Target namespace
- Returns: Confirmation with new replica count

**restart_deployment(name: str, namespace: str = "default")**
- Restart deployment (kubectl rollout restart)
- Parameters:
  - `name`: Deployment name
  - `namespace`: Target namespace
- Returns: Confirmation message

### Service & Config Operations

**get_services(namespace: str = "default")**
- Lists services in namespace
- Parameters:
  - `namespace`: Target namespace
- Returns: Service list with type, cluster-IP, ports

**get_configmaps(namespace: str = "default")**
- Lists configmaps in namespace
- Parameters:
  - `namespace`: Target namespace
- Returns: ConfigMap list with keys

**get_secrets(namespace: str = "default", decode: bool = False)**
- Lists secrets (names only by default for security)
- Parameters:
  - `namespace`: Target namespace
  - `decode`: Return decoded secret values (use with caution)
- Returns: Secret list with type

### Cluster Operations

**get_nodes()**
- Lists all cluster nodes with status
- Returns: Node list with roles, status, age, version

**get_events(namespace: str = "default", limit: int = 50)**
- Recent cluster events
- Parameters:
  - `namespace`: Target namespace
  - `limit`: Maximum events to return
- Returns: Event list with type, reason, message, age

**port_forward_info(name: str, namespace: str = "default")**
- Get port forward command (doesn't actually forward)
- Parameters:
  - `name`: Pod or service name
  - `namespace`: Target namespace
- Returns: kubectl port-forward command string

### Manifest Application

**apply_manifest(manifest: str, namespace: str = "default", dry_run: bool = True)**
- Apply YAML manifest
- Parameters:
  - `manifest`: YAML manifest as string
  - `namespace`: Target namespace
  - `dry_run`: Preview changes without applying (default: True)
- Returns: Applied resources or dry-run preview

## Context and Namespace Handling

The skill maintains state for the current context and uses it for all operations:

1. **Default Context**: Uses current context from kubeconfig
2. **Context Switching**: `use_context()` persists for session
3. **Namespace Scoping**: Most operations default to "default" namespace but accept explicit namespace parameter
4. **In-Cluster Config**: Automatically detects when running inside Kubernetes cluster

### Best Practices

- Always verify context before destructive operations
- Use label selectors for targeted pod queries
- Enable dry-run for manifest application testing
- Use tail parameter for log retrieval to limit output
- Never decode secrets unless absolutely necessary

## Examples

### Basic Pod Inspection

```python
# List all pods in default namespace
pods = get_pods()

# List pods in specific namespace with labels
web_pods = get_pods(namespace="production", labels="app=nginx,tier=frontend")

# Get detailed pod info
pod_details = get_pod(name="nginx-7d8b4d8c9f-x7k2m", namespace="production")

# Get pod logs (last 100 lines)
logs = pod_logs(name="nginx-7d8b4d8c9f-x7k2m", tail=100)

# Get logs from previous crashed container
crash_logs = pod_logs(name="app-pod", previous=True)
```

### Deployment Management

```python
# List deployments
deployments = get_deployments(namespace="production")

# Scale deployment
scale_deployment(name="nginx", replicas=5, namespace="production")

# Restart deployment
restart_deployment(name="nginx", namespace="production")

# Get deployment details
deployment_info = get_deployment(name="nginx", namespace="production")
```

### Cluster Inspection

```python
# List all contexts
contexts = get_contexts()

# Switch context
use_context("production-cluster")

# List namespaces
namespaces = get_namespaces()

# List all nodes
nodes = get_nodes()

# Get recent events
events = get_events(namespace="production", limit=20)
```

### Service and Config Operations

```python
# List services
services = get_services(namespace="production")

# List configmaps
configmaps = get_configmaps(namespace="production")

# List secrets (names only)
secrets = get_secrets(namespace="production")

# Describe pod (kubectl describe equivalent)
pod_description = describe_pod(name="nginx-pod", namespace="production")
```

### Advanced Operations

```python
# Execute command in pod
result = exec_command(
    pod="nginx-7d8b4d8c9f-x7k2m",
    command=["cat", "/etc/nginx/nginx.conf"],
    namespace="production"
)

# Apply manifest with dry-run
dry_run_result = apply_manifest(
    manifest="""
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: nginx
    image: nginx:latest
""",
    namespace="default",
    dry_run=True
)

# Actually apply manifest
apply_result = apply_manifest(manifest=yaml_content, dry_run=False)

# Get port forward command
port_cmd = port_forward_info(name="nginx-service", namespace="production")
# Returns: "kubectl port-forward service/nginx-service 8080:80 -n production"
```

### Multi-Cluster Workflow

```python
# List available clusters
contexts = get_contexts()

# Switch to staging
use_context("staging-cluster")

# Check staging pods
staging_pods = get_pods(namespace="default")

# Switch to production
use_context("production-cluster")

# Check production pods
prod_pods = get_pods(namespace="default")

# Compare deployments
staging_deploys = get_deployments(namespace="default")
prod_deploys = get_deployments(namespace="default")
```

### Troubleshooting Workflow

```python
# 1. Check pod status
pods = get_pods(namespace="production", labels="app=failing-app")

# 2. Get pod details
pod_info = get_pod(name="failing-pod", namespace="production")

# 3. Check logs
logs = pod_logs(name="failing-pod", namespace="production", tail=200)

# 4. Check events
events = get_events(namespace="production", limit=30)

# 5. Describe pod for full details
description = describe_pod(name="failing-pod", namespace="production")

# 6. Check previous container logs if pod crashed
crash_logs = pod_logs(name="failing-pod", previous=True)

# 7. Exec into running pod if needed
exec_command(pod="failing-pod", command=["sh", "-c", "env"], namespace="production")
```

## Configuration

The skill uses standard kubeconfig locations:
- `~/.kube/config` (default)
- `KUBECONFIG` environment variable
- In-cluster service account (when running in Kubernetes)

No additional configuration required for standard setups.

## Security Considerations

- Secret values are hidden by default (use `decode=True` cautiously)
- Exec commands should be used carefully in production
- Manifest application defaults to dry-run for safety
- Context switching requires appropriate kubeconfig permissions
- All operations respect RBAC policies of the current context
