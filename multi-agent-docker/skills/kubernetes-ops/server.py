#!/usr/bin/env python3
"""
Kubernetes Operations MCP Server
Native Python client for cluster management - no kubectl wrapper
"""

import os
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import base64
import json

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    from kubernetes.stream import stream
except ImportError:
    print("Error: kubernetes package not installed. Run: pip install kubernetes", file=sys.stderr)
    sys.exit(1)

# Initialize FastMCP server
mcp = FastMCP("kubernetes-ops")

# Global state for current context
CURRENT_CONTEXT = None
KUBECONFIG_PATH = None


def load_kube_config():
    """Load kubeconfig with in-cluster fallback"""
    global CURRENT_CONTEXT, KUBECONFIG_PATH

    try:
        # Try loading from kubeconfig
        KUBECONFIG_PATH = os.environ.get('KUBECONFIG', os.path.expanduser('~/.kube/config'))
        config.load_kube_config(config_file=KUBECONFIG_PATH)
        contexts, active_context = config.list_kube_config_contexts(config_file=KUBECONFIG_PATH)
        CURRENT_CONTEXT = active_context['name']
        return True
    except Exception as e:
        # Try in-cluster config
        try:
            config.load_incluster_config()
            CURRENT_CONTEXT = "in-cluster"
            return True
        except Exception:
            raise Exception(f"Failed to load kubeconfig: {e}")


def get_age(timestamp) -> str:
    """Calculate age from timestamp"""
    if not timestamp:
        return "unknown"

    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    delta = now - timestamp

    if delta.days > 365:
        return f"{delta.days // 365}y"
    elif delta.days > 0:
        return f"{delta.days}d"
    elif delta.seconds > 3600:
        return f"{delta.seconds // 3600}h"
    elif delta.seconds > 60:
        return f"{delta.seconds // 60}m"
    else:
        return f"{delta.seconds}s"


# Context and Namespace Management

@mcp.tool()
def get_contexts() -> str:
    """List all available kubeconfig contexts"""
    try:
        contexts, active_context = config.list_kube_config_contexts(config_file=KUBECONFIG_PATH)

        result = ["Available Contexts:", ""]
        for ctx in contexts:
            name = ctx['name']
            cluster = ctx['context'].get('cluster', 'N/A')
            user = ctx['context'].get('user', 'N/A')
            namespace = ctx['context'].get('namespace', 'default')

            marker = "* " if name == active_context['name'] else "  "
            result.append(f"{marker}{name}")
            result.append(f"    Cluster: {cluster}")
            result.append(f"    User: {user}")
            result.append(f"    Namespace: {namespace}")
            result.append("")

        return "\n".join(result)
    except Exception as e:
        return f"Error listing contexts: {str(e)}"


@mcp.tool()
def use_context(context: str) -> str:
    """Switch to a different Kubernetes context"""
    global CURRENT_CONTEXT

    try:
        config.load_kube_config(config_file=KUBECONFIG_PATH, context=context)
        CURRENT_CONTEXT = context
        return f"Switched to context: {context}"
    except Exception as e:
        return f"Error switching context: {str(e)}"


@mcp.tool()
def get_namespaces() -> str:
    """List all namespaces in the current cluster"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        namespaces = v1.list_namespace()

        result = [f"Namespaces in context '{CURRENT_CONTEXT}':", ""]
        result.append(f"{'NAME':<30} {'STATUS':<10} {'AGE':<10}")
        result.append("-" * 52)

        for ns in namespaces.items:
            name = ns.metadata.name
            status = ns.status.phase
            age = get_age(ns.metadata.creation_timestamp)
            result.append(f"{name:<30} {status:<10} {age:<10}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error listing namespaces: {e.reason}"


# Pod Operations

@mcp.tool()
def get_pods(namespace: str = "default", labels: Optional[str] = None) -> str:
    """List pods in a namespace with optional label selector"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        pods = v1.list_namespaced_pod(namespace=namespace, label_selector=labels)

        result = [f"Pods in namespace '{namespace}':", ""]
        result.append(f"{'NAME':<40} {'STATUS':<15} {'RESTARTS':<10} {'AGE':<10}")
        result.append("-" * 77)

        for pod in pods.items:
            name = pod.metadata.name
            status = pod.status.phase
            age = get_age(pod.metadata.creation_timestamp)

            # Calculate restarts
            restarts = 0
            if pod.status.container_statuses:
                restarts = sum(cs.restart_count for cs in pod.status.container_statuses)

            result.append(f"{name:<40} {status:<15} {restarts:<10} {age:<10}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error listing pods: {e.reason}"


@mcp.tool()
def get_pod(name: str, namespace: str = "default") -> str:
    """Get detailed information about a specific pod"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        pod = v1.read_namespaced_pod(name=name, namespace=namespace)

        result = [f"Pod: {name}", f"Namespace: {namespace}", ""]

        # Basic info
        result.append("Metadata:")
        result.append(f"  Created: {pod.metadata.creation_timestamp}")
        result.append(f"  Age: {get_age(pod.metadata.creation_timestamp)}")
        if pod.metadata.labels:
            result.append(f"  Labels: {pod.metadata.labels}")
        result.append("")

        # Status
        result.append("Status:")
        result.append(f"  Phase: {pod.status.phase}")
        result.append(f"  Host IP: {pod.status.host_ip}")
        result.append(f"  Pod IP: {pod.status.pod_ip}")
        result.append("")

        # Containers
        result.append("Containers:")
        if pod.status.container_statuses:
            for cs in pod.status.container_statuses:
                result.append(f"  {cs.name}:")
                result.append(f"    Image: {cs.image}")
                result.append(f"    Ready: {cs.ready}")
                result.append(f"    Restarts: {cs.restart_count}")
                result.append(f"    State: {list(cs.state.to_dict().keys())[0]}")

        # Conditions
        if pod.status.conditions:
            result.append("")
            result.append("Conditions:")
            for cond in pod.status.conditions:
                result.append(f"  {cond.type}: {cond.status}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error getting pod: {e.reason}"


@mcp.tool()
def describe_pod(name: str, namespace: str = "default") -> str:
    """Describe pod with full details (kubectl describe equivalent)"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        pod = v1.read_namespaced_pod(name=name, namespace=namespace)

        result = [f"Name: {pod.metadata.name}"]
        result.append(f"Namespace: {pod.metadata.namespace}")
        result.append(f"Priority: {pod.spec.priority or 0}")
        result.append(f"Node: {pod.spec.node_name}/{pod.status.host_ip}")
        result.append(f"Start Time: {pod.status.start_time}")
        result.append(f"Labels: {pod.metadata.labels or {}}")
        result.append(f"Annotations: {pod.metadata.annotations or {}}")
        result.append(f"Status: {pod.status.phase}")
        result.append(f"IP: {pod.status.pod_ip}")
        result.append("")

        # Containers
        result.append("Containers:")
        for container in pod.spec.containers:
            result.append(f"  {container.name}:")
            result.append(f"    Image: {container.image}")
            result.append(f"    Ports: {container.ports or []}")
            result.append(f"    Environment: {len(container.env or [])} variables")
            if container.resources:
                result.append(f"    Requests: {container.resources.requests or {}}")
                result.append(f"    Limits: {container.resources.limits or {}}")

        # Conditions
        result.append("")
        result.append("Conditions:")
        if pod.status.conditions:
            for cond in pod.status.conditions:
                result.append(f"  {cond.type}: {cond.status} ({cond.reason or 'N/A'})")

        # Events
        events = v1.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={name}"
        )

        if events.items:
            result.append("")
            result.append("Events:")
            for event in sorted(events.items, key=lambda x: x.last_timestamp or x.event_time):
                age = get_age(event.last_timestamp or event.event_time)
                result.append(f"  {age} {event.type} {event.reason}: {event.message}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error describing pod: {e.reason}"


@mcp.tool()
def pod_logs(
    name: str,
    namespace: str = "default",
    container: Optional[str] = None,
    tail: int = 100,
    previous: bool = False
) -> str:
    """Retrieve pod logs"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        logs = v1.read_namespaced_pod_log(
            name=name,
            namespace=namespace,
            container=container,
            tail_lines=tail,
            previous=previous
        )

        header = f"Logs for pod '{name}'"
        if container:
            header += f" (container: {container})"
        if previous:
            header += " [PREVIOUS]"
        header += f" (last {tail} lines):"

        return f"{header}\n\n{logs}"
    except ApiException as e:
        return f"Error getting pod logs: {e.reason}"


@mcp.tool()
def exec_command(
    pod: str,
    command: List[str],
    namespace: str = "default",
    container: Optional[str] = None
) -> str:
    """Execute command in pod container"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        resp = stream(
            v1.connect_get_namespaced_pod_exec,
            pod,
            namespace,
            command=command,
            container=container,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False
        )

        return f"Command output from pod '{pod}':\n\n{resp}"
    except ApiException as e:
        return f"Error executing command: {e.reason}"


# Deployment Operations

@mcp.tool()
def get_deployments(namespace: str = "default") -> str:
    """List deployments in namespace"""
    load_kube_config()
    apps_v1 = client.AppsV1Api()

    try:
        deployments = apps_v1.list_namespaced_deployment(namespace=namespace)

        result = [f"Deployments in namespace '{namespace}':", ""]
        result.append(f"{'NAME':<35} {'READY':<10} {'UP-TO-DATE':<12} {'AVAILABLE':<12} {'AGE':<10}")
        result.append("-" * 81)

        for deploy in deployments.items:
            name = deploy.metadata.name
            replicas = deploy.spec.replicas or 0
            ready = deploy.status.ready_replicas or 0
            updated = deploy.status.updated_replicas or 0
            available = deploy.status.available_replicas or 0
            age = get_age(deploy.metadata.creation_timestamp)

            result.append(f"{name:<35} {ready}/{replicas:<9} {updated:<12} {available:<12} {age:<10}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error listing deployments: {e.reason}"


@mcp.tool()
def get_deployment(name: str, namespace: str = "default") -> str:
    """Get detailed deployment information"""
    load_kube_config()
    apps_v1 = client.AppsV1Api()

    try:
        deploy = apps_v1.read_namespaced_deployment(name=name, namespace=namespace)

        result = [f"Deployment: {name}", f"Namespace: {namespace}", ""]

        result.append("Metadata:")
        result.append(f"  Created: {deploy.metadata.creation_timestamp}")
        result.append(f"  Age: {get_age(deploy.metadata.creation_timestamp)}")
        if deploy.metadata.labels:
            result.append(f"  Labels: {deploy.metadata.labels}")
        result.append("")

        result.append("Spec:")
        result.append(f"  Replicas: {deploy.spec.replicas}")
        result.append(f"  Strategy: {deploy.spec.strategy.type}")
        result.append(f"  Selector: {deploy.spec.selector.match_labels}")
        result.append("")

        result.append("Status:")
        result.append(f"  Ready Replicas: {deploy.status.ready_replicas or 0}")
        result.append(f"  Available Replicas: {deploy.status.available_replicas or 0}")
        result.append(f"  Updated Replicas: {deploy.status.updated_replicas or 0}")
        result.append(f"  Unavailable Replicas: {deploy.status.unavailable_replicas or 0}")

        if deploy.status.conditions:
            result.append("")
            result.append("Conditions:")
            for cond in deploy.status.conditions:
                result.append(f"  {cond.type}: {cond.status} - {cond.message or ''}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error getting deployment: {e.reason}"


@mcp.tool()
def scale_deployment(name: str, replicas: int, namespace: str = "default") -> str:
    """Scale deployment to specified replicas"""
    load_kube_config()
    apps_v1 = client.AppsV1Api()

    try:
        # Update replica count
        body = {"spec": {"replicas": replicas}}
        apps_v1.patch_namespaced_deployment_scale(name=name, namespace=namespace, body=body)

        return f"Scaled deployment '{name}' to {replicas} replicas in namespace '{namespace}'"
    except ApiException as e:
        return f"Error scaling deployment: {e.reason}"


@mcp.tool()
def restart_deployment(name: str, namespace: str = "default") -> str:
    """Restart deployment (kubectl rollout restart)"""
    load_kube_config()
    apps_v1 = client.AppsV1Api()

    try:
        # Trigger restart by updating annotation
        now = datetime.now(timezone.utc).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": now
                        }
                    }
                }
            }
        }

        apps_v1.patch_namespaced_deployment(name=name, namespace=namespace, body=body)

        return f"Restarted deployment '{name}' in namespace '{namespace}'"
    except ApiException as e:
        return f"Error restarting deployment: {e.reason}"


# Service and Config Operations

@mcp.tool()
def get_services(namespace: str = "default") -> str:
    """List services in namespace"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        services = v1.list_namespaced_service(namespace=namespace)

        result = [f"Services in namespace '{namespace}':", ""]
        result.append(f"{'NAME':<35} {'TYPE':<15} {'CLUSTER-IP':<18} {'EXTERNAL-IP':<18} {'PORTS':<20} {'AGE':<10}")
        result.append("-" * 118)

        for svc in services.items:
            name = svc.metadata.name
            svc_type = svc.spec.type
            cluster_ip = svc.spec.cluster_ip or "None"
            external_ip = svc.status.load_balancer.ingress[0].ip if (
                svc.status.load_balancer and svc.status.load_balancer.ingress
            ) else "<none>"

            ports = []
            if svc.spec.ports:
                for port in svc.spec.ports:
                    port_str = f"{port.port}"
                    if port.target_port:
                        port_str += f":{port.target_port}"
                    if port.protocol != "TCP":
                        port_str += f"/{port.protocol}"
                    ports.append(port_str)
            ports_str = ",".join(ports)

            age = get_age(svc.metadata.creation_timestamp)

            result.append(f"{name:<35} {svc_type:<15} {cluster_ip:<18} {external_ip:<18} {ports_str:<20} {age:<10}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error listing services: {e.reason}"


@mcp.tool()
def get_configmaps(namespace: str = "default") -> str:
    """List configmaps in namespace"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        configmaps = v1.list_namespaced_config_map(namespace=namespace)

        result = [f"ConfigMaps in namespace '{namespace}':", ""]
        result.append(f"{'NAME':<40} {'DATA':<10} {'AGE':<10}")
        result.append("-" * 62)

        for cm in configmaps.items:
            name = cm.metadata.name
            data_count = len(cm.data) if cm.data else 0
            age = get_age(cm.metadata.creation_timestamp)

            result.append(f"{name:<40} {data_count:<10} {age:<10}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error listing configmaps: {e.reason}"


@mcp.tool()
def get_secrets(namespace: str = "default", decode: bool = False) -> str:
    """List secrets (names only by default for security)"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        secrets = v1.list_namespaced_secret(namespace=namespace)

        result = [f"Secrets in namespace '{namespace}':", ""]

        if decode:
            result.append("⚠️  WARNING: Displaying decoded secret values")
            result.append("")

        result.append(f"{'NAME':<40} {'TYPE':<35} {'DATA':<10} {'AGE':<10}")
        result.append("-" * 97)

        for secret in secrets.items:
            name = secret.metadata.name
            secret_type = secret.type
            data_count = len(secret.data) if secret.data else 0
            age = get_age(secret.metadata.creation_timestamp)

            result.append(f"{name:<40} {secret_type:<35} {data_count:<10} {age:<10}")

            if decode and secret.data:
                result.append("  Data:")
                for key, value in secret.data.items():
                    decoded = base64.b64decode(value).decode('utf-8')
                    result.append(f"    {key}: {decoded}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error listing secrets: {e.reason}"


# Cluster Operations

@mcp.tool()
def get_nodes() -> str:
    """List all cluster nodes with status"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        nodes = v1.list_node()

        result = [f"Nodes in cluster:", ""]
        result.append(f"{'NAME':<35} {'STATUS':<12} {'ROLES':<20} {'AGE':<10} {'VERSION':<15}")
        result.append("-" * 94)

        for node in nodes.items:
            name = node.metadata.name

            # Status
            status = "Unknown"
            if node.status.conditions:
                for cond in node.status.conditions:
                    if cond.type == "Ready":
                        status = "Ready" if cond.status == "True" else "NotReady"

            # Roles
            roles = []
            if node.metadata.labels:
                for key, value in node.metadata.labels.items():
                    if key.startswith("node-role.kubernetes.io/"):
                        role = key.split("/")[1]
                        roles.append(role if role else "worker")
            roles_str = ",".join(roles) if roles else "<none>"

            age = get_age(node.metadata.creation_timestamp)
            version = node.status.node_info.kubelet_version

            result.append(f"{name:<35} {status:<12} {roles_str:<20} {age:<10} {version:<15}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error listing nodes: {e.reason}"


@mcp.tool()
def get_events(namespace: str = "default", limit: int = 50) -> str:
    """Get recent cluster events"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        events = v1.list_namespaced_event(namespace=namespace)

        # Sort by timestamp (newest first)
        sorted_events = sorted(
            events.items,
            key=lambda x: x.last_timestamp or x.event_time or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True
        )[:limit]

        result = [f"Recent events in namespace '{namespace}' (last {limit}):", ""]
        result.append(f"{'AGE':<8} {'TYPE':<10} {'REASON':<25} {'OBJECT':<30} {'MESSAGE':<50}")
        result.append("-" * 125)

        for event in sorted_events:
            age = get_age(event.last_timestamp or event.event_time)
            event_type = event.type
            reason = event.reason
            obj = f"{event.involved_object.kind}/{event.involved_object.name}"
            message = event.message[:47] + "..." if len(event.message) > 50 else event.message

            result.append(f"{age:<8} {event_type:<10} {reason:<25} {obj:<30} {message:<50}")

        return "\n".join(result)
    except ApiException as e:
        return f"Error listing events: {e.reason}"


@mcp.tool()
def port_forward_info(name: str, namespace: str = "default") -> str:
    """Get port forward command (doesn't actually forward)"""
    load_kube_config()
    v1 = client.CoreV1Api()

    try:
        # Try as pod first
        try:
            pod = v1.read_namespaced_pod(name=name, namespace=namespace)
            ports = []
            for container in pod.spec.containers:
                if container.ports:
                    for port in container.ports:
                        ports.append(f"{port.container_port}:{port.container_port}")

            port_str = " ".join(ports) if ports else "8080:8080"
            return f"kubectl port-forward pod/{name} {port_str} -n {namespace}"
        except:
            pass

        # Try as service
        svc = v1.read_namespaced_service(name=name, namespace=namespace)
        ports = []
        if svc.spec.ports:
            for port in svc.spec.ports:
                ports.append(f"{port.port}:{port.target_port or port.port}")

        port_str = " ".join(ports) if ports else "8080:80"
        return f"kubectl port-forward service/{name} {port_str} -n {namespace}"
    except ApiException as e:
        return f"Error getting port forward info: {e.reason}"


# Manifest Application

@mcp.tool()
def apply_manifest(manifest: str, namespace: str = "default", dry_run: bool = True) -> str:
    """Apply YAML manifest (dry-run by default for safety)"""
    load_kube_config()

    try:
        import yaml
        from kubernetes import utils

        # Parse YAML
        resources = list(yaml.safe_load_all(manifest))

        result = [f"{'DRY-RUN: ' if dry_run else ''}Applying manifest to namespace '{namespace}':", ""]

        api_client = client.ApiClient()

        for resource in resources:
            if not resource:
                continue

            kind = resource.get('kind', 'Unknown')
            name = resource.get('metadata', {}).get('name', 'unknown')

            if dry_run:
                result.append(f"Would apply: {kind}/{name}")
            else:
                # Actually apply
                utils.create_from_dict(api_client, resource, namespace=namespace)
                result.append(f"Applied: {kind}/{name}")

        if dry_run:
            result.append("")
            result.append("⚠️  DRY-RUN mode: No changes were made")
            result.append("Set dry_run=False to actually apply")

        return "\n".join(result)
    except Exception as e:
        return f"Error applying manifest: {str(e)}"


if __name__ == "__main__":
    # Initialize on startup
    try:
        load_kube_config()
        print(f"Loaded kubeconfig, current context: {CURRENT_CONTEXT}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not load kubeconfig: {e}", file=sys.stderr)
        print("In-cluster config will be attempted when tools are called", file=sys.stderr)

    mcp.run()
