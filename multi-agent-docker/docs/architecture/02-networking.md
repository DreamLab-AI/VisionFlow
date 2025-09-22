# Networking Deep Dive

This document provides a detailed explanation of the networking model used in the Multi-Agent Docker Environment.

### 1. Custom Docker Network: `docker_ragflow`

The entire environment operates on a custom Docker bridge network named `docker_ragflow`. This network provides a private and isolated communication channel between the containers, enabling them to resolve each other's addresses using their service names as hostnames.

**Key Characteristics**:
- **Isolation**: Containers on this network are isolated from the host machine's network, except for explicitly published ports.
- **Service Discovery**: Docker's embedded DNS server allows containers to look up the IP address of other containers on the same network using their service names (e.g., `multi-agent-container` can reach `gui-tools-container` at `http://gui-tools-service`).
- **Scalability**: The bridge network design allows for easy addition of new services without complex network configuration.

### 2. Inter-Container Communication

Communication between the `multi-agent-container` and the `gui-tools-container` is primarily achieved via TCP sockets. The MCP bridge tools in the `multi-agent-container` connect to the TCP servers running in the `gui-tools-container`.

#### Service Hostnames

The hostnames for the GUI application services are defined as environment variables in the `docker-compose.yml` file for the `multi-agent` service. This allows the bridge clients to dynamically connect to the correct host.

- `BLENDER_HOST=gui-tools-service`
- `QGIS_HOST=gui-tools-service`
- `PBR_HOST=gui-tools-service`

### 3. Port Mapping

The following table details the ports exposed by the environment to the host machine:

| Port | Service | Container | Purpose |
| :--- | :--- | :--- | :--- |
| `3000` | `multi-agent` | `multi-agent-container` | Claude Flow UI |
| `3002` | `multi-agent` | `multi-agent-container` | MCP WebSocket Bridge for external control |
| `9500` | `multi-agent` | `multi-agent-container` | MCP TCP Server for high-performance connections |
| `9501` | `multi-agent` | `multi-agent-container` | MCP Health Check endpoint |
| `5901` | `gui-tools-service` | `gui-tools-container` | VNC access to the XFCE desktop environment |
| `9876` | `gui-tools-service` | `gui-tools-container` | Blender MCP TCP Server |
| `9877` | `gui-tools-service` | `gui-tools-container` | QGIS MCP TCP Server |
| `9878` | `gui-tools-service` | `gui-tools-container` | PBR Generator MCP TCP Server |

### 4. MCP Connectivity Endpoints

The environment provides two primary endpoints for MCP communication, managed by `supervisord` in the `multi-agent-container`.

#### MCP TCP Server (`mcp-tcp-server.js`)
- **Port**: `9500`
- **Purpose**: A high-performance, persistent TCP server. It maintains a single `claude-flow` instance and multiplexes requests from multiple TCP clients to it. This is ideal for performance-critical applications and inter-service communication.
- **Authentication**: Uses a token-based authentication mechanism defined by `TCP_AUTH_TOKEN`.

#### MCP WebSocket Bridge (`mcp-ws-relay.js`)
- **Port**: `3002`
- **Purpose**: A bridge that translates WebSocket connections to `stdio` for `claude-flow`. For each new WebSocket connection, it spawns a dedicated `claude-flow` process, ensuring session isolation. This is ideal for web-based clients and external control systems.
- **Authentication**: Uses a bearer token in the `Authorization` header, defined by `WS_AUTH_TOKEN`.

### 5. Network Security Considerations

- **Internal Communication**: All inter-container communication happens over the private Docker network, isolated from the host.
- **Authentication**: Both WebSocket and TCP endpoints require authentication tokens to prevent unauthorized access.
- **Port Exposure**: Only necessary ports are exposed to the host, minimizing the attack surface.
- **SSL/TLS**: Production deployments should enable SSL/TLS encryption by setting `SSL_ENABLED=true` and providing certificates.

### 6. Troubleshooting Network Issues

If you experience connectivity issues:

1. **Verify Container Status**: Use `docker ps` to ensure all containers are running.
2. **Check Network Membership**: Use `docker network inspect docker_ragflow` to verify containers are on the correct network.
3. **Test Internal Connectivity**: Use `docker exec` to run `nc` (netcat) commands between containers.
4. **Review Logs**: Check container logs with `docker logs <container-name>` for connection errors.
5. **Verify Port Availability**: Ensure no other services on the host are using the exposed ports.