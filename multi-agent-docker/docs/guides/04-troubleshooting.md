# Troubleshooting Guide

This guide provides solutions to common issues you might encounter while using the Multi-Agent Docker Environment.

## Quick Fixes for Common Setup Issues

### Setup Script Hangs on Goal Planner

**Problem**: The setup script hangs when initializing the Goal Planner agent.

**Solution**:
- The script now has a 30-second timeout for agent initialization
- If it fails, you can manually initialize agents later:
  ```bash
  npx claude-flow@alpha goal init
  npx claude-flow@alpha neural init
  ```
- Check if MCP services are running: `mcp-tcp-status`

### Rust Toolchain Errors

**Problem**: `rustup could not choose a version of rustc to run`

**Solution**:
- The setup script now automatically sets the default toolchain
- If you still see errors, manually set it:
  ```bash
  rustup default stable
  ```

### Playwright Not Working

**Problem**: Playwright commands fail or browsers not found.

**Solution**:
- Playwright now runs in the GUI container with visual debugging
- Access it via VNC on port 5901
- Check proxy status: `playwright-proxy-status`
- Ensure GUI container is running: `docker-compose ps gui-tools-service`

### MCP Services Not Starting

**Problem**: MCP TCP server or WebSocket bridge not responding.

**Solution**:
- Check service status: `mcp-status`
- View logs: `mcp-tcp-logs` or `mcp-ws-logs`
- Restart services: `mcp-restart`
- The setup script now checks TCP port 9500 directly instead of just the health endpoint

## 1. Networking Issues

Networking problems are common in multi-container setups. Here's how to diagnose them.

### Issue: "Connection refused" from a Bridge Tool

If an MCP bridge tool (e.g., `blender-mcp`, `qgis-mcp`) returns a "Connection refused" error, it means the client in the `multi-agent-container` could not reach the server in the `gui-tools-container`.

**Debugging Steps:**

1.  **Verify Both Containers are Running**:
    ```bash
    # Run this on your host machine
    ./multi-agent.sh status
    ```
    Ensure both `multi-agent-container` and `gui-tools-container` are `Up`.

2.  **Check Network Connectivity from Inside the Container**:
    Access the `multi-agent-container` shell:
    ```bash
    ./multi-agent.sh shell
    ```
    Inside the container, use `ping` to test if the service name of the GUI container is resolvable:
    ```bash
    ping gui-tools-service
    ```
    You should see a reply. If not, there's a problem with the Docker network itself.

3.  **Inspect the Docker Network**:
    From your host machine, inspect the `docker_ragflow` network to ensure both containers are attached:
    ```bash
    docker network inspect docker_ragflow
    ```
    Look for the `"Containers"` section in the JSON output. It should list both `multi-agent-container` and `gui-tools-container`.

4.  **Check Container Logs**:
    The issue might be with the server application inside the `gui-tools-container`. Check its logs:
    ```bash
    # Run this on your host machine
    ./multi-agent.sh logs gui-tools-service
    ```
    Look for any error messages from Blender, QGIS, or the PBR generator server.

### Issue: Cannot Connect to MCP Services

If you can't connect to the MCP TCP server or WebSocket bridge:

1. **Check Service Status**:
   ```bash
   multi-agent services
   ```
   All MCP services should show as RUNNING.

2. **Test Direct Connection**:
   ```bash
   # Test TCP server
   multi-agent test-mcp
   
   # Test WebSocket bridge
   mcp-test-ws
   ```

3. **Review Security Logs**:
   ```bash
   mcp-security-audit
   ```
   Look for authentication failures or blocked connections.

## 2. VNC Issues

### Issue: Cannot Connect to VNC on `localhost:5901`

If you can't connect to the GUI environment using a VNC client, follow these steps.

1.  **Verify Port Mapping**:
    Run `docker ps` on your host machine and check the `PORTS` column for `gui-tools-container`. It should include `0.0.0.0:5901->5901/tcp`.

2.  **Check VNC Server Logs**:
    The VNC server (`x11vnc`) runs inside the `gui-tools-container`. Check its logs for errors:
    ```bash
    # Run this on your host machine
    ./multi-agent.sh logs gui-tools-service | grep x11vnc
    ```

3.  **Check XFCE and Xvfb Logs**:
    The VNC server depends on the XFCE desktop environment and the Xvfb virtual frame buffer. Check their logs as well:
    ```bash
    # Run this on your host machine
    ./multi-agent.sh logs gui-tools-service
    ```
    Look for errors related to `Xvfb` or `xfce4-session`.

### Issue: Black Screen in VNC

If you connect but see a black screen:

1. **Wait for Desktop to Load**: The XFCE desktop can take 30-60 seconds to fully initialize.

2. **Check Display Variable**:
   ```bash
   docker exec gui-tools-container env | grep DISPLAY
   ```
   Should show `DISPLAY=:99`.

3. **Restart VNC Service**: From within the gui-tools-container, restart x11vnc.

## 3. MCP Tool Issues

### Issue: "Tool not found"

If `claude-flow` or the `mcp-helper.sh` script reports that a tool is not found:

1.  **Verify `.mcp.json`**:
    Inside the `multi-agent-container`, check the contents of `/workspace/.mcp.json`. Ensure the tool is defined correctly.

2.  **Re-run Setup Script**:
    The workspace might be out of sync with the core assets. Re-run the setup script to copy the latest configurations:
    ```bash
    /app/setup-workspace.sh --force
    ```

3.  **Check File Permissions**:
    Ensure the tool's script is executable:
    ```bash
    ls -l /workspace/mcp-tools/
    ```

### Issue: Tool Execution Failures

If a tool runs but fails:

1. **Check Tool Logs**:
   ```bash
   # View specific tool logs
   blender-log  # for Blender MCP
   qgis-log     # for QGIS MCP
   tcp-log      # for TCP server
   ```

2. **Validate JSON Payload**:
   Ensure your JSON is valid:
   ```bash
   echo '{"your":"json"}' | jq .
   ```

3. **Test with Simple Command**:
   Try the simplest possible command for the tool to isolate the issue.

## 4. Permission Issues

### Issue: "Permission denied" when running scripts

If you get permission errors in the `/workspace` directory, it's likely due to a UID/GID mismatch between your host and the container.

1.  **Set `HOST_UID` and `HOST_GID`**:
    On your host machine, find your user and group ID:
    ```bash
    id -u
    id -g
    ```
    Update these values in your `.env` file.

2.  **Rebuild the Container**:
    ```bash
    ./multi-agent.sh build
    ```
    This will rebuild the image with a `dev` user that matches your host permissions.

3.  **Fix Existing Files**:
    You may need to fix the ownership of existing files in your workspace directory from your host machine:
    ```bash
    sudo chown -R $(id -u):$(id -g) ./workspace
    ```

## 5. Claude Authentication Issues

### Issue: Claude not authenticated in container

1. **Verify Host Authentication**:
   ```bash
   claude --version  # Should work on host
   ```

2. **Check Mount Points**:
   ```bash
   docker exec multi-agent-container ls -la /home/dev/.claude/
   ```

3. **Re-authenticate on Host**:
   ```bash
   claude login
   ```
   Then restart the container.

## 6. Performance Issues

### Issue: Container running slowly

1. **Check Resource Allocation**:
   Ensure adequate resources in `.env`:
   ```
   DOCKER_MEMORY=16g
   DOCKER_CPUS=4
   ```

2. **Monitor Resource Usage**:
   ```bash
   docker stats
   ```

3. **Check for Memory Leaks**:
   ```bash
   mcp-memory
   ```

## 7. Security-Related Issues

### Issue: Authentication failures

1. **Verify Tokens**:
   Ensure tokens in `.env` match what you're using:
   - `WS_AUTH_TOKEN`
   - `TCP_AUTH_TOKEN`

2. **Check Rate Limiting**:
   You might be hitting rate limits:
   ```bash
   grep "RATE_LIMIT" /app/mcp-logs/security/*.log
   ```

3. **Review Blocked IPs**:
   ```bash
   grep "BLOCKED" /app/mcp-logs/security/*.log
   ```

## 8. Automated Setup Issues

### Issue: Automated setup failing

1. **Check Setup Logs**:
   ```bash
   setup-logs
   ```

2. **View Setup Status**:
   ```bash
   setup-status
   ```

3. **Re-run Setup Manually**:
   ```bash
   rerun-setup
   ```

## Getting Additional Help

If these solutions don't resolve your issue:

1. **Check Container Health**:
   ```bash
   multi-agent health
   ```

2. **Collect Diagnostic Information**:
   ```bash
   # From host
   ./multi-agent.sh logs > diagnostics.log
   
   # From container
   multi-agent status >> diagnostics.log
   ```

3. **Review Recent Changes**:
   Check if recent updates or configuration changes might have caused the issue.

Remember to always check the logs first - they usually contain the specific error message that will point you to the solution.