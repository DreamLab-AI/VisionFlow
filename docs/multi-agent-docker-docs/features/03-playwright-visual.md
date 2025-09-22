# Playwright GUI Container Integration

## Overview

This document outlines the architecture and benefits of moving the Playwright MCP server from the main multi-agent container to the GUI tools container, enabling visual browser automation and debugging capabilities.

## Current Architecture

Currently, Playwright runs in the main container with:
- Headless mode using Xvfb virtual display
- No visual feedback during automation
- Limited debugging capabilities
- Resource sharing with development tools

## Proposed Architecture

Move Playwright and its MCP server to the `gui-tools-service` container alongside Blender, QGIS, and other GUI tools.

### Benefits

#### 1. Visual Debugging and Development
- **Real-time browser visualization** through VNC (port 5901)
- **Interactive debugging** - see what the automation is doing
- **Visual selector development** - point and click to generate selectors
- **Screenshot/video recording** with actual UI rendering
- **DevTools access** for network and console debugging

#### 2. Better Resource Management
- **Dedicated GUI container** with its own resources
- **GPU acceleration** already configured for graphics
- **Isolated browser processes** from development environment
- **Shared X11 display** with other GUI tools

#### 3. Enhanced Capabilities
- **Native dialog handling** - file pickers, print dialogs, etc.
- **Complex interactions** - drag & drop, hover effects, animations
- **Multi-window support** - popup handling, tab management
- **Browser extensions** - install and use extensions visually

#### 4. Simplified Architecture
- **Single display server** for all GUI applications
- **Unified VNC access** to all visual tools
- **Consistent GUI tool management** through supervisor
- **Shared authentication** and security model

## Implementation Plan

### Phase 1: Container Preparation

1. **Update GUI Container Dockerfile**
   ```dockerfile
   # Add Playwright dependencies
   RUN apt-get update && apt-get install -y \
       # Browser dependencies
       chromium \
       chromium-driver \
       firefox \
       firefox-geckodriver \
       # Additional fonts for better rendering
       fonts-liberation \
       fonts-noto-color-emoji \
       # Video recording support
       ffmpeg
   
   # Install Playwright
   RUN npm install -g playwright @executeautomation/playwright-mcp-server
   RUN npx playwright install-deps
   ```

2. **Configure Playwright Environment**
   ```yaml
   environment:
     - DISPLAY=:0
     - PLAYWRIGHT_MCP_HOST=0.0.0.0
     - PLAYWRIGHT_MCP_PORT=9879
     - PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
   ```

### Phase 2: MCP Server Migration

1. **Move Playwright MCP Configuration**
   ```javascript
   // gui-container supervisord.conf addition
   [program:playwright-mcp-server]
   command=npx @executeautomation/playwright-mcp-server
   directory=/workspace
   autorestart=true
   stdout_logfile=/var/log/playwright-mcp.log
   stderr_logfile=/var/log/playwright-mcp-error.log
   priority=40
   environment=DISPLAY=":0",NODE_ENV="production"
   ```

2. **Update Port Mappings**
   ```yaml
   ports:
     - "5901:5901"   # VNC port
     - "9876:9876"   # MCP Blender port
     - "9877:9877"   # MCP QGIS port
     - "9878:9878"   # MCP PBR Generator port
     - "9879:9879"   # MCP Playwright port (NEW)
   ```

### Phase 3: Integration Updates

1. **Update Main Container MCP Configuration**
   ```json
   {
     "mcpServers": {
       "playwright": {
         "command": "nc",
         "args": ["gui-tools-service", "9879"],
         "env": {
           "MCP_MODE": "proxy"
         }
       }
     }
   }
   ```

2. **Create Proxy Script**
   ```javascript
   // core-assets/scripts/playwright-proxy.js
   const net = require('net');
   
   // Proxy MCP commands to GUI container
   const proxy = net.createServer(socket => {
     const client = net.connect(9879, 'gui-tools-service');
     socket.pipe(client);
     client.pipe(socket);
   });
   
   proxy.listen(9879, '127.0.0.1');
   ```

### Phase 4: Enhanced Features

1. **Visual Test Runner UI**
   ```javascript
   // Add web UI for test management
   const express = require('express');
   const app = express();
   
   app.get('/tests', (req, res) => {
     // List available tests
   });
   
   app.post('/tests/:id/run', (req, res) => {
     // Run test with live preview
   });
   ```

2. **Recording Studio**
   - Browser action recording through VNC
   - Code generation from recorded actions
   - Visual assertion builder
   - Test case management UI

## Usage Examples

### Accessing Playwright Visually

1. **Connect to VNC**
   ```bash
   # Using VNC viewer
   vncviewer localhost:5901
   
   # Or using web-based noVNC
   http://localhost:6080
   ```

2. **Run Visual Tests**
   ```javascript
   // From main container
   await mcp.call('playwright', 'navigate', {
     url: 'https://example.com',
     options: { 
       headless: false,  // Run in visible mode
       slowMo: 500      // Slow down for visibility
     }
   });
   ```

3. **Interactive Debugging**
   ```javascript
   // Launch browser with DevTools
   await mcp.call('playwright', 'launch', {
     devtools: true,
     headless: false
   });
   
   // Pause for manual inspection
   await mcp.call('playwright', 'pause');
   ```

## Security Considerations

1. **VNC Security**
   - Password-protected VNC access
   - Optional SSL/TLS encryption
   - IP whitelist for connections

2. **Browser Isolation**
   - Separate browser profiles per session
   - Sandboxed browser processes
   - No access to host filesystem

3. **MCP Security**
   - Authenticated MCP connections
   - Rate limiting for browser operations
   - Resource usage monitoring

## Performance Optimization

1. **Browser Pooling**
   - Pre-warmed browser instances
   - Connection reuse for faster tests
   - Automatic cleanup of idle browsers

2. **Display Optimization**
   - Hardware-accelerated rendering
   - Optimized VNC compression
   - Selective screen updates

3. **Resource Management**
   - CPU/memory limits per browser
   - Automatic browser restart on high memory
   - Test timeout enforcement

## Monitoring and Debugging

1. **Metrics Collection**
   - Browser performance metrics
   - Test execution times
   - Resource usage statistics

2. **Logging**
   - Centralized browser console logs
   - Network request/response logging
   - Screenshot on failure

3. **Health Checks**
   - Browser availability monitoring
   - Display server health
   - MCP server responsiveness

## Migration Checklist

- [ ] Update GUI container Dockerfile with Playwright
- [ ] Configure Playwright MCP in GUI container
- [ ] Add port mapping for Playwright MCP (9879)
- [ ] Create proxy configuration in main container
- [ ] Update documentation with visual access instructions
- [ ] Test browser automation through VNC
- [ ] Implement recording studio UI
- [ ] Add performance monitoring
- [ ] Update security configurations
- [ ] Create migration guide for existing tests

## Future Enhancements

1. **Multi-browser Support**
   - Simultaneous Chrome, Firefox, Safari testing
   - Browser version management
   - Cross-browser visual comparison

2. **AI-Powered Testing**
   - Visual regression detection
   - Automatic selector healing
   - Test generation from user behavior

3. **Collaborative Features**
   - Shared VNC sessions for pair testing
   - Test result sharing and annotation
   - Real-time test execution viewing

## Conclusion

Moving Playwright to the GUI container provides significant advantages for browser automation development and debugging. The visual capabilities, combined with the existing GUI infrastructure, create a powerful platform for web testing and automation that goes beyond traditional headless approaches.