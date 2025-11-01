# VisionFlow Architecture Diagrams

## 1. C4 Model - Context Diagram (Level 1)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                          VISIONFLOW ECOSYSTEM                                │
│                    WebXR Knowledge Graph Visualization                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

        [Web Browsers]                    [Mobile Devices]              [VR Headsets]
             │                                   │                             │
             └───────────────────┬───────────────┴─────────────────────────────┘
                                 │
                                 │ HTTPS/WebSocket
                                 │ Port 3001
                                 │
                    ┌────────────▼─────────────┐
                    │                          │
                    │   VisionFlow System      │
                    │                          │
                    │  • 3D Graph Rendering    │
                    │  • Real-time Physics     │
                    │  • GPU Acceleration      │
                    │  • WebXR Support         │
                    │                          │
                    └────────┬─────────┬───────┘
                             │         │
                  ┌──────────┘         └──────────┐
                  │                                 │
                  │                                 │
     ┌────────────▼─────────────┐      ┌──────────▼───────────┐
     │                          │      │                       │
     │  Claude Flow / MCP       │      │  File Storage         │
     │  (AI Agent System)       │      │  (Copyparty)          │
     │                          │      │                       │
     │  • Agent Orchestration   │      │  • Document Storage   │
     │  • Tool Invocation       │      │  • File Browsing      │
     │  • Telemetry             │      │  • Upload/Download    │
     │                          │      │                       │
     └──────────────────────────┘      └───────────────────────┘
```

---

## 2. C4 Model - Container Diagram (Level 2)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VISIONFLOW CONTAINER ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

                              [Browser Client]
                                    │
                                    │ HTTPS/WSS
                                    │ Port 3001
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VisionFlow Container                                 │
│                           (172.18.0.11)                                      │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         Nginx Reverse Proxy                           │  │
│   │                            Port: 3001                                 │  │
│   │                                                                       │  │
│   │  Technology: Nginx 1.x                                               │  │
│   │  Purpose: Request routing, SSL, WebSocket upgrades                   │  │
│   │  Routes:                                                              │  │
│   │    • /api/* → Rust Backend                                           │  │
│   │    • /wss, /ws/* → WebSocket services                                │  │
│   │    • /* → Vite Frontend                                              │  │
│   │    • /browser/* → Copyparty (external)                               │  │
│   └───────────┬──────────────────────────────────┬────────────────────────┘  │
│               │                                   │                           │
│               │ HTTP                              │ HTTP                      │
│               ▼                                   ▼                           │
│   ┌─────────────────────────┐        ┌────────────────────────────┐         │
│   │   Rust Backend          │        │   Vite Dev Server          │         │
│   │   (webxr)               │        │   (React + Babylon.js)     │         │
│   │   Port: 4000            │        │   Port: 5173               │         │
│   │                         │        │                            │         │
│   │  Technology:            │        │  Technology:               │         │
│   │    • Actix-web 4.11     │        │    • Vite 6.3.6            │         │
│   │    • Rust 2021          │        │    • React 18              │         │
│   │    • CUDA/PTX           │        │    • TypeScript 5          │         │
│   │    • WebSocket actors   │        │    • Babylon.js 8.28       │         │
│   │                         │        │                            │         │
│   │  Features:              │        │  Features:                 │         │
│   │    • REST API           │        │    • HMR (Hot Reload)      │         │
│   │    • Graph WebSocket    │        │    • 3D Rendering          │         │
│   │    • GPU Physics        │        │    • WebXR Support         │         │
│   │    • MCP Relay          │        │    • UI Components         │         │
│   │    • Speech services    │        │    • State Management      │         │
│   │                         │        │                            │         │
│   │  Data Stores:           │        │  Build Output:             │         │
│   │    • settings.yaml      │        │    • /app/client/dist/     │         │
│   │    • /app/data/         │        │                            │         │
│   └───────────┬─────────────┘        └────────────────────────────┘         │
│               │                                                               │
│               │ CUDA API                                                      │
│               ▼                                                               │
│   ┌─────────────────────────┐                                                │
│   │   NVIDIA GPU            │                                                │
│   │   CUDA Runtime          │                                                │
│   │                         │                                                │
│   │  • Graph Physics        │                                                │
│   │  • Parallel Compute     │                                                │
│   │  • PTX Compilation      │                                                │
│   └─────────────────────────┘                                                │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                        Supervisord                                    │  │
│   │                      Process Manager                                  │  │
│   │                                                                       │  │
│   │  Manages: nginx, rust-backend, vite-dev                              │  │
│   │  Logs: /app/logs/*.log                                               │  │
│   │  Socket: /tmp/supervisor.sock                                        │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                │ TCP :9500
                                │ MCP Protocol
                                ▼
                    ┌───────────────────────────┐
                    │  agentic-workstation      │
                    │  (172.18.0.7)             │
                    │                           │
                    │  • Claude Flow Server     │
                    │  • MCP Protocol Handler   │
                    │  • AI Agent Coordination  │
                    └───────────────────────────┘
```

---

## 3. C4 Model - Component Diagram (Level 3) - Rust Backend

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RUST BACKEND COMPONENTS                               │
│                         (webxr binary)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

                                [Nginx Proxy]
                                     │
                                     │ HTTP/WebSocket
                                     ▼
                    ┌────────────────────────────────┐
                    │    Actix-web HTTP Server       │
                    │         Port 4000              │
                    └─────────────┬──────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
    ┌──────────────────┐ ┌─────────────────┐ ┌──────────────────┐
    │  REST API        │ │  WebSocket      │ │  Health Check    │
    │  Handlers        │ │  Handlers       │ │  Endpoint        │
    │                  │ │                 │ │                  │
    │  • /api/health   │ │  • /wss         │ │  • /health       │
    │  • /api/graph    │ │  • /ws/speech   │ │                  │
    │  • /api/nodes    │ │  • /ws/mcp      │ │                  │
    │  • /api/edges    │ │                 │ │                  │
    └────────┬─────────┘ └────────┬────────┘ └──────────────────┘
             │                    │
             └─────────┬──────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │    Actor System            │
          │    (Actix Actors)          │
          │                            │
          │  ┌──────────────────────┐  │
          │  │  Graph Actor         │  │
          │  │  • Node management   │  │
          │  │  • Edge management   │  │
          │  │  • State updates     │  │
          │  └──────────┬───────────┘  │
          │             │               │
          │  ┌──────────▼───────────┐  │
          │  │  GPU Actor           │  │
          │  │  • Physics sim       │  │
          │  │  • CUDA kernels      │  │
          │  │  • Force calculations│  │
          │  └──────────┬───────────┘  │
          │             │               │
          │  ┌──────────▼───────────┐  │
          │  │  Settings Actor      │  │
          │  │  • Config management │  │
          │  │  • User preferences  │  │
          │  └──────────┬───────────┘  │
          │             │               │
          │  ┌──────────▼───────────┐  │
          │  │  MCP Relay Actor     │  │
          │  │  • TCP connection    │  │
          │  │  • Message routing   │  │
          │  │  • Tool invocation   │  │
          │  └──────────────────────┘  │
          └────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │  GPU    │  │ Config  │  │  MCP    │
    │  Module │  │  YAML   │  │  TCP    │
    │         │  │  Parser │  │  Client │
    │ • CUDA  │  │         │  │         │
    │ • PTX   │  │         │  │         │
    └─────────┘  └─────────┘  └─────────┘
```

---

## 4. C4 Model - Component Diagram (Level 3) - React Frontend

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REACT FRONTEND COMPONENTS                              │
│                      (Vite + React + Babylon.js)                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              [Browser]
                                 │
                                 │ HTTP/WebSocket
                                 ▼
                    ┌────────────────────────────┐
                    │    Vite Dev Server         │
                    │       Port 5173            │
                    │                            │
                    │  • HMR WebSocket           │
                    │  • Static serving          │
                    │  • Module bundling         │
                    └────────────┬───────────────┘
                                 │
                                 │ Serves
                                 ▼
                    ┌────────────────────────────┐
                    │      React Application     │
                    │                            │
                    │  ┌──────────────────────┐  │
                    │  │   App Router         │  │
                    │  │   (React Router)     │  │
                    │  └──────────┬───────────┘  │
                    │             │               │
                    │  ┌──────────▼───────────┐  │
                    │  │  Feature Modules     │  │
                    │  │                      │  │
                    │  │  ┌────────────────┐ │  │
                    │  │  │ Visualization  │ │  │
                    │  │  │   Module       │ │  │
                    │  │  │                │ │  │
                    │  │  │ • 3D Scene     │ │  │
                    │  │  │ • Camera ctrl  │ │  │
                    │  │  │ • Node render  │ │  │
                    │  │  │ • Edge render  │ │  │
                    │  │  └────────┬───────┘ │  │
                    │  │           │         │  │
                    │  │  ┌────────▼───────┐ │  │
                    │  │  │ Control Panel  │ │  │
                    │  │  │                │ │  │
                    │  │  │ • Settings UI  │ │  │
                    │  │  │ • Sliders      │ │  │
                    │  │  │ • Toggles      │ │  │
                    │  │  └────────┬───────┘ │  │
                    │  │           │         │  │
                    │  │  ┌────────▼───────┐ │  │
                    │  │  │ Graph Manager  │ │  │
                    │  │  │                │ │  │
                    │  │  │ • Data fetch   │ │  │
                    │  │  │ • State mgmt   │ │  │
                    │  │  │ • WebSocket    │ │  │
                    │  │  └────────────────┘ │  │
                    │  └──────────────────────┘  │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │  Babylon.js  │  │  React State │  │  API Client  │
        │   Engine     │  │              │  │              │
        │              │  │  • Context   │  │  • REST      │
        │  • Scene     │  │  • Hooks     │  │  • WebSocket │
        │  • Camera    │  │  • Zustand?  │  │  • Axios?    │
        │  • Lights    │  │              │  │              │
        │  • Materials │  │              │  │              │
        │  • Meshes    │  │              │  │              │
        └──────────────┘  └──────────────┘  └──────────────┘
                                                    │
                                                    │ HTTP/WS
                                                    ▼
                                          [Rust Backend API]
```

---

## 5. Data Flow Diagram - Request Lifecycle

```
USER ACTION: Click "Load Graph"
│
│ 1. User clicks button in React UI
▼
┌─────────────────────────────────────────────────────────────┐
│  React Component                                             │
│  GraphViewer.tsx                                            │
│                                                              │
│  handleLoadGraph() → fetch('/api/graph/load')              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 2. HTTP GET /api/graph/load
                      │    Host: localhost:3001
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Nginx Reverse Proxy (Port 3001)                            │
│                                                              │
│  location /api/ {                                           │
│      proxy_pass http://127.0.0.1:4000;                     │
│  }                                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 3. Proxied to backend
                      │    http://127.0.0.1:4000/api/graph/load
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Actix-web HTTP Server (Port 4000)                          │
│                                                              │
│  Route: GET /api/graph/load                                 │
│  Handler: load_graph_handler()                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 4. Fetch graph data
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Graph Actor                                                 │
│                                                              │
│  • Load nodes from storage                                  │
│  • Load edges from storage                                  │
│  • Calculate positions                                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 5. Trigger GPU physics
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  GPU Actor                                                   │
│                                                              │
│  • Initialize CUDA context                                  │
│  • Load graph into GPU memory                               │
│  • Run force-directed layout                                │
│  • Return calculated positions                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 6. Format response
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Actix-web Response                                          │
│                                                              │
│  HTTP 200 OK                                                │
│  Content-Type: application/json                             │
│  {                                                           │
│    "nodes": [...],                                          │
│    "edges": [...],                                          │
│    "metadata": {...}                                        │
│  }                                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 7. Response through proxy
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Nginx Proxy                                                 │
│                                                              │
│  • Add CORS headers                                         │
│  • Add security headers                                     │
│  • Forward to client                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 8. JSON response
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  React Component                                             │
│  GraphViewer.tsx                                            │
│                                                              │
│  .then(data => {                                            │
│    setGraphData(data);                                      │
│    initializeBabylonScene(data);                           │
│  })                                                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 9. Render 3D scene
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Babylon.js Engine                                           │
│                                                              │
│  • Create node meshes                                       │
│  • Create edge lines                                        │
│  • Apply materials                                          │
│  • Start render loop                                        │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ 10. User sees 3D graph
                      ▼
                   [Browser Display]
```

---

## 6. WebSocket Data Flow - Real-time Graph Updates

```
GRAPH UPDATE: Node position changed
│
│ 1. GPU Actor calculates new positions
▼
┌─────────────────────────────────────────────────────────────┐
│  GPU Actor                                                   │
│                                                              │
│  • Physics tick (60 FPS)                                    │
│  • Calculate forces                                         │
│  • Update node positions                                    │
│  • Detect changes                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 2. Notify Graph Actor
                      │    Message: PositionUpdate
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Graph Actor                                                 │
│                                                              │
│  • Receive position updates                                 │
│  • Update internal state                                    │
│  • Format WebSocket message                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 3. Broadcast to WebSocket clients
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  WebSocket Actor (/wss)                                      │
│                                                              │
│  • Serialize message to JSON                                │
│  • Send to all connected clients                            │
│  {                                                           │
│    "type": "position_update",                               │
│    "nodes": [                                               │
│      {"id": "node1", "x": 1.5, "y": 2.3, "z": -0.8},      │
│      ...                                                     │
│    ]                                                         │
│  }                                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 4. WebSocket frame through Nginx
                      │    ws://localhost:3001/wss
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Nginx WebSocket Proxy                                       │
│                                                              │
│  location /wss {                                            │
│      proxy_pass http://127.0.0.1:4000;                     │
│      proxy_http_version 1.1;                                │
│      proxy_set_header Upgrade $http_upgrade;               │
│      proxy_set_header Connection $connection_upgrade;      │
│  }                                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 5. WebSocket message to browser
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  React WebSocket Client                                      │
│  useGraphWebSocket.ts                                       │
│                                                              │
│  ws.onmessage = (event) => {                                │
│    const update = JSON.parse(event.data);                  │
│    handlePositionUpdate(update);                           │
│  }                                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 6. Update 3D scene
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Babylon.js Scene                                            │
│                                                              │
│  • Update mesh positions                                    │
│  • Smooth interpolation                                     │
│  • Re-render frame                                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 7. Smooth animation
                      ▼
                [Browser Display - Animated Graph]
```

---

## 7. MCP Integration Flow

```
AGENT ACTION: Request graph analysis
│
│ 1. Claude agent invokes tool
▼
┌─────────────────────────────────────────────────────────────┐
│  Claude Flow Server                                          │
│  (agentic-workstation:9500)                                 │
│                                                              │
│  Tool: analyze_graph                                        │
│  Parameters: {graphId: "123"}                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 2. MCP Protocol Message
                      │    TCP Socket :9500
                      │    {
                      │      "method": "tools/call",
                      │      "params": {...}
                      │    }
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  VisionFlow MCP Relay Actor                                  │
│  (Port 4000, WebSocket /ws/mcp-relay)                       │
│                                                              │
│  • Receive MCP message                                      │
│  • Parse tool invocation                                    │
│  • Route to appropriate handler                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 3. Execute graph analysis
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Graph Actor                                                 │
│                                                              │
│  • Analyze graph structure                                  │
│  • Calculate metrics                                        │
│    - Node count: 1234                                       │
│    - Edge count: 5678                                       │
│    - Clustering coefficient: 0.45                           │
│    - Average degree: 4.6                                    │
│  • Generate insights                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 4. Format MCP response
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  MCP Relay Actor                                             │
│                                                              │
│  {                                                           │
│    "jsonrpc": "2.0",                                        │
│    "id": "123",                                             │
│    "result": {                                              │
│      "content": [                                           │
│        {                                                    │
│          "type": "text",                                    │
│          "text": "Graph analysis complete:\n..."           │
│        }                                                    │
│      ]                                                      │
│    }                                                         │
│  }                                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 5. Send response via TCP
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Claude Flow Server                                          │
│                                                              │
│  • Receive tool result                                      │
│  • Parse metrics                                            │
│  • Generate natural language response                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 6. Present to user
                      ▼
                 [Claude Response]
          "The graph contains 1,234 nodes
           with 5,678 connections. The
           clustering coefficient of 0.45
           suggests moderate community
           structure..."
```

---

## 8. Deployment Architecture - Development vs Production

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEVELOPMENT DEPLOYMENT                              │
│                        (Current Configuration)                               │
└─────────────────────────────────────────────────────────────────────────────┘

                              [Docker Host]
                                   │
                                   │ Port 3001
                                   ▼
                    ┌──────────────────────────────┐
                    │  Single Container            │
                    │  visionflow_container        │
                    │                              │
                    │  ┌─────────────────────┐    │
                    │  │   Supervisord       │    │
                    │  │   (Process Mgr)     │    │
                    │  └──────────┬──────────┘    │
                    │             │                │
                    │  ┌──────────┼──────────┐    │
                    │  │          │          │    │
                    │  ▼          ▼          ▼    │
                    │ Nginx    Rust       Vite    │
                    │ :3001    :4000      :5173   │
                    │                              │
                    │  Volumes:                    │
                    │  • Source code (bind)        │
                    │  • Build cache (volume)      │
                    │  • Logs (bind)               │
                    │                              │
                    │  Pros:                       │
                    │  ✓ Simple to manage          │
                    │  ✓ Fast iteration            │
                    │  ✓ Unified logging           │
                    │                              │
                    │  Cons:                       │
                    │  ✗ Not scalable              │
                    │  ✗ No isolation              │
                    │  ✗ Debug builds              │
                    └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRODUCTION DEPLOYMENT                               │
│                          (Recommended)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              [Load Balancer]
                             Cloudflare / Nginx
                                   │
                                   │ HTTPS :443
                                   ▼
                    ┌──────────────────────────────┐
                    │   Nginx Container            │
                    │   (Reverse Proxy)            │
                    │   ┌──────────────────┐       │
                    │   │   nginx:alpine   │       │
                    │   │   • SSL term     │       │
                    │   │   • Caching      │       │
                    │   │   • Rate limit   │       │
                    │   │   • Gzip         │       │
                    │   └────────┬─────────┘       │
                    └────────────┼─────────────────┘
                                 │
                    ┌────────────┼─────────────┐
                    │            │             │
                    ▼            ▼             ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │  Backend     │  │  Backend     │  │  Frontend    │
        │  Container 1 │  │  Container 2 │  │  Container   │
        │              │  │              │  │              │
        │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │
        │  │ Rust   │  │  │  │ Rust   │  │  │  │ Nginx  │  │
        │  │ webxr  │  │  │  │ webxr  │  │  │  │ static │  │
        │  │ :4000  │  │  │  │ :4000  │  │  │  │ serve  │  │
        │  │        │  │  │  │        │  │  │  │        │  │
        │  │ GPU 1  │  │  │  │ GPU 2  │  │  │  │ Built  │  │
        │  └────────┘  │  │  └────────┘  │  │  │ assets │  │
        │              │  │              │  │  └────────┘  │
        │  Release     │  │  Release     │  │              │
        │  build       │  │  build       │  │  Production  │
        └──────────────┘  └──────────────┘  └──────────────┘
                 │                 │                │
                 └────────┬────────┴────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Redis Cache         │
              │   (Session store)     │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   PostgreSQL          │
              │   (Graph data)        │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Prometheus          │
              │   (Metrics)           │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Grafana             │
              │   (Dashboards)        │
              └───────────────────────┘

              Pros:
              ✓ Horizontal scaling
              ✓ Service isolation
              ✓ Zero-downtime deploys
              ✓ Better security
              ✓ Production optimizations

              Requires:
              • Docker Swarm or Kubernetes
              • Service mesh (optional)
              • CI/CD pipeline
              • Monitoring stack
```

---

## 9. Sequence Diagram - Container Startup

```
┌─────────┐   ┌──────────────┐   ┌───────┐   ┌──────┐   ┌──────┐
│ Docker  │   │ Supervisord  │   │ Nginx │   │ Rust │   │ Vite │
└────┬────┘   └──────┬───────┘   └───┬───┘   └───┬──┘   └───┬──┘
     │               │                │           │          │
     │ Start         │                │           │          │
     │ container     │                │           │          │
     ├──────────────>│                │           │          │
     │               │                │           │          │
     │               │ Start nginx    │           │          │
     │               ├───────────────>│           │          │
     │               │                │           │          │
     │               │                │ Test      │          │
     │               │                │ config    │          │
     │               │                │ nginx -t  │          │
     │               │                │◄──────────┤          │
     │               │                │           │          │
     │               │                │ Listen    │          │
     │               │                │ :3001     │          │
     │               │                │◄──────────┤          │
     │               │                │           │          │
     │               │   Ready        │           │          │
     │               │◄───────────────┤           │          │
     │               │                │           │          │
     │               │ Start rust-backend-wrapper │          │
     │               ├───────────────────────────>│          │
     │               │                │           │          │
     │               │                │        cargo build   │
     │               │                │        --features gpu│
     │               │                │           │◄─────────┤
     │               │                │           │          │
     │               │                │           │ Build    │
     │               │                │           │ (30-60s) │
     │               │                │           │          │
     │               │                │     Start binary     │
     │               │                │     /app/target/     │
     │               │                │     debug/webxr      │
     │               │                │           │◄─────────┤
     │               │                │           │          │
     │               │                │        Initialize    │
     │               │                │        • CUDA        │
     │               │                │        • MCP         │
     │               │                │        • Actors      │
     │               │                │           │          │
     │               │                │        Listen :4000  │
     │               │                │           │◄─────────┤
     │               │                │           │          │
     │               │   Ready        │           │          │
     │               │◄───────────────────────────┤          │
     │               │                │           │          │
     │               │ Start vite-dev │           │          │
     │               ├───────────────────────────────────────>│
     │               │                │           │          │
     │               │                │           │    npm   │
     │               │                │           │    run   │
     │               │                │           │    dev   │
     │               │                │           │          │
     │               │                │           │    Vite  │
     │               │                │           │    startup
     │               │                │           │          │
     │               │                │           │    Listen │
     │               │                │           │    :5173  │
     │               │                │           │          │
     │               │   Ready        │           │          │
     │               │◄───────────────────────────────────────┤
     │               │                │           │          │
     │  Container    │                │           │          │
     │  healthy      │                │           │          │
     │◄──────────────┤                │           │          │
     │               │                │           │          │
     │               │           All services running         │
     │               │           Port 3001 accessible         │
     │               │                │           │          │
```

---

## 10. Network Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCKER NETWORK TOPOLOGY                              │
│                       Network: docker_ragflow                                │
│                       Subnet: 172.18.0.0/16                                  │
│                       Gateway: 172.18.0.1                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                          [Docker Bridge: docker_ragflow]
                                    172.18.0.1
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        │                               │                               │
┌───────▼────────┐            ┌─────────▼────────┐          ┌──────────▼──────┐
│ visionflow     │            │ agentic          │          │ ragflow-es-01   │
│ _container     │            │ -workstation     │          │ (Copyparty)     │
│ 172.18.0.11    │            │ 172.18.0.7       │          │ 172.18.0.4      │
│                │            │                  │          │                 │
│ Services:      │            │ Services:        │          │ Services:       │
│ • Nginx :3001  │◄──┐        │ • Claude Flow    │          │ • Copyparty     │
│ • Rust :4000   │   │        │ • MCP Server     │          │   :3923         │
│ • Vite :5173   │   │        │   :9500          │          │                 │
│                │   │        │ • Management API │          └─────────────────┘
│ Exposed:       │   │        │   :9090          │
│ • 3001 → Host  │   │        │                  │
└────────┬───────┘   │        └──────────────────┘
         │           │
         │ MCP TCP   │ Copyparty HTTP
         │ :9500     │ :3923/browser
         │           │
         └───────────┘

┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ vircadia_world │   │ vircadia_world │   │ vircadia_world │   │ whisper-webui  │
│ _postgres      │   │ _pgweb         │   │ _api_manager   │   │                │
│ 172.18.0.2     │   │ 172.18.0.3     │   │ 172.18.0.5     │   │ 172.18.0.6     │
│                │   │                │   │                │   │                │
│ PostgreSQL     │   │ PgWeb UI       │   │ Vircadia API   │   │ Whisper ASR    │
│ :5432          │   │                │   │                │   │                │
└────────────────┘   └────────────────┘   └────────────────┘   └────────────────┘

┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ ragflow-minio  │   │ ragflow-mysql  │   │ kokoro-tts     │   │ ragflow-redis  │
│ 172.18.0.8     │   │ 172.18.0.9     │   │ _container     │   │ 172.18.0.12    │
│                │   │                │   │ 172.18.0.10    │   │                │
│ S3 Storage     │   │ MySQL          │   │                │   │ Redis Cache    │
│ :9000          │   │ :3306          │   │ TTS Service    │   │ :6379          │
└────────────────┘   └────────────────┘   └────────────────┘   └────────────────┘

                             ┌────────────────┐
                             │ ragflow-server │
                             │ 172.18.0.13    │
                             │                │
                             │ RAGFlow        │
                             │                │
                             └────────────────┘

Legend:
  ──────> HTTP/HTTPS connection
  ◄─────► Bidirectional WebSocket
  ────┐   Service dependency
```

---

## 11. Technology Stack Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TECHNOLOGY STACK                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────┐
│   PRESENTATION LAYER   │
├────────────────────────┤
│ • React 18             │
│ • TypeScript 5         │
│ • Babylon.js 8.28      │
│ • Radix UI 2.x         │
│ • React Three Fiber    │
└───────────┬────────────┘
            │
┌───────────▼────────────┐
│   APPLICATION LAYER    │
├────────────────────────┤
│ • Vite 6.3.6           │
│ • React Router         │
│ • State management     │
│ • API client           │
└───────────┬────────────┘
            │
┌───────────▼────────────┐
│    NETWORKING LAYER    │
├────────────────────────┤
│ • Nginx 1.x            │
│ • HTTP/2               │
│ • WebSocket            │
│ • CORS                 │
└───────────┬────────────┘
            │
┌───────────▼────────────┐
│     BACKEND LAYER      │
├────────────────────────┤
│ • Rust 2021            │
│ • Actix-web 4.11       │
│ • Actix actors         │
│ • Tungstenite WS       │
└───────────┬────────────┘
            │
┌───────────▼────────────┐
│   COMPUTE LAYER        │
├────────────────────────┤
│ • NVIDIA CUDA          │
│ • PTX compiler         │
│ • GPU kernels          │
└───────────┬────────────┘
            │
┌───────────▼────────────┐
│    STORAGE LAYER       │
├────────────────────────┤
│ • YAML config          │
│ • Markdown data        │
│ • File system          │
└────────────────────────┘

┌────────────────────────┐
│   INTEGRATION LAYER    │
├────────────────────────┤
│ • MCP Protocol         │
│ • Claude Flow          │
│ • Docker API           │
└────────────────────────┘

┌────────────────────────┐
│    RUNTIME LAYER       │
├────────────────────────┤
│ • Docker Engine        │
│ • NVIDIA runtime       │
│ • Linux kernel         │
└────────────────────────┘
```

---

## Document Metadata

**Created:** 2025-10-23
**Version:** 1.0
**Format:** Markdown + ASCII diagrams
**Diagrams:** 11 total
**Levels:** C4 Levels 1-3 + sequence/data flow

**Diagram Types:**
1. C4 Context (Level 1)
2. C4 Container (Level 2)
3. C4 Component - Backend (Level 3)
4. C4 Component - Frontend (Level 3)
5. Data Flow - Request Lifecycle
6. Data Flow - WebSocket Updates
7. Data Flow - MCP Integration
8. Deployment Comparison
9. Sequence - Startup
10. Network Topology
11. Technology Stack

**Tools Used:**
- ASCII diagrams (terminal-friendly)
- Box-drawing characters
- Sequence diagram notation
- Network topology visualization
