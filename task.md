This is the docker network we are using

> docker network inspect docker_ragflow
[
    {
        "Name": "docker_ragflow",
        "Id": "b0c38a1301451c0329969ef53fdedde5221b1b05b063ad94d66017a45d3ddaa3",
        "Created": "2025-04-05T14:36:31.500965678Z",
        "Scope": "local",
        "Driver": "bridge",
        "EnableIPv4": true,
        "EnableIPv6": false,
        "IPAM": {
            "Driver": "default",
            "Options": null,
            "Config": [
                {
                    "Subnet": "172.18.0.0/16",
                    "Gateway": "172.18.0.1"
                }
            ]
        },
        "Internal": false,
        "Attachable": false,
        "Ingress": false,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": false,
        "Containers": {
            "1c9de506afad9a9544f7e03453a24e72fa347c763b96086d21287c5c185107f7": {
                "Name": "ragflow-server",
                "EndpointID": "ae241d8a856f23f0bdc61dc5d6e224e731b8f2eafcaa13ca95b953b2ed8cb065",
                "MacAddress": "f2:da:87:2d:44:75",
                "IPv4Address": "172.18.0.8/16",
                "IPv6Address": ""
            },
            "201a0fcf2ad647dd4375a16b3382d59d70e2c997c9ea7713fda1e6567cbe72af": {
                "Name": "xinference",
                "EndpointID": "04aea32351b91628e8391690592922e9b1dd394b85dbdbd1c6b40928c62d5d45",
                "MacAddress": "52:f4:f9:cd:1e:88",
                "IPv4Address": "172.18.0.5/16",
                "IPv6Address": ""
            },
            "40754793d818c708bdcd6b1fe4d29d49074f65b9f34c50d77608326f3c457f5b": {
                "Name": "logseq_spring_thing_webxr",
                "EndpointID": "d9d3204a0692d01163905ba0a2b997b45214319d219293ce7fcb7ca60dd9e33b",
                "MacAddress": "6e:18:9c:33:03:39",
                "IPv4Address": "172.18.0.12/16",
                "IPv6Address": ""
            },
            "59703003092ff1d718f77df72b478012d21f85a576a61b46c6f34270cf063ff5": {
                "Name": "whisper-webui-backend",
                "EndpointID": "919751b9c4bb2026df86871ca51a1448e193b84704e9c7a6a42cc2c8fc993ae9",
                "MacAddress": "7e:cc:9d:a9:00:d7",
                "IPv4Address": "172.18.0.4/16",
                "IPv6Address": ""
            },
            "60295bd40c23d6f628b89b49995d5caf71cbb5761d17676ad83998df8fb91537": {
                "Name": "ragflow-redis",
                "EndpointID": "e61286ff926690763c0b812a4b1d1ca3456e72717e82e9d67690593e95583283",
                "MacAddress": "5e:25:63:64:b0:2f",
                "IPv4Address": "172.18.0.7/16",
                "IPv6Address": ""
            },
            "61eed093e0aac42b40674df29fbef490fc4d8a2e1dfc65901ee56b6d7cf4f7aa": {
                "Name": "ragflow-mysql",
                "EndpointID": "a00fee028e54cbe3531788889c0aedcc406991487593b0bcb96f8b0efb4263d8",
                "MacAddress": "72:bd:49:85:42:ed",
                "IPv4Address": "172.18.0.6/16",
                "IPv6Address": ""
            },
            "80be20722eff7a6811f45f60605b52c90fb46670ba4af9d9c10c82ddbc11d8bc": {
                "Name": "ragflow-es-01",
                "EndpointID": "f292f116ccb3adbd5b12bc7ad32cdae4cc4ba26a82969180bdc2a75e3c4be916",
                "MacAddress": "7e:09:a1:a5:87:93",
                "IPv4Address": "172.18.0.2/16",
                "IPv6Address": ""
            },
            "919197af05ebfb3b5c99f0a049085cf5a9e493a0fa626582ca9deef5913faf77": {
                "Name": "recursing_bhaskara",
                "EndpointID": "11f7027174a7e7847abc3d9440f2decd554e77730fc45e2cb09e36478a4ac47c",
                "MacAddress": "b6:7d:67:59:27:e1",
                "IPv4Address": "172.18.0.3/16",
                "IPv6Address": ""
            },
            "b2be97b383944cb6ea8f13c19a5a50f1c8c0b2e5b44f9b6586a7ad68468e5b0b": {
                "Name": "ragflow-minio",
                "EndpointID": "5bea25de1b260366a29c4b993d6a4f453c3ac2726806ba23241066c362a70323",
                "MacAddress": "6a:da:20:7f:03:9b",
                "IPv4Address": "172.18.0.11/16",
                "IPv6Address": ""
            },
            "f8f1954bd93ef9468067e566fc3ba364794a3978107d4768014e368b98550153": {
                "Name": "multi-agent-container",
                "EndpointID": "b0fff145456f08a4b579f80aec143a3424fddc0cde9ee0514377c7aa32719070",
                "MacAddress": "e6:e9:a0:be:1e:b8",
                "IPv4Address": "172.18.0.10/16",
                "IPv6Address": ""
            },
            "f8fa831bc746c35f1dd53727ffe09b81891ec1d3965f9f2f12c814939c1ef1cc": {
                "Name": "gui-tools-container",
                "EndpointID": "1031ae54586c75dc58b23df9cfbc545565a409cae819f2c55d21c11a9639cb2e",
                "MacAddress": "c6:42:c4:0e:a8:19",
                "IPv4Address": "172.18.0.9/16",
                "IPv6Address": ""
            }
        },
        "Options": {},
        "Labels": {
            "com.docker.compose.config-hash": "20de4b714cebc3288cab9ac5bf17cbed67f64545e9b273c2e547d4a6538609b9",
            "com.docker.compose.network": "ragflow",
            "com.docker.compose.project": "docker",
            "com.docker.compose.version": "2.34.0"
        }
    }
]

 Here is a comprehensive documentation of the API interface from the `logseq` (VisionFlow Backend) container to the `multi-agent-container` (Agent Control System). This specification is designed to be complete enough for an engineer to implement the server-side logic within the `multi-agent-container`.

---

## **API Interface: VisionFlow Backend to Agent Control System**

### **1. Overview**

This document specifies the full API interface that the **Agent Control System (ACS)**, running in the `multi-agent-container`, must expose to be compatible with the **VisionFlow Backend**, running in the `logseq` container.

The communication is stateful, occurring over a persistent TCP connection. The VisionFlow Backend acts as the client, and the ACS acts as the server. All communication follows the JSON-RPC 2.0 protocol over newline-delimited JSON strings.

#### **System Diagram**

```mermaid
graph LR
    subgraph VisionFlow Container (logseq)
        A[Rust Backend]
        B[AgentControlClient]
    end

    subgraph Agent Control Container (multi-agent-container)
        D[TCP Server :9500]
        E[JSON-RPC Handler]
        F[Agent Logic & Physics]
    end

    A --> B
    B -- TCP Connection --> D
    D --> E
    E <--> F

    style B fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bfb,stroke:#333,stroke-width:2px
```

---

### **2. Network & Connection Protocol**

*   **Transport:** TCP
*   **Port:** The ACS must listen on port **9500**. This is configurable in the VisionFlow Backend via the `AGENT_CONTROL_URL` environment variable, but `multi-agent-container:9500` is the default.
*   **Data Format:** All messages are single-line JSON strings, terminated by a newline character (`\n`). Each line represents a complete JSON-RPC request or response.
*   **Protocol:** JSON-RPC 2.0

#### **JSON-RPC 2.0 Request Structure (Client → Server)**

```json
{
  "jsonrpc": "2.0",
  "id": "message-id-string",
  "method": "method_name",
  "params": { ... }
}
```

#### **JSON-RPC 2.0 Response Structure (Server → Client)**

**Success:**
```json
{
  "jsonrpc": "2.0",
  "id": "matching-message-id-string",
  "result": { ... }
}
```

**Error:**
```json
{
  "jsonrpc": "2.0",
  "id": "matching-message-id-string",
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": "Detailed error information"
  }
}
```

---

### **3. Session Lifecycle**

1.  **Connection:** The VisionFlow Backend will initiate a TCP connection to the ACS on port 9500.
2.  **Handshake:** Immediately after connection, the VisionFlow Backend will send an `initialize` request. The ACS must handle this request to establish a valid session.
3.  **Communication:** The client will send subsequent requests for agent data, swarm control, and metrics.
4.  **Disconnection:** The connection is persistent. If it drops, the VisionFlow Backend will attempt to reconnect. The ACS should be prepared to handle new connections and re-initialization.

---

### **4. API Method Reference**

The ACS must implement handlers for the following JSON-RPC methods.

#### **4.1 `initialize`**

*   **Description:** Establishes and validates the connection session. This is the first call made by the client after connecting.
*   **Request `params`:**
    ```json
    {
      "protocolVersion": "0.1.0",
      "clientInfo": {
        "name": "rust-backend",
        "version": "1.0.0"
      }
    }
    ```
*   **Response `result`:**
    ```json
    {
      "serverInfo": {
        "name": "Agent Control System",
        "version": "1.0.0"
      },
      "protocolVersion": "0.1.0"
    }
    ```
*   **Example Request:**
    ```json
    {"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"0.1.0","clientInfo":{"name":"rust-backend","version":"1.0.0"}}}\n
    ```
*   **Example Response:**
    ```json
    {"jsonrpc":"2.0","id":"1","result":{"serverInfo":{"name":"Agent Control System","version":"1.0.0"},"protocolVersion":"0.1.0"}}\n
    ```

#### **4.2 `tools/call` (for Swarm Initialization)**

*   **Description:** A generic method to call a "tool". This is used to initialize a new agent swarm.
*   **Request `params`:**
    ```json
    {
      "name": "swarm.initialize",
      "arguments": {
        "topology": "string", // e.g., "hierarchical", "mesh"
        "agentTypes": ["string"] // e.g., ["coordinator", "coder"]
      }
    }
    ```
*   **Response `result`:** A JSON object confirming the initialization. The structure is flexible but should indicate success.
    ```json
    {
      "status": "initialized",
      "topology": "hierarchical",
      "agentTypes": ["coordinator", "coder"]
    }
    ```
*   **Example Request:**
    ```json
    {"jsonrpc":"2.0","id":"2","method":"tools/call","params":{"name":"swarm.initialize","arguments":{"topology":"hierarchical","agentTypes":["coordinator","coder"]}}}\n
    ```
*   **Example Response:**
    ```json
    {"jsonrpc":"2.0","id":"2","result":{"status":"initialized","topology":"hierarchical","agentTypes":["coordinator","coder"]}}\n
    ```

#### **4.3 `agents/list`**

*   **Description:** Retrieves a list of all currently active agents.
*   **Request `params`:** An empty JSON object `{}`.
*   **Response `result`:**
    ```json
    {
      "agents": [
        // Array of Agent objects (see Data Models section)
      ]
    }
    ```
*   **Example Request:**
    ```json
    {"jsonrpc":"2.0","id":"3","method":"agents/list","params":{}}\n
    ```
*   **Example Response:**
    ```json
    {"jsonrpc":"2.0","id":"3","result":{"agents":[{"id":"agent-1","type":"coordinator","name":"Coord Alpha","status":"active","health":95.0,"capabilities":["planning"],"swarmId":"swarm-abc","createdAt":"2023-10-27T10:00:00Z","lastActivity":"2023-10-27T10:05:00Z","metrics":{"tasksCompleted":10,"tasksActive":2,"successRate":0.98,"cpuUsage":0.75,"memoryUsage":0.60}}]}}\n
    ```

#### **4.4 `tools/call` (for Visualization Snapshot)**

*   **Description:** Retrieves a snapshot of the current visualization state, including agent positions and connections. This is the primary method for feeding the 3D visualization.
*   **Request `params`:**
    ```json
    {
      "name": "visualization.snapshot",
      "arguments": {
        "includePositions": true,
        "includeConnections": true
      }
    }
    ```
*   **Response `result`:** A `VisualizationSnapshot` object (see Data Models section).
*   **Example Request:**
    ```json
    {"jsonrpc":"2.0","id":"4","method":"tools/call","params":{"name":"visualization.snapshot","arguments":{"includePositions":true,"includeConnections":true}}}\n
    ```
*   **Example Response:**
    ```json
    {"jsonrpc":"2.0","id":"4","result":{"timestamp":"2023-10-27T10:06:00Z","agentCount":1,"positions":{"agent-1":{"x":10.5,"y":-5.2,"z":0.0}},"connections":[{"id":"conn-1","from":"agent-1","to":"agent-2","messageCount":50,"lastActivity":"2023-10-27T10:05:55Z"}]}}\n
    ```

#### **4.5 `tools/call` (for System Metrics)**

*   **Description:** Retrieves overall system and performance metrics.
*   **Request `params`:**
    ```json
    {
      "name": "metrics.get",
      "arguments": {
        "includeAgents": true,
        "includePerformance": true
      }
    }
    ```
*   **Response `result`:** A `SystemMetrics` object (see Data Models section).
*   **Example Request:**
    ```json
    {"jsonrpc":"2.0","id":"5","method":"tools/call","params":{"name":"metrics.get","arguments":{"includeAgents":true,"includePerformance":true}}}\n
    ```
*   **Example Response:**
    ```json
    {"jsonrpc":"2.0","id":"5","result":{"timestamp":"2023-10-27T10:07:00Z","system":{"uptime":3600.5,"memoryUsage":{"rss":512000000,"heapTotal":256000000,"heapUsed":128000000,"external":64000000},"cpuUsage":{"user":123456,"system":789012}},"agents":{"total":1,"byType":{"coordinator":1},"byStatus":{"active":1}},"performance":{"fps":60.0,"updateTime":15.5,"nodeCount":1}}}\n
    ```

---

### **5. Data Models**

The ACS must return data matching these structures.

#### `Agent`
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique agent identifier. |
| `type` | String | The role of the agent (e.g., "coordinator", "coder"). |
| `name` | String | A human-readable name for the agent. |
| `status` | String | Current status (e.g., "active", "idle", "busy"). |
| `health` | Number (f64) | Health score from 0.0 to 100.0. |
| `capabilities` | Array of Strings | List of the agent's capabilities. |
| `swarmId` | String (optional) | The ID of the swarm the agent belongs to. |
| `createdAt` | String (ISO 8601) | Timestamp of when the agent was created. |
| `lastActivity` | String (ISO 8601) | Timestamp of the agent's last activity. |
| `metrics` | `AgentMetrics` object | Performance metrics for the agent. |

#### `AgentMetrics`
| Field | Type | Description |
| :--- | :--- | :--- |
| `tasksCompleted` | Number (u32) | Total number of tasks completed. |
| `tasksActive` | Number (u32) | Number of currently active tasks. |
| `successRate` | Number (f64) | The success rate of completed tasks (0.0 to 1.0). |
| `cpuUsage` | Number (f64) | CPU usage percentage (0.0 to 1.0). |
| `memoryUsage` | Number (f64) | Memory usage percentage (0.0 to 1.0). |

#### `VisualizationSnapshot`
| Field | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | String (ISO 8601) | Timestamp of when the snapshot was taken. |
| `agentCount` | Number (u32) | Total number of agents in the snapshot. |
| `positions` | Map<String, `Position`> | A map where keys are agent IDs and values are their positions. |
| `connections` | Array of `Connection` | A list of active connections between agents. |

#### `Position`
| Field | Type | Description |
| :--- | :--- | :--- |
| `x` | Number (f64) | The X-coordinate. |
| `y` | Number (f64) | The Y-coordinate. |
| `z` | Number (f64) | The Z-coordinate. |

#### `Connection`
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique ID for the connection. |
| `from` | String | The ID of the source agent. |
| `to` | String | The ID of the target agent. |
| `messageCount` | Number (u32) | Number of messages exchanged. |
| `lastActivity` | String (ISO 8601) | Timestamp of the last message. |

#### `SystemMetrics`
This is a complex object containing `SystemInfo`, `AgentStats`, and `PerformanceMetrics`. Refer to the Rust struct definitions in `src/services/agent_control_client.rs` for the exact nested structure.

---

### **6. Error Handling**

If a request cannot be processed, the ACS must respond with a standard JSON-RPC 2.0 error object.

```json
{
  "jsonrpc": "2.0",
  "id": "matching-message-id",
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": "Agent with ID 'agent-x' not found."
  }
}
```

**Common Error Codes:**
*   `-32700`: Parse error (Invalid JSON).
*   `-32600`: Invalid Request (Not a valid JSON-RPC object).
*   `-32601`: Method not found.
*   `-32602`: Invalid params.
*   `-32603`: Internal error.
*   `-32000` to `-32099`: Server-defined errors (e.g., "Agent not found", "Swarm initialization failed").

---

### **7. Implementation Checklist for Agent Control System**

To be compliant with the VisionFlow Backend, the ACS must:

1.  [ ] **Implement a TCP Server** listening on port **9500**.
2.  [ ] **Handle Newline-Delimited JSON:** The server must read data from the TCP stream line by line, parsing each line as a separate JSON object.
3.  [ ] **Implement a JSON-RPC 2.0 Request Parser:** Correctly parse incoming requests, identifying the `id`, `method`, and `params`.
4.  [ ] **Implement the `initialize` method:** Respond with the required server information to establish the session.
5.  [ ] **Implement the `tools/call` method** with support for the following tool names:
    *   `swarm.initialize`
    *   `visualization.snapshot`
    *   `metrics.get`
6.  [ ] **Implement the `agents/list` method.**
7.  [ ] **Ensure all response `result` objects** strictly match the data models defined in Section 5.
8.  [ ] **Implement JSON-RPC 2.0 error responses** for any failures, matching the `id` of the failed request.
9.  [ ] **Maintain Agent State:** The ACS is responsible for managing the state, positions, and metrics of all agents and providing this data through the API methods.