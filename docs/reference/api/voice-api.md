# Voice System API Reference

## Overview

The VisionFlow Voice System provides a complete voice-to-agent integration pipeline, enabling users to control agent swarms through natural speech commands. The system integrates Whisper STT, Kokoro TTS, and real MCP task orchestration.

**Base Endpoints**:
- WebSocket: `ws://localhost:3001/ws/voice`
- REST: `http://localhost:3001/api/voice`

## Architecture

```
Voice Input → Whisper STT → Intent Parser → MCP Orchestration → Agent Execution → Results → Kokoro TTS → Voice Output
```

### Components

1. **Speech-to-Text**: Whisper at `172.18.0.5:8080`
2. **Text-to-Speech**: Kokoro at `172.18.0.9:5000`
3. **Intent Recognition**: Voice command parsing
4. **MCP Integration**: Real agent spawning and task orchestration
5. **Context Management**: Session-based conversation memory

## WebSocket Voice API

### Connection Setup

```javascript
const voiceWs = new WebSocket('ws://localhost:3001/ws/voice');

voiceWs.onopen = () => {
    console.log('Voice WebSocket connected');
};

voiceWs.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleVoiceResponse(message);
};
```

### Voice Command Execution

#### Send Voice Command

```javascript
// Direct text command
voiceWs.send(JSON.stringify({
    type: 'voice_command',
    command: 'spawn a researcher agent',
    sessionId: 'session_123',
    userId: 'user_456'
}));

// Audio data (base64 encoded)
voiceWs.send(JSON.stringify({
    type: 'audio_command',
    audioData: base64AudioData,
    format: 'wav',
    sessionId: 'session_123',
    userId: 'user_456'
}));
```

#### Response Format

```javascript
{
    "type": "voice_response",
    "sessionId": "session_123",
    "response": {
        "intent": "SpawnAgent",
        "success": true,
        "message": "Successfully spawned researcher agent in swarm swarm_1757880683494_yl81sece5",
        "data": {
            "agentId": "agent_1757967065850_dv2zg7",
            "agentType": "researcher",
            "swarmId": "swarm_1757880683494_yl81sece5",
            "mcpTaskId": "mcp_task_1757967065850_xyz789",
            "status": "spawning",
            "estimatedReadyTime": "2025-01-22T10:00:30Z"
        }
    },
    "context": {
        "conversationTurn": 5,
        "pendingOperations": 1,
        "lastIntent": "SpawnAgent",
        "followUpExpected": false
    },
    "audioResponse": {
        "available": true,
        "url": "/api/voice/tts/audio_1757967065850.wav",
        "duration": 3.2
    }
}
```

### Error Handling

```javascript
{
    "type": "voice_error",
    "sessionId": "session_123",
    "error": {
        "code": "MCP_CONNECTION_FAILED",
        "message": "Failed to connect to MCP server",
        "details": {
            "service": "multi-agent-container:9500",
            "retryAfter": 5000
        }
    },
    "audioResponse": {
        "available": true,
        "url": "/api/voice/tts/error_audio.wav"
    }
}
```

## Supported Voice Commands

### Agent Management Commands

#### Spawn Agent
- **Pattern**: `"spawn a {agent_type} agent"`
- **Examples**:
  - "spawn a researcher agent"
  - "create a coder agent"
  - "launch a testing agent"
- **MCP Action**: `call_agent_spawn(agent_type, swarm_id)`

#### Agent Status
- **Pattern**: `"what's the status of {agent_type|all} agents"`
- **Examples**:
  - "what's the status of all agents"
  - "how are the coder agents doing"
  - "list all active agents"
- **MCP Action**: `call_agent_list("all")`

#### Stop Agent
- **Pattern**: `"stop the {agent_type} agent"`
- **Examples**:
  - "stop the researcher agent"
  - "terminate the coder agent"
- **MCP Action**: `call_agent_stop(agent_id)`

### Task Execution Commands

#### Execute Task
- **Pattern**: `"execute task: {description} with {priority} priority"`
- **Examples**:
  - "execute task: analyse the authentication module with high priority"
  - "run analysis on the data with medium priority"
  - "process the files with low priority"
- **MCP Action**: `call_task_orchestrate(description, priority, strategy)`

#### Task Status
- **Pattern**: `"what's the status of task {task_id}"`
- **Examples**:
  - "what's the status of task task_1757967065850"
  - "how is the analysis task going"
- **MCP Action**: `call_task_status(task_id)`

### System Query Commands

#### System Status
- **Pattern**: `"what's the system status"`
- **Examples**:
  - "what's the system status"
  - "how is the system performing"
  - "give me a system overview"
- **MCP Action**: `call_system_status()`

#### Agent Count
- **Pattern**: `"how many agents are {status}"`
- **Examples**:
  - "how many agents are running"
  - "how many agents are active"
  - "count all agents"
- **MCP Action**: `call_agent_count(status)`

#### Help Commands
- **Pattern**: `"help"` or `"what can I do"`
- **Examples**:
  - "help"
  - "what commands are available"
  - "what can I do"
- **Action**: Returns command list and examples

## REST Voice API

### Process Voice Command

```http
POST /api/voice/command
Content-Type: application/json

{
    "command": "spawn a researcher agent",
    "sessionId": "session_123",
    "userId": "user_456",
    "context": {
        "previousCommand": "list agents",
        "expectingFollowUp": false
    }
}
```

**Response:**
```json
{
    "intent": "SpawnAgent",
    "success": true,
    "message": "Successfully spawned researcher agent in swarm swarm_1757880683494_yl81sece5",
    "data": {
        "agentId": "agent_1757967065850_dv2zg7",
        "agentType": "researcher",
        "swarmId": "swarm_1757880683494_yl81sece5",
        "mcpTaskId": "mcp_task_1757967065850_xyz789"
    },
    "executionTime": 1247,
    "mcpCalls": [
        {
            "method": "agent_spawn",
            "duration": 892,
            "success": true
        }
    ]
}
```

### Upload Audio Command

```http
POST /api/voice/audio
Content-Type: multipart/form-data

# Form fields:
# - audio: Audio file (WAV, MP3, M4A)
# - sessionId: Session identifier
# - userId: User identifier
```

**Response:**
```json
{
    "transcription": "spawn a researcher agent",
    "intent": "SpawnAgent",
    "success": true,
    "message": "Successfully spawned researcher agent",
    "audioResponse": {
        "url": "/api/voice/tts/response_audio.wav",
        "duration": 2.8
    }
}
```

### Get Audio Response

```http
GET /api/voice/tts/{audioId}.wav
```

Returns: Audio file (WAV format) for TTS response

### Session Management

#### Get Session Info

```http
GET /api/voice/session/{sessionId}
```

**Response:**
```json
{
    "sessionId": "session_123",
    "userId": "user_456",
    "createdAt": "2025-01-22T10:00:00Z",
    "lastActivity": "2025-01-22T10:15:30Z",
    "conversationTurns": 8,
    "totalCommands": 12,
    "successfulCommands": 11,
    "pendingOperations": [
        {
            "operationType": "AgentSpawn",
            "parameters": {
                "agentType": "researcher",
                "swarmId": "swarm_1757880683494_yl81sece5"
            },
            "expectedCompletion": "2025-01-22T10:16:00Z"
        }
    ],
    "context": {
        "lastIntent": "SpawnAgent",
        "followUpExpected": false,
        "recentTopics": ["agent management", "task execution"]
    }
}
```

#### Clear Session

```http
DELETE /api/voice/session/{sessionId}
```

## Intent Recognition

### Supported Intents

| Intent | Description | Parameters | MCP Method |
|--------|-------------|------------|------------|
| `SpawnAgent` | Create new agent | `agentType`, `swarmId` | `agent_spawn` |
| `QueryStatus` | Get agent/system status | `target`, `scope` | `agent_list`, `swarm_status` |
| `ListAgents` | List all agents | `filter`, `status` | `agent_list` |
| `ExecuteTask` | Submit task to swarm | `description`, `priority` | `task_orchestrate` |
| `StopAgent` | Terminate agent | `agentId`, `agentType` | `agent_stop` |
| `Help` | Get command help | none | local |
| `SystemStatus` | Get system overview | none | `system_status` |

### Intent Confidence Scores

```json
{
    "recognisedIntent": "SpawnAgent",
    "confidence": 0.95,
    "alternativeIntents": [
        {"intent": "ListAgents", "confidence": 0.12},
        {"intent": "QueryStatus", "confidence": 0.08}
    ],
    "parameters": {
        "agentType": {
            "value": "researcher",
            "confidence": 0.92
        }
    }
}
```

## Real MCP Integration

### Agent Spawning Flow

1. **Voice Command**: "spawn a researcher agent"
2. **Intent Recognition**: `SpawnAgent` with `agentType: "researcher"`
3. **MCP Call**: `call_agent_spawn("researcher", current_swarm_id)`
4. **TCP Request**: JSON-RPC to `multi-agent-container:9500`
5. **Agent Creation**: Real agent spawned in MCP swarm
6. **Response**: Agent ID and status returned
7. **TTS Response**: Success message converted to speech

### Task Orchestration Flow

1. **Voice Command**: "execute task: analyse security with high priority"
2. **Intent Recognition**: `ExecuteTask` with parameters
3. **MCP Call**: `call_task_orchestrate(description, "high", "adaptive")`
4. **Task Assignment**: Real task distributed to agent swarm
5. **Progress Tracking**: Task status monitored via MCP
6. **Results**: Real execution results returned
7. **Voice Feedback**: Results summarized in speech

## Error Handling

### MCP Connection Errors

```json
{
    "error": {
        "type": "MCP_CONNECTION_ERROR",
        "message": "Unable to connect to MCP server",
        "details": {
            "host": "multi-agent-container",
            "port": 9500,
            "retryAfter": 5000
        }
    },
    "fallbackResponse": "I'm having trouble connecting to the agent system. Please try again in a moment.",
    "audioFallback": true
}
```

### Speech Recognition Errors

```json
{
    "error": {
        "type": "STT_ERROR",
        "message": "Speech recognition failed",
        "details": {
            "service": "whisper",
            "audioQuality": "poor"
        }
    },
    "fallbackResponse": "I couldn't understand your command. Could you please repeat it more clearly?",
    "audioFallback": true
}
```

### Intent Recognition Errors

```json
{
    "error": {
        "type": "INTENT_NOT_RECOGNIZED",
        "message": "Command not understood",
        "transcription": "make the system do the thing",
        "suggestions": [
            "spawn a [type] agent",
            "execute task: [description]",
            "what's the status of agents"
        ]
    },
    "fallbackResponse": "I didn't understand that command. Try saying 'help' for available commands.",
    "audioFallback": true
}
```

## Performance Metrics

### Response Times

- **Voice-to-Intent**: ~100ms
- **MCP Execution**: ~500-2000ms (varies by operation)
- **Response-to-TTS**: ~200ms
- **Total Pipeline**: ~1-3 seconds end-to-end

### Success Rates

- **Command Recognition**: ~95% for supported patterns
- **MCP Execution**: ~90% when server available
- **Context Preservation**: ~100% within session
- **Error Recovery**: ~100% graceful handling

### Service Dependencies

| Service | Endpoint | Purpose | Timeout |
|---------|----------|---------|---------|
| Whisper STT | `172.18.0.5:8080` | Speech recognition | 5s |
| Kokoro TTS | `172.18.0.9:5000` | Speech synthesis | 3s |
| MCP Server | `multi-agent-container:9500` | Agent orchestration | 10s |
| Redis | `localhost:6379` | Session storage | 1s |

## Integration Examples

### Complete Voice-to-Agent Flow

```javascript
class VoiceAgentController {
    constructor() {
        this.ws = new WebSocket('ws://localhost:3001/ws/voice');
        this.sessionId = this.generateSessionId();
        this.setupEventHandlers();
    }

    async sendVoiceCommand(audioBlob) {
        const base64Audio = await this.blobToBase64(audioBlob);

        this.ws.send(JSON.stringify({
            type: 'audio_command',
            audioData: base64Audio,
            format: 'wav',
            sessionId: this.sessionId,
            userId: 'user_123'
        }));
    }

    setupEventHandlers() {
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);

            switch (message.type) {
                case 'voice_response':
                    this.handleAgentResponse(message);
                    break;
                case 'voice_error':
                    this.handleError(message);
                    break;
            }
        };
    }

    handleAgentResponse(message) {
        console.log('Agent executed:', message.response.data);

        // Play audio response
        if (message.audioResponse.available) {
            const audio = new Audio(message.audioResponse.url);
            audio.play();
        }

        // Update UI with agent data
        this.updateAgentStatus(message.response.data);
    }
}

// Usage
const voiceController = new VoiceAgentController();

// Record and send voice command
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (event) => {
            voiceController.sendVoiceCommand(event.data);
        };

        // Start/stop recording with button
        document.getElementById('recordBtn').onclick = () => {
            recorder.start();
            setTimeout(() => recorder.stop(), 3000); // 3 second recording
        };
    });
```

## Related Documentation

- [MCP Protocol](mcp-protocol.md) - Agent orchestration protocol
- [WebSocket API](websocket-api.md) - Real-time communication
- [Voice Integration Summary](../../VOICE_INTEGRATION_SUMMARY.md) - Implementation details

---

**[← WebSocket API](websocket-api.md)** | **[Back to API Index →](index.md)**