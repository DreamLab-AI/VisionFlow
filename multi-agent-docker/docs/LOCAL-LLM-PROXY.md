# Local LLM Proxy — Nemotron 3 120B via llama.cpp

## Overview

The Local LLM Proxy enables Claude Code CLI to communicate with a locally-hosted Nemotron 3 Super 120B model running on llama.cpp. It uses agentic-flow's `AnthropicToOpenRouterProxy` to translate between the Anthropic API format (used by Claude CLI) and the OpenAI-compatible format (served by llama.cpp).

## Architecture

```
local-private user shell
+---------------------+
| claude CLI          |
| ANTHROPIC_BASE_URL  |
| = localhost:3100    |
+--------+------------+
         | Anthropic API format
         v
+---------------------+
| agentic-flow proxy  |  supervisord: local-llm-proxy
| port 3100           |  /opt/scripts/local-llm-proxy.mjs
+--------+------------+
         | OpenAI API format
         v
+---------------------+
| llama.cpp server    |
| 192.168.2.48:8080   |  (host machine, LAN)
| Nemotron 3 120B     |
+---------------------+
```

## Model Specifications

| Property | Value |
|----------|-------|
| Model | NVIDIA Nemotron 3 Super 120B |
| Architecture | MoE (12B active parameters) |
| Quantization | IQ4_XS |
| Context Length | 262,144 tokens |
| Vocabulary | 131K tokens |
| Training Context | 1M tokens |
| Speed | ~56 tok/s |
| Server | llama.cpp (`llama-server`) |
| Host | `192.168.2.48` |
| Port | `8080` |
| API | OpenAI-compatible (`/v1/chat/completions`) |

## API Translation

The proxy handles format translation between two incompatible APIs:

### Request Translation (Anthropic → OpenAI)

| Anthropic Field | OpenAI Field |
|----------------|-------------|
| `system` (string or content blocks) | `messages[0]` with role `system` |
| `messages[]` | `messages[]` (role mapping preserved) |
| `tools[]` with `input_schema` | `tools[].function` with `parameters` |
| `max_tokens` | `max_tokens` |
| `temperature` | `temperature` |
| `stream` | `stream` |
| `model` (claude-*) | Overridden to `defaultModel` (Nemotron) |

### Response Translation (OpenAI → Anthropic)

| OpenAI Field | Anthropic Field |
|-------------|----------------|
| `choices[0].message.content` | `content[].text` |
| `choices[0].message.tool_calls[]` | `content[].tool_use` |
| `finish_reason: "stop"` | `stop_reason: "end_turn"` |
| `finish_reason: "length"` | `stop_reason: "max_tokens"` |
| `finish_reason: "function_call"` | `stop_reason: "tool_use"` |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |

### Streaming

SSE streaming is supported. OpenAI `data: {"choices":[...]}` chunks are converted to Anthropic `content_block_delta` events.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_HOST` | `192.168.2.48` | llama.cpp server host |
| `LOCAL_LLM_PORT` | `8080` | llama.cpp server port |
| `LOCAL_LLM_MODEL` | `NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ4_XS-00001-of-00003.gguf` | Model identifier |
| `LOCAL_LLM_CONTEXT` | `262144` | Context window size |
| `PROXY_PORT` | `3100` | Proxy listen port |

These are set in `docker-compose.unified.yml` and available to all container services.

### Supervisord Service

The proxy runs as `[program:local-llm-proxy]` in supervisord with `autostart=false` (on-demand):

```ini
[program:local-llm-proxy]
command=/usr/local/bin/node /opt/scripts/local-llm-proxy.mjs
directory=/home/local-private
user=local-private
environment=HOME="/home/local-private",LOCAL_LLM_HOST="...",LOCAL_LLM_PORT="...",LOCAL_LLM_MODEL="...",PROXY_PORT="3100",NODE_ENV="production",NODE_PATH="/usr/local/lib/node_modules"
autostart=false
autorestart=true
priority=360
```

### Proxy Script

`/opt/scripts/local-llm-proxy.mjs` — instantiates agentic-flow's `AnthropicToOpenRouterProxy` with the llama.cpp base URL:

```javascript
import { AnthropicToOpenRouterProxy } from 'agentic-flow/dist/proxy/anthropic-to-openrouter.js';

const proxy = new AnthropicToOpenRouterProxy({
    openrouterApiKey: 'sk-local-no-key-needed',  // llama.cpp ignores auth
    openrouterBaseUrl: `http://${host}:${port}/v1`,
    defaultModel: model
});

proxy.start(proxyPort);
```

## Usage

### From devuser (primary shell)

```bash
# Start the proxy
llm-proxy-start

# Check status
llm-proxy-status

# View logs
llm-proxy-logs

# Stop
llm-proxy-stop

# Switch to local-private user (Claude CLI auto-routes to Nemotron)
as-local
```

### From local-private user

The user environment is pre-configured with proxy settings:

```bash
# These are already set in .zshrc:
# export ANTHROPIC_BASE_URL="http://localhost:3100"
# export ANTHROPIC_API_KEY="sk-ant-proxy-local"

# Start proxy if not running
proxy-start

# Use Claude CLI — routes to Nemotron automatically
claude

# Direct LLM access (bypasses proxy)
llm-health              # Check llama.cpp health
llm-models              # List loaded models
llm-ask "question"      # Quick chat completion
```

### Direct API Access

```bash
# Proxy health check
curl http://localhost:3100/health

# Send Anthropic-format request through proxy
curl -X POST http://localhost:3100/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-proxy-local" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
# Note: model name is overridden to Nemotron by the proxy

# Direct llama.cpp access (OpenAI format)
curl http://192.168.2.48:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ4_XS-00001-of-00003.gguf",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024
  }'
```

## Tool Use Support

The proxy translates Anthropic tool definitions to OpenAI function calling format:

- Anthropic `tools[].input_schema` → OpenAI `tools[].function.parameters`
- OpenAI `tool_calls[]` responses → Anthropic `content[].tool_use` blocks
- For models without native function calling, the proxy falls back to tool emulation via structured prompts

Nemotron 3 120B supports basic function calling through llama.cpp's tool use implementation, though complex multi-tool chains may be less reliable than with native Claude models.

## Limitations

1. **Not auto-started** — The proxy has `autostart=false` because the local LLM may not always be available. Start manually with `llm-proxy-start` or `proxy-start`.
2. **No streaming parity** — Anthropic streaming events are approximated from OpenAI SSE chunks. Some event types (e.g., `message_start`, `content_block_start`) are simplified.
3. **Tool use fidelity** — Complex tool chains and multi-tool responses may not translate perfectly between API formats.
4. **Model capabilities** — Nemotron 3 120B is capable but differs from Claude models in reasoning style, code generation quality, and instruction following.
5. **Network dependency** — The llama.cpp server runs on a separate host (`192.168.2.48`). Network availability affects proxy operation.

## Files

| File | Purpose |
|------|---------|
| `unified-config/scripts/local-llm-proxy.mjs` | Proxy startup script (copied to `/opt/scripts/`) |
| `unified-config/supervisord.unified.conf` | `[program:local-llm-proxy]` service definition |
| `unified-config/entrypoint-unified.sh` | local-private user provisioning + proxy env vars |
| `unified-config/turbo-flow-aliases.sh` | `llm-proxy-*` aliases for devuser |
| `unified-config/tmux-autostart.sh` | Window 12 (LocalLLM) help text |
| `docker-compose.unified.yml` | `LOCAL_LLM_*` environment variables |
| `Dockerfile.unified` | local-private user creation + proxy script copy |

## Troubleshooting

```bash
# Check if llama.cpp is reachable
curl -s http://192.168.2.48:8080/health

# Check proxy logs
tail -f /var/log/local-llm-proxy.log
tail -f /var/log/local-llm-proxy.error.log

# Restart proxy
sudo supervisorctl restart local-llm-proxy

# Test proxy translation manually
curl -s http://localhost:3100/health

# Verify local-private user environment
sudo -u local-private -i bash -c 'echo $ANTHROPIC_BASE_URL'
```
