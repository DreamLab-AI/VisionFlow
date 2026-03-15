#!/usr/bin/env node
/**
 * Local LLM Proxy — Anthropic-to-OpenAI translation for llama.cpp
 *
 * Bridges Claude CLI (Anthropic API format) to local Nemotron 3 120B
 * running on llama.cpp with OpenAI-compatible API.
 *
 * Environment:
 *   LOCAL_LLM_HOST     - llama.cpp host (default: 192.168.2.48)
 *   LOCAL_LLM_PORT     - llama.cpp port (default: 8080)
 *   LOCAL_LLM_MODEL    - Model name for llama.cpp
 *   PROXY_PORT         - Port for this proxy (default: 3100)
 *
 * Usage in Claude CLI:
 *   export ANTHROPIC_BASE_URL=http://localhost:3100
 *   export ANTHROPIC_API_KEY=sk-ant-proxy-local
 *   claude
 */

// NODE_PATH must include /usr/local/lib/node_modules for global package resolution
import { AnthropicToOpenRouterProxy } from 'agentic-flow/dist/proxy/anthropic-to-openrouter.js';

const host = process.env.LOCAL_LLM_HOST || '192.168.2.48';
const port = process.env.LOCAL_LLM_PORT || '8080';
const model = process.env.LOCAL_LLM_MODEL || 'NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ4_XS-00001-of-00003.gguf';
const proxyPort = parseInt(process.env.PROXY_PORT || '3100');

const baseUrl = `http://${host}:${port}/v1`;

console.log(`Local LLM Proxy starting...`);
console.log(`  Target: ${baseUrl}`);
console.log(`  Model:  ${model}`);
console.log(`  Proxy:  http://localhost:${proxyPort}`);

const proxy = new AnthropicToOpenRouterProxy({
    openrouterApiKey: 'sk-local-no-key-needed',
    openrouterBaseUrl: baseUrl,
    defaultModel: model
});

proxy.start(proxyPort);

console.log(`\nConfigure Claude CLI:`);
console.log(`  export ANTHROPIC_BASE_URL=http://localhost:${proxyPort}`);
console.log(`  export ANTHROPIC_API_KEY=sk-ant-proxy-local`);
console.log(`  claude`);
