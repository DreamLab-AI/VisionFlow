---
name: perplexity
description: "DEPRECATED -- This skill has been merged into perplexity-research. Use /perplexity-research instead."
status: deprecated
superseded_by: perplexity-research
version: 2.0.0
author: turbo-flow-claude
mcp_server: true
protocol: mcp-sdk
entry_point: mcp-server/server.js
dependencies:
  - perplexity-sdk
---

# Perplexity AI Research Skill (DEPRECATED)

**This skill has been merged into `perplexity-research`. Use `/perplexity-research` instead.**

The `perplexity-research` skill provides all the same capabilities (real-time web search, source citations, UK-centric prompts, MCP server integration) plus a more practical API query template, structured prompt guidelines, batch research support, and clearer usage patterns.

## Migration

Replace any references to `/perplexity` with `/perplexity-research`. The API key configuration, model selection, and all MCP tools remain identical.
