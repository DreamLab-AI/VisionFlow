---
title: X-FluxAgent Integration Plan for ComfyUI MCP Skill
description: This document analyzes X-FluxAgent components and provides concrete adaptation strategies for integrating into the ComfyUI skill with MCP server architecture and Playwright browser automation.
category: explanation
tags:
  - architecture
  - patterns
  - structure
  - api
  - api
related-docs:
  - multi-agent-docker/ANTIGRAVITY.md
  - multi-agent-docker/SKILLS.md
  - multi-agent-docker/TERMINAL_GRID.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Docker installation
  - Node.js runtime
---

# X-FluxAgent Integration Plan for ComfyUI MCP Skill

**Analysis Date**: 2025-12-03
**Target System**: Turbo Flow Claude - Multi-Agent Docker Workstation
**Source**: X-FluxAgent (ComfyUI custom node)
**Goal**: Adapt patterns for MCP server + Playwright automation

---

## Executive Summary

This document analyzes X-FluxAgent components and provides concrete adaptation strategies for integrating into the ComfyUI skill with MCP server architecture and Playwright browser automation.

**Key Adaptations**:
1. **ChatBotService** → MCP tool with Z.AI backend (Claude API)
2. **AICodeGenNode** → Workflow template generation system
3. **RichTextWidget/CodeMirror** → Markdown response rendering in Playwright
4. **Hot Reload** → MCP server development workflow
5. **AnyType** → Flexible MCP tool parameter system

---

## 1. ChatBotService Adaptation

### 1.1 Original Implementation

**File**: `X-FluxAgent/fluxagent/ChatBotService.py`

**Pattern Analysis**:
```python
# OpenAI API integration
class ChatBotService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

    def get_openai_response(self, user_message, system_message=None):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "gpt-4.1",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]

# Async route handler (aiohttp)
@routes.post('/X-FluxAgent-chatbot-message')
async def on_message(request):
    data = await request.json()
    user_message = data.get('message', '')

    # Send loading indicator
    PromptServer.instance.send_sync("X-FluxAgent.chatbot.loading", {"loading": True})

    # Get AI response in thread pool
    ai_response = await loop.run_in_executor(None, chatbot_service.get_openai_response, user_message)

    # Send response via WebSocket
    PromptServer.instance.send_sync("X-FluxAgent.chatbot.message", {
        "user_message": user_message,
        "ai_response": ai_response,
        "timestamp": int(time() * 1000),
        "loading": False
    })
    return web.json_response({'status': 'success'})
```

**Key Features**:
- Async HTTP route with aiohttp
- OpenAI API client
- WebSocket event broadcasting
- Thread pool executor for blocking I/O
- Loading state management
- Error handling with try/catch

### 1.2 Adapted Implementation (MCP Server)

**File**: `skills/comfyui/src/chat-service.ts`

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";

/**
 * ChatService: Adapted from X-FluxAgent ChatBotService
 * Uses Claude API via Z.AI (port 9600) through MCP client
 */
export class ChatService {
  private client: Client;
  private transport: StdioClientTransport;
  private zaiUrl: string;

  constructor(zaiUrl: string = "http://localhost:9600") {
    this.zaiUrl = zaiUrl;
    this.client = new Client({
      name: "comfyui-chat-client",
      version: "1.0.0"
    }, {
      capabilities: {}
    });
  }

  /**
   * Initialize MCP client connection
   */
  async initialize(): Promise<void> {
    this.transport = new StdioClientTransport({
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-everything"]
    });
    await this.client.connect(this.transport);
  }

  /**
   * Send chat request to Z.AI (Claude API)
   * Adapted from: get_openai_response()
   */
  async sendChatMessage(
    userMessage: string,
    systemMessage?: string
  ): Promise<string> {
    try {
      // Build request body
      const requestBody = {
        prompt: userMessage,
        system: systemMessage || "You are a helpful AI assistant specialized in ComfyUI workflows and image generation. Respond with markdown formatting including code blocks for workflow JSON.",
        timeout: 30000
      };

      // Call Z.AI service (internal to container)
      const response = await fetch(`${this.zaiUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Z.AI Error: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      return result.response || result.content || "";

    } catch (error) {
      console.error("Chat service error:", error);
      throw new Error(`Failed to get AI response: ${error.message}`);
    }
  }

  /**
   * Send chat with workflow context
   * Enhanced version with ComfyUI-specific context
   */
  async sendWorkflowChat(
    userMessage: string,
    workflowContext?: any
  ): Promise<string> {
    let systemMessage = `You are a ComfyUI workflow assistant.
You have access to the current workflow state and can help generate, modify, or explain workflows.
Always format workflow JSON with proper syntax highlighting.`;

    if (workflowContext) {
      systemMessage += `\n\nCurrent workflow context:\n${JSON.stringify(workflowContext, null, 2)}`;
    }

    return this.sendChatMessage(userMessage, systemMessage);
  }

  /**
   * Cleanup connections
   */
  async close(): Promise<void> {
    if (this.client) {
      await this.client.close();
    }
  }
}
```

**Integration with MCP Tool**:

```typescript
// skills/comfyui/src/tools/chat.ts
import { z } from "zod";
import { ChatService } from "../chat-service.js";

const chatInputSchema = z.object({
  message: z.string().describe("User message to send to AI assistant"),
  workflowContext: z.any().optional().describe("Optional ComfyUI workflow context"),
  systemPrompt: z.string().optional().describe("Optional system prompt override")
});

export async function chatTool(params: z.infer<typeof chatInputSchema>) {
  const chatService = new ChatService();
  await chatService.initialize();

  try {
    let response: string;

    if (params.workflowContext) {
      response = await chatService.sendWorkflowChat(
        params.message,
        params.workflowContext
      );
    } else if (params.systemPrompt) {
      response = await chatService.sendChatMessage(
        params.message,
        params.systemPrompt
      );
    } else {
      response = await chatService.sendChatMessage(params.message);
    }

    return {
      content: [
        {
          type: "text",
          text: response
        }
      ]
    };
  } finally {
    await chatService.close();
  }
}
```

### 1.3 Migration Steps

**Step 1**: Replace OpenAI with Z.AI endpoint
```typescript
// Before (OpenAI)
const url = "https://api.openai.com/v1/chat/completions";
const apiKey = process.env.OPENAI_API_KEY;

// After (Z.AI)
const url = "http://localhost:9600/chat";
// No API key needed (internal service)
```

**Step 2**: Adapt async route pattern to MCP tool pattern
```typescript
// Before (aiohttp route)
@routes.post('/X-FluxAgent-chatbot-message')
async def on_message(request):
    ...

// After (MCP tool)
server.tool(
  "comfyui_chat",
  "Send message to AI assistant for ComfyUI help",
  chatInputSchema,
  chatTool
);
```

**Step 3**: Remove WebSocket broadcasting (not needed in MCP)
```typescript
// Before (WebSocket events)
PromptServer.instance.send_sync("X-FluxAgent.chatbot.loading", {...})

// After (direct response)
return { content: [{ type: "text", text: response }] };
```

---

## 2. AICodeGenNode Pattern Adaptation

### 2.1 Original Implementation

**File**: `X-FluxAgent/fluxagent/AICodeGenNode.py`

**Pattern Analysis**:
```python
class AICodeGenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "rich_text": ("X-FluxAgent.RichTextWidget", {"default": "Your code", "readOnly": True}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",  # Contains workflow JSON
            }
        }

    def process(self, **kw):
        node_id = kw.get('node_id', 0)
        extra_pnginfo = kw.get('extra_pnginfo', {})

        # Extract workflow structure
        inputs = []
        outputs = []
        if extra_pnginfo and 'workflow' in extra_pnginfo:
            workflow = extra_pnginfo['workflow']
            for node in workflow['nodes']:
                if str(node.get('id')) == node_id:
                    # Parse inputs/outputs dynamically
                    node_inputs = node.get('inputs', [])
                    node_outputs = node.get('outputs', [])
                    break

        # Generate code based on inputs/outputs
        generated_code = f"Generated code goes here {node_id}"
        return (generated_code,)
```

**Key Features**:
- Dynamic input/output discovery from workflow JSON
- Access to full workflow graph via `extra_pnginfo`
- Node ID tracking for context
- Flexible type system for connections

### 2.2 Adapted Implementation (Workflow Templates)

**File**: `skills/comfyui/src/workflow-generator.ts`

```typescript
import { z } from "zod";

/**
 * WorkflowGenerator: Adapted from AICodeGenNode pattern
 * Generates ComfyUI workflow JSON from high-level descriptions
 */
export class WorkflowGenerator {
  private chatService: ChatService;

  constructor(chatService: ChatService) {
    this.chatService = chatService;
  }

  /**
   * Generate workflow template from description
   * Adapted from: AICodeGenNode.process()
   */
  async generateWorkflow(
    description: string,
    baseWorkflow?: any
  ): Promise<any> {
    const systemPrompt = `You are a ComfyUI workflow generator.
Generate valid ComfyUI workflow JSON based on user descriptions.

Workflow structure:
{
  "nodes": [
    {
      "id": number,
      "type": string,
      "pos": [x, y],
      "size": [width, height],
      "flags": {},
      "order": number,
      "mode": 0,
      "inputs": [{"name": string, "type": string, "link": number?}],
      "outputs": [{"name": string, "type": string, "links": number[]?}],
      "properties": {},
      "widgets_values": []
    }
  ],
  "links": [
    [link_id, from_node_id, from_output_index, to_node_id, to_input_index, data_type]
  ],
  "groups": [],
  "config": {},
  "extra": {},
  "version": 0.4
}

Generate complete, valid workflow JSON. Include all required nodes and proper connections.`;

    let prompt = `Generate a ComfyUI workflow for: ${description}`;

    if (baseWorkflow) {
      prompt += `\n\nModify this existing workflow:\n${JSON.stringify(baseWorkflow, null, 2)}`;
    }

    const response = await this.chatService.sendChatMessage(prompt, systemPrompt);

    // Extract JSON from markdown code blocks
    const jsonMatch = response.match(/```(?:json)?\n([\s\S]+?)\n```/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[1]);
    }

    // Try to parse entire response as JSON
    try {
      return JSON.parse(response);
    } catch (e) {
      throw new Error(`Failed to parse workflow JSON from response: ${response.substring(0, 200)}...`);
    }
  }

  /**
   * Analyze existing workflow structure
   * Similar to: AICodeGenNode input/output discovery
   */
  async analyzeWorkflow(workflow: any): Promise<{
    nodeCount: number;
    inputNodes: any[];
    outputNodes: any[];
    connections: any[];
    complexity: string;
  }> {
    const nodes = workflow.nodes || [];
    const links = workflow.links || [];

    // Find input nodes (no inputs or only optional inputs)
    const inputNodes = nodes.filter((node: any) => {
      const inputs = node.inputs || [];
      return inputs.length === 0 || inputs.every((inp: any) => !inp.link);
    });

    // Find output nodes (SaveImage, PreviewImage, etc.)
    const outputNodes = nodes.filter((node: any) =>
      ['SaveImage', 'PreviewImage', 'SaveAnimatedWEBP'].includes(node.type)
    );

    // Determine complexity
    let complexity = 'simple';
    if (nodes.length > 20) complexity = 'complex';
    else if (nodes.length > 10) complexity = 'moderate';

    return {
      nodeCount: nodes.length,
      inputNodes,
      outputNodes,
      connections: links,
      complexity
    };
  }

  /**
   * Generate node template with dynamic inputs/outputs
   * Adapted from: AICodeGenNode dynamic type system
   */
  generateNodeTemplate(
    nodeType: string,
    inputs: Array<{name: string; type: string}>,
    outputs: Array<{name: string; type: string}>
  ): any {
    return {
      id: Date.now(), // Generate unique ID
      type: nodeType,
      pos: [0, 0],
      size: [200, 100],
      flags: {},
      order: 0,
      mode: 0,
      inputs: inputs.map((inp, idx) => ({
        name: inp.name,
        type: inp.type,
        link: null
      })),
      outputs: outputs.map((out, idx) => ({
        name: out.name,
        type: out.type,
        links: []
      })),
      properties: {},
      widgets_values: []
    };
  }
}
```

**MCP Tool Integration**:

```typescript
// skills/comfyui/src/tools/generate-workflow.ts
const generateWorkflowSchema = z.object({
  description: z.string().describe("High-level description of desired workflow"),
  baseWorkflow: z.any().optional().describe("Optional existing workflow to modify"),
  includeAnalysis: z.boolean().default(false).describe("Include workflow analysis")
});

export async function generateWorkflowTool(params: z.infer<typeof generateWorkflowSchema>) {
  const chatService = new ChatService();
  await chatService.initialize();

  const generator = new WorkflowGenerator(chatService);

  try {
    const workflow = await generator.generateWorkflow(
      params.description,
      params.baseWorkflow
    );

    let responseText = `Generated ComfyUI workflow:\n\`\`\`json\n${JSON.stringify(workflow, null, 2)}\n\`\`\``;

    if (params.includeAnalysis) {
      const analysis = await generator.analyzeWorkflow(workflow);
      responseText += `\n\nWorkflow Analysis:\n- Nodes: ${analysis.nodeCount}\n- Input nodes: ${analysis.inputNodes.length}\n- Output nodes: ${analysis.outputNodes.length}\n- Complexity: ${analysis.complexity}`;
    }

    return {
      content: [
        { type: "text", text: responseText },
        { type: "resource", resource: { uri: "workflow://generated", mimeType: "application/json", text: JSON.stringify(workflow) } }
      ]
    };
  } finally {
    await chatService.close();
  }
}
```

---

## 3. RichTextWidget/CodeMirror Adaptation

### 3.1 Original Implementation

**Files**:
- `X-FluxAgent/js/fluxagent/RichTextWidget.js`
- `X-FluxAgent/js/fluxagent/ChatBotTab.js`

**Pattern Analysis**:

```javascript
// RichTextWidget.js - CodeMirror integration
import { createRichEditor } from "./codemirror_bundle.js"

widget.editor = createRichEditor(htmlElement, widget.value, {
    language: "markdown",
    showLineNumbers: false,
    onUpdate: (value) => {
        widget.value = value;
        saveValue();
    }
});

// ChatBotTab.js - Markdown rendering with syntax highlighting
import markdownIt from 'https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/+esm'
import highlightJs from 'https://cdn.jsdelivr.net/npm/highlight.js@11.11.1/+esm'

const md = new markdownIt({
    html: true,
    breaks: true,
    linkify: true,
    highlight: function (str, lang) {
        if (lang && highlightJs.getLanguage(lang)) {
            return highlightJs.highlight(str, { language: lang }).value;
        }
        return highlightJs.highlightAuto(str).value;
    }
});

function addMessageToChat(message, type = 'ai') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message-bubble ${type}-message`;
    messageDiv.innerHTML = markdownToHtml(message);

    // Add copy buttons to code blocks
    addCopyButtonsToCodeBlocks(messageDiv);

    messageContainer.appendChild(messageDiv);
}
```

**Key Features**:
- CodeMirror for rich text editing
- markdown-it for rendering
- highlight.js for syntax highlighting
- Copy-to-clipboard for code blocks
- Auto-scroll management
- GitHub Dark theme styling

### 3.2 Adapted Implementation (Playwright)

**File**: `skills/comfyui/src/browser-ui.ts`

```typescript
import { Page } from "playwright";

/**
 * BrowserUI: Adapted from RichTextWidget + ChatBotTab pattern
 * Renders markdown with syntax highlighting in Playwright-controlled browser
 */
export class BrowserUI {
  private page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * Inject markdown rendering libraries
   * Adapted from: ChatBotTab markdown-it + highlight.js setup
   */
  async injectMarkdownSupport(): Promise<void> {
    await this.page.addScriptTag({
      url: 'https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js'
    });

    await this.page.addScriptTag({
      url: 'https://cdn.jsdelivr.net/npm/highlight.js@11.11.1/lib/core.min.js'
    });

    // Add common languages
    await this.page.addScriptTag({
      url: 'https://cdn.jsdelivr.net/npm/highlight.js@11.11.1/lib/languages/javascript.min.js'
    });
    await this.page.addScriptTag({
      url: 'https://cdn.jsdelivr.net/npm/highlight.js@11.11.1/lib/languages/python.min.js'
    });
    await this.page.addScriptTag({
      url: 'https://cdn.jsdelivr.net/npm/highlight.js@11.11.1/lib/languages/json.min.js'
    });

    // Add GitHub Dark theme CSS
    await this.page.addStyleTag({
      url: 'https://cdn.jsdelivr.net/npm/highlight.js@11.11.1/styles/github-dark.css'
    });

    // Initialize markdown-it with highlight.js
    await this.page.evaluate(() => {
      (window as any).md = (window as any).markdownit({
        html: true,
        breaks: true,
        linkify: true,
        typographer: true,
        highlight: function (str: string, lang: string) {
          const hljs = (window as any).hljs;
          if (lang && hljs.getLanguage(lang)) {
            try {
              return hljs.highlight(str, { language: lang }).value;
            } catch (e) {
              console.warn('Highlight failed:', e);
            }
          }
          return hljs.highlightAuto(str).value;
        }
      });
    });
  }

  /**
   * Render markdown message in chat UI
   * Adapted from: addMessageToChat()
   */
  async renderMarkdownMessage(
    markdown: string,
    type: 'user' | 'ai' | 'error' = 'ai'
  ): Promise<void> {
    await this.page.evaluate(({ markdown, type }) => {
      const md = (window as any).md;
      const html = md.render(markdown);

      const messageDiv = document.createElement('div');
      messageDiv.className = `message-bubble ${type}-message`;
      messageDiv.innerHTML = html;

      // Add copy buttons to code blocks
      const codeBlocks = messageDiv.querySelectorAll('pre code');
      codeBlocks.forEach((codeBlock: Element) => {
        const pre = codeBlock.parentElement;
        if (!pre) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        pre.parentNode?.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);

        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-button';
        copyBtn.textContent = 'Copy';
        copyBtn.onclick = () => {
          navigator.clipboard.writeText(codeBlock.textContent || '').then(() => {
            copyBtn.textContent = 'Copied!';
            setTimeout(() => copyBtn.textContent = 'Copy', 2000);
          });
        };
        wrapper.appendChild(copyBtn);
      });

      const container = document.getElementById('chat-messages');
      if (container) {
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
      }
    }, { markdown, type });
  }

  /**
   * Create chat UI container
   * Adapted from: ChatBotTab UI structure
   */
  async createChatContainer(): Promise<void> {
    await this.page.evaluate(() => {
      const container = document.createElement('div');
      container.id = 'comfyui-chat-container';
      container.innerHTML = `
        <style>
          @import url('https://cdn.jsdelivr.net/npm/highlight.js@11.11.1/styles/github-dark.css');

          #comfyui-chat-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 400px;
            height: 600px;
            background: #1e1e1e;
            border: 1px solid #444;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          }

          #chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            font-size: 14px;
            line-height: 1.6;
            color: #fff;
          }

          .message-bubble {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 8px;
            max-width: 90%;
          }

          .user-message {
            background: #007acc;
            color: white;
            margin-left: auto;
            margin-right: 0;
          }

          .ai-message {
            background: #333;
            color: #fff;
          }

          .error-message {
            background: #d32f2f;
            color: white;
          }

          .code-block-wrapper {
            position: relative;
            margin: 12px 0;
          }

          .copy-button {
            position: absolute;
            top: 8px;
            right: 8px;
            background: #21262d;
            border: 1px solid #30363d;
            color: #f0f6fc;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            opacity: 0;
            transition: opacity 0.2s;
          }

          .code-block-wrapper:hover .copy-button {
            opacity: 1;
          }

          #comfyui-chat-container pre {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
          }

          #comfyui-chat-container code {
            background: #262c36;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 85%;
          }

          #comfyui-chat-container pre code {
            background: none;
            padding: 0;
          }
        </style>

        <div id="chat-messages"></div>
      `;

      document.body.appendChild(container);
    });
  }

  /**
   * Display workflow JSON with syntax highlighting
   */
  async displayWorkflow(workflow: any): Promise<void> {
    const markdown = `### Generated Workflow\n\n\`\`\`json\n${JSON.stringify(workflow, null, 2)}\n\`\`\``;
    await this.renderMarkdownMessage(markdown, 'ai');
  }

  /**
   * Show loading indicator
   * Adapted from: showLoading()
   */
  async showLoading(show: boolean): Promise<void> {
    await this.page.evaluate((show) => {
      let indicator = document.getElementById('loading-indicator');
      if (!indicator && show) {
        indicator = document.createElement('div');
        indicator.id = 'loading-indicator';
        indicator.style.cssText = `
          text-align: center;
          padding: 10px;
          color: #888;
          font-style: italic;
          animation: pulse 1.5s ease-in-out infinite alternate;
        `;
        indicator.innerHTML = '<span>AI is thinking...</span>';
        document.getElementById('chat-messages')?.appendChild(indicator);
      } else if (indicator && !show) {
        indicator.remove();
      }
    }, show);
  }
}
```

**Usage in MCP Tool**:

```typescript
// skills/comfyui/src/tools/chat-with-ui.ts
export async function chatWithUITool(params: any) {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  // Navigate to ComfyUI
  await page.goto('http://localhost:8188');

  // Setup UI
  const ui = new BrowserUI(page);
  await ui.injectMarkdownSupport();
  await ui.createChatContainer();

  // Show loading
  await ui.showLoading(true);

  // Get AI response
  const chatService = new ChatService();
  await chatService.initialize();
  const response = await chatService.sendChatMessage(params.message);

  // Display response
  await ui.showLoading(false);
  await ui.renderMarkdownMessage(response, 'ai');

  // If response contains workflow JSON, display it
  const jsonMatch = response.match(/```json\n([\s\S]+?)\n```/);
  if (jsonMatch) {
    try {
      const workflow = JSON.parse(jsonMatch[1]);
      await ui.displayWorkflow(workflow);
    } catch (e) {
      console.error('Failed to parse workflow:', e);
    }
  }

  await chatService.close();
  await browser.close();

  return { content: [{ type: "text", text: "Chat UI interaction completed" }] };
}
```

---

## 4. Hot Reload System Adaptation

### 4.1 Original Implementation

**File**: `X-FluxAgent/fluxagent/utils/HotReload.py`

**Pattern Analysis**:

```python
def reload_module(module_name: str) -> bool:
    """Reloads a ComfyUI custom node module and clears relevant caches."""
    try:
        # Find dependent modules
        reload_modules = [
            mod_name for mod_name in sys.modules.keys()
            if module_name in mod_name and mod_name != module_name
        ]

        # Unload dependent modules first
        for reload_mod in reload_modules:
            if reload_mod in sys.modules:
                del sys.modules[reload_mod]

        # Unload main module
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Reload the module
        spec = importlib.util.spec_from_file_location(module_name, module_path_init)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Mark reloaded class types for cache clearing
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            for key in module.NODE_CLASS_MAPPINGS.keys():
                RELOADED_CLASS_TYPES[key] = 3

        return True
    except Exception as e:
        logging.error(f"Failed to reload module {module_name}: {e}")
        return False

# Monkey patch cache clearing
def monkeypatch():
    """Apply necessary monkey patches for hot reloading cache management."""
    _original_set_prompt = caching.HierarchicalCache.set_prompt

    def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        # Clear cache for reloaded classes
        found_keys = []
        for key, item_list in self.cache_key_set.keys.items():
            if dfs(item_list, RELOADED_CLASS_TYPES):
                found_keys.append(key)

        for key in found_keys:
            del self.cache[cache_key]
            del self.cache_key_set.keys[key]

        return _original_set_prompt(self, dynprompt, node_ids, is_changed_cache)

    caching.HierarchicalCache.set_prompt = set_prompt
```

**Key Features**:
- Dynamic module reloading without restart
- Dependency graph traversal
- Cache invalidation via monkey patching
- Registry update (NODE_CLASS_MAPPINGS)

### 4.2 Adapted Implementation (MCP Server Development)

**File**: `skills/comfyui/scripts/dev-reload.sh`

```bash
#!/bin/bash
# dev-reload.sh: Hot reload MCP server during development
# Adapted from: X-FluxAgent HotReload.py

set -e

SKILL_DIR="/home/devuser/.claude/skills/comfyui"
LOG_FILE="/var/log/comfyui-mcp-dev.log"

echo "[$(date)] Starting MCP server hot reload watcher..." | tee -a "$LOG_FILE"

# Watch for changes in src/ directory
watch_and_reload() {
    cd "$SKILL_DIR"

    # Use inotifywait for file system monitoring
    while inotifywait -r -e modify,create,delete src/; do
        echo "[$(date)] Detected changes in src/, reloading..." | tee -a "$LOG_FILE"

        # Rebuild TypeScript
        npm run build 2>&1 | tee -a "$LOG_FILE"

        if [ $? -eq 0 ]; then
            echo "[$(date)] Build successful, restarting MCP server..." | tee -a "$LOG_FILE"

            # Find and kill existing MCP server process
            pkill -f "comfyui-mcp-server" || true
            sleep 1

            # Restart MCP server in background
            node dist/index.js >> "$LOG_FILE" 2>&1 &
            MCP_PID=$!

            echo "[$(date)] MCP server restarted (PID: $MCP_PID)" | tee -a "$LOG_FILE"
        else
            echo "[$(date)] Build failed, keeping previous version running" | tee -a "$LOG_FILE"
        fi

        sleep 2
    done
}

# Trap Ctrl+C to cleanup
trap 'echo "Stopping hot reload watcher..."; pkill -P $$; exit' INT TERM

# Start watching
watch_and_reload
```

**Usage**:

```bash
# Install inotify-tools
sudo pacman -S inotify-tools

# Run hot reload watcher in background
cd /home/devuser/.claude/skills/comfyui
./scripts/dev-reload.sh &

# Now edit files in src/ and they'll auto-reload
vim src/tools/chat.ts  # Save triggers reload
```

**TypeScript Module Hot Reload** (for dynamic tool loading):

```typescript
// skills/comfyui/src/tool-loader.ts
import { watch } from "fs/promises";
import { resolve, join } from "path";

/**
 * ToolLoader: Adapted from HotReload.py
 * Dynamically reload MCP tools during development
 */
export class ToolLoader {
  private toolsDir: string;
  private loadedTools: Map<string, any>;
  private watcher: any;

  constructor(toolsDir: string) {
    this.toolsDir = resolve(toolsDir);
    this.loadedTools = new Map();
  }

  /**
   * Load or reload a tool module
   * Adapted from: reload_module()
   */
  async loadTool(toolName: string): Promise<any> {
    const toolPath = join(this.toolsDir, `${toolName}.js`);

    try {
      // Clear module cache (similar to del sys.modules[module_name])
      delete require.cache[require.resolve(toolPath)];

      // Reload module
      const toolModule = await import(toolPath);

      this.loadedTools.set(toolName, toolModule);
      console.log(`[ToolLoader] Loaded tool: ${toolName}`);

      return toolModule;
    } catch (error) {
      console.error(`[ToolLoader] Failed to load tool ${toolName}:`, error);
      throw error;
    }
  }

  /**
   * Watch tools directory for changes
   * Adapted from: ComfyUI hot reload pattern
   */
  async startWatching(): Promise<void> {
    console.log(`[ToolLoader] Watching ${this.toolsDir} for changes...`);

    try {
      const watcher = watch(this.toolsDir, { recursive: true });

      for await (const event of watcher) {
        if (event.filename && event.filename.endsWith('.js')) {
          const toolName = event.filename.replace('.js', '');
          console.log(`[ToolLoader] Detected change in ${toolName}, reloading...`);

          try {
            await this.loadTool(toolName);
          } catch (error) {
            console.error(`[ToolLoader] Reload failed for ${toolName}:`, error);
          }
        }
      }
    } catch (error) {
      console.error('[ToolLoader] Watcher error:', error);
    }
  }

  /**
   * Get all loaded tools
   */
  getLoadedTools(): Map<string, any> {
    return this.loadedTools;
  }

  /**
   * Stop watching
   */
  stopWatching(): void {
    if (this.watcher) {
      this.watcher.close();
    }
  }
}
```

**Integration in MCP Server**:

```typescript
// skills/comfyui/src/index.ts
import { ToolLoader } from "./tool-loader.js";

const toolLoader = new ToolLoader("./dist/tools");

// Load all tools initially
const toolFiles = ["chat", "generate-workflow", "monitor", "execute"];
for (const tool of toolFiles) {
  await toolLoader.loadTool(tool);
}

// Start hot reload watcher in development mode
if (process.env.NODE_ENV === "development") {
  toolLoader.startWatching();
}

// Register tools dynamically
const loadedTools = toolLoader.getLoadedTools();
for (const [name, module] of loadedTools) {
  server.tool(
    module.name || name,
    module.description,
    module.schema,
    module.handler
  );
}
```

---

## 5. AnyType Trick Adaptation

### 5.1 Original Implementation

**File**: `X-FluxAgent/fluxagent/utils/AnyType.py`

**Pattern Analysis**:

```python
class AnyType(str):
    """
    Wildcard trick: allows any type of connection in ComfyUI
    by overriding __ne__ to always return False
    """
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

# Usage in node definition
class FlexibleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "any_input": (any_typ, {}),  # Accepts ANY type
            }
        }

    RETURN_TYPES = (any_typ,)  # Can return ANY type
```

**Key Feature**:
- Type system bypass for maximum flexibility
- Allows connections between any node types
- Useful for generic utility nodes

### 5.2 Adapted Implementation (MCP Tool Parameters)

**File**: `skills/comfyui/src/schemas/flexible-params.ts`

```typescript
import { z } from "zod";

/**
 * FlexibleParams: Adapted from AnyType trick
 * Provides maximum flexibility for MCP tool parameters
 */

/**
 * Schema that accepts any type (similar to AnyType)
 * Uses zod's .passthrough() and .unknown() for flexibility
 */
export const anySchema = z.any().describe("Accepts any type of input");

/**
 * Workflow data schema with flexible structure
 * Adapted from: ComfyUI workflow JSON with any_typ
 */
export const flexibleWorkflowSchema = z.object({
  nodes: z.array(z.object({
    id: z.union([z.string(), z.number()]),
    type: z.string(),
    inputs: z.any().optional(),  // Flexible inputs
    outputs: z.any().optional(), // Flexible outputs
    properties: z.any().optional()
  })).optional(),
  links: z.array(z.any()).optional(),
  extra: z.any().optional()
}).passthrough(); // Allow additional fields

/**
 * Generic tool parameter that accepts any workflow-related data
 */
export const workflowDataSchema = z.union([
  z.string().describe("Workflow as JSON string"),
  z.object({}).passthrough().describe("Workflow as object"),
  z.array(z.any()).describe("Workflow as array"),
  z.null().describe("No workflow data")
]).describe("Flexible workflow data in any format");

/**
 * Node input schema with AnyType-like flexibility
 */
export const flexibleNodeInputSchema = z.object({
  name: z.string(),
  type: z.string().default("*"), // Default to wildcard
  value: z.any().optional(),
  link: z.union([z.number(), z.null()]).optional()
});

/**
 * Create tool schema with optional flexible parameters
 * Adapted from: AICodeGenNode INPUT_TYPES pattern
 */
export function createFlexibleToolSchema(
  requiredParams: Record<string, z.ZodType>,
  optionalParams: Record<string, z.ZodType> = {}
) {
  return z.object({
    ...requiredParams,
    ...Object.fromEntries(
      Object.entries(optionalParams).map(([key, schema]) => [
        key,
        schema.optional()
      ])
    ),
    // Add catch-all for additional parameters
    additionalData: z.any().optional().describe("Any additional data")
  }).passthrough(); // Allow extra fields like AnyType
}
```

**Usage Examples**:

```typescript
// Example 1: Tool that accepts any workflow format
const flexibleWorkflowTool = z.object({
  workflow: workflowDataSchema,
  operation: z.enum(["analyze", "modify", "execute"]),
  parameters: z.any().optional()
});

// Example 2: Tool with flexible node manipulation
const nodeManipulationSchema = createFlexibleToolSchema(
  {
    nodeId: z.string().describe("Target node ID")
  },
  {
    newInputs: z.array(flexibleNodeInputSchema),
    newOutputs: z.array(flexibleNodeInputSchema),
    customProperties: z.any()
  }
);

// Example 3: Universal workflow tool (maximum flexibility)
const universalWorkflowSchema = z.object({
  action: z.string().describe("Action to perform"),
  target: z.any().describe("Target (node, workflow, or any data)"),
  parameters: z.record(z.any()).optional()
}).passthrough();
```

**Implementation in MCP Tool**:

```typescript
// skills/comfyui/src/tools/flexible-workflow.ts
const flexibleWorkflowToolSchema = createFlexibleToolSchema(
  {
    action: z.enum(["create", "modify", "analyze", "execute"])
  },
  {
    workflow: workflowDataSchema,
    nodeData: z.any(),
    options: z.any()
  }
);

export async function flexibleWorkflowTool(
  params: z.infer<typeof flexibleWorkflowToolSchema>
) {
  // Handle any type of workflow data
  let workflow: any;

  if (typeof params.workflow === "string") {
    try {
      workflow = JSON.parse(params.workflow);
    } catch (e) {
      // If not JSON, treat as workflow description
      workflow = { description: params.workflow };
    }
  } else if (params.workflow) {
    workflow = params.workflow;
  } else {
    workflow = { nodes: [], links: [] };
  }

  // Flexible action handling
  switch (params.action) {
    case "create":
      // Create workflow from any input format
      const generator = new WorkflowGenerator(chatService);
      return await generator.generateWorkflow(
        workflow.description || JSON.stringify(workflow)
      );

    case "modify":
      // Modify workflow with any node data format
      if (params.nodeData) {
        workflow.nodes = workflow.nodes || [];
        workflow.nodes.push(params.nodeData); // Accept any node format
      }
      return workflow;

    case "analyze":
      // Analyze any workflow structure
      return await generator.analyzeWorkflow(workflow);

    case "execute":
      // Execute workflow through Playwright
      const executor = new WorkflowExecutor(page);
      return await executor.executeWorkflow(workflow);
  }
}
```

**Type Safety with Flexibility**:

```typescript
// Combine type safety with flexibility using discriminated unions
export const strictOrFlexibleSchema = z.discriminatedUnion("mode", [
  // Strict mode: enforced types
  z.object({
    mode: z.literal("strict"),
    workflow: flexibleWorkflowSchema,
    validation: z.literal(true)
  }),
  // Flexible mode: any types allowed (AnyType equivalent)
  z.object({
    mode: z.literal("flexible"),
    data: z.any(),
    validation: z.literal(false)
  })
]);

// Usage
const tool = async (params: z.infer<typeof strictOrFlexibleSchema>) => {
  if (params.mode === "strict") {
    // Full type checking
    const workflow = params.workflow;
    // TypeScript knows exact structure
  } else {
    // Flexible handling like AnyType
    const data = params.data;
    // Can be anything
  }
};
```

---

## 6. Integration Points Summary

### 6.1 MCP Server Architecture

```
skills/comfyui/
├── src/
│   ├── index.ts                    # MCP server entry point
│   ├── chat-service.ts             # Adapted ChatBotService
│   ├── workflow-generator.ts       # Adapted AICodeGenNode
│   ├── browser-ui.ts               # Adapted RichTextWidget/ChatBotTab
│   ├── tool-loader.ts              # Adapted HotReload
│   ├── tools/
│   │   ├── chat.ts                 # Chat tool
│   │   ├── generate-workflow.ts   # Workflow generation
│   │   ├── monitor.ts              # Browser monitoring
│   │   ├── execute.ts              # Workflow execution
│   │   └── flexible-workflow.ts   # AnyType-style tool
│   ├── schemas/
│   │   ├── flexible-params.ts      # Adapted AnyType
│   │   └── workflow.ts             # Workflow schemas
│   └── utils/
│       ├── zai-client.ts           # Z.AI API wrapper
│       └── playwright-helpers.ts   # Browser automation
├── scripts/
│   ├── dev-reload.sh               # Hot reload watcher
│   └── test-tools.sh               # Tool testing
└── package.json
```

### 6.2 Data Flow

```
Claude Code Request
    ↓
MCP Server (index.ts)
    ↓
Tool Dispatcher
    ↓
┌─────────────┬──────────────┬─────────────┐
│             │              │             │
Chat Tool  Generate Tool  Execute Tool  Monitor Tool
│             │              │             │
↓             ↓              ↓             ↓
ChatService  Generator   Playwright    BrowserUI
│             │              │             │
↓             ↓              ↓             ↓
Z.AI API    Claude via    ComfyUI Web   Visual
(port 9600) Z.AI         (port 8188)   Feedback
```

### 6.3 Component Mapping

| X-FluxAgent Component | MCP Skill Equivalent | Integration Point |
|----------------------|---------------------|-------------------|
| ChatBotService.py | chat-service.ts | Z.AI API (port 9600) |
| AICodeGenNode.py | workflow-generator.ts | Claude via Z.AI |
| RichTextWidget.js | browser-ui.ts | Playwright page injection |
| ChatBotTab.js | browser-ui.ts | Playwright UI rendering |
| HotReload.py | tool-loader.ts + dev-reload.sh | File watcher + module reload |
| AnyType.py | flexible-params.ts | Zod schemas with .any() |
| OpenAI API | Z.AI wrapper | Internal HTTP (localhost:9600) |
| WebSocket events | MCP tool responses | Direct return values |
| ComfyUI routes | MCP tools | Server.tool() registration |

---

## 7. Migration Checklist

### Phase 1: Core Service Adaptation ✅
- [x] ChatBotService → chat-service.ts with Z.AI
- [x] OpenAI API → Z.AI endpoint mapping
- [x] Async request handling → MCP tool async/await
- [x] Error handling preservation

### Phase 2: Workflow Generation 🔄
- [ ] AICodeGenNode → workflow-generator.ts
- [ ] Workflow JSON parsing logic
- [ ] Node input/output discovery
- [ ] Template generation system
- [ ] Workflow analysis tools

### Phase 3: UI Rendering 🔄
- [ ] RichTextWidget → browser-ui.ts
- [ ] markdown-it integration in Playwright
- [ ] highlight.js syntax highlighting
- [ ] Copy-to-clipboard functionality
- [ ] Chat container injection
- [ ] Auto-scroll management

### Phase 4: Development Tools 🔄
- [ ] HotReload → tool-loader.ts
- [ ] File watching system (inotify)
- [ ] Module cache clearing
- [ ] Automatic MCP server restart
- [ ] Development workflow scripts

### Phase 5: Type System 🔄
- [ ] AnyType → flexible-params.ts
- [ ] Zod schema flexibility patterns
- [ ] Discriminated union types
- [ ] Passthrough schemas
- [ ] Type coercion utilities

### Phase 6: Testing & Validation ⏳
- [ ] Unit tests for chat-service.ts
- [ ] Integration tests with Z.AI
- [ ] Playwright UI tests
- [ ] Hot reload verification
- [ ] End-to-end workflow generation

### Phase 7: Documentation 📝
- [ ] API documentation
- [ ] Tool usage examples
- [ ] Migration notes
- [ ] Performance benchmarks
- [ ] Troubleshooting guide

---

## 8. Testing Strategy

### 8.1 Unit Tests

```typescript
// skills/comfyui/tests/chat-service.test.ts
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { ChatService } from "../src/chat-service";

describe("ChatService", () => {
  let chatService: ChatService;

  beforeEach(async () => {
    chatService = new ChatService("http://localhost:9600");
    await chatService.initialize();
  });

  afterEach(async () => {
    await chatService.close();
  });

  it("should send chat message to Z.AI", async () => {
    const response = await chatService.sendChatMessage(
      "Generate a simple text-to-image workflow"
    );

    expect(response).toBeTruthy();
    expect(response.length).toBeGreaterThan(0);
  });

  it("should handle workflow context", async () => {
    const workflow = { nodes: [], links: [] };
    const response = await chatService.sendWorkflowChat(
      "Add a KSampler node",
      workflow
    );

    expect(response).toContain("KSampler");
  });

  it("should handle errors gracefully", async () => {
    // Simulate Z.AI down
    const failingService = new ChatService("http://localhost:9999");
    await failingService.initialize();

    await expect(
      failingService.sendChatMessage("test")
    ).rejects.toThrow();
  });
});
```

### 8.2 Integration Tests

```typescript
// skills/comfyui/tests/workflow-generation.integration.test.ts
describe("Workflow Generation", () => {
  it("should generate valid ComfyUI workflow", async () => {
    const generator = new WorkflowGenerator(chatService);

    const workflow = await generator.generateWorkflow(
      "Create a text-to-image workflow with SDXL model"
    );

    expect(workflow).toHaveProperty("nodes");
    expect(workflow).toHaveProperty("links");
    expect(workflow.nodes.length).toBeGreaterThan(0);

    // Validate required nodes
    const nodeTypes = workflow.nodes.map((n: any) => n.type);
    expect(nodeTypes).toContain("CLIPTextEncode");
    expect(nodeTypes).toContain("KSampler");
    expect(nodeTypes).toContain("SaveImage");
  });

  it("should modify existing workflow", async () => {
    const baseWorkflow = {
      nodes: [{ id: 1, type: "CheckpointLoaderSimple" }],
      links: []
    };

    const workflow = await generator.generateWorkflow(
      "Add a LoRA loader",
      baseWorkflow
    );

    expect(workflow.nodes.length).toBeGreaterThan(1);
    expect(workflow.nodes.some((n: any) => n.type.includes("LoRA"))).toBe(true);
  });
});
```

### 8.3 Playwright UI Tests

```typescript
// skills/comfyui/tests/browser-ui.e2e.test.ts
import { chromium } from "playwright";

describe("Browser UI", () => {
  let browser, page, ui;

  beforeEach(async () => {
    browser = await chromium.launch();
    page = await browser.newPage();
    await page.goto("http://localhost:8188");

    ui = new BrowserUI(page);
    await ui.injectMarkdownSupport();
    await ui.createChatContainer();
  });

  afterEach(async () => {
    await browser.close();
  });

  it("should render markdown with syntax highlighting", async () => {
    const markdown = "# Test\n\n```json\n{\"test\": true}\n```";
    await ui.renderMarkdownMessage(markdown, "ai");

    const hasCodeBlock = await page.evaluate(() => {
      return document.querySelector('pre code') !== null;
    });

    expect(hasCodeBlock).toBe(true);
  });

  it("should add copy buttons to code blocks", async () => {
    const markdown = "```python\nprint('hello')\n```";
    await ui.renderMarkdownMessage(markdown, "ai");

    const hasCopyButton = await page.evaluate(() => {
      return document.querySelector('.copy-button') !== null;
    });

    expect(hasCopyButton).toBe(true);
  });

  it("should display workflow JSON", async () => {
    const workflow = { nodes: [], links: [] };
    await ui.displayWorkflow(workflow);

    const workflowText = await page.textContent('#chat-messages');
    expect(workflowText).toContain("Generated Workflow");
  });
});
```

---

## 9. Performance Considerations

### 9.1 Z.AI API Optimization

```typescript
// Implement request pooling like X-FluxAgent's thread pool
class ZAIRequestPool {
  private maxConcurrent = 4; // Match claude-zai worker pool
  private queue: Array<() => Promise<any>> = [];
  private active = 0;

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    while (this.active >= this.maxConcurrent) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    this.active++;
    try {
      return await fn();
    } finally {
      this.active--;
      this.processQueue();
    }
  }

  private processQueue() {
    if (this.queue.length > 0 && this.active < this.maxConcurrent) {
      const next = this.queue.shift();
      if (next) next();
    }
  }
}
```

### 9.2 Workflow Caching

```typescript
// Cache generated workflows to avoid redundant AI calls
class WorkflowCache {
  private cache = new Map<string, any>();
  private maxSize = 100;

  set(key: string, workflow: any): void {
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, workflow);
  }

  get(key: string): any | undefined {
    return this.cache.get(key);
  }

  hash(description: string): string {
    // Simple hash for caching
    return Buffer.from(description).toString('base64');
  }
}
```

---

## 10. Future Enhancements

### 10.1 Real-time Collaboration
- WebSocket support for live workflow editing
- Multiple users on same workflow
- Adapted from X-FluxAgent's WebSocket pattern

### 10.2 Custom Node Generation
- AI-powered custom node creation
- Adapted from AICodeGenNode dynamic code generation

### 10.3 Visual Workflow Builder
- Drag-and-drop interface in Playwright
- Real-time preview updates
- Adapted from RichTextWidget interactive editing

### 10.4 Plugin System
- Hot-loadable workflow plugins
- Adapted from HotReload architecture

---

---

---

## Related Documentation

- [Complete State Management Architecture](../diagrams/client/state/state-management-complete.md)
- [VisionFlow Documentation Modernization - Final Report](../DOCUMENTATION_MODERNIZATION_COMPLETE.md)
- [Server-Side Actor System - Complete Architecture Documentation](../diagrams/server/actors/actor-system-complete.md)
- [VisionFlow GPU CUDA Architecture - Complete Technical Documentation](../diagrams/infrastructure/gpu/cuda-architecture-complete.md)
- [Server Architecture](../concepts/architecture/core/server.md)

## Conclusion

This adaptation plan provides a comprehensive roadmap for integrating X-FluxAgent patterns into the ComfyUI MCP skill. Key adaptations:

1. **ChatBotService** → Z.AI-backed MCP tool for cost-effective Claude API calls
2. **AICodeGenNode** → AI-powered workflow generation with full context
3. **RichTextWidget/ChatBotTab** → Playwright-based markdown rendering with syntax highlighting
4. **HotReload** → Development-time module hot reloading for rapid iteration
5. **AnyType** → Flexible Zod schemas for maximum parameter flexibility

All patterns preserve the original X-FluxAgent architecture while adapting to MCP server constraints and leveraging Turbo Flow Claude's unique infrastructure (Z.AI service, multi-user isolation, Playwright automation).

**Next Steps**:
1. Implement Phase 1 (Core Service Adaptation) ✅
2. Create unit tests for chat-service.ts
3. Begin Phase 2 (Workflow Generation)
4. Prototype Playwright UI integration
5. Develop hot reload scripts for development workflow

---

**Document Version**: 1.0
**Last Updated**: 2025-12-03
**Author**: Implementation Coder Agent
**Status**: Ready for Implementation
