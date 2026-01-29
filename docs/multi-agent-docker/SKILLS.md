---
title: Skills Documentation
description: The Turbo Flow Claude multi-agent environment includes **54 specialized skills** that extend Claude Code's capabilities across AI/ML, DevOps, graphics, system administration, and development workfl...
category: how-to
tags:
  - architecture
  - design
  - patterns
  - structure
  - api
related-docs:
  - multi-agent-docker/ANTIGRAVITY.md
  - multi-agent-docker/TERMINAL_GRID.md
  - multi-agent-docker/comfyui-sam3d-setup.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Docker installation
  - Rust toolchain
---

# Skills Documentation

## Overview

The Turbo Flow Claude multi-agent environment includes **54 specialized skills** that extend Claude Code's capabilities across AI/ML, DevOps, graphics, system administration, and development workflows. Skills provide pre-configured tools, templates, and integrations that enable rapid development across diverse domains.

**Key Features**:
- Auto-discovery by Claude Flow and Claude Code
- MCP (Model Context Protocol) integration for 15+ skills
- Seamless integration with 610+ agent templates
- File extension and keyword-based auto-invocation
- AgentDB pattern learning for optimized skill selection

## Skills System Architecture

### Skill Location
All skills are installed in `/home/devuser/.claude/skills/` (or `~/.claude/skills/` for DevPod mode).

### Skill Structure
```
skills/
├── skill-name/
│   ├── SKILL.md                # Skill documentation
│   ├── mcp-server/            # MCP server (if applicable)
│   ├── tools/                 # Command-line tools
│   ├── examples/              # Usage examples
│   ├── config/                # Configuration files
│   └── tests/                 # Test suite
```

### MCP Integration
15+ skills provide MCP servers for autonomous tool usage:
- **CUDA**: GPU development tools
- **Blender**: 52 3D modeling tools via WebSocket
- **ComfyUI**: AI image/video generation workflows
- **Playwright**: Browser automation
- **QGIS**: Geospatial analysis
- **Perplexity**: Real-time web research
- **DeepSeek Reasoning**: Advanced reasoning chains
- **ImageMagick**: Image processing automation
- Plus 7 additional MCP-enabled skills

## Complete Skills Reference

### AI & Machine Learning (5 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **cuda** | **AI-powered CUDA development assistant** with 4 specialist agents (General, Optimizer, Debugger, Analyzer), kernel compilation, and GPU profiling | Yes | CUDA kernel development, GPU optimization, parallel computing, nvcc compilation, performance profiling |
| **pytorch-ml** | Deep learning with PyTorch, CUDA GPU acceleration, model training, and data science workflows | No | Neural networks, computer vision, NLP, transfer learning, model deployment |
| **agentic-lightning** | Reinforcement learning with AgentDB + RuVector (9 algorithms: Q-Learning, SARSA, PPO, DQN, A3C, etc.) | Yes | Training agents, pattern learning, experience replay, transfer learning, GNN models |
| **agentic-qe** | Quality engineering fleet with 20 agents and 46 QE skills for comprehensive testing | Yes | Test generation, TDD, coverage analysis, parallel testing, security scanning, quality gates |
| **deepseek-reasoning** | DeepSeek R1 reasoning integration for complex problem-solving | Yes | Mathematical proofs, multi-step reasoning, chain-of-thought analysis |

### 3D Graphics & Visualization (5 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **blender** | Blender 5.x integration with 52 tools for 3D modeling, materials, physics, and animation via WebSocket | Yes | 3D modeling, rendering, animation, physics simulation, material creation |
| **comfyui** | AI image/video generation with Stable Diffusion, FLUX, AnimateDiff, and Salad Cloud deployment | Yes | Text-to-image, image-to-image, video generation, AI workflows, batch processing |
| **comfyui-3d** | 3D asset generation from text/images using SAM3D and FLUX2 pipelines | Yes | AI-powered 3D object creation, mesh generation, texture baking |
| **imagemagick** | Image processing and manipulation via command-line tools | Yes | Batch conversion, effects, resizing, format transformation |
| **algorithmic-art** | Generative art creation with procedural graphics and patterns | No | SVG generation, algorithmic patterns, procedural graphics |

### DevOps & Infrastructure (7 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **kubernetes-ops** | Kubernetes cluster management with kubectl and helm | No | Deploy pods, manage clusters, helm charts, K8s debugging |
| **docker-orchestrator** | Docker Swarm and Compose orchestration | No | Multi-container apps, swarm mode, service scaling |
| **docker-manager** | Docker container lifecycle management | No | Build images, manage containers, inspect logs |
| **infrastructure-manager** | Terraform and Ansible infrastructure-as-code | No | Provision cloud resources, configuration management |
| **grafana-monitor** | Grafana and Prometheus monitoring and alerting | No | Create dashboards, configure alerts, metrics visualization |
| **network-analysis** | Network diagnostics with tcpdump, wireshark, and traffic analysis | No | Packet capture, traffic analysis, network troubleshooting |
| **git-architect** | Advanced Git workflows and repository management | No | Complex branching, rebasing, conflict resolution |

### System Administration (4 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **linux-admin** | CachyOS/Arch Linux system administration | No | Package management (pacman/yay), systemd, user management |
| **tmux-ops** | tmux session and workspace management | No | Create sessions, scripted layouts, terminal multiplexing |
| **text-processing** | Text manipulation with awk, sed, jq, and grep | No | Log parsing, data transformation, JSON processing |

### Document Processing (5 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **pdf** | PDF creation, manipulation, and generation | No | Generate reports, merge/split PDFs, form filling |
| **docx** | Microsoft Word document generation | No | Create formatted Word documents programmatically |
| **xlsx** | Excel spreadsheet operations and data export | No | Complex spreadsheets, data analysis, charting |
| **pptx** | PowerPoint presentation generation | No | Automated slide decks, business presentations |
| **latex-documents** | LaTeX document compilation for academic papers | No | Academic papers, technical documentation, math typesetting |

### Web Development & Testing (5 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **playwright** | Browser automation and E2E testing with Playwright | Yes | Web scraping, E2E testing, browser automation |
| **chrome-devtools** | Chrome DevTools Protocol integration | No | Web debugging, performance profiling, network analysis |
| **frontend-creator** | React, Vue, Svelte project scaffolding | No | Generate frontend projects, component libraries |
| **webapp-testing** | Web application functional and accessibility testing | No | Functional testing, a11y audits, visual regression |
| **mcp-builder** | Build custom MCP servers and integrations | No | Create MCP tools, extend Claude capabilities |

### Research & Knowledge (8 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **perplexity** | Perplexity AI real-time web research with citations | Yes | Current events, web research, fact-checking with sources |
| **web-summary** | YouTube transcript extraction and web content summarization | No | Summarize videos, articles, long-form content |
| **ontology-core** | Knowledge graph operations and semantic relationships | No | Build ontologies, manage knowledge graphs |
| **ontology-enrich** | Ontology enrichment and expansion | No | Expand knowledge graphs, add semantic data |
| **import-to-ontology** | Import structured data into ontologies | No | Convert CSV/JSON to knowledge graphs |
| **logseq-formatted** | Logseq knowledge base formatting | No | Generate Logseq-compatible notes |
| **wardley-maps** | Wardley mapping for strategic analysis | No | Strategic planning, value chain mapping |
| **docs-alignment** | Documentation verification against codebase | No | Validate docs match code, link checking, diagram verification |

### Engineering & Electronics (3 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **qgis** | QGIS geospatial analysis and mapping | Yes | Geographic data, GIS operations, spatial analysis |
| **kicad** | KiCad PCB design and electronic schematics | No | Electronics design, PCB layout, circuit boards |
| **ngspice** | SPICE circuit simulation and analysis | No | Circuit simulation, electronics analysis |

### Media Processing (1 Skill)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **ffmpeg-processing** | Video and audio processing, transcoding, and streaming | No | Video transcoding, audio extraction, streaming setup |

### Development Tools (3 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **rust-development** | Rust toolchain with cargo, clippy, rustfmt, and WASM | No | Rust projects, systems programming, WASM compilation |
| **jupyter-notebooks** | Jupyter notebook creation and execution | No | Data science, interactive Python, research |
| **skill-creator** | Create new Claude Code skills from templates | No | Develop custom skills, extend capabilities |

### Design & Communication (5 Skills)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **brand-guidelines** | Brand identity management and style guides | No | Generate brand assets, style guides |
| **canvas-design** | Visual design and layout generation | No | Create graphics, design layouts |
| **theme-factory** | UI theme generation with color schemes | No | Design tokens, color palettes, CSS themes |
| **slack-gif-creator** | Animated GIF creation for communications | No | Communication assets, team memes |
| **internal-comms** | Internal communication drafting | No | Announcements, team updates, newsletters |

### Anthropic Templates (1 Skill)

| Skill | Description | MCP | When to Use |
|-------|-------------|-----|-------------|
| **anthropic-examples-and-templates** | Official Anthropic examples and templates | No | Learn Claude best practices, example workflows |

## Skill Usage with Claude Code

### Direct Invocation

```bash
# Direct skill usage in prompts
claude "Use cuda skill to compile kernel.cu for RTX 4090"
claude "Use pytorch-ml to train a ResNet50 model on my dataset"
claude "Use blender to create a 3D cube with physics simulation"
```

### Claude Flow Integration

Claude Flow automatically selects appropriate skills based on:

1. **File Extensions**: `.cu` → cuda, `.blend` → blender, `.qgs` → qgis
2. **Keywords**: "test coverage" → agentic-qe, "train model" → pytorch-ml
3. **Task Type**: "kubernetes deploy" → kubernetes-ops
4. **Memory Patterns**: AgentDB learns successful skill combinations

```bash
# Skills auto-invoked during workflows
npx claude-flow@alpha sparc run dev "optimize CUDA kernel performance"
# → Automatically uses cuda skill

cf-swarm "generate comprehensive test suite"
# → Automatically uses agentic-qe skill

cf-hive "create 3D product visualization"
# → Automatically uses blender + comfyui skills
```

### MCP Tool Access

Skills with MCP servers expose tools for autonomous usage:

```javascript
// Example: CUDA skill MCP tools
{
  "tool": "cuda_compile",
  "args": {
    "source_file": "kernel.cu",
    "auto_arch": true,
    "optimization_level": "O3"
  }
}

// Example: Blender skill MCP tools
{
  "tool": "create_primitive",
  "params": {
    "type": "cube",
    "size": 2
  }
}

// Example: ComfyUI skill MCP tools
{
  "tool": "image_generate",
  "params": {
    "prompt": "futuristic cityscape at sunset",
    "width": 1024,
    "height": 1024
  }
}
```

## Featured Skill: CUDA Development

The **cuda** skill is the premier CUDA development environment with AI-powered assistance.

### 4 Specialist Agents

1. **General Assistant** - CUDA questions, kernel creation, learning
2. **Optimizer Agent** - Performance optimization, memory coalescing, shared memory
3. **Debugger Agent** - Compilation errors, race conditions, memory issues
4. **Analyzer Agent** - Code review, best practices, complexity analysis

### Available Tools

**Kernel Development**:
- `cuda_create_kernel` - Generate optimized kernels from specs
- `cuda_compile` - Compile with nvcc and auto-architecture detection
- `cuda_analyze` - Deep analysis for optimization opportunities
- `cuda_read_kernel` / `cuda_write_kernel` - File operations

**GPU Management**:
- `cuda_gpu_status` - Get GPU info via nvidia-smi
- `cuda_detect_arch` - Auto-detect compute capability
- `cuda_profile` - Profile kernel execution
- `cuda_benchmark` - Performance benchmarking

**Agent Routing**:
- `cuda_route_query` - Route to appropriate specialist
- `cuda_general_assist` - General CUDA assistance
- `cuda_optimize_code` - Optimization specialist
- `cuda_debug_code` - Debugging specialist
- `cuda_analyze_quality` - Code analysis specialist

### Environment

The container includes:
- **CUDA Toolkit**: 13.0
- **GPUs Available**: 3x (RTX A6000 48GB, 2x Quadro RTX 6000 24GB)
- **PyTorch CUDA**: 2.9.1+cu128
- **nvcc Location**: `/opt/cuda/bin/nvcc`

### Usage Examples

```bash
# Create optimized kernel
claude "Use cuda to create a matrix transpose kernel with shared memory tiling"

# Optimize existing code
cuda_optimize --file matmul.cu --target-gpu rtx4090

# Debug kernel
cuda_debug --file buggy_kernel.cu --error "incorrect results for large arrays"

# Analyze quality
cuda_analyze --file my_kernel.cu --report-format markdown
```

### Performance Optimization Checklist

The analyzer checks for:
1. **Memory Access Patterns** - Coalescing, bank conflicts, strided access
2. **Occupancy** - Register usage, shared memory, thread block size
3. **Synchronization** - `__syncthreads()` overhead, warp divergence, race conditions
4. **Arithmetic Intensity** - Compute-to-memory ratio, loop unrolling, vectorization

### Integration with Other Skills

- **pytorch-ml** - Train models, optimize bottlenecks with custom kernels
- **rust-development** - Write safe CUDA kernel wrappers in Rust
- **docker-orchestrator** - Deploy GPU containers
- **infrastructure-manager** - Provision GPU cloud instances

## Skill Development

To create new skills, use the **skill-creator** skill:

```bash
# Generate new skill template
claude "Use skill-creator to generate a new skill for PostgreSQL database management"

# Skill structure
~/.claude/skills/my-skill/
├── SKILL.md              # Documentation (required)
├── tools/                # Command-line tools
├── mcp-server/          # MCP server (optional)
│   └── server.js
├── examples/            # Usage examples
└── tests/               # Test suite
```

### SKILL.md Format

```markdown
---
skill: my-skill
name: My Skill
version: 1.0.0
description: Brief description
tags: [tag1, tag2, tag3]
mcp_server: true/false
entry_point: mcp-server/server.js (if MCP enabled)
---

# My Skill

Description and usage guide...
```

## Best Practices

1. **Skill Discovery**: Let Claude Flow auto-select skills when possible
2. **MCP Tools**: Use MCP tools for autonomous operations
3. **Skill Chaining**: Combine skills (e.g., comfyui → blender for 3D workflows)
4. **Pattern Learning**: AgentDB learns successful skill combinations over time
5. **Verification**: Use agentic-qe for testing skill integrations

## Configuration

### Enable Specific Skills

Skills are auto-loaded from `~/.claude/skills/`. To disable a skill, move it out of the directory:

```bash
mv ~/.claude/skills/unwanted-skill /tmp/
```

### MCP Server Configuration

For skills with MCP servers, configure in `~/.config/claude/mcp.json`:

```json
{
  "mcpServers": {
    "cuda": {
      "command": "python",
      "args": ["/home/devuser/.claude/skills/cuda/mcp-server/server.py"]
    },
    "blender": {
      "command": "node",
      "args": ["/home/devuser/.claude/skills/blender/mcp-server/server.js"]
    }
  }
}
```

## Performance Metrics

Skills contribute to Claude Flow's performance improvements:
- **84.8%** SWE-Bench solve rate
- **32.3%** token reduction (through efficient skill usage)
- **2.8-4.4x** speed improvement (skill-based parallelization)
- **>95%** integration success rate

## Troubleshooting

### Skill Not Found

```bash
# Check skill installation
ls ~/.claude/skills/

# Verify skill structure
cat ~/.claude/skills/cuda/SKILL.md
```

### MCP Server Not Starting

```bash
# Check MCP configuration
cat ~/.config/claude/mcp.json

# Test MCP server directly
python ~/.claude/skills/cuda/mcp-server/server.py
```

### Skill Auto-Selection Issues

```bash
# View AgentDB patterns for skill selection
npx claude-flow@alpha memory search "skill-selection"

# Manually specify skill
claude "Use cuda skill (not pytorch-ml) to compile this kernel"
```

## Resources

- **Skill Templates**: `/home/devuser/.claude/skills/skill-creator/`
- **Agent Library**: `/home/devuser/agents/` (610+ templates)
- **SPARC Methodology**: `cat agents/doc-planner.md`
- **Claude Flow**: https://github.com/ruvnet/claude-flow
- **MCP Documentation**: https://github.com/anthropic/mcp

---

## Related Documentation

- [Working with Agents](../archive/docs/guides/user/working-with-agents.md)
- [Testing Guide](../archive/docs/guides/developer/05-testing-guide.md)
- [Adding Features](../guides/developer/04-adding-features.md)
- [ComfyUI SAM3D Docker Setup](comfyui-sam3d-setup.md)
- [RAGFlow Knowledge Management Integration](../guides/ai-models/ragflow-integration.md)

## Summary

The 54 skills in the Turbo Flow Claude environment provide comprehensive coverage across:
- **AI/ML**: 5 skills (cuda, pytorch-ml, agentic-lightning, agentic-qe, deepseek-reasoning)
- **3D Graphics**: 5 skills (blender, comfyui, comfyui-3d, imagemagick, algorithmic-art)
- **DevOps**: 7 skills (kubernetes-ops, docker-orchestrator, infrastructure-manager, etc.)
- **System Admin**: 4 skills (linux-admin, tmux-ops, text-processing, git-architect)
- **Documents**: 5 skills (pdf, docx, xlsx, pptx, latex-documents)
- **Web Dev**: 5 skills (playwright, chrome-devtools, frontend-creator, etc.)
- **Research**: 8 skills (perplexity, web-summary, ontology-core, wardley-maps, etc.)
- **Engineering**: 3 skills (qgis, kicad, ngspice)
- **Media**: 1 skill (ffmpeg-processing)
- **Development**: 3 skills (rust-development, jupyter-notebooks, skill-creator)
- **Design**: 5 skills (brand-guidelines, canvas-design, theme-factory, etc.)
- **Templates**: 1 skill (anthropic-examples-and-templates)

With 15+ MCP-enabled skills, auto-discovery, and AgentDB pattern learning, the skills system provides a powerful foundation for rapid development across all domains.
