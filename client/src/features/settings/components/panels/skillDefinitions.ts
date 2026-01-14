/**
 * Skill definitions extracted from multi-agent-docker/skills/
 * These represent the available skills for explicit invocation.
 */

export interface SkillDefinition {
  id: string;
  name: string;
  version: string;
  description: string;
  tags: string[];
  mcpServer: boolean;
  icon: string;
  category: 'ai-generation' | 'development' | 'infrastructure' | 'document' | 'testing' | 'analysis' | 'media';
}

export const skillDefinitions: SkillDefinition[] = [
  // AI Generation
  {
    id: 'comfyui',
    name: 'ComfyUI',
    version: '1.0.0',
    description: 'AI image/video generation with FLUX, Stable Diffusion, and distributed GPU compute',
    tags: ['image-generation', 'video', 'stable-diffusion', 'flux', 'gpu'],
    mcpServer: true,
    icon: '🎨',
    category: 'ai-generation'
  },
  {
    id: 'comfyui-3d',
    name: 'ComfyUI 3D',
    version: '1.0.0',
    description: 'Text-to-3D model generation using FLUX2 and SAM3D reconstruction',
    tags: ['3d', 'mesh', 'texture', 'sam3d'],
    mcpServer: false,
    icon: '🧊',
    category: 'ai-generation'
  },
  {
    id: 'blender',
    name: 'Blender',
    version: '1.0.0',
    description: 'Blender 5.x MCP with 52+ tools for 3D modeling, materials, physics, animation',
    tags: ['3d', 'modeling', 'rendering', 'animation'],
    mcpServer: true,
    icon: '🎬',
    category: 'ai-generation'
  },
  {
    id: 'algorithmic-art',
    name: 'Algorithmic Art',
    version: '1.0.0',
    description: 'Create generative art using p5.js with seeded randomness and flow fields',
    tags: ['art', 'p5js', 'generative', 'creative-coding'],
    mcpServer: false,
    icon: '🌀',
    category: 'ai-generation'
  },

  // Development
  {
    id: 'cuda',
    name: 'CUDA',
    version: '1.0.0',
    description: 'AI-powered CUDA development with kernel optimization and GPU profiling',
    tags: ['cuda', 'gpu', 'nvidia', 'parallel-computing'],
    mcpServer: true,
    icon: '⚡',
    category: 'development'
  },
  {
    id: 'rust-development',
    name: 'Rust Development',
    version: '1.0.0',
    description: 'Complete Rust toolchain with cargo, rustfmt, clippy, and WASM support',
    tags: ['rust', 'cargo', 'wasm', 'systems'],
    mcpServer: true,
    icon: '🦀',
    category: 'development'
  },
  {
    id: 'pytorch-ml',
    name: 'PyTorch ML',
    version: '1.0.0',
    description: 'Deep learning with PyTorch - model training, GPU acceleration, data science',
    tags: ['pytorch', 'ml', 'deep-learning', 'neural-networks'],
    mcpServer: true,
    icon: '🔥',
    category: 'development'
  },
  {
    id: 'frontend-creator',
    name: 'Frontend Creator',
    version: '2.0.0',
    description: 'Production-grade React + TypeScript + Vite + Tailwind + shadcn/ui interfaces',
    tags: ['react', 'typescript', 'frontend', 'ui'],
    mcpServer: false,
    icon: '🖼️',
    category: 'development'
  },
  {
    id: 'mcp-builder',
    name: 'MCP Builder',
    version: '1.0.0',
    description: 'Create MCP servers with tools, resources, and prompts',
    tags: ['mcp', 'server', 'tools', 'integration'],
    mcpServer: true,
    icon: '🔌',
    category: 'development'
  },
  {
    id: 'skill-creator',
    name: 'Skill Creator',
    version: '1.0.0',
    description: 'Create new Claude Code skills with YAML frontmatter and proper structure',
    tags: ['skills', 'claude-code', 'templates'],
    mcpServer: false,
    icon: '🛠️',
    category: 'development'
  },

  // Testing
  {
    id: 'agentic-qe',
    name: 'Agentic QE',
    version: '2.8.2',
    description: 'AI-powered QE fleet with 20 agents, 46 skills, 100 MCP tools, self-learning',
    tags: ['testing', 'qa', 'tdd', 'coverage', 'security'],
    mcpServer: true,
    icon: '🧪',
    category: 'testing'
  },
  {
    id: 'playwright',
    name: 'Playwright',
    version: '1.0.0',
    description: 'Browser automation, visual testing, and web scraping on Display :1',
    tags: ['browser', 'e2e', 'automation', 'testing'],
    mcpServer: true,
    icon: '🎭',
    category: 'testing'
  },
  {
    id: 'webapp-testing',
    name: 'WebApp Testing',
    version: '1.0.0',
    description: 'Comprehensive web application testing workflows',
    tags: ['testing', 'e2e', 'integration', 'web'],
    mcpServer: false,
    icon: '🔍',
    category: 'testing'
  },

  // Infrastructure
  {
    id: 'docker-manager',
    name: 'Docker Manager',
    version: '1.0.0',
    description: 'Manage containers via Docker API and launch.sh wrapper',
    tags: ['docker', 'containers', 'orchestration'],
    mcpServer: true,
    icon: '🐳',
    category: 'infrastructure'
  },
  {
    id: 'docker-orchestrator',
    name: 'Docker Orchestrator',
    version: '2.0.0',
    description: 'Container orchestration using Python SDK - logs, inspect, network mapping',
    tags: ['docker', 'compose', 'networking'],
    mcpServer: true,
    icon: '🚢',
    category: 'infrastructure'
  },
  {
    id: 'kubernetes-ops',
    name: 'Kubernetes Ops',
    version: '1.0.0',
    description: 'K8s cluster operations - pods, deployments, services via Python client',
    tags: ['kubernetes', 'k8s', 'cluster', 'pods'],
    mcpServer: true,
    icon: '☸️',
    category: 'infrastructure'
  },
  {
    id: 'infrastructure-manager',
    name: 'Infrastructure Manager',
    version: '1.0.0',
    description: 'IaC with Ansible, Terraform, Pulumi - unified management with safety controls',
    tags: ['ansible', 'terraform', 'pulumi', 'iac'],
    mcpServer: true,
    icon: '🏗️',
    category: 'infrastructure'
  },
  {
    id: 'grafana-monitor',
    name: 'Grafana Monitor',
    version: '1.0.0',
    description: 'Observability stack - Grafana dashboards, Prometheus metrics, Loki logs',
    tags: ['grafana', 'prometheus', 'loki', 'observability'],
    mcpServer: true,
    icon: '📊',
    category: 'infrastructure'
  },
  {
    id: 'linux-admin',
    name: 'Linux Admin',
    version: '1.0.0',
    description: 'Linux system administration and management tasks',
    tags: ['linux', 'sysadmin', 'shell', 'ops'],
    mcpServer: true,
    icon: '🐧',
    category: 'infrastructure'
  },
  {
    id: 'tmux-ops',
    name: 'Tmux Ops',
    version: '1.0.0',
    description: 'Terminal multiplexer operations and session management',
    tags: ['tmux', 'terminal', 'sessions'],
    mcpServer: true,
    icon: '💻',
    category: 'infrastructure'
  },

  // Document Processing
  {
    id: 'docx',
    name: 'DOCX',
    version: '1.0.0',
    description: 'Word document creation, editing, tracked changes, and text extraction',
    tags: ['word', 'documents', 'office'],
    mcpServer: false,
    icon: '📄',
    category: 'document'
  },
  {
    id: 'pdf',
    name: 'PDF',
    version: '1.0.0',
    description: 'PDF generation, manipulation, and text extraction',
    tags: ['pdf', 'documents', 'export'],
    mcpServer: true,
    icon: '📑',
    category: 'document'
  },
  {
    id: 'xlsx',
    name: 'XLSX',
    version: '1.0.0',
    description: 'Excel spreadsheet operations and data manipulation',
    tags: ['excel', 'spreadsheet', 'data'],
    mcpServer: false,
    icon: '📊',
    category: 'document'
  },
  {
    id: 'pptx',
    name: 'PPTX',
    version: '1.0.0',
    description: 'PowerPoint presentation creation and editing',
    tags: ['powerpoint', 'presentations', 'slides'],
    mcpServer: false,
    icon: '📽️',
    category: 'document'
  },
  {
    id: 'latex-documents',
    name: 'LaTeX Documents',
    version: '1.0.0',
    description: 'Professional document preparation - papers, presentations, technical docs',
    tags: ['latex', 'academic', 'typesetting'],
    mcpServer: true,
    icon: '📐',
    category: 'document'
  },
  {
    id: 'jupyter-notebooks',
    name: 'Jupyter Notebooks',
    version: '1.0.0',
    description: 'Notebook operations - create, execute, analyze with cell manipulation',
    tags: ['jupyter', 'notebooks', 'data-science'],
    mcpServer: true,
    icon: '📓',
    category: 'document'
  },
  {
    id: 'docs-alignment',
    name: 'Docs Alignment',
    version: '1.0.0',
    description: 'Enterprise documentation validation with 15-agent swarm, Diataxis framework',
    tags: ['documentation', 'validation', 'diataxis'],
    mcpServer: false,
    icon: '📚',
    category: 'document'
  },

  // Analysis & Research
  {
    id: 'perplexity',
    name: 'Perplexity',
    version: '1.0.0',
    description: 'Real-time web research with source citations and UK-centric prompts',
    tags: ['research', 'web-search', 'citations'],
    mcpServer: true,
    icon: '🔎',
    category: 'analysis'
  },
  {
    id: 'deepseek-reasoning',
    name: 'DeepSeek Reasoning',
    version: '2.0.0',
    description: 'Advanced multi-step reasoning via MCP with Chain-of-Thought outputs',
    tags: ['reasoning', 'ai', 'chain-of-thought'],
    mcpServer: true,
    icon: '🧠',
    category: 'analysis'
  },
  {
    id: 'network-analysis',
    name: 'Network Analysis',
    version: '1.0.0',
    description: 'Network topology analysis and visualization',
    tags: ['networks', 'graph', 'topology'],
    mcpServer: true,
    icon: '🌐',
    category: 'analysis'
  },
  {
    id: 'git-architect',
    name: 'Git Architect',
    version: '1.0.0',
    description: 'High-level repo management - semantic search, smart diffs, architecture maps',
    tags: ['git', 'repository', 'codebase-analysis'],
    mcpServer: true,
    icon: '🏛️',
    category: 'analysis'
  },
  {
    id: 'ontology-core',
    name: 'Ontology Core',
    version: '1.0.0',
    description: 'OWL2 ontology management and reasoning',
    tags: ['ontology', 'owl2', 'knowledge-graph'],
    mcpServer: true,
    icon: '🔗',
    category: 'analysis'
  },
  {
    id: 'wardley-maps',
    name: 'Wardley Maps',
    version: '1.0.0',
    description: 'Strategic planning with Wardley mapping methodology',
    tags: ['strategy', 'planning', 'wardley'],
    mcpServer: false,
    icon: '🗺️',
    category: 'analysis'
  },

  // Media Processing
  {
    id: 'ffmpeg-processing',
    name: 'FFmpeg Processing',
    version: '1.0.0',
    description: 'Professional video/audio processing - transcode, edit, stream, analyze',
    tags: ['video', 'audio', 'transcoding', 'streaming'],
    mcpServer: true,
    icon: '🎥',
    category: 'media'
  },
  {
    id: 'imagemagick',
    name: 'ImageMagick',
    version: '2.0.0',
    description: 'Image processing - format conversion, resize, crop, filter, batch ops',
    tags: ['images', 'conversion', 'batch-processing'],
    mcpServer: true,
    icon: '🖼️',
    category: 'media'
  },
  {
    id: 'web-summary',
    name: 'Web Summary',
    version: '1.0.0',
    description: 'Summarize web content including YouTube with semantic topic links',
    tags: ['summarization', 'web', 'youtube'],
    mcpServer: true,
    icon: '📰',
    category: 'media'
  },
  {
    id: 'chrome-devtools',
    name: 'Chrome DevTools',
    version: '1.0.0',
    description: 'Debug web pages with DevTools - performance, network, console, DOM',
    tags: ['chrome', 'debugging', 'performance'],
    mcpServer: true,
    icon: '🔧',
    category: 'media'
  },
  {
    id: 'qgis',
    name: 'QGIS',
    version: '1.0.0',
    description: 'Geospatial analysis - distances, buffers, transforms, layer operations',
    tags: ['gis', 'geospatial', 'mapping'],
    mcpServer: true,
    icon: '🗺️',
    category: 'media'
  },
  {
    id: 'kicad',
    name: 'KiCad',
    version: '2.0.0',
    description: 'Electronic circuit design - schematics, PCB layout, Gerber export',
    tags: ['pcb', 'electronics', 'circuits'],
    mcpServer: true,
    icon: '⚡',
    category: 'media'
  },
  {
    id: 'ngspice',
    name: 'NGSpice',
    version: '1.0.0',
    description: 'Circuit simulation and analysis',
    tags: ['spice', 'simulation', 'electronics'],
    mcpServer: true,
    icon: '📈',
    category: 'media'
  },

  // Learning & Training
  {
    id: 'agentic-lightning',
    name: 'Agentic Lightning',
    version: '1.0.0',
    description: 'RL training with AgentDB + RuVector for self-improving agents',
    tags: ['rl', 'reinforcement-learning', 'agent-training'],
    mcpServer: true,
    icon: '⚡',
    category: 'analysis'
  }
];

// Group skills by category
export function getSkillsByCategory(): Record<string, SkillDefinition[]> {
  const categories: Record<string, SkillDefinition[]> = {};

  for (const skill of skillDefinitions) {
    if (!categories[skill.category]) {
      categories[skill.category] = [];
    }
    categories[skill.category].push(skill);
  }

  return categories;
}

// Category display names
export const categoryLabels: Record<string, string> = {
  'ai-generation': 'AI Generation',
  'development': 'Development',
  'testing': 'Testing & QE',
  'infrastructure': 'Infrastructure',
  'document': 'Documents',
  'analysis': 'Analysis & Research',
  'media': 'Media Processing'
};

// Category icons
export const categoryIcons: Record<string, string> = {
  'ai-generation': '🎨',
  'development': '💻',
  'testing': '🧪',
  'infrastructure': '🏗️',
  'document': '📄',
  'analysis': '🔍',
  'media': '🎬'
};
