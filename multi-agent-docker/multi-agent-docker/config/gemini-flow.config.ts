// Gemini-Flow Production Configuration for CachyOS Workstation
// Enterprise-grade AI orchestration with A2A + MCP dual protocol support

export default {
  // ===== Protocol Configuration =====
  protocols: {
    // Agent-to-Agent (A2A) Protocol
    a2a: {
      enabled: true,
      messageTimeout: 5000,
      retryAttempts: 3,
      encryption: 'AES-256-GCM',
      healthChecks: true,
      consensusMechanism: 'byzantine-fault-tolerant',
      faultTolerance: 0.33, // Tolerates up to 33% compromised agents
      routing: {
        algorithm: 'weighted-expertise',
        latencyOptimization: true,
        loadBalancing: 'intelligent'
      }
    },

    // Model Context Protocol (MCP)
    mcp: {
      enabled: true,
      contextSyncInterval: 100,
      modelCoordination: 'intelligent',
      fallbackStrategy: 'round-robin',
      sharedMemory: true,
      streaming: true,
      toolCalling: true
    }
  },

  // ===== Swarm Configuration =====
  swarm: {
    maxAgents: 66,
    topology: 'hierarchical', // hierarchical | mesh | hub-spoke
    consensus: 'byzantine-fault-tolerant',
    coordinationProtocol: 'a2a',

    // Agent specializations (66 total)
    specializations: {
      'system-architect': 5,
      'master-coder': 12,
      'research-scientist': 8,
      'data-analyst': 10,
      'strategic-planner': 6,
      'security-expert': 5,
      'performance-optimizer': 8,
      'documentation-writer': 4,
      'qa-specialist': 4,
      'devops-engineer': 4
    },

    // Spawn configuration
    spawn: {
      strategy: 'on-demand', // on-demand | pre-allocated | hybrid
      warmupTime: 100, // ms
      pooling: {
        enabled: true,
        minIdle: 5,
        maxIdle: 20
      }
    }
  },

  // ===== Performance Targets =====
  performance: {
    // Core system
    sqliteOps: 396610, // ops/sec
    agentSpawnTime: 100, // ms
    routingLatency: 75, // ms
    memoryPerAgent: 4.2, // MB
    parallelTasks: 10000,

    // A2A protocol
    a2aLatency: 25, // ms (target: <50ms)
    consensusSpeed: 2400, // ms for 1000 nodes
    messageThroughput: 50000, // msgs/sec
    faultRecovery: 500, // ms

    // Resource limits
    maxConcurrentRequests: 100,
    requestTimeout: 60000,
    maxRetries: 3
  },

  // ===== Google AI Services Integration =====
  google: {
    projectId: process.env.GOOGLE_CLOUD_PROJECT || '',
    credentials: process.env.GOOGLE_APPLICATION_CREDENTIALS || '',

    services: {
      // Veo3: Video Generation
      veo3: {
        enabled: true,
        quota: 'enterprise',
        quality: '4K',
        fps: 60,
        processingTimeout: 300000, // 5 minutes
        caching: true
      },

      // Imagen4: Image Generation
      imagen4: {
        enabled: true,
        quota: 'enterprise',
        quality: 'ultra-high',
        batchProcessing: true,
        styleConsistency: true
      },

      // Lyria: Music Composition
      lyria: {
        enabled: true,
        quota: 'enterprise',
        qualityScore: 92, // musician approval rate
        caching: true
      },

      // Chirp: Speech Synthesis
      chirp: {
        enabled: true,
        quota: 'enterprise',
        realtime: true,
        naturalness: 96, // % score
        languages: ['en-US', 'en-GB', 'es-ES', 'fr-FR', 'de-DE']
      },

      // Co-Scientist: Research Automation
      'co-scientist': {
        enabled: true,
        quota: 'enterprise',
        papersPerHour: 840,
        validationSuccess: 94, // %
        citationTracking: true
      },

      // Project Mariner: Browser Automation
      mariner: {
        enabled: true,
        quota: 'enterprise',
        taskCompletion: 98.4, // %
        dataExtraction: true,
        browserPool: 10
      },

      // AgentSpace: Agent Coordination
      agentspace: {
        enabled: true,
        quota: 'enterprise',
        maxConcurrentAgents: 10000,
        coordinationLatency: 15, // ms
        taskSuccess: 97.2 // %
      },

      // Multi-modal Streaming
      streaming: {
        enabled: true,
        quota: 'enterprise',
        endToEndLatency: 45, // ms
        accuracy: 98.7, // %
        throughput: 15000000 // ops/sec
      }
    },

    // Service optimization
    optimization: {
      costPerformance: 'balanced', // aggressive | balanced | quality
      caching: {
        enabled: true,
        ttl: 3600, // seconds
        maxSize: '10GB'
      },
      batchProcessing: {
        enabled: true,
        maxBatchSize: 100,
        timeout: 5000
      }
    }
  },

  // ===== Monitoring & Observability =====
  monitoring: {
    enabled: true,
    metricsEndpoint: process.env.METRICS_ENDPOINT || 'http://localhost:9090',

    dashboards: {
      performance: true,
      agents: true,
      costs: true,
      protocols: true,
      services: true
    },

    alerting: {
      enabled: true,
      channels: ['log', 'webhook'],
      thresholds: {
        latency: 100, // ms
        errorRate: 0.05, // 5%
        cpuUsage: 0.8, // 80%
        memoryUsage: 0.9 // 90%
      }
    },

    logging: {
      level: process.env.LOG_LEVEL || 'info',
      format: 'json',
      destination: 'file',
      rotation: {
        enabled: true,
        maxSize: '100MB',
        maxFiles: 10
      }
    }
  },

  // ===== Cost Management =====
  costs: {
    tracking: {
      enabled: true,
      granularity: 'per-service',
      reporting: 'real-time'
    },

    budgets: {
      daily: parseFloat(process.env.DAILY_BUDGET || '50'),
      weekly: parseFloat(process.env.WEEKLY_BUDGET || '250'),
      monthly: parseFloat(process.env.MONTHLY_BUDGET || '1000')
    },

    optimization: {
      autoScale: true,
      preferFreeServices: true, // Prefer Xinference, ONNX when possible
      cachingStrategy: 'aggressive',
      batchWhenPossible: true
    }
  },

  // ===== Security Configuration =====
  security: {
    encryption: {
      atRest: true,
      inTransit: true,
      algorithm: 'AES-256-GCM'
    },

    authentication: {
      required: true,
      method: 'api-key', // api-key | oauth | jwt
      apiKeys: {
        google: process.env.GOOGLE_API_KEY,
        anthropic: process.env.ANTHROPIC_API_KEY,
        openai: process.env.OPENAI_API_KEY
      }
    },

    rateLimit: {
      enabled: true,
      perMinute: 1000,
      perHour: 50000,
      perDay: 1000000
    },

    audit: {
      enabled: true,
      logAllRequests: true,
      retentionDays: 90
    }
  },

  // ===== Network Configuration =====
  network: {
    timeout: 5000,
    retryAttempts: 3,
    keepAlive: true,
    compression: true,
    batchRequests: true,

    // RAGFlow integration
    ragflow: {
      enabled: true,
      network: 'docker_ragflow',
      xinferenceUrl: 'http://172.18.0.11:9997/v1'
    }
  },

  // ===== Development & Testing =====
  development: {
    mode: process.env.NODE_ENV === 'production' ? 'production' : 'development',
    debugging: {
      enabled: process.env.DEBUG === 'true',
      verboseLogging: false,
      tracing: true
    },

    testing: {
      mockServices: false,
      integrationTests: true,
      performanceTests: true
    }
  },

  // ===== Experimental Features =====
  experimental: {
    quantumProcessing: false, // Q1 2025
    planetaryScale: false, // Q2 2025
    edgeComputing: false, // Q4 2025
    multiModal: true,
    streaming: true
  }
};
