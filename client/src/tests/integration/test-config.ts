/**
 * Integration Test Configuration
 * Centralized configuration for integration tests with common setup and utilities
 *
 * NOTE: Tests are prepared but disabled due to security concerns.
 * Re-enable once testing dependencies are secure.
 */

import { vi } from 'vitest';
import { createMockApiClient, createPreConfiguredApiClient } from '../__mocks__/apiClient';
import { createMockWebSocket, MockWebSocketServer } from '../__mocks__/websocket';

// Test configuration constants
export const TEST_CONFIG = {
  API_BASE_URL: 'http://localhost:3001',
  WEBSOCKET_URL: 'ws://localhost:3002',
  TIMEOUTS: {
    SHORT: 1000,
    MEDIUM: 5000,
    LONG: 10000
  },
  RETRY_ATTEMPTS: 3,
  DEBOUNCE_MS: 300
};

// Common test scenarios
export const TEST_SCENARIOS = {
  WORKSPACE_CRUD: {
    CREATE: 'workspace-create',
    UPDATE: 'workspace-update',
    DELETE: 'workspace-delete',
    FAVORITE: 'workspace-favorite',
    ARCHIVE: 'workspace-archive'
  },
  ANALYTICS: {
    STRUCTURAL: 'analytics-structural',
    SEMANTIC: 'analytics-semantic',
    GPU_METRICS: 'analytics-gpu-metrics',
    PROGRESS: 'analytics-progress'
  },
  OPTIMIZATION: {
    LAYOUT: 'optimization-layout',
    CLUSTERING: 'optimization-clustering',
    CANCELLATION: 'optimization-cancel',
    GPU_STATUS: 'optimization-gpu'
  },
  EXPORT: {
    JSON: 'export-json',
    CSV: 'export-csv',
    GRAPHML: 'export-graphml',
    PNG: 'export-png',
    SHARE: 'export-share',
    EMBED: 'export-embed'
  }
};

// Pre-configured API responses for common scenarios
export const API_RESPONSES = {
  // Workspace responses
  WORKSPACE_LIST: {
    data: [
      {
        id: 'ws-1',
        name: 'Test Workspace 1',
        description: 'First test workspace',
        type: 'personal',
        status: 'active',
        memberCount: 1,
        lastAccessed: new Date('2024-09-24T10:00:00Z'),
        createdAt: new Date('2024-09-20T10:00:00Z'),
        favorite: false
      },
      {
        id: 'ws-2',
        name: 'Team Workspace',
        description: 'Collaborative team workspace',
        type: 'team',
        status: 'active',
        memberCount: 5,
        lastAccessed: new Date('2024-09-24T09:00:00Z'),
        createdAt: new Date('2024-09-15T10:00:00Z'),
        favorite: true
      }
    ]
  },

  WORKSPACE_CREATE: (data: any) => ({
    data: {
      id: `ws-${Date.now()}`,
      ...data,
      type: 'personal',
      status: 'active',
      memberCount: 1,
      lastAccessed: new Date(),
      createdAt: new Date(),
      favorite: false
    }
  }),

  // Analytics responses
  ANALYTICS_PARAMS: {
    data: {
      visualAnalytics: {
        enabled: true,
        algorithm: 'stress_majorization',
        iterations: 100,
        convergenceThreshold: 0.01
      },
      gpuAcceleration: {
        enabled: true,
        device: 'CUDA:0',
        memoryUsage: 0.65,
        computeCapability: '8.6'
      }
    }
  },

  ANALYTICS_STATS: {
    data: {
      performance: {
        avgFrameTime: 16.7,
        gpuUtilization: 78.2,
        memoryUsage: 1024 * 1024 * 512,
        activeOperations: 3
      },
      graph: {
        nodeCount: 1500,
        edgeCount: 3200,
        clusterCount: 8,
        centralityMetrics: {
          betweenness: 0.34,
          closeness: 0.62,
          eigenvector: 0.51
        }
      }
    }
  },

  STRUCTURAL_ANALYSIS: {
    data: {
      similarity: { overall: 0.73, structural: 0.68, semantic: 0.78 },
      matches: 127,
      differences: 42,
      clusters: 8,
      centrality: {
        betweenness: 0.34,
        closeness: 0.62,
        eigenvector: 0.51
      },
      communities: [
        { id: 0, size: 45, density: 0.8, modularity: 0.42 },
        { id: 1, size: 32, density: 0.7, modularity: 0.38 }
      ]
    },
    status: 'completed',
    progress: 100
  },

  // Optimization responses
  OPTIMIZATION_RESULT: {
    data: {
      algorithm: 'Adaptive Force-Directed',
      confidence: 0.87,
      performanceGain: 0.34,
      clusters: 8,
      iterations: 245,
      convergenceReached: true,
      executionTime: 12500,
      gpuUtilization: 85.2,
      recommendations: [
        {
          type: 'layout',
          priority: 'high',
          description: 'Adjust node spacing for better clarity',
          confidence: 0.92,
          impact: 0.25
        },
        {
          type: 'clustering',
          priority: 'medium',
          description: 'Group related nodes for improved navigation',
          confidence: 0.78,
          impact: 0.18
        }
      ],
      metrics: {
        stressMajorization: { stressReduction: 0.85, finalStress: 0.023 },
        clustering: { modularity: 0.72, silhouette: 0.65 },
        performance: { computeTime: 12.5, efficiency: 0.89 }
      }
    }
  },

  // Export responses
  EXPORT_SUCCESS: (format: string) => ({
    data: {
      taskId: `export-${Date.now()}`,
      status: 'processing',
      format,
      downloadUrl: null,
      progress: 0
    }
  }),

  EXPORT_COMPLETE: (taskId: string) => ({
    data: {
      taskId,
      status: 'completed',
      progress: 100,
      downloadUrl: `https://api.example.com/download/${taskId}`,
      fileSize: 2048
    }
  }),

  SHARE_SUCCESS: {
    data: {
      shareId: 'shared-abc123',
      url: 'https://example.com/shared/shared-abc123',
      expires: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      public: true,
      passwordProtected: false
    }
  }
};

// Create pre-configured mock API client for integration tests
export const createIntegrationApiClient = () => {
  return createPreConfiguredApiClient({
    // Workspace endpoints
    '/api/workspace/list': API_RESPONSES.WORKSPACE_LIST,
    '/api/workspace/create': API_RESPONSES.WORKSPACE_CREATE,
    '/api/workspace/*': (url: string, data?: any) => {
      if (url.includes('/favorite')) {
        return { data: { success: true, favorite: !data?.favorite } };
      }
      if (url.includes('/archive')) {
        return { data: { success: true, status: data?.archive ? 'archived' : 'active' } };
      }
      return { data: { success: true } };
    },

    // Analytics endpoints
    '/api/analytics/params': API_RESPONSES.ANALYTICS_PARAMS,
    '/api/analytics/stats': API_RESPONSES.ANALYTICS_STATS,
    '/api/analytics/structural': API_RESPONSES.STRUCTURAL_ANALYSIS,
    '/api/analytics/semantic': { data: { semanticSimilarity: 0.78 }, status: 'completed' },
    '/api/analytics/constraints': { data: { constraints: [] } },

    // Optimization endpoints
    '/api/optimization/layout': { data: { taskId: 'layout-123', status: 'queued' } },
    '/api/optimization/clustering': { data: { taskId: 'clustering-123', status: 'queued' } },
    '/api/optimization/result/*': API_RESPONSES.OPTIMIZATION_RESULT,
    '/api/optimization/cancel/*': { data: { cancelled: true } },

    // Export endpoints
    '/api/export/graph': (url: string, data: any) => API_RESPONSES.EXPORT_SUCCESS(data.format),
    '/api/export/share': API_RESPONSES.SHARE_SUCCESS,
    '/api/export/status/*': (url: string) => {
      const taskId = url.split('/').pop();
      return API_RESPONSES.EXPORT_COMPLETE(taskId!);
    },
    '/api/export/download/*': { data: JSON.stringify({ test: 'data' }) }
  });
};

// WebSocket message templates
export const WEBSOCKET_MESSAGES = {
  WORKSPACE_UPDATED: (id: string, updates: any) => ({
    type: 'workspace_updated',
    data: { id, ...updates }
  }),

  WORKSPACE_CREATED: (workspace: any) => ({
    type: 'workspace_created',
    data: workspace
  }),

  ANALYTICS_PROGRESS: (taskId: string, progress: number) => ({
    type: 'analytics_progress',
    data: { taskId, progress }
  }),

  ANALYTICS_COMPLETE: (taskId: string, results: any) => ({
    type: 'analytics_complete',
    data: { taskId, results }
  }),

  OPTIMIZATION_PROGRESS: (taskId: string, progress: number) => ({
    type: 'optimization_progress',
    data: { taskId, progress, currentIteration: Math.floor(progress * 2.5) }
  }),

  OPTIMIZATION_COMPLETE: (taskId: string, result: any) => ({
    type: 'optimization_complete',
    data: { taskId, result }
  }),

  GPU_METRICS: (metrics: any) => ({
    type: 'gpu_metrics',
    data: metrics
  })
};

// Test data generators
export const TEST_DATA = {
  GRAPH_DATA: {
    nodes: [
      { id: '1', label: 'Node 1', x: 0, y: 0, size: 10 },
      { id: '2', label: 'Node 2', x: 100, y: 100, size: 15 },
      { id: '3', label: 'Node 3', x: -50, y: 75, size: 8 }
    ],
    edges: [
      { id: 'e1', source: '1', target: '2', weight: 1.0 },
      { id: 'e2', source: '2', target: '3', weight: 0.8 }
    ]
  },

  LARGE_GRAPH_DATA: (nodeCount: number = 1000, edgeCount: number = 2000) => ({
    nodes: Array.from({ length: nodeCount }, (_, i) => ({
      id: `node-${i}`,
      label: `Node ${i}`,
      x: Math.random() * 2000 - 1000,
      y: Math.random() * 2000 - 1000,
      size: Math.random() * 20 + 5
    })),
    edges: Array.from({ length: edgeCount }, (_, i) => ({
      id: `edge-${i}`,
      source: `node-${Math.floor(Math.random() * nodeCount)}`,
      target: `node-${Math.floor(Math.random() * nodeCount)}`,
      weight: Math.random()
    }))
  })
};

// Common test utilities
export const TEST_UTILS = {
  // Wait for condition with timeout
  waitForCondition: async (condition: () => boolean, timeout = 5000) => {
    const start = Date.now();
    while (!condition() && Date.now() - start < timeout) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    if (!condition()) {
      throw new Error(`Condition not met within ${timeout}ms`);
    }
  },

  // Simulate user interaction delay
  userDelay: (ms: number = 100) => new Promise(resolve => setTimeout(resolve, ms)),

  // Mock user events
  mockUserEvent: {
    click: (element: HTMLElement) => {
      element.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    },
    type: (element: HTMLInputElement, value: string) => {
      element.value = value;
      element.dispatchEvent(new Event('input', { bubbles: true }));
    },
    keyPress: (element: HTMLElement, key: string) => {
      element.dispatchEvent(new KeyboardEvent('keypress', { key, bubbles: true }));
    }
  },

  // Performance monitoring
  measureRenderTime: async (renderFn: () => Promise<void>) => {
    const start = performance.now();
    await renderFn();
    return performance.now() - start;
  },

  // Memory usage tracking
  trackMemoryUsage: () => {
    const initial = (performance as any).memory?.usedJSHeapSize || 0;
    return {
      getIncrease: () => {
        const current = (performance as any).memory?.usedJSHeapSize || 0;
        return current - initial;
      },
      getMB: () => {
        const current = (performance as any).memory?.usedJSHeapSize || 0;
        return (current - initial) / (1024 * 1024);
      }
    };
  }
};

// Error scenarios for testing
export const ERROR_SCENARIOS = {
  NETWORK_ERROR: () => Promise.reject(new Error('Network Error')),
  TIMEOUT_ERROR: () => Promise.reject(new Error('Request timeout')),
  AUTH_ERROR: () => Promise.reject(new Error('Unauthorized')),
  VALIDATION_ERROR: () => Promise.reject(new Error('Validation failed')),
  SERVER_ERROR: () => Promise.reject(new Error('Internal server error')),

  // Progressive error (fails first N times, then succeeds)
  INTERMITTENT_ERROR: (failCount: number = 2) => {
    let attempts = 0;
    return () => {
      attempts++;
      if (attempts <= failCount) {
        return Promise.reject(new Error('Intermittent failure'));
      }
      return Promise.resolve({ data: { success: true } });
    };
  }
};

// Coverage tracking helpers
export const COVERAGE_UTILS = {
  // Track which components were tested
  testedComponents: new Set<string>(),

  // Track which API endpoints were called
  calledEndpoints: new Set<string>(),

  // Track which features were exercised
  exercisedFeatures: new Set<string>(),

  // Register component as tested
  registerComponent: (componentName: string) => {
    COVERAGE_UTILS.testedComponents.add(componentName);
  },

  // Register endpoint as called
  registerEndpoint: (endpoint: string) => {
    COVERAGE_UTILS.calledEndpoints.add(endpoint);
  },

  // Register feature as exercised
  registerFeature: (featureName: string) => {
    COVERAGE_UTILS.exercisedFeatures.add(featureName);
  },

  // Generate coverage report
  generateReport: () => ({
    components: Array.from(COVERAGE_UTILS.testedComponents),
    endpoints: Array.from(COVERAGE_UTILS.calledEndpoints),
    features: Array.from(COVERAGE_UTILS.exercisedFeatures),
    timestamp: new Date().toISOString()
  }),

  // Reset coverage tracking
  reset: () => {
    COVERAGE_UTILS.testedComponents.clear();
    COVERAGE_UTILS.calledEndpoints.clear();
    COVERAGE_UTILS.exercisedFeatures.clear();
  }
};

// Test environment setup
export const setupTestEnvironment = () => {
  const mockApiClient = createIntegrationApiClient();
  const mockWebSocketServer = new MockWebSocketServer(TEST_CONFIG.WEBSOCKET_URL);
  const mockWebSocket = createMockWebSocket(TEST_CONFIG.WEBSOCKET_URL);

  return {
    mockApiClient,
    mockWebSocketServer,
    mockWebSocket,
    cleanup: () => {
      mockWebSocketServer.close();
      vi.clearAllMocks();
      COVERAGE_UTILS.reset();
    }
  };
};