/**
 * Mock API Client for Integration Tests
 * Provides consistent mock implementations for backend API calls
 */

import { vi } from 'vitest';

export interface MockApiClient {
  get: ReturnType<typeof vi.fn>;
  post: ReturnType<typeof vi.fn>;
  put: ReturnType<typeof vi.fn>;
  patch: ReturnType<typeof vi.fn>;
  delete: ReturnType<typeof vi.fn>;
  upload: ReturnType<typeof vi.fn>;
}

export const createMockApiClient = (): MockApiClient => {
  return {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
    upload: vi.fn()
  };
};

// Mock response creators
export const createSuccessResponse = (data: any, status: number = 200) => ({
  data,
  status,
  statusText: 'OK',
  headers: {
    'content-type': 'application/json'
  },
  config: {}
});

export const createErrorResponse = (message: string, status: number = 500) => ({
  response: {
    data: { error: message, message },
    status,
    statusText: status === 404 ? 'Not Found' : status === 400 ? 'Bad Request' : 'Internal Server Error',
    headers: {}
  },
  message,
  code: status.toString()
});

// Common mock data generators
export const generateMockGraphData = (nodeCount: number = 10, edgeCount: number = 15) => ({
  nodes: Array.from({ length: nodeCount }, (_, i) => ({
    id: `node-${i}`,
    label: `Node ${i}`,
    x: Math.random() * 1000,
    y: Math.random() * 1000,
    size: Math.random() * 20 + 5
  })),
  edges: Array.from({ length: edgeCount }, (_, i) => ({
    id: `edge-${i}`,
    source: `node-${Math.floor(Math.random() * nodeCount)}`,
    target: `node-${Math.floor(Math.random() * nodeCount)}`,
    weight: Math.random()
  }))
});

export const generateMockWorkspace = (id: string, overrides: any = {}) => ({
  id,
  name: `Workspace ${id}`,
  description: `Description for workspace ${id}`,
  type: 'personal' as const,
  status: 'active' as const,
  memberCount: 1,
  lastAccessed: new Date(),
  createdAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000),
  favorite: Math.random() > 0.7,
  ...overrides
});

export const generateMockAnalysisResult = () => ({
  clusters: Array.from({ length: 5 }, (_, i) => ({
    id: i,
    size: Math.floor(Math.random() * 20) + 10,
    density: Math.random(),
    modularity: Math.random()
  })),
  centrality: {
    betweenness: Math.random(),
    closeness: Math.random(),
    eigenvector: Math.random(),
    degree: Math.random()
  },
  connectivity: {
    components: Math.floor(Math.random() * 5) + 1,
    diameter: Math.floor(Math.random() * 10) + 3,
    averagePathLength: Math.random() * 5 + 1
  },
  similarity: {
    overall: Math.random(),
    structural: Math.random(),
    semantic: Math.random()
  }
});

export const generateMockOptimizationResult = () => ({
  algorithm: 'Adaptive Force-Directed',
  confidence: Math.random(),
  performanceGain: Math.random() * 0.5,
  clusters: Math.floor(Math.random() * 10) + 3,
  iterations: Math.floor(Math.random() * 500) + 100,
  convergenceReached: Math.random() > 0.2,
  executionTime: Math.random() * 30000 + 5000,
  gpuUtilization: Math.random() * 100,
  recommendations: [
    {
      type: 'layout',
      priority: 'high',
      description: 'Adjust node spacing for better clarity',
      confidence: Math.random(),
      impact: Math.random() * 0.3
    },
    {
      type: 'clustering',
      priority: 'medium',
      description: 'Group related nodes for improved navigation',
      confidence: Math.random(),
      impact: Math.random() * 0.2
    }
  ],
  metrics: {
    stressMajorization: {
      stressReduction: Math.random(),
      finalStress: Math.random() * 0.1
    },
    clustering: {
      modularity: Math.random(),
      silhouette: Math.random()
    },
    performance: {
      computeTime: Math.random() * 10,
      efficiency: Math.random()
    }
  }
});

// Request matching utilities
export const matchesUrl = (actualUrl: string, expectedPattern: string | RegExp) => {
  if (typeof expectedPattern === 'string') {
    return actualUrl === expectedPattern || actualUrl.includes(expectedPattern);
  }
  return expectedPattern.test(actualUrl);
};

export const matchesMethod = (method: string, expectedMethod: string) => {
  return method.toLowerCase() === expectedMethod.toLowerCase();
};

// Mock API client with predefined responses
export const createPreConfiguredApiClient = (responses: Record<string, any>) => {
  const client = createMockApiClient();

  // Configure GET responses
  client.get.mockImplementation((url: string) => {
    for (const [pattern, response] of Object.entries(responses)) {
      if (matchesUrl(url, pattern)) {
        return typeof response === 'function' ? response(url) : Promise.resolve(response);
      }
    }
    return Promise.reject(createErrorResponse(`No mock configured for GET ${url}`, 404));
  });

  // Configure POST responses
  client.post.mockImplementation((url: string, data?: any) => {
    for (const [pattern, response] of Object.entries(responses)) {
      if (matchesUrl(url, pattern)) {
        return typeof response === 'function' ? response(url, data) : Promise.resolve(response);
      }
    }
    return Promise.reject(createErrorResponse(`No mock configured for POST ${url}`, 404));
  });

  // Similar for other methods
  ['put', 'patch', 'delete'].forEach(method => {
    client[method as keyof MockApiClient].mockImplementation((url: string, data?: any) => {
      for (const [pattern, response] of Object.entries(responses)) {
        if (matchesUrl(url, pattern)) {
          return typeof response === 'function' ? response(url, data) : Promise.resolve(response);
        }
      }
      return Promise.reject(createErrorResponse(`No mock configured for ${method.toUpperCase()} ${url}`, 404));
    });
  });

  return client;
};

// Rate limiting and throttling simulation
export const withDelay = (response: any, delayMs: number = 100) => {
  return new Promise(resolve => {
    setTimeout(() => resolve(response), delayMs);
  });
};

export const withRandomDelay = (response: any, minMs: number = 50, maxMs: number = 500) => {
  const delay = Math.random() * (maxMs - minMs) + minMs;
  return withDelay(response, delay);
};

// Error simulation helpers
export const withRandomError = (response: any, errorRate: number = 0.1) => {
  if (Math.random() < errorRate) {
    return Promise.reject(createErrorResponse('Simulated network error', 500));
  }
  return Promise.resolve(response);
};

export const withTimeoutError = (response: any, timeoutMs: number = 5000) => {
  return Promise.race([
    response,
    new Promise((_, reject) => {
      setTimeout(() => reject(createErrorResponse('Request timeout', 408)), timeoutMs);
    })
  ]);
};