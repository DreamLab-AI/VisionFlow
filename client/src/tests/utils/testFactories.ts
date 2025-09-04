import { vi } from 'vitest';

// Mock settings data factory
export const createMockSettings = (overrides: Partial<any> = {}) => ({
  visualisation: {
    glow: {
      nodeGlowStrength: 1.5,
      edgeGlowStrength: 2.0,
      environmentGlowStrength: 1.0,
      baseColor: '#00ffff',
      emissionColor: '#ffffff',
      enabled: true,
      ...overrides.visualisation?.glow
    },
    graphs: {
      logseq: {
        physics: {
          springK: 0.1,
          repelK: 100.0,
          attractionK: 0.02,
          maxVelocity: 5.0,
          boundsSize: 1000.0,
          separationRadius: 50.0,
          centerGravityK: 0.1,
          coolingRate: 0.95,
          ...overrides.visualisation?.graphs?.logseq?.physics
        },
        nodeRadius: 10.0,
        edgeThickness: 2.0,
        ...overrides.visualisation?.graphs?.logseq
      }
    },
    colorSchemes: [],
    ...overrides.visualisation
  },
  system: {
    debugMode: false,
    maxConnections: 100,
    connectionTimeout: 5000,
    autoSave: true,
    logLevel: 'info',
    websocket: {
      heartbeatInterval: 30000,
      reconnectDelay: 1000,
      maxRetries: 5,
      ...overrides.system?.websocket
    },
    audit: {
      auditLogPath: '/var/log/audit.log',
      maxLogSize: 10485760,
      ...overrides.system?.audit
    },
    ...overrides.system
  },
  xr: {
    handMeshColor: '#ffffff',
    handRayColor: '#0099ff',
    teleportRayColor: '#00ff00',
    controllerRayColor: '#ff0000',
    planeColor: '#333333',
    portalEdgeColor: '#ffff00',
    spaceType: 'room-scale',
    locomotionMethod: 'teleport',
    ...overrides.xr
  },
  ...overrides
});

// Mock WebSocket factory
export const createMockWebSocket = (overrides: Partial<WebSocket> = {}) => {
  const mockWs = {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    send: vi.fn(),
    close: vi.fn(),
    readyState: WebSocket.OPEN,
    onopen: null,
    onmessage: null,
    onclose: null,
    onerror: null,
    url: 'ws://localhost:8000/ws',
    protocol: '',
    extensions: '',
    bufferedAmount: 0,
    binaryType: 'blob' as BinaryType,
    dispatchEvent: vi.fn(),
    CONNECTING: WebSocket.CONNECTING,
    OPEN: WebSocket.OPEN,
    CLOSING: WebSocket.CLOSING,
    CLOSED: WebSocket.CLOSED,
    ...overrides
  };

  // Helper methods for testing
  (mockWs as any).triggerOpen = () => {
    if (mockWs.onopen) mockWs.onopen({} as Event);
  };
  
  (mockWs as any).triggerMessage = (data: any) => {
    if (mockWs.onmessage) {
      mockWs.onmessage({ 
        data: typeof data === 'string' ? data : JSON.stringify(data),
        type: 'message',
        target: mockWs
      } as MessageEvent);
    }
  };
  
  (mockWs as any).triggerError = (error: any = new Error('WebSocket error')) => {
    if (mockWs.onerror) mockWs.onerror({ error } as ErrorEvent);
  };
  
  (mockWs as any).triggerClose = (code = 1000, reason = '') => {
    if (mockWs.onclose) {
      mockWs.onclose({ code, reason, wasClean: true } as CloseEvent);
    }
  };

  return mockWs;
};

// Mock fetch response factory
export const createMockFetchResponse = (data: any, options: Partial<Response> = {}) => {
  return Promise.resolve({
    ok: true,
    status: 200,
    statusText: 'OK',
    headers: new Headers(),
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
    blob: () => Promise.resolve(new Blob([JSON.stringify(data)])),
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
    formData: () => Promise.resolve(new FormData()),
    clone: () => createMockFetchResponse(data, options),
    body: null,
    bodyUsed: false,
    redirected: false,
    type: 'basic' as ResponseType,
    url: '',
    ...options
  } as Response);
};

// Mock API endpoints
export const mockApiEndpoints = {
  getSettings: (paths: string[]) => 
    createMockFetchResponse(createMockSettings()),
  
  updateSettings: (updates: Array<{ path: string; value: any }>) =>
    createMockFetchResponse({ success: true, updated: updates.length }),
  
  getHealth: () =>
    createMockFetchResponse({ status: 'healthy', timestamp: Date.now() })
};

// Test utilities
export const waitForNextTick = () => new Promise(resolve => setTimeout(resolve, 0));
export const waitFor = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Mock performance timing for benchmarks
export const createMockPerformance = () => {
  let startTime = 0;
  return {
    now: vi.fn(() => Date.now()),
    mark: vi.fn((name: string) => {}),
    measure: vi.fn((name: string, start?: string, end?: string) => ({
      name,
      duration: 10,
      startTime: startTime++
    })),
    getEntriesByName: vi.fn(() => []),
    clearMarks: vi.fn(),
    clearMeasures: vi.fn()
  };
};

// Storage mock helpers
export const createStorageMock = () => {
  let store: Record<string, string> = {};
  
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = String(value);
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
    key: vi.fn((index: number) => {
      const keys = Object.keys(store);
      return keys[index] || null;
    }),
    get length() {
      return Object.keys(store).length;
    },
    // Test helpers
    _getStore: () => ({ ...store }),
    _setStore: (newStore: Record<string, string>) => {
      store = { ...newStore };
    }
  };
};

// Error factory for testing error scenarios
export const createTestError = (message: string, name = 'TestError') => {
  const error = new Error(message);
  error.name = name;
  return error;
};

// Async test helpers
export const expectAsync = {
  toResolve: (promise: Promise<any>) => expect(promise).resolves,
  toReject: (promise: Promise<any>) => expect(promise).rejects,
  toResolveWith: (promise: Promise<any>, value: any) => 
    expect(promise).resolves.toEqual(value),
  toRejectWith: (promise: Promise<any>, error: any) => 
    expect(promise).rejects.toThrow(error)
};

// Component test helpers
export const createMockComponent = (name: string) => {
  const MockComponent = vi.fn(({ children, ...props }) => {
    return children || null;
  });
  MockComponent.displayName = name;
  return MockComponent;
};

// Settings path validation helpers
export const isValidSettingsPath = (path: string): boolean => {
  const pathRegex = /^[a-zA-Z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)*$/;
  return pathRegex.test(path);
};

export const isValidCamelCase = (str: string): boolean => {
  return /^[a-z][a-zA-Z0-9]*$/.test(str);
};

// Performance testing utilities
export const measurePerformance = async (fn: () => Promise<void> | void, iterations = 1) => {
  const times: number[] = [];
  
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    const end = performance.now();
    times.push(end - start);
  }
  
  const total = times.reduce((sum, time) => sum + time, 0);
  const average = total / times.length;
  const min = Math.min(...times);
  const max = Math.max(...times);
  
  return {
    total,
    average,
    min,
    max,
    times,
    iterations
  };
};

// Concurrency testing helpers
export const runConcurrently = async (tasks: (() => Promise<any>)[], maxConcurrency = 10) => {
  const results = [];
  const running: Promise<any>[] = [];
  
  for (const task of tasks) {
    if (running.length >= maxConcurrency) {
      await Promise.race(running);
    }
    
    const promise = task().then(result => {
      running.splice(running.indexOf(promise), 1);
      return result;
    });
    
    running.push(promise);
    results.push(promise);
  }
  
  return Promise.all(results);
};