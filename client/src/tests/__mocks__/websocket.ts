/**
 * Mock WebSocket for Integration Tests
 * Provides consistent WebSocket mock implementations for real-time features
 */

import { vi } from 'vitest';

export interface MockWebSocket {
  url: string;
  readyState: number;
  onopen?: ((event: Event) => void) | null;
  onclose?: ((event: CloseEvent) => void) | null;
  onmessage?: ((event: MessageEvent) => void) | null;
  onerror?: ((event: Event) => void) | null;
  send: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  addEventListener: ReturnType<typeof vi.fn>;
  removeEventListener: ReturnType<typeof vi.fn>;
  dispatchEvent: ReturnType<typeof vi.fn>;
}

export class MockWebSocketServer {
  private clients: MockWebSocket[] = [];
  private url: string;

  constructor(url: string) {
    this.url = url;
  }

  // Simulate server sending message to all clients
  send(data: string) {
    this.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN && client.onmessage) {
        const event = new MessageEvent('message', { data });
        client.onmessage(event);
      }
    });
  }

  // Simulate server sending message to specific client
  sendTo(client: MockWebSocket, data: string) {
    if (client.readyState === WebSocket.OPEN && client.onmessage) {
      const event = new MessageEvent('message', { data });
      client.onmessage(event);
    }
  }

  // Add client to server
  addClient(client: MockWebSocket) {
    this.clients.push(client);
  }

  // Remove client from server
  removeClient(client: MockWebSocket) {
    const index = this.clients.indexOf(client);
    if (index > -1) {
      this.clients.splice(index, 1);
    }
  }

  // Close all connections
  close() {
    this.clients.forEach(client => {
      client.readyState = WebSocket.CLOSED;
      if (client.onclose) {
        const event = new CloseEvent('close', { code: 1000, reason: 'Server shutdown' });
        client.onclose(event);
      }
    });
    this.clients = [];
  }

  // Reset server state
  reset() {
    this.clients.forEach(client => {
      client.readyState = WebSocket.CLOSED;
    });
    this.clients = [];
  }

  // Get connected clients count
  get clientCount() {
    return this.clients.filter(client => client.readyState === WebSocket.OPEN).length;
  }
}

export const createMockWebSocket = (url: string = 'ws://localhost:3002'): MockWebSocket => {
  const mockSocket: MockWebSocket = {
    url,
    readyState: WebSocket.CONNECTING,
    onopen: null,
    onclose: null,
    onmessage: null,
    onerror: null,
    send: vi.fn(),
    close: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn()
  };

  // Simulate connection opening after a short delay
  setTimeout(() => {
    mockSocket.readyState = WebSocket.OPEN;
    if (mockSocket.onopen) {
      const event = new Event('open');
      mockSocket.onopen(event);
    }
  }, 10);

  // Implement close method
  mockSocket.close.mockImplementation((code?: number, reason?: string) => {
    mockSocket.readyState = WebSocket.CLOSING;
    setTimeout(() => {
      mockSocket.readyState = WebSocket.CLOSED;
      if (mockSocket.onclose) {
        const event = new CloseEvent('close', {
          code: code || 1000,
          reason: reason || 'Normal closure'
        });
        mockSocket.onclose(event);
      }
    }, 10);
  });

  // Implement addEventListener
  mockSocket.addEventListener.mockImplementation((type: string, listener: EventListener) => {
    switch (type) {
      case 'open':
        mockSocket.onopen = listener as (event: Event) => void;
        break;
      case 'close':
        mockSocket.onclose = listener as (event: CloseEvent) => void;
        break;
      case 'message':
        mockSocket.onmessage = listener as (event: MessageEvent) => void;
        break;
      case 'error':
        mockSocket.onerror = listener as (event: Event) => void;
        break;
    }
  });

  // Implement removeEventListener
  mockSocket.removeEventListener.mockImplementation((type: string) => {
    switch (type) {
      case 'open':
        mockSocket.onopen = null;
        break;
      case 'close':
        mockSocket.onclose = null;
        break;
      case 'message':
        mockSocket.onmessage = null;
        break;
      case 'error':
        mockSocket.onerror = null;
        break;
    }
  });

  return mockSocket;
};

// WebSocket message helpers
export const createWebSocketMessage = (type: string, data: any) => ({
  type,
  data,
  timestamp: new Date().toISOString()
});

// Common WebSocket message types
export const createWorkspaceUpdateMessage = (workspaceId: string, updates: any) =>
  createWebSocketMessage('workspace_updated', {
    id: workspaceId,
    ...updates
  });

export const createWorkspaceCreatedMessage = (workspace: any) =>
  createWebSocketMessage('workspace_created', workspace);

export const createWorkspaceDeletedMessage = (workspaceId: string) =>
  createWebSocketMessage('workspace_deleted', { id: workspaceId });

export const createAnalyticsProgressMessage = (taskId: string, progress: number) =>
  createWebSocketMessage('analytics_progress', {
    taskId,
    progress,
    timestamp: new Date().toISOString()
  });

export const createAnalyticsCompleteMessage = (taskId: string, results: any) =>
  createWebSocketMessage('analytics_complete', {
    taskId,
    results,
    timestamp: new Date().toISOString()
  });

export const createOptimizationProgressMessage = (taskId: string, progress: number, details?: any) =>
  createWebSocketMessage('optimization_progress', {
    taskId,
    progress,
    ...details
  });

export const createOptimizationCompleteMessage = (taskId: string, result: any) =>
  createWebSocketMessage('optimization_complete', {
    taskId,
    result,
    timestamp: new Date().toISOString()
  });

export const createGpuMetricsMessage = (metrics: any) =>
  createWebSocketMessage('gpu_metrics', metrics);

// WebSocket event simulation utilities
export const simulateConnectionLoss = (mockSocket: MockWebSocket) => {
  mockSocket.readyState = WebSocket.CLOSED;
  if (mockSocket.onclose) {
    const event = new CloseEvent('close', { code: 1006, reason: 'Connection lost' });
    mockSocket.onclose(event);
  }
};

export const simulateConnectionRestore = (mockSocket: MockWebSocket) => {
  mockSocket.readyState = WebSocket.OPEN;
  if (mockSocket.onopen) {
    const event = new Event('open');
    mockSocket.onopen(event);
  }
};

export const simulateNetworkError = (mockSocket: MockWebSocket) => {
  if (mockSocket.onerror) {
    const event = new Event('error');
    mockSocket.onerror(event);
  }
};

// Progress simulation helpers
export const simulateProgressUpdates = (
  mockSocket: MockWebSocket,
  taskId: string,
  totalSteps: number = 10,
  intervalMs: number = 100,
  messageCreator: (taskId: string, progress: number) => any = createAnalyticsProgressMessage
) => {
  let step = 0;
  const interval = setInterval(() => {
    step++;
    const progress = (step / totalSteps) * 100;

    if (mockSocket.onmessage) {
      const message = messageCreator(taskId, progress);
      const event = new MessageEvent('message', { data: JSON.stringify(message) });
      mockSocket.onmessage(event);
    }

    if (step >= totalSteps) {
      clearInterval(interval);
    }
  }, intervalMs);

  return interval;
};

// Batch message simulation
export const simulateBatchMessages = (
  mockSocket: MockWebSocket,
  messages: any[],
  intervalMs: number = 50
) => {
  let index = 0;
  const interval = setInterval(() => {
    if (index >= messages.length) {
      clearInterval(interval);
      return;
    }

    if (mockSocket.onmessage) {
      const event = new MessageEvent('message', {
        data: JSON.stringify(messages[index])
      });
      mockSocket.onmessage(event);
    }

    index++;
  }, intervalMs);

  return interval;
};

// Connection state management
export const createWebSocketStateMachine = () => {
  let state: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' = 'disconnected';
  let retryCount = 0;
  const maxRetries = 3;

  return {
    getState: () => state,
    getRetryCount: () => retryCount,

    connect: (mockSocket: MockWebSocket) => {
      state = 'connecting';
      setTimeout(() => {
        mockSocket.readyState = WebSocket.OPEN;
        state = 'connected';
        retryCount = 0;
        if (mockSocket.onopen) {
          mockSocket.onopen(new Event('open'));
        }
      }, 100);
    },

    disconnect: (mockSocket: MockWebSocket) => {
      state = 'disconnected';
      mockSocket.readyState = WebSocket.CLOSED;
      if (mockSocket.onclose) {
        mockSocket.onclose(new CloseEvent('close', { code: 1000 }));
      }
    },

    handleConnectionLoss: (mockSocket: MockWebSocket) => {
      state = 'reconnecting';
      retryCount++;

      if (retryCount <= maxRetries) {
        setTimeout(() => {
          this.connect(mockSocket);
        }, Math.pow(2, retryCount) * 1000); // Exponential backoff
      } else {
        state = 'disconnected';
      }
    },

    reset: () => {
      state = 'disconnected';
      retryCount = 0;
    }
  };
};