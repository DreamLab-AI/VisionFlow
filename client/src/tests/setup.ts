import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock WebSocket for tests
global.WebSocket = vi.fn(() => ({
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  send: vi.fn(),
  close: vi.fn(),
  readyState: WebSocket.OPEN,
})) as any;

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    store: {} as Record<string, string>,
    getItem: vi.fn((key: string) => {
      return window.localStorage.store[key] || null;
    }),
    setItem: vi.fn((key: string, value: string) => {
      window.localStorage.store[key] = String(value);
    }),
    removeItem: vi.fn((key: string) => {
      delete window.localStorage.store[key];
    }),
    clear: vi.fn(() => {
      window.localStorage.store = {};
    }),
    key: vi.fn((index: number) => {
      const keys = Object.keys(window.localStorage.store);
      return keys[index] || null;
    }),
    get length() {
      return Object.keys(this.store).length;
    }
  },
  writable: true,
});

// Mock sessionStorage
Object.defineProperty(window, 'sessionStorage', {
  value: {
    store: {} as Record<string, string>,
    getItem: vi.fn((key: string) => {
      return window.sessionStorage.store[key] || null;
    }),
    setItem: vi.fn((key: string, value: string) => {
      window.sessionStorage.store[key] = String(value);
    }),
    removeItem: vi.fn((key: string) => {
      delete window.sessionStorage.store[key];
    }),
    clear: vi.fn(() => {
      window.sessionStorage.store = {};
    }),
    key: vi.fn((index: number) => {
      const keys = Object.keys(window.sessionStorage.store);
      return keys[index] || null;
    }),
    get length() {
      return Object.keys(this.store).length;
    }
  },
  writable: true,
});

// Mock URL.createObjectURL
global.URL.createObjectURL = vi.fn(() => 'mock-url');
global.URL.revokeObjectURL = vi.fn();

// Mock fetch
global.fetch = vi.fn();

// Mock performance.now for testing
Object.defineProperty(global, 'performance', {
  value: {
    now: vi.fn(() => Date.now()),
  },
  writable: true,
});

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock requestAnimationFrame
global.requestAnimationFrame = vi.fn((cb) => setTimeout(cb, 16));
global.cancelAnimationFrame = vi.fn((id) => clearTimeout(id));

// Mock console methods for cleaner test output
global.console = {
  ...console,
  warn: vi.fn(),
  error: vi.fn(),
  log: vi.fn(),
};

// Clean up between tests
beforeEach(() => {
  // Clear all mocks
  vi.clearAllMocks();
  
  // Reset localStorage and sessionStorage
  window.localStorage.clear();
  window.sessionStorage.clear();
  
  // Reset fetch mock
  (global.fetch as any).mockClear();
});

afterEach(() => {
  // Additional cleanup
  vi.restoreAllMocks();
});