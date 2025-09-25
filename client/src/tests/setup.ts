/**
 * Test Setup Configuration
 * Global setup for integration tests with proper mocks and utilities
 *
 * NOTE: These tests are prepared but disabled due to security concerns.
 * Re-enable once testing dependencies are secure.
 */

import { vi, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock global APIs
beforeAll(() => {
  // Mock window.matchMedia
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))
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

  // Mock navigator.clipboard
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      writeText: vi.fn().mockResolvedValue(undefined),
      readText: vi.fn().mockResolvedValue(''),
      write: vi.fn().mockResolvedValue(undefined),
      read: vi.fn().mockResolvedValue([])
    }
  });

  // Mock localStorage
  Object.defineProperty(window, 'localStorage', {
    value: {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn(),
      clear: vi.fn(),
      length: 0,
      key: vi.fn()
    }
  });

  // Mock sessionStorage
  Object.defineProperty(window, 'sessionStorage', {
    value: {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn(),
      clear: vi.fn(),
      length: 0,
      key: vi.fn()
    }
  });

  // Mock URL.createObjectURL and revokeObjectURL
  global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
  global.URL.revokeObjectURL = vi.fn();

  // Mock Blob
  global.Blob = vi.fn().mockImplementation((content, options) => ({
    size: content?.[0]?.length || 0,
    type: options?.type || 'text/plain',
    text: vi.fn().mockResolvedValue(content?.[0] || ''),
    arrayBuffer: vi.fn().mockResolvedValue(new ArrayBuffer(0)),
    stream: vi.fn().mockReturnValue(new ReadableStream())
  }));

  // Mock File
  global.File = vi.fn().mockImplementation((parts, name, options) => ({
    name,
    size: parts?.[0]?.length || 0,
    type: options?.type || 'text/plain',
    lastModified: Date.now(),
    text: vi.fn().mockResolvedValue(parts?.[0] || ''),
    arrayBuffer: vi.fn().mockResolvedValue(new ArrayBuffer(0)),
    stream: vi.fn().mockReturnValue(new ReadableStream())
  }));

  // Mock performance.now
  Object.defineProperty(performance, 'now', {
    value: vi.fn(() => Date.now())
  });

  // Mock performance.mark and measure
  Object.defineProperty(performance, 'mark', {
    value: vi.fn()
  });

  Object.defineProperty(performance, 'measure', {
    value: vi.fn()
  });

  // Mock requestAnimationFrame
  global.requestAnimationFrame = vi.fn((callback) => {
    setTimeout(callback, 16);
    return 1;
  });

  global.cancelAnimationFrame = vi.fn();

  // Mock console methods to reduce noise in tests
  console.warn = vi.fn();
  console.error = vi.fn();
  console.debug = vi.fn();

  // Suppress specific warnings
  const originalConsoleWarn = console.warn;
  console.warn = (...args) => {
    const message = args[0];
    if (
      typeof message === 'string' &&
      (
        message.includes('ReactDOM.render is no longer supported') ||
        message.includes('Warning: Function components cannot be given refs') ||
        message.includes('Warning: Failed prop type')
      )
    ) {
      return;
    }
    originalConsoleWarn.apply(console, args);
  };
});

// Clean up after each test
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.clearAllTimers();
});

// Reset mocks before each test
beforeEach(() => {
  vi.clearAllMocks();

  // Reset localStorage
  (window.localStorage.getItem as any).mockReturnValue(null);
  (window.localStorage.setItem as any).mockClear();
  (window.localStorage.removeItem as any).mockClear();
  (window.localStorage.clear as any).mockClear();

  // Reset sessionStorage
  (window.sessionStorage.getItem as any).mockReturnValue(null);
  (window.sessionStorage.setItem as any).mockClear();
  (window.sessionStorage.removeItem as any).mockClear();
  (window.sessionStorage.clear as any).mockClear();

  // Reset clipboard
  (navigator.clipboard.writeText as any).mockResolvedValue(undefined);
  (navigator.clipboard.readText as any).mockResolvedValue('');

  // Reset URL mocks
  (global.URL.createObjectURL as any).mockReturnValue('blob:mock-url');
  (global.URL.revokeObjectURL as any).mockClear();

  // Reset performance mocks
  (performance.now as any).mockReturnValue(Date.now());
});

afterAll(() => {
  vi.restoreAllMocks();
});

// Test utilities
export const waitForNextTick = () => new Promise(resolve => setTimeout(resolve, 0));

export const waitFor = async (condition: () => boolean, timeout = 5000) => {
  const start = Date.now();
  while (!condition() && Date.now() - start < timeout) {
    await waitForNextTick();
  }
  if (!condition()) {
    throw new Error(`Condition not met within ${timeout}ms`);
  }
};

// Custom test environment detection
export const isTestEnvironment = () => process.env.NODE_ENV === 'test';

// Mock data generators for consistency
export const createTestId = (prefix: string = 'test') =>
  `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

export const createMockDate = (daysAgo: number = 0) =>
  new Date(Date.now() - daysAgo * 24 * 60 * 60 * 1000);

// Error simulation utilities
export const simulateNetworkError = () => {
  throw new Error('Network Error: Simulated connection failure');
};

export const simulateTimeoutError = () => {
  throw new Error('Timeout Error: Request timed out');
};

export const simulateAuthError = () => {
  throw new Error('Authentication Error: Invalid credentials');
};

// Performance testing helpers
export const measureExecutionTime = async (fn: () => Promise<void> | void) => {
  const start = performance.now();
  await fn();
  const end = performance.now();
  return end - start;
};

export const expectExecutionTimeUnder = async (
  fn: () => Promise<void> | void,
  maxMs: number
) => {
  const executionTime = await measureExecutionTime(fn);
  if (executionTime >= maxMs) {
    throw new Error(`Execution took ${executionTime.toFixed(2)}ms, expected under ${maxMs}ms`);
  }
};

// Memory leak detection (simplified)
export const detectMemoryLeaks = () => {
  const initialMemory = (performance as any).memory?.usedJSHeapSize;

  return () => {
    if ((performance as any).memory?.usedJSHeapSize) {
      const currentMemory = (performance as any).memory.usedJSHeapSize;
      const memoryIncrease = currentMemory - initialMemory;

      // Warn if memory increased by more than 50MB
      if (memoryIncrease > 50 * 1024 * 1024) {
        console.warn(`Potential memory leak detected: +${(memoryIncrease / 1024 / 1024).toFixed(2)}MB`);
      }
    }
  };
};

// Accessibility testing helpers
export const checkAccessibility = (element: HTMLElement) => {
  const issues: string[] = [];

  // Check for alt text on images
  element.querySelectorAll('img').forEach(img => {
    if (!img.alt) {
      issues.push(`Image without alt text: ${img.src || img.outerHTML.substring(0, 50)}`);
    }
  });

  // Check for form labels
  element.querySelectorAll('input, select, textarea').forEach(input => {
    const id = input.getAttribute('id');
    const label = element.querySelector(`label[for="${id}"]`);
    const ariaLabel = input.getAttribute('aria-label');
    const ariaLabelledBy = input.getAttribute('aria-labelledby');

    if (!label && !ariaLabel && !ariaLabelledBy) {
      issues.push(`Form input without label: ${input.outerHTML.substring(0, 50)}`);
    }
  });

  // Check for keyboard navigation
  element.querySelectorAll('button, a, input, select, textarea').forEach(interactive => {
    if (interactive.getAttribute('tabindex') === '-1' && !interactive.hasAttribute('aria-hidden')) {
      issues.push(`Interactive element with tabindex="-1": ${interactive.outerHTML.substring(0, 50)}`);
    }
  });

  return issues;
};

// Test data cleanup
export const cleanupTestData = () => {
  // Clear any test data that might persist between tests
  localStorage.clear();
  sessionStorage.clear();

  // Reset any global state
  if (typeof window !== 'undefined') {
    // Reset any window properties that tests might have modified
    delete (window as any).testData;
    delete (window as any).mockResponses;
  }
};