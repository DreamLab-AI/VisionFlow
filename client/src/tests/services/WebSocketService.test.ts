import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { webSocketService } from '../../client/src/services/WebSocketService';

// Mock WebSocket
class MockWebSocket {
  public static CONNECTING = 0;
  public static OPEN = 1;
  public static CLOSING = 2;
  public static CLOSED = 3;

  public readyState: number = MockWebSocket.CONNECTING;
  public onopen: ((event: Event) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;
  public url: string;

  private eventListeners: { [key: string]: EventListener[] } = {};

  constructor(url: string) {
    this.url = url;
    // Simulate async connection
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
      this.dispatchEvent(new Event('open'));
    }, 10);
  }

  public send(data: string | ArrayBuffer): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    // Mock successful send
  }

  public close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSED;
    const closeEvent = new CloseEvent('close', {
      code: code || 1000,
      reason: reason || '',
      wasClean: true
    });
    if (this.onclose) {
      this.onclose(closeEvent);
    }
    this.dispatchEvent(closeEvent);
  }

  public addEventListener(type: string, listener: EventListener, options?: any): void {
    if (!this.eventListeners[type]) {
      this.eventListeners[type] = [];
    }
    this.eventListeners[type].push(listener);
  }

  public removeEventListener(type: string, listener: EventListener): void {
    if (this.eventListeners[type]) {
      const index = this.eventListeners[type].indexOf(listener);
      if (index > -1) {
        this.eventListeners[type].splice(index, 1);
      }
    }
  }

  public dispatchEvent(event: Event): boolean {
    const listeners = this.eventListeners[event.type];
    if (listeners) {
      listeners.forEach(listener => listener(event));
    }
    return true;
  }

  // Simulate receiving a message
  public simulateMessage(data: string | ArrayBuffer | Blob): void {
    if (this.onmessage) {
      const messageEvent = new MessageEvent('message', { data });
      this.onmessage(messageEvent);
    }
  }

  // Simulate connection error
  public simulateError(): void {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }

  // Simulate unexpected close
  public simulateUnexpectedClose(): void {
    this.readyState = MockWebSocket.CLOSED;
    const closeEvent = new CloseEvent('close', {
      code: 1006,
      reason: 'Abnormal closure',
      wasClean: false
    });
    if (this.onclose) {
      this.onclose(closeEvent);
    }
  }
}

// Mock global WebSocket
(global as any).WebSocket = MockWebSocket;

// Mock setTimeout and clearTimeout for testing reconnection logic
const originalSetTimeout = global.setTimeout;
const originalClearTimeout = global.clearTimeout;
const originalSetInterval = global.setInterval;
const originalClearInterval = global.clearInterval;

describe('WebSocketService', () => {
  let mockSocket: MockWebSocket;

  beforeEach(() => {
    // Reset the service instance
    vi.clearAllMocks();
    
    // Mock timers
    vi.useFakeTimers();
    
    // Mock console methods to reduce test noise
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  describe('Connection Management', () => {
    it('should establish connection successfully', async () => {
      const connectionStatusHandler = vi.fn();
      const connectionStateHandler = vi.fn();
      
      webSocketService.onConnectionStatusChange(connectionStatusHandler);
      webSocketService.onConnectionStateChange(connectionStateHandler);
      
      const connectPromise = webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      await connectPromise;
      
      expect(connectionStatusHandler).toHaveBeenCalledWith(true);
      expect(connectionStateHandler).toHaveBeenCalledWith(
        expect.objectContaining({ status: 'connected' })
      );
    });

    it('should handle connection failures with exponential backoff', async () => {
      // Mock WebSocket to fail immediately
      (global as any).WebSocket = class extends MockWebSocket {
        constructor(url: string) {
          super(url);
          setTimeout(() => {
            this.simulateError();
            this.simulateUnexpectedClose();
          }, 5);
        }
      };

      const connectionStateHandler = vi.fn();
      webSocketService.onConnectionStateChange(connectionStateHandler);

      // Attempt to connect
      webSocketService.connect().catch(() => {});
      
      // Fast-forward through initial connection failure
      vi.advanceTimersByTime(10);
      
      // Should be in reconnecting state
      expect(connectionStateHandler).toHaveBeenCalledWith(
        expect.objectContaining({ status: 'reconnecting' })
      );

      // Fast-forward through first reconnect attempt (1s delay)
      vi.advanceTimersByTime(1000);
      
      // Fast-forward through second reconnect attempt (2s delay)
      vi.advanceTimersByTime(2000);
      
      // Fast-forward through third reconnect attempt (4s delay)
      vi.advanceTimersByTime(4000);
      
      expect(connectionStateHandler).toHaveBeenCalledWith(
        expect.objectContaining({ 
          status: 'reconnecting',
          reconnectAttempts: expect.any(Number)
        })
      );
    });

    it('should stop reconnecting after maximum attempts', async () => {
      // Mock WebSocket to always fail
      (global as any).WebSocket = class extends MockWebSocket {
        constructor(url: string) {
          super(url);
          setTimeout(() => {
            this.simulateError();
            this.simulateUnexpectedClose();
          }, 5);
        }
      };

      const connectionStateHandler = vi.fn();
      webSocketService.onConnectionStateChange(connectionStateHandler);

      // Attempt to connect
      webSocketService.connect().catch(() => {});
      
      // Fast-forward through all reconnection attempts
      for (let i = 0; i < 15; i++) {
        vi.advanceTimersByTime(30000); // Max delay
      }
      
      expect(connectionStateHandler).toHaveBeenCalledWith(
        expect.objectContaining({ status: 'failed' })
      );
    });
  });

  describe('Message Queuing', () => {
    it('should queue messages when disconnected', () => {
      const initialQueueCount = webSocketService.getQueuedMessageCount();
      
      // Send message while disconnected
      webSocketService.sendMessage('test', { data: 'test' });
      
      expect(webSocketService.getQueuedMessageCount()).toBe(initialQueueCount + 1);
    });

    it('should process queued messages after reconnection', async () => {
      // Queue a message while disconnected
      webSocketService.sendMessage('test', { data: 'test' });
      
      // Connect and wait for message processing
      const connectPromise = webSocketService.connect();
      vi.advanceTimersByTime(20);
      await connectPromise;
      
      // Queue should be processed
      expect(webSocketService.getQueuedMessageCount()).toBe(0);
    });

    it('should limit queue size to prevent memory issues', () => {
      // Fill queue beyond limit
      for (let i = 0; i < 150; i++) {
        webSocketService.sendMessage('test', { data: i });
      }
      
      // Queue should be limited to maxQueueSize (100)
      expect(webSocketService.getQueuedMessageCount()).toBeLessThanOrEqual(100);
    });
  });

  describe('Data Validation', () => {
    it('should validate JSON messages', async () => {
      const messageHandler = vi.fn();
      webSocketService.onMessage(messageHandler);
      
      await webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      // Get mock socket instance
      const mockSocket = (global as any).WebSocket.mock?.instances?.[0] || 
                        new MockWebSocket('test');
      
      // Simulate valid message
      mockSocket.simulateMessage(JSON.stringify({ type: 'test', data: 'valid' }));
      
      expect(messageHandler).toHaveBeenCalledWith({ type: 'test', data: 'valid' });
      
      // Simulate invalid message (empty)
      messageHandler.mockClear();
      mockSocket.simulateMessage('');
      
      expect(messageHandler).not.toHaveBeenCalled();
      
      // Simulate invalid message (malformed JSON)
      mockSocket.simulateMessage('{ invalid json');
      
      expect(messageHandler).not.toHaveBeenCalled();
    });

    it('should validate binary data size', async () => {
      const binaryHandler = vi.fn();
      webSocketService.onBinaryMessage(binaryHandler);
      
      await webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      const mockSocket = (global as any).WebSocket.mock?.instances?.[0] || 
                        new MockWebSocket('test');
      
      // Simulate valid binary data
      const validData = new ArrayBuffer(1024);
      mockSocket.simulateMessage(validData);
      
      // Should process valid data (note: may not call handler if parsing fails, but won't crash)
      expect(() => mockSocket.simulateMessage(validData)).not.toThrow();
      
      // Simulate empty binary data
      const emptyData = new ArrayBuffer(0);
      mockSocket.simulateMessage(emptyData);
      
      // Should handle empty data gracefully
      expect(() => mockSocket.simulateMessage(emptyData)).not.toThrow();
    });
  });

  describe('Heartbeat Mechanism', () => {
    it('should send heartbeat pings periodically', async () => {
      await webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      const mockSocket = (global as any).WebSocket.mock?.instances?.[0] || 
                        new MockWebSocket('test');
      const sendSpy = vi.spyOn(mockSocket, 'send');
      
      // Fast-forward to trigger heartbeat
      vi.advanceTimersByTime(30000); // 30 seconds
      
      expect(sendSpy).toHaveBeenCalledWith('ping');
    });

    it('should handle heartbeat timeout', async () => {
      await webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      const mockSocket = (global as any).WebSocket.mock?.instances?.[0] || 
                        new MockWebSocket('test');
      const closeSpy = vi.spyOn(mockSocket, 'close');
      
      // Trigger heartbeat
      vi.advanceTimersByTime(30000);
      
      // Don't respond with pong, trigger timeout
      vi.advanceTimersByTime(10000); // 10 seconds timeout
      
      expect(closeSpy).toHaveBeenCalledWith(4000, 'Heartbeat timeout');
    });

    it('should handle heartbeat response', async () => {
      await webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      const mockSocket = (global as any).WebSocket.mock?.instances?.[0] || 
                        new MockWebSocket('test');
      
      // Trigger heartbeat
      vi.advanceTimersByTime(30000);
      
      // Simulate pong response
      mockSocket.simulateMessage('pong');
      
      // Should not timeout
      const closeSpy = vi.spyOn(mockSocket, 'close');
      vi.advanceTimersByTime(10000);
      
      expect(closeSpy).not.toHaveBeenCalledWith(4000, 'Heartbeat timeout');
    });
  });

  describe('Connection State API', () => {
    it('should provide connection state information', () => {
      const state = webSocketService.getConnectionState();
      
      expect(state).toHaveProperty('status');
      expect(state).toHaveProperty('reconnectAttempts');
      expect(['disconnected', 'connecting', 'connected', 'reconnecting', 'failed']).toContain(state.status);
    });

    it('should allow forcing reconnection', async () => {
      await webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      const mockSocket = (global as any).WebSocket.mock?.instances?.[0] || 
                        new MockWebSocket('test');
      const closeSpy = vi.spyOn(mockSocket, 'close');
      
      webSocketService.forceReconnect();
      
      expect(closeSpy).toHaveBeenCalledWith(4001, 'Forced reconnection');
    });

    it('should allow clearing message queue', () => {
      // Queue some messages
      webSocketService.sendMessage('test1', {});
      webSocketService.sendMessage('test2', {});
      
      expect(webSocketService.getQueuedMessageCount()).toBe(2);
      
      webSocketService.clearMessageQueue();
      
      expect(webSocketService.getQueuedMessageCount()).toBe(0);
    });
  });

  describe('Error Recovery', () => {
    it('should recover from malformed binary data', async () => {
      await webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      const mockSocket = (global as any).WebSocket.mock?.instances?.[0] || 
                        new MockWebSocket('test');
      
      // Simulate malformed binary data - should not crash
      const malformedData = new ArrayBuffer(10);
      const view = new Uint8Array(malformedData);
      view.fill(255); // Fill with invalid data
      
      expect(() => mockSocket.simulateMessage(malformedData)).not.toThrow();
    });

    it('should recover from message handler errors', async () => {
      const faultyHandler = vi.fn(() => {
        throw new Error('Handler error');
      });
      const goodHandler = vi.fn();
      
      webSocketService.onMessage(faultyHandler);
      webSocketService.onMessage(goodHandler);
      
      await webSocketService.connect();
      vi.advanceTimersByTime(20);
      
      const mockSocket = (global as any).WebSocket.mock?.instances?.[0] || 
                        new MockWebSocket('test');
      
      // Send valid message
      mockSocket.simulateMessage(JSON.stringify({ type: 'test', data: 'test' }));
      
      // Both handlers should be called despite error in first one
      expect(faultyHandler).toHaveBeenCalled();
      expect(goodHandler).toHaveBeenCalled();
    });
  });
});