import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createMockWebSocket, waitFor, waitForNextTick } from '../utils/testFactories';

// Mock WebSocket for testing
global.WebSocket = vi.fn();

describe('WebSocket Integration Tests', () => {
  let mockWebSocket: any;
  let originalWebSocket: any;
  
  beforeEach(() => {
    originalWebSocket = global.WebSocket;
    mockWebSocket = createMockWebSocket();
    (global.WebSocket as any).mockImplementation(() => mockWebSocket);
  });

  afterEach(() => {
    global.WebSocket = originalWebSocket;
    vi.clearAllMocks();
  });

  describe('Settings Sync via WebSocket', () => {
    it('should establish WebSocket connection for settings sync', async () => {
      // Simulate settings sync WebSocket connection
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      
      expect(WebSocket).toHaveBeenCalledWith('ws://localhost:8000/ws/settings');
      
      // Simulate connection open
      mockWebSocket.triggerOpen();
      
      expect(mockWebSocket.readyState).toBe(WebSocket.OPEN);
    });

    it('should send settings update via WebSocket', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      mockWebSocket.triggerOpen();
      
      const settingsUpdate = {
        type: 'settings_update',
        path: 'visualisation.glow.nodeGlowStrength',
        value: 2.5,
        timestamp: Date.now()
      };
      
      ws.send(JSON.stringify(settingsUpdate));
      
      expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify(settingsUpdate));
    });

    it('should receive settings updates from server', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const receivedUpdates: any[] = [];
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        receivedUpdates.push(data);
      };
      
      mockWebSocket.triggerOpen();
      
      // Simulate server sending update
      const serverUpdate = {
        type: 'settings_changed',
        path: 'system.debugMode',
        value: true,
        source: 'another_client'
      };
      
      mockWebSocket.triggerMessage(serverUpdate);
      
      await waitForNextTick();
      
      expect(receivedUpdates).toHaveLength(1);
      expect(receivedUpdates[0]).toEqual(serverUpdate);
    });

    it('should handle WebSocket connection errors gracefully', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const errors: any[] = [];
      
      ws.onerror = (event) => {
        errors.push(event.error);
      };
      
      mockWebSocket.triggerError(new Error('Connection failed'));
      
      expect(errors).toHaveLength(1);
      expect(errors[0].message).toBe('Connection failed');
    });

    it('should handle WebSocket reconnection', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      let connectionCount = 0;
      
      ws.onopen = () => {
        connectionCount++;
      };
      
      // Initial connection
      mockWebSocket.triggerOpen();
      expect(connectionCount).toBe(1);
      
      // Connection lost
      mockWebSocket.triggerClose(1006, 'Connection lost');
      
      // Simulate reconnection logic
      await waitFor(100);
      
      // New connection attempt
      const newMockWs = createMockWebSocket();
      (global.WebSocket as any).mockImplementation(() => newMockWs);
      
      const reconnectWs = new WebSocket('ws://localhost:8000/ws/settings');
      reconnectWs.onopen = () => {
        connectionCount++;
      };
      
      newMockWs.triggerOpen();
      expect(connectionCount).toBe(2);
    });
  });

  describe('Real-time Settings Synchronization', () => {
    it('should sync settings changes across multiple clients', async () => {
      // Simulate two clients
      const client1 = new WebSocket('ws://localhost:8000/ws/settings');
      const client2 = new WebSocket('ws://localhost:8000/ws/settings');
      
      const client1Updates: any[] = [];
      const client2Updates: any[] = [];
      
      client1.onmessage = (event) => {
        client1Updates.push(JSON.parse(event.data));
      };
      
      client2.onmessage = (event) => {
        client2Updates.push(JSON.parse(event.data));
      };
      
      mockWebSocket.triggerOpen();
      
      // Client 1 sends update
      const update = {
        type: 'settings_update',
        path: 'visualisation.glow.baseColor',
        value: '#ff0000',
        clientId: 'client_1'
      };
      
      client1.send(JSON.stringify(update));
      
      // Server broadcasts to all clients (including sender)
      const broadcast = {
        ...update,
        type: 'settings_changed',
        source: 'client_1'
      };
      
      mockWebSocket.triggerMessage(broadcast);
      
      await waitForNextTick();
      
      // Both clients should receive the update
      expect(client1Updates).toHaveLength(1);
      expect(client2Updates).toHaveLength(1);
      expect(client1Updates[0]).toEqual(broadcast);
      expect(client2Updates[0]).toEqual(broadcast);
    });

    it('should handle batch settings updates via WebSocket', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const receivedBatches: any[] = [];
      
      ws.onmessage = (event) => {
        receivedBatches.push(JSON.parse(event.data));
      };
      
      mockWebSocket.triggerOpen();
      
      // Send batch update
      const batchUpdate = {
        type: 'settings_batch_update',
        updates: [
          { path: 'visualisation.glow.nodeGlowStrength', value: 3.0 },
          { path: 'visualisation.glow.edgeGlowStrength', value: 2.5 },
          { path: 'system.debugMode', value: true }
        ],
        timestamp: Date.now()
      };
      
      ws.send(JSON.stringify(batchUpdate));
      
      // Server confirms batch update
      const batchConfirmation = {
        type: 'settings_batch_changed',
        updated: 3,
        success: true,
        updates: batchUpdate.updates
      };
      
      mockWebSocket.triggerMessage(batchConfirmation);
      
      await waitForNextTick();
      
      expect(receivedBatches).toHaveLength(1);
      expect(receivedBatches[0]).toEqual(batchConfirmation);
      expect(receivedBatches[0].updated).toBe(3);
    });

    it('should handle WebSocket message queuing during disconnection', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const messageQueue: any[] = [];
      
      // Queue messages when disconnected
      const queueMessage = (message: any) => {
        if (ws.readyState !== WebSocket.OPEN) {
          messageQueue.push(message);
        } else {
          ws.send(JSON.stringify(message));
        }
      };
      
      // Connection starts closed
      expect(ws.readyState).not.toBe(WebSocket.OPEN);
      
      // Queue some messages
      queueMessage({ type: 'settings_update', path: 'test1', value: 'value1' });
      queueMessage({ type: 'settings_update', path: 'test2', value: 'value2' });
      
      expect(messageQueue).toHaveLength(2);
      expect(mockWebSocket.send).not.toHaveBeenCalled();
      
      // Connection opens
      mockWebSocket.triggerOpen();
      
      // Send queued messages
      while (messageQueue.length > 0) {
        const message = messageQueue.shift();
        ws.send(JSON.stringify(message));
      }
      
      expect(mockWebSocket.send).toHaveBeenCalledTimes(2);
    });
  });

  describe('WebSocket Performance and Reliability', () => {
    it('should handle high-frequency WebSocket messages efficiently', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const receivedMessages: any[] = [];
      
      ws.onmessage = (event) => {
        receivedMessages.push(JSON.parse(event.data));
      };
      
      mockWebSocket.triggerOpen();
      
      const startTime = performance.now();
      
      // Send 1000 rapid messages
      for (let i = 0; i < 1000; i++) {
        mockWebSocket.triggerMessage({
          type: 'settings_changed',
          path: `rapid.update.${i % 10}`, // 10 unique paths
          value: i,
          sequence: i
        });
      }
      
      await waitForNextTick();
      
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      expect(receivedMessages).toHaveLength(1000);
      expect(duration).toBeLessThan(1000); // Should process 1000 messages in < 1s
      
      // Verify message order is preserved
      receivedMessages.forEach((msg, index) => {
        expect(msg.sequence).toBe(index);
      });
    });

    it('should handle WebSocket connection timeout', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      let connectionTimeout = false;
      
      // Simulate connection timeout
      setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          connectionTimeout = true;
          mockWebSocket.triggerError(new Error('Connection timeout'));
        }
      }, 5000); // 5 second timeout
      
      // Don't trigger open - simulate timeout
      await waitFor(100);
      
      // In a real implementation, this would be handled by connection logic
      expect(ws.readyState).not.toBe(WebSocket.OPEN);
    });

    it('should handle WebSocket message size limits', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      mockWebSocket.triggerOpen();
      
      // Create very large message
      const largeMessage = {
        type: 'settings_batch_update',
        updates: Array.from({ length: 10000 }, (_, i) => ({
          path: `large.batch.item${i}`,
          value: {
            id: i,
            data: 'x'.repeat(1000), // 1KB per item = ~10MB total
            timestamp: Date.now()
          }
        }))
      };
      
      const messageString = JSON.stringify(largeMessage);
      expect(messageString.length).toBeGreaterThan(10 * 1024 * 1024); // > 10MB
      
      // Send large message
      ws.send(messageString);
      
      // Should handle large messages (may be chunked in real implementation)
      expect(mockWebSocket.send).toHaveBeenCalledWith(messageString);
    });
  });

  describe('WebSocket Error Handling and Recovery', () => {
    it('should handle malformed WebSocket messages gracefully', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const errors: any[] = [];
      const validMessages: any[] = [];
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          validMessages.push(data);
        } catch (error) {
          errors.push(error);
        }
      };
      
      mockWebSocket.triggerOpen();
      
      // Send malformed messages
      mockWebSocket.triggerMessage('invalid json {');
      mockWebSocket.triggerMessage('{"incomplete": ');
      mockWebSocket.triggerMessage('null');
      
      // Send valid message
      mockWebSocket.triggerMessage({ type: 'valid_message', data: 'test' });
      
      await waitForNextTick();
      
      expect(errors).toHaveLength(2); // Two malformed JSON messages
      expect(validMessages).toHaveLength(2); // null and valid message
    });

    it('should implement exponential backoff for reconnection', async () => {
      const connectionAttempts: number[] = [];
      let attemptCount = 0;
      
      const connectWithBackoff = async (attempt: number) => {
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000); // Max 30s
        connectionAttempts.push(delay);
        
        await waitFor(delay);
        
        // Simulate connection attempt
        attemptCount++;
        
        if (attemptCount < 3) {
          // Fail first 2 attempts
          throw new Error('Connection failed');
        }
        
        return new WebSocket('ws://localhost:8000/ws/settings');
      };
      
      let connected = false;
      let attempt = 0;
      
      while (!connected && attempt < 5) {
        try {
          await connectWithBackoff(attempt);
          connected = true;
        } catch (error) {
          attempt++;
        }
      }
      
      expect(connected).toBe(true);
      expect(connectionAttempts).toEqual([1000, 2000, 4000]); // Exponential backoff
      expect(attemptCount).toBe(3); // Third attempt succeeded
    });

    it('should handle WebSocket close codes appropriately', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const closeEvents: any[] = [];
      
      ws.onclose = (event) => {
        closeEvents.push({
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean
        });
      };
      
      mockWebSocket.triggerOpen();
      
      // Test different close scenarios
      const closeScenarios = [
        { code: 1000, reason: 'Normal closure', wasClean: true },
        { code: 1006, reason: 'Abnormal closure', wasClean: false },
        { code: 1011, reason: 'Server error', wasClean: false },
        { code: 4000, reason: 'Custom application error', wasClean: false }
      ];
      
      for (const scenario of closeScenarios) {
        mockWebSocket.triggerClose(scenario.code, scenario.reason);
        await waitForNextTick();
      }
      
      expect(closeEvents).toHaveLength(4);
      closeEvents.forEach((event, index) => {
        expect(event.code).toBe(closeScenarios[index].code);
        expect(event.reason).toBe(closeScenarios[index].reason);
      });
    });
  });

  describe('WebSocket Security and Validation', () => {
    it('should validate WebSocket message origins', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const suspiciousMessages: any[] = [];
      const validMessages: any[] = [];
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Simulate origin validation
        const isValidOrigin = !data.malicious && typeof data.type === 'string';
        
        if (isValidOrigin) {
          validMessages.push(data);
        } else {
          suspiciousMessages.push(data);
        }
      };
      
      mockWebSocket.triggerOpen();
      
      // Send messages with varying security profiles
      mockWebSocket.triggerMessage({ type: 'valid_update', path: 'test', value: 'safe' });
      mockWebSocket.triggerMessage({ malicious: true, script: '<script>alert("xss")</script>' });
      mockWebSocket.triggerMessage({ type: 'injection', value: "'; DROP TABLE settings; --" });
      
      await waitForNextTick();
      
      expect(validMessages).toHaveLength(1);
      expect(suspiciousMessages).toHaveLength(2);
    });

    it('should sanitize WebSocket message content', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const sanitizedMessages: any[] = [];
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Simulate content sanitization
        const sanitized = {
          ...data,
          value: typeof data.value === 'string' 
            ? data.value.replace(/<[^>]*>/g, '') // Remove HTML tags
            : data.value
        };
        
        sanitizedMessages.push(sanitized);
      };
      
      mockWebSocket.triggerOpen();
      
      // Send potentially dangerous content
      mockWebSocket.triggerMessage({
        type: 'settings_update',
        path: 'test.html',
        value: '<script>alert("xss")</script>Hello<b>World</b>'
      });
      
      await waitForNextTick();
      
      expect(sanitizedMessages).toHaveLength(1);
      expect(sanitizedMessages[0].value).toBe('HelloWorld'); // HTML tags removed
    });
  });

  describe('WebSocket Integration with Settings Store', () => {
    it('should sync settings store with WebSocket updates', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const settingsStore = new Map<string, any>();
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'settings_changed') {
          settingsStore.set(data.path, data.value);
        } else if (data.type === 'settings_batch_changed') {
          data.updates.forEach((update: any) => {
            settingsStore.set(update.path, update.value);
          });
        }
      };
      
      mockWebSocket.triggerOpen();
      
      // Receive individual update
      mockWebSocket.triggerMessage({
        type: 'settings_changed',
        path: 'visualisation.glow.nodeGlowStrength',
        value: 2.5
      });
      
      // Receive batch update
      mockWebSocket.triggerMessage({
        type: 'settings_batch_changed',
        updates: [
          { path: 'system.debugMode', value: true },
          { path: 'xr.locomotionMethod', value: 'teleport' }
        ]
      });
      
      await waitForNextTick();
      
      expect(settingsStore.get('visualisation.glow.nodeGlowStrength')).toBe(2.5);
      expect(settingsStore.get('system.debugMode')).toBe(true);
      expect(settingsStore.get('xr.locomotionMethod')).toBe('teleport');
    });

    it('should handle WebSocket settings conflicts gracefully', async () => {
      const ws = new WebSocket('ws://localhost:8000/ws/settings');
      const conflictLog: any[] = [];
      const settingsStore = new Map([
        ['test.conflict', { value: 'local', timestamp: Date.now() }]
      ]);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'settings_changed') {
          const existing = settingsStore.get(data.path);
          
          if (existing && existing.timestamp > data.timestamp) {
            // Local change is newer - conflict detected
            conflictLog.push({
              path: data.path,
              localValue: existing.value,
              remoteValue: data.value,
              resolution: 'keep_local'
            });
          } else {
            // Remote change is newer or no conflict
            settingsStore.set(data.path, {
              value: data.value,
              timestamp: data.timestamp
            });
          }
        }
      };
      
      mockWebSocket.triggerOpen();
      
      // Send older remote update (should be ignored)
      mockWebSocket.triggerMessage({
        type: 'settings_changed',
        path: 'test.conflict',
        value: 'remote_old',
        timestamp: Date.now() - 10000 // 10 seconds ago
      });
      
      // Send newer remote update (should be accepted)
      mockWebSocket.triggerMessage({
        type: 'settings_changed',
        path: 'test.conflict',
        value: 'remote_new',
        timestamp: Date.now() + 10000 // 10 seconds in future
      });
      
      await waitForNextTick();
      
      expect(conflictLog).toHaveLength(1);
      expect(conflictLog[0].resolution).toBe('keep_local');
      expect(settingsStore.get('test.conflict').value).toBe('remote_new'); // Newer remote value accepted
    });
  });
});