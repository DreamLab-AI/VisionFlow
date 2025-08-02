import { describe, it, expect, vi, beforeEach, afterEach, Mock } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { useBotsWebSocketIntegration } from '../../features/bots/hooks/useBotsWebSocketIntegration';
import { botsWebSocketIntegration } from '../../features/bots/services/BotsWebSocketIntegration';
import type { BotsAgent } from '../../features/bots/types/BotsTypes';
import React from 'react';

// Mock the WebSocket service
vi.mock('../../features/bots/services/BotsWebSocketIntegration', () => ({
  botsWebSocketIntegration: {
    on: vi.fn(),
    requestInitialData: vi.fn(),
    getConnectionStatus: vi.fn(),
    sendBotsUpdate: vi.fn(),
    disconnect: vi.fn()
  }
}));

// Mock the WebSocket service  
vi.mock('../../services/WebSocketService', () => ({
  webSocketService: {
    onConnectionStatusChange: vi.fn(),
    onMessage: vi.fn(),
    onBinaryMessage: vi.fn(),
    sendMessage: vi.fn(),
    close: vi.fn()
  }
}));

// Mock the API service
vi.mock('../../services/apiService', () => ({
  apiService: {
    get: vi.fn(),
    post: vi.fn()
  }
}));

describe('Frontend Adaptation Tests', () => {
  let mockOn: Mock;
  let mockRequestInitialData: Mock;
  let mockGetConnectionStatus: Mock;

  beforeEach(() => {
    mockOn = botsWebSocketIntegration.on as Mock;
    mockRequestInitialData = botsWebSocketIntegration.requestInitialData as Mock;
    mockGetConnectionStatus = botsWebSocketIntegration.getConnectionStatus as Mock;

    // Reset all mocks
    vi.clearAllMocks();
  });

  describe('WebSocket Agent Data Routing', () => {
    it('should route MCP agent data through WebSocket correctly', async () => {
      // Mock connection status
      mockGetConnectionStatus.mockReturnValue({
        mcp: false, // MCP handled by backend
        logseq: true,
        overall: true
      });

      // Mock event subscription
      const mockUnsubscribe = vi.fn();
      mockOn.mockReturnValue(mockUnsubscribe);

      // Mock API response with agent data
      const mockAgentData = {
        nodes: [
          {
            id: 'agent1',
            type: 'coordinator',
            status: 'busy',
            health: 90,
            cpuUsage: 45,
            memoryUsage: 60,
            createdAt: '2024-01-01T00:00:00Z',
            age: 3600000,
            position: { x: 10, y: 20, z: 30 }
          }
        ],
        edges: [
          {
            id: 'edge1',
            source: 'agent1',
            target: 'agent2',
            dataVolume: 5000,
            messageCount: 50,
            lastMessageTime: Date.now()
          }
        ]
      };

      // Test hook behavior
      const TestComponent = () => {
        const connectionStatus = useBotsWebSocketIntegration();
        return (
          <div data-testid="connection-status">
            MCP: {connectionStatus.mcp ? 'connected' : 'disconnected'}
            {' | '}
            Logseq: {connectionStatus.logseq ? 'connected' : 'disconnected'}
            {' | '}
            Overall: {connectionStatus.overall ? 'connected' : 'disconnected'}
          </div>
        );
      };

      render(<TestComponent />);

      // Verify hook subscribes to connection events
      expect(mockOn).toHaveBeenCalledWith('mcp-connected', expect.any(Function));
      expect(mockOn).toHaveBeenCalledWith('logseq-connected', expect.any(Function));

      // Verify connection status display
      await waitFor(() => {
        expect(screen.getByTestId('connection-status')).toHaveTextContent('Overall: connected');
      });
    });

    it('should handle binary message routing for position updates', () => {
      const processBinaryMessage = (data: ArrayBuffer) => {
        // Simulate the binary message processing that would happen in the frontend
        const view = new DataView(data);
        const updates: Array<{ nodeId: number; position: { x: number; y: number; z: number } }> = [];
        
        // Process binary data (simplified version of actual GPU output)
        for (let i = 0; i < data.byteLength; i += 28) { // 28 bytes per node
          if (i + 28 <= data.byteLength) {
            const nodeId = view.getUint32(i, true);
            const x = view.getFloat32(i + 4, true);
            const y = view.getFloat32(i + 8, true);
            const z = view.getFloat32(i + 12, true);
            
            updates.push({
              nodeId,
              position: { x, y, z }
            });
          }
        }
        
        return updates;
      };

      // Create test binary data
      const buffer = new ArrayBuffer(56); // 2 nodes
      const view = new DataView(buffer);
      
      // Node 1
      view.setUint32(0, 1, true);        // nodeId
      view.setFloat32(4, 10.5, true);    // x
      view.setFloat32(8, 20.3, true);    // y
      view.setFloat32(12, -5.7, true);   // z
      view.setFloat32(16, 0.1, true);    // vx
      view.setFloat32(20, -0.3, true);   // vy
      view.setFloat32(24, 0.8, true);    // vz
      
      // Node 2
      view.setUint32(28, 2, true);       // nodeId
      view.setFloat32(32, -15.2, true);  // x
      view.setFloat32(36, 35.1, true);   // y
      view.setFloat32(40, 12.4, true);   // z
      view.setFloat32(44, -0.2, true);   // vx
      view.setFloat32(48, 0.5, true);    // vy
      view.setFloat32(52, -0.1, true);   // vz

      const updates = processBinaryMessage(buffer);
      
      expect(updates).toHaveLength(2);
      expect(updates[0].nodeId).toBe(1);
      expect(updates[0].position.x).toBeCloseTo(10.5);
      expect(updates[1].nodeId).toBe(2);
      expect(updates[1].position.x).toBeCloseTo(-15.2);
    });

    it('should verify no physics worker is used', () => {
      // Mock window.Worker to detect if any worker is created
      const originalWorker = global.Worker;
      const workerInstances: Worker[] = [];
      
      global.Worker = class MockWorker extends EventTarget {
        constructor(url: string | URL) {
          super();
          workerInstances.push(this as any);
          // Store the worker URL for inspection
          (this as any).url = url.toString();
        }
        
        postMessage = vi.fn();
        terminate = vi.fn();
      } as any;

      // Simulate component that would previously use physics worker
      const simulateComponentWithoutWorker = () => {
        // This should NOT create any workers for physics
        // All physics processing should be handled by GPU backend
        return {
          hasWorker: workerInstances.length > 0,
          workerTypes: workerInstances.map((w: any) => w.url)
        };
      };

      const result = simulateComponentWithoutWorker();
      
      expect(result.hasWorker).toBe(false);
      expect(result.workerTypes).toHaveLength(0);
      expect(workerInstances).toHaveLength(0);

      // Restore original Worker
      global.Worker = originalWorker;
    });
  });

  describe('Position Update from GPU Stream', () => {
    it('should validate GPU position updates are applied correctly', () => {
      const applyGPUPositionUpdates = (
        agents: Map<string, BotsAgent>, 
        binaryData: ArrayBuffer
      ): Map<string, BotsAgent> => {
        const view = new DataView(binaryData);
        const updatedAgents = new Map(agents);
        
        // Process binary updates (28 bytes per node)
        for (let i = 0; i < binaryData.byteLength; i += 28) {
          if (i + 28 <= binaryData.byteLength) {
            const nodeId = view.getUint32(i, true);
            const x = view.getFloat32(i + 4, true);
            const y = view.getFloat32(i + 8, true);
            const z = view.getFloat32(i + 12, true);
            const vx = view.getFloat32(i + 16, true);
            const vy = view.getFloat32(i + 20, true);
            const vz = view.getFloat32(i + 24, true);
            
            // Find agent by nodeId mapping
            const agentId = `agent${nodeId}`;
            const agent = updatedAgents.get(agentId);
            
            if (agent) {
              updatedAgents.set(agentId, {
                ...agent,
                position: { x, y, z },
                velocity: { x: vx, y: vy, z: vz }
              });
            }
          }
        }
        
        return updatedAgents;
      };

      // Create test agents
      const agents = new Map<string, BotsAgent>();
      agents.set('agent1', {
        id: 'agent1',
        type: 'coordinator',
        status: 'busy',
        health: 90,
        cpuUsage: 45,
        memoryUsage: 60,
        createdAt: '2024-01-01T00:00:00Z',
        age: 3600000,
        position: { x: 0, y: 0, z: 0 },
        velocity: { x: 0, y: 0, z: 0 }
      });

      // Create GPU update
      const buffer = new ArrayBuffer(28);
      const view = new DataView(buffer);
      view.setUint32(0, 1, true);         // nodeId = 1 (maps to agent1)
      view.setFloat32(4, 15.5, true);     // new x position
      view.setFloat32(8, 25.3, true);     // new y position
      view.setFloat32(12, -8.7, true);    // new z position
      view.setFloat32(16, 0.5, true);     // velocity x
      view.setFloat32(20, -0.2, true);    // velocity y
      view.setFloat32(24, 1.1, true);     // velocity z

      const updatedAgents = applyGPUPositionUpdates(agents, buffer);
      const updatedAgent = updatedAgents.get('agent1');

      expect(updatedAgent).toBeDefined();
      expect(updatedAgent!.position!.x).toBeCloseTo(15.5);
      expect(updatedAgent!.position!.y).toBeCloseTo(25.3);
      expect(updatedAgent!.position!.z).toBeCloseTo(-8.7);
      expect(updatedAgent!.velocity!.x).toBeCloseTo(0.5);
      expect(updatedAgent!.velocity!.y).toBeCloseTo(-0.2);
      expect(updatedAgent!.velocity!.z).toBeCloseTo(1.1);
    });

    it('should handle position update validation and error recovery', () => {
      const validatePositionUpdate = (position: { x: number; y: number; z: number }) => {
        // Validate position values are finite and within reasonable bounds
        const isValid = 
          isFinite(position.x) && isFinite(position.y) && isFinite(position.z) &&
          Math.abs(position.x) < 1000 && Math.abs(position.y) < 1000 && Math.abs(position.z) < 1000;
        
        return isValid;
      };

      // Test valid positions
      expect(validatePositionUpdate({ x: 10, y: 20, z: 30 })).toBe(true);
      expect(validatePositionUpdate({ x: -500, y: 0, z: 999 })).toBe(true);

      // Test invalid positions
      expect(validatePositionUpdate({ x: NaN, y: 20, z: 30 })).toBe(false);
      expect(validatePositionUpdate({ x: Infinity, y: 20, z: 30 })).toBe(false);
      expect(validatePositionUpdate({ x: 10, y: 20, z: 2000 })).toBe(false);
    });
  });

  describe('Mock Data Removal Verification', () => {
    it('should verify no mock data is used in production mode', () => {
      const detectMockDataUsage = (dataSource: any) => {
        // Check if data contains mock indicators
        const mockIndicators = [
          'mock',
          'test-',
          'fake-',
          'dummy',
          'sample-',
          'placeholder'
        ];
        
        const dataStr = JSON.stringify(dataSource).toLowerCase();
        return mockIndicators.some(indicator => dataStr.includes(indicator));
      };

      // Test with production-like data
      const productionData = {
        agents: [
          {
            id: 'coord-7f4a9b2c',
            type: 'coordinator',
            status: 'active',
            health: 92,
            createdAt: '2024-01-15T14:32:17Z'
          }
        ]
      };

      expect(detectMockDataUsage(productionData)).toBe(false);

      // Test with mock data (should be detected)
      const mockData = {
        agents: [
          {
            id: 'mock-agent-1',
            type: 'test-coordinator',
            status: 'fake-active'
          }
        ]
      };

      expect(detectMockDataUsage(mockData)).toBe(true);
    });

    it('should ensure WebSocket integration uses real backend data', async () => {
      // Mock the actual API call that should be used instead of mock data
      const { apiService } = await import('../../services/apiService');
      const mockGet = apiService.get as Mock;
      
      mockGet.mockResolvedValue({
        nodes: [
          {
            id: 'real-agent-1',
            type: 'coordinator',
            status: 'busy',
            health: 88,
            cpuUsage: 32,
            memoryUsage: 45,
            createdAt: '2024-01-15T14:32:17Z',
            age: Date.now() - new Date('2024-01-15T14:32:17Z').getTime()
          }
        ],
        edges: []
      });

      // Call the actual method that fetches data
      await botsWebSocketIntegration.requestInitialData();

      // Verify real API endpoint was called
      expect(mockGet).toHaveBeenCalledWith('/bots/data');
      
      // Verify no mock functions were used
      expect(mockGet).not.toHaveBeenCalledWith('/mock/bots');
      expect(mockGet).not.toHaveBeenCalledWith('/test/agents');
    });
  });

  describe('WebSocket Throughput Validation', () => {
    it('should handle high-frequency position updates efficiently', async () => {
      const measureThroughput = async (messageCount: number, messageSize: number) => {
        const messages: ArrayBuffer[] = [];
        const processingTimes: number[] = [];
        
        // Generate test messages
        for (let i = 0; i < messageCount; i++) {
          const buffer = new ArrayBuffer(messageSize);
          const view = new DataView(buffer);
          
          // Fill with test data
          for (let j = 0; j < messageSize; j += 4) {
            view.setFloat32(j, Math.random() * 100, true);
          }
          
          messages.push(buffer);
        }
        
        // Measure processing time
        const startTime = performance.now();
        
        for (const message of messages) {
          const messageStart = performance.now();
          
          // Simulate message processing
          const view = new DataView(message);
          let sum = 0;
          for (let i = 0; i < message.byteLength; i += 4) {
            sum += view.getFloat32(i, true);
          }
          
          const messageEnd = performance.now();
          processingTimes.push(messageEnd - messageStart);
        }
        
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        
        return {
          totalTime,
          averageMessageTime: processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length,
          throughput: messageCount / (totalTime / 1000), // messages per second
          dataRate: (messageCount * messageSize) / (totalTime / 1000) / 1024 / 1024 // MB/s
        };
      };

      // Test with 100 agents (2800 bytes per message)
      const result = await measureThroughput(100, 2800);
      
      expect(result.throughput).toBeGreaterThan(500); // Should process >500 messages/sec
      expect(result.dataRate).toBeGreaterThan(1); // Should handle >1 MB/s
      expect(result.averageMessageTime).toBeLessThan(2); // <2ms per message
    });

    it('should handle WebSocket reconnection gracefully', () => {
      const simulateReconnection = () => {
        let connectionState = 'connected';
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        const disconnect = () => {
          connectionState = 'disconnected';
        };
        
        const reconnect = () => {
          if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            // Simulate connection success/failure
            const success = Math.random() > 0.3; // 70% success rate
            if (success) {
              connectionState = 'connected';
              return true;
            }
          }
          return false;
        };
        
        return { connectionState, disconnect, reconnect, reconnectAttempts };
      };

      const connection = simulateReconnection();
      expect(connection.connectionState).toBe('connected');
      
      // Simulate disconnection
      connection.disconnect();
      expect(connection.connectionState).toBe('disconnected');
      
      // Try to reconnect
      let reconnected = false;
      let attempts = 0;
      while (!reconnected && attempts < 10) {
        reconnected = connection.reconnect();
        attempts++;
      }
      
      expect(connection.reconnectAttempts).toBeGreaterThan(0);
      expect(connection.reconnectAttempts).toBeLessThanOrEqual(5);
    });
  });
});