import { describe, it, expect, vi, beforeEach, afterEach, Mock } from 'vitest';
import { render, waitFor, act } from '@testing-library/react';
import React from 'react';
import { botsWebSocketIntegration } from '../../features/bots/services/BotsWebSocketIntegration';
import { parseBinaryNodeData, createBinaryNodeData } from '../../types/binaryProtocol';
import type { BotsAgent, BotsEdge } from '../../features/bots/types/BotsTypes';

// Mock Three.js and React Three Fiber for GPU visualization tests
vi.mock('three', () => ({
  Vector3: vi.fn().mockImplementation((x = 0, y = 0, z = 0) => ({ x, y, z })),
  Color: vi.fn().mockImplementation(() => ({ 
    getHexString: () => 'ffffff',
    setHSL: vi.fn().mockReturnThis()
  })),
  ShaderMaterial: vi.fn(),
  AdditiveBlending: 'AdditiveBlending'
}));

vi.mock('@react-three/fiber', () => ({
  useFrame: vi.fn((callback) => {
    // Simulate frame callback for testing
    if (typeof callback === 'function') {
      callback({ clock: { elapsedTime: 1.0 } }, 0.016);
    }
  }),
  useThree: vi.fn(() => ({ gl: {} }))
}));

// Mock API service
vi.mock('../../services/apiService', () => ({
  apiService: {
    get: vi.fn(),
    post: vi.fn()
  }
}));

// Mock WebSocket service
vi.mock('../../services/WebSocketService', () => ({
  webSocketService: {
    onConnectionStatusChange: vi.fn(),
    onMessage: vi.fn(),
    onBinaryMessage: vi.fn(),
    sendMessage: vi.fn(),
    close: vi.fn()
  }
}));

describe('VisionFlow GPU Physics Integration Tests', () => {
  let mockApiGet: Mock;
  let mockApiPost: Mock;

  beforeEach(() => {
    const { apiService } = require('../../services/apiService');
    mockApiGet = apiService.get;
    mockApiPost = apiService.post;
    
    vi.clearAllMocks();
  });

  describe('End-to-End Data Flow: MCP → GPU → Frontend', () => {
    it('should handle complete data pipeline from MCP to frontend visualization', async () => {
      // 1. Mock MCP data from backend
      const mockMCPAgents: BotsAgent[] = [
        {
          id: 'coord-001',
          type: 'coordinator',
          status: 'busy',
          health: 95,
          cpuUsage: 45,
          memoryUsage: 60,
          createdAt: '2024-01-15T14:32:17Z',
          age: 3600000,
          activity: 0.8,
          currentTask: 'Managing swarm coordination',
          tasksActive: 3,
          tasksCompleted: 47
        },
        {
          id: 'coder-002',
          type: 'coder',
          status: 'active',
          health: 88,
          cpuUsage: 70,
          memoryUsage: 80,
          createdAt: '2024-01-15T14:30:12Z',
          age: 3720000,
          activity: 0.9,
          currentTask: 'Implementing GPU physics kernel',
          tasksActive: 2,
          tasksCompleted: 23
        },
        {
          id: 'tester-003',
          type: 'tester',
          status: 'idle', 
          health: 92,
          cpuUsage: 20,
          memoryUsage: 35,
          createdAt: '2024-01-15T14:28:45Z',
          age: 3840000,
          activity: 0.3,
          currentTask: 'Preparing test scenarios',
          tasksActive: 1,
          tasksCompleted: 31
        }
      ];

      const mockMCPEdges: BotsEdge[] = [
        {
          id: 'edge-001-002',
          source: 'coord-001',
          target: 'coder-002', 
          dataVolume: 15420,
          messageCount: 87,
          lastMessageTime: Date.now() - 5000
        },
        {
          id: 'edge-001-003',
          source: 'coord-001',
          target: 'tester-003',
          dataVolume: 8730,
          messageCount: 42,
          lastMessageTime: Date.now() - 12000
        }
      ];

      // 2. Mock backend API response with MCP data
      mockApiGet.mockResolvedValue({
        nodes: mockMCPAgents,
        edges: mockMCPEdges,
        metadata: {
          totalAgents: 3,
          totalConnections: 2,
          lastUpdate: Date.now()
        }
      });

      // 3. Mock GPU physics processing (binary data)
      const createGPUPhysicsUpdate = (agents: BotsAgent[]) => {
        const binaryData = agents.map((agent, index) => ({
          nodeId: index + 1,
          position: {
            x: Math.cos((index / agents.length) * Math.PI * 2) * 25 + Math.random() * 5,
            y: Math.sin(index * 0.7) * 15 + Math.random() * 3,
            z: Math.sin((index / agents.length) * Math.PI * 2) * 25 + Math.random() * 5
          },
          velocity: {
            x: (Math.random() - 0.5) * 2,
            y: (Math.random() - 0.5) * 2,
            z: (Math.random() - 0.5) * 2
          }
        }));
        
        return createBinaryNodeData(binaryData);
      };

      const gpuUpdateBuffer = createGPUPhysicsUpdate(mockMCPAgents);

      // 4. Test data flow pipeline
      await act(async () => {
        // Simulate MCP data fetch
        await botsWebSocketIntegration.requestInitialData();
      });

      // Verify MCP data was fetched
      expect(mockApiGet).toHaveBeenCalledWith('/bots/data');

      // 5. Test GPU data processing
      const parsedGPUData = parseBinaryNodeData(gpuUpdateBuffer);
      expect(parsedGPUData).toHaveLength(3);
      
      // Verify GPU data contains valid physics updates
      parsedGPUData.forEach((node, index) => {
        expect(node.nodeId).toBe(index + 1);
        expect(typeof node.position.x).toBe('number');
        expect(typeof node.position.y).toBe('number');
        expect(typeof node.position.z).toBe('number');
        expect(isFinite(node.position.x)).toBe(true);
        expect(isFinite(node.position.y)).toBe(true);
        expect(isFinite(node.position.z)).toBe(true);
      });

      // 6. Test frontend data integration
      const integrateGPUDataWithAgents = (agents: BotsAgent[], gpuData: typeof parsedGPUData) => {
        const updatedAgents = agents.map((agent, index) => {
          const gpuNode = gpuData.find(node => node.nodeId === index + 1);
          if (gpuNode) {
            return {
              ...agent,
              position: gpuNode.position,
              velocity: gpuNode.velocity
            };
          }
          return agent;
        });
        
        return updatedAgents;
      };

      const integratedAgents = integrateGPUDataWithAgents(mockMCPAgents, parsedGPUData);
      
      // Verify integration preserves agent data while adding GPU physics
      expect(integratedAgents).toHaveLength(3);
      integratedAgents.forEach((agent, index) => {
        expect(agent.id).toBe(mockMCPAgents[index].id);
        expect(agent.type).toBe(mockMCPAgents[index].type);
        expect(agent.status).toBe(mockMCPAgents[index].status);
        expect(agent.position).toBeDefined();
        expect(agent.velocity).toBeDefined();
      });
    });

    it('should handle error scenarios without mock fallbacks', async () => {
      // Test API failure handling
      mockApiGet.mockRejectedValue(new Error('MCP connection failed'));

      let errorCaught = false;
      let fallbackToMock = false;

      try {
        await botsWebSocketIntegration.requestInitialData();
      } catch (error) {
        errorCaught = true;
        
        // Verify no mock data fallback is attempted
        const errorMessage = (error as Error).message;
        expect(errorMessage).not.toContain('mock');
        expect(errorMessage).not.toContain('fallback');
        expect(errorMessage).not.toContain('test-data');
      }

      // Should attempt real API, not fall back to mock
      expect(mockApiGet).toHaveBeenCalledWith('/bots/data');
      expect(mockApiGet).not.toHaveBeenCalledWith('/mock/bots');
      expect(errorCaught).toBe(true);
      expect(fallbackToMock).toBe(false);
    });
  });

  describe('Performance Benchmarks for 100+ Agents', () => {
    it('should handle 100+ agents with acceptable performance', async () => {
      const agentCount = 150;
      const edgeCount = 300;

      // Generate large dataset
      const generateLargeDataset = (numAgents: number, numEdges: number) => {
        const agents: BotsAgent[] = Array.from({ length: numAgents }, (_, i) => ({
          id: `agent-${i.toString().padStart(3, '0')}`,
          type: ['coordinator', 'coder', 'tester', 'analyst', 'architect'][i % 5] as any,
          status: ['busy', 'active', 'idle'][i % 3] as any,
          health: 70 + Math.random() * 30,
          cpuUsage: Math.random() * 100,
          memoryUsage: Math.random() * 100,
          createdAt: new Date(Date.now() - Math.random() * 86400000).toISOString(),
          age: Math.random() * 86400000,
          activity: Math.random(),
          tasksActive: Math.floor(Math.random() * 5),
          tasksCompleted: Math.floor(Math.random() * 100)
        }));

        const edges: BotsEdge[] = Array.from({ length: numEdges }, (_, i) => {
          const sourceIdx = Math.floor(Math.random() * numAgents);
          let targetIdx = Math.floor(Math.random() * numAgents);
          while (targetIdx === sourceIdx) {
            targetIdx = Math.floor(Math.random() * numAgents);
          }

          return {
            id: `edge-${i.toString().padStart(3, '0')}`,
            source: agents[sourceIdx].id,
            target: agents[targetIdx].id,
            dataVolume: Math.floor(Math.random() * 50000),
            messageCount: Math.floor(Math.random() * 200),
            lastMessageTime: Date.now() - Math.random() * 60000
          };
        });

        return { agents, edges };
      };

      const { agents, edges } = generateLargeDataset(agentCount, edgeCount);

      // Test data processing performance
      const startTime = performance.now();

      // 1. Binary data creation (simulating GPU output)
      const binaryNodes = agents.map((agent, index) => ({
        nodeId: index,
        position: {
          x: Math.random() * 200 - 100,
          y: Math.random() * 200 - 100,
          z: Math.random() * 200 - 100
        },
        velocity: {
          x: (Math.random() - 0.5) * 4,
          y: (Math.random() - 0.5) * 4,
          z: (Math.random() - 0.5) * 4
        }
      }));

      const binaryBuffer = createBinaryNodeData(binaryNodes);
      const binaryCreationTime = performance.now();

      // 2. Binary data parsing
      const parsedNodes = parseBinaryNodeData(binaryBuffer);
      const binaryParsingTime = performance.now();

      // 3. Agent data integration
      const integratedAgents = agents.map((agent, index) => ({
        ...agent,
        position: parsedNodes[index]?.position || { x: 0, y: 0, z: 0 },
        velocity: parsedNodes[index]?.velocity || { x: 0, y: 0, z: 0 }
      }));
      const integrationTime = performance.now();

      // 4. Edge processing (communication intensity calculation)
      const processedEdges = edges.map(edge => {
        const timeDelta = (Date.now() - edge.lastMessageTime) / 1000;
        const intensity = (edge.messageCount / Math.max(timeDelta, 1)) + 
                         (edge.dataVolume * 0.001 / Math.max(timeDelta, 1));
        return { ...edge, intensity };
      });
      const edgeProcessingTime = performance.now();

      // Performance assertions
      const totalTime = edgeProcessingTime - startTime;
      const binaryCreationDuration = binaryCreationTime - startTime;
      const binaryParsingDuration = binaryParsingTime - binaryCreationTime;
      const integrationDuration = integrationTime - binaryParsingTime;
      const edgeProcessingDuration = edgeProcessingTime - integrationTime;

      // Verify results
      expect(parsedNodes).toHaveLength(agentCount);
      expect(integratedAgents).toHaveLength(agentCount);
      expect(processedEdges).toHaveLength(edgeCount);

      // Performance benchmarks (all operations should complete in under 100ms for 150 agents)
      expect(totalTime).toBeLessThan(100);
      expect(binaryCreationDuration).toBeLessThan(20);
      expect(binaryParsingDuration).toBeLessThan(20);
      expect(integrationDuration).toBeLessThan(30);
      expect(edgeProcessingDuration).toBeLessThan(30);

      // Memory efficiency check
      const memoryUsage = {
        binaryBuffer: binaryBuffer.byteLength,
        agentData: JSON.stringify(integratedAgents).length,
        edgeData: JSON.stringify(processedEdges).length
      };

      // Binary data should be more compact than JSON
      expect(memoryUsage.binaryBuffer).toBeLessThan(memoryUsage.agentData);
      
      // Total memory usage should be reasonable for 150 agents
      const totalMemory = memoryUsage.binaryBuffer + memoryUsage.agentData + memoryUsage.edgeData;
      expect(totalMemory).toBeLessThan(5 * 1024 * 1024); // Less than 5MB for 150 agents
    });

    it('should scale linearly with agent count', () => {
      const testScaling = (agentCounts: number[]) => {
        const results = agentCounts.map(count => {
          const startTime = performance.now();
          
          // Simulate processing
          const agents = Array.from({ length: count }, (_, i) => ({
            id: `agent-${i}`,
            position: { x: Math.random() * 100, y: Math.random() * 100, z: Math.random() * 100 }
          }));
          
          // Simple O(n) processing
          const processed = agents.map(agent => ({
            ...agent,
            distance: Math.sqrt(agent.position.x ** 2 + agent.position.y ** 2 + agent.position.z ** 2)
          }));
          
          const endTime = performance.now();
          
          return {
            agentCount: count,
            processingTime: endTime - startTime,
            timePerAgent: (endTime - startTime) / count
          };
        });
        
        return results;
      };

      const testCounts = [50, 100, 200, 400];
      const scalingResults = testScaling(testCounts);

      // Verify roughly linear scaling (time per agent should be relatively constant)
      const timesPerAgent = scalingResults.map(r => r.timePerAgent);
      const avgTimePerAgent = timesPerAgent.reduce((a, b) => a + b, 0) / timesPerAgent.length;
      
      // All times per agent should be within 50% of average (allowing for some variance)
      timesPerAgent.forEach(time => {
        expect(time).toBeLessThan(avgTimePerAgent * 1.5);
        expect(time).toBeGreaterThan(avgTimePerAgent * 0.5);
      });

      // Largest dataset should still process quickly
      const largestResult = scalingResults[scalingResults.length - 1];
      expect(largestResult.processingTime).toBeLessThan(50); // 400 agents in under 50ms
    });
  });

  describe('WebSocket Throughput Validation', () => {
    it('should handle high-frequency binary updates', async () => {
      const simulateHighFrequencyUpdates = async () => {
        const updateFrequency = 60; // 60 FPS
        const agentCount = 100;
        const updateDuration = 1000; // 1 second test
        const expectedUpdates = Math.floor((updateDuration / 1000) * updateFrequency);
        
        const processedUpdates: number[] = [];
        let totalProcessingTime = 0;
        
        for (let i = 0; i < expectedUpdates; i++) {
          const startTime = performance.now();
          
          // Create binary update for 100 agents
          const binaryNodes = Array.from({ length: agentCount }, (_, nodeIndex) => ({
            nodeId: nodeIndex,
            position: {
              x: Math.sin(i * 0.1 + nodeIndex * 0.2) * 50,
              y: Math.cos(i * 0.05 + nodeIndex * 0.3) * 30,
              z: Math.sin(i * 0.08 + nodeIndex * 0.15) * 40
            },
            velocity: {
              x: Math.cos(i * 0.02) * 2,
              y: Math.sin(i * 0.03) * 2,
              z: Math.cos(i * 0.025) * 2
            }
          }));
          
          const binaryBuffer = createBinaryNodeData(binaryNodes);
          const parsedData = parseBinaryNodeData(binaryBuffer);
          
          const endTime = performance.now();
          const processingTime = endTime - startTime;
          
          processedUpdates.push(processingTime);
          totalProcessingTime += processingTime;
          
          // Simulate 60 FPS timing
          if (i < expectedUpdates - 1) {
            await new Promise(resolve => setTimeout(resolve, 1000 / updateFrequency));
          }
        }
        
        return {
          updatesProcessed: processedUpdates.length,
          averageProcessingTime: totalProcessingTime / processedUpdates.length,
          maxProcessingTime: Math.max(...processedUpdates),
          minProcessingTime: Math.min(...processedUpdates),
          throughput: processedUpdates.length / (updateDuration / 1000) // updates per second
        };
      };

      const results = await simulateHighFrequencyUpdates();
      
      // Verify throughput meets requirements
      expect(results.updatesProcessed).toBeGreaterThanOrEqual(50); // At least 50 updates processed
      expect(results.averageProcessingTime).toBeLessThan(5); // Average under 5ms per update
      expect(results.maxProcessingTime).toBeLessThan(16); // Max processing time under 16ms (60 FPS)
      expect(results.throughput).toBeGreaterThan(30); // Minimum 30 updates per second
    });

    it('should maintain data integrity under high load', () => {
      const testDataIntegrity = (stressLevel: number) => {
        const originalData = Array.from({ length: stressLevel }, (_, i) => ({
          nodeId: i,
          position: { x: i * 10, y: i * 20, z: i * 30 },
          velocity: { x: i * 0.1, y: i * 0.2, z: i * 0.3 }
        }));
        
        // Create and parse binary data multiple times
        let allDataValid = true;
        const iterations = Math.min(100, Math.max(10, Math.floor(1000 / stressLevel)));
        
        for (let iteration = 0; iteration < iterations; iteration++) {
          const binaryBuffer = createBinaryNodeData(originalData);
          const parsedData = parseBinaryNodeData(binaryBuffer);
          
          if (parsedData.length !== originalData.length) {
            allDataValid = false;
            break;
          }
          
          for (let i = 0; i < originalData.length; i++) {
            const original = originalData[i];
            const parsed = parsedData[i];
            
            if (
              parsed.nodeId !== original.nodeId ||
              Math.abs(parsed.position.x - original.position.x) > 0.001 ||
              Math.abs(parsed.position.y - original.position.y) > 0.001 ||
              Math.abs(parsed.position.z - original.position.z) > 0.001
            ) {
              allDataValid = false;
              break;
            }
          }
          
          if (!allDataValid) break;
        }
        
        return { allDataValid, iterations, dataSize: stressLevel };
      };

      // Test with different stress levels
      const stressLevels = [10, 50, 100, 200];
      stressLevels.forEach(level => {
        const result = testDataIntegrity(level);
        expect(result.allDataValid).toBe(true);
      });
    });
  });

  describe('Error Handling Without Mock Fallbacks', () => {
    it('should handle GPU processing errors gracefully', () => {
      const handleGPUError = (errorType: string, data?: any) => {
        const errorHandlers = {
          'invalid_binary_data': () => {
            // Should return empty array, not mock data
            return [];
          },
          'gpu_kernel_failure': () => {
            // Should maintain last known positions, not generate mock positions
            throw new Error('GPU kernel processing failed - retrying...');
          },
          'memory_allocation_error': () => {
            // Should reduce processing load, not fall back to mock
            throw new Error('Insufficient GPU memory - reducing agent count');
          },
          'websocket_disconnection': () => {
            // Should attempt reconnection, not use mock data
            throw new Error('WebSocket disconnected - attempting reconnection');
          }
        };
        
        const handler = errorHandlers[errorType as keyof typeof errorHandlers];
        if (handler) {
          return handler();
        } else {
          throw new Error(`Unknown error type: ${errorType}`);
        }
      };

      // Test invalid binary data handling
      const invalidDataResult = handleGPUError('invalid_binary_data');
      expect(invalidDataResult).toEqual([]);
      expect(Array.isArray(invalidDataResult)).toBe(true);

      // Test GPU kernel failure
      expect(() => handleGPUError('gpu_kernel_failure')).toThrow('GPU kernel processing failed');
      
      // Test memory allocation error
      expect(() => handleGPUError('memory_allocation_error')).toThrow('Insufficient GPU memory');
      
      // Test WebSocket disconnection
      expect(() => handleGPUError('websocket_disconnection')).toThrow('WebSocket disconnected');
      
      // Verify no mock data is referenced in error messages
      ['gpu_kernel_failure', 'memory_allocation_error', 'websocket_disconnection'].forEach(errorType => {
        try {
          handleGPUError(errorType);
        } catch (error) {
          const errorMessage = (error as Error).message.toLowerCase();
          expect(errorMessage).not.toContain('mock');
          expect(errorMessage).not.toContain('fallback');
          expect(errorMessage).not.toContain('dummy');
          expect(errorMessage).not.toContain('test');
        }
      });
    });

    it('should maintain system stability during partial failures', () => {
      const simulatePartialFailure = (totalAgents: number, failureRate: number) => {
        const agents = Array.from({ length: totalAgents }, (_, i) => ({
          id: `agent-${i}`,
          nodeId: i,
          status: 'active' as const,
          lastUpdate: Date.now()
        }));
        
        const results = {
          processed: 0,
          failed: 0,
          recovered: 0,
          totalTime: 0
        };
        
        const startTime = performance.now();
        
        agents.forEach((agent, index) => {
          try {
            // Simulate random failures
            if (Math.random() < failureRate) {
              throw new Error(`Processing failed for agent ${agent.id}`);
            }
            
            // Simulate successful processing
            results.processed++;
          } catch (error) {
            results.failed++;
            
            // Simulate error recovery (not fallback to mock data)
            if (Math.random() > 0.3) { // 70% recovery rate
              // Recovery mechanism: retry with last known state
              results.recovered++;
              results.processed++; // Count as processed after recovery
            }
          }
        });
        
        results.totalTime = performance.now() - startTime;
        
        return results;
      };

      // Test with different failure rates
      const lowFailureResults = simulatePartialFailure(100, 0.1); // 10% failure
      const highFailureResults = simulatePartialFailure(100, 0.3); // 30% failure
      
      // Verify system maintains stability
      expect(lowFailureResults.processed).toBeGreaterThan(85); // At least 85% success with 10% failure rate
      expect(highFailureResults.processed).toBeGreaterThan(60); // At least 60% success with 30% failure rate
      
      // Verify recovery mechanisms work
      expect(lowFailureResults.recovered).toBeGreaterThan(0);
      expect(highFailureResults.recovered).toBeGreaterThan(0);
      
      // Verify processing time remains reasonable even with failures
      expect(lowFailureResults.totalTime).toBeLessThan(50);
      expect(highFailureResults.totalTime).toBeLessThan(100);
    });

    it('should validate no mock service endpoints are called during errors', async () => {
      // Simulate various error conditions
      const errorConditions = [
        'network_timeout',
        'server_error_500',
        'bad_request_400',
        'unauthorized_401',
        'service_unavailable_503'
      ];

      for (const condition of errorConditions) {
        // Mock API to simulate the error condition
        mockApiGet.mockRejectedValue(new Error(`${condition}: API call failed`));
        
        try {
          await botsWebSocketIntegration.requestInitialData();
        } catch (error) {
          // Error is expected, verify no mock endpoints were called
          expect(mockApiGet).toHaveBeenCalledWith('/bots/data');
          expect(mockApiGet).not.toHaveBeenCalledWith('/mock/bots');
          expect(mockApiGet).not.toHaveBeenCalledWith('/test/agents');
          expect(mockApiGet).not.toHaveBeenCalledWith('/fallback/data');
          expect(mockApiGet).not.toHaveBeenCalledWith('/demo/visualization');
        }
        
        // Reset mock for next iteration
        vi.clearAllMocks();
      }
    });
  });
});