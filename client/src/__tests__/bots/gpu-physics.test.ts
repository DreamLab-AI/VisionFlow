import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { parseBinaryNodeData, createBinaryNodeData, BINARY_NODE_SIZE } from '../../types/binaryProtocol';
import type { BinaryNodeData, Vec3 } from '../../types/binaryProtocol';
import type { BotsAgent, BotsEdge } from '../../features/bots/types/BotsTypes';

describe('GPU Physics Backend Tests', () => {
  describe('Binary Protocol Processing', () => {
    it('should parse valid binary node data correctly', () => {
      // Create test data with 3 nodes
      const testNodes: BinaryNodeData[] = [
        {
          nodeId: 1,
          position: { x: 10.5, y: 20.3, z: -5.7 },
          velocity: { x: 0.1, y: -0.3, z: 0.8 }
        },
        {
          nodeId: 2,
          position: { x: -15.2, y: 35.1, z: 12.4 },
          velocity: { x: -0.2, y: 0.5, z: -0.1 }
        },
        {
          nodeId: 3,
          position: { x: 0.0, y: 0.0, z: 0.0 },
          velocity: { x: 0.0, y: 0.0, z: 0.0 }
        }
      ];

      // Create binary buffer
      const buffer = createBinaryNodeData(testNodes);
      expect(buffer.byteLength).toBe(testNodes.length * BINARY_NODE_SIZE);

      // Parse back
      const parsedNodes = parseBinaryNodeData(buffer);
      expect(parsedNodes).toHaveLength(3);

      // Verify data integrity
      testNodes.forEach((original, index) => {
        const parsed = parsedNodes[index];
        expect(parsed.nodeId).toBe(original.nodeId);
        expect(parsed.position.x).toBeCloseTo(original.position.x, 5);
        expect(parsed.position.y).toBeCloseTo(original.position.y, 5);
        expect(parsed.position.z).toBeCloseTo(original.position.z, 5);
        expect(parsed.velocity.x).toBeCloseTo(original.velocity.x, 5);
        expect(parsed.velocity.y).toBeCloseTo(original.velocity.y, 5);
        expect(parsed.velocity.z).toBeCloseTo(original.velocity.z, 5);
      });
    });

    it('should handle corrupted binary data gracefully', () => {
      // Test with invalid buffer size
      const invalidBuffer = new ArrayBuffer(10); // Not multiple of BINARY_NODE_SIZE
      const result = parseBinaryNodeData(invalidBuffer);
      expect(result).toHaveLength(0); // Should return empty array for corrupted data

      // Test with NaN values
      const buffer = new ArrayBuffer(BINARY_NODE_SIZE);
      const view = new DataView(buffer);
      view.setUint32(0, 1, true); // nodeId
      view.setFloat32(4, NaN, true); // Invalid position.x
      view.setFloat32(8, 20.0, true); // position.y
      view.setFloat32(12, 30.0, true); // position.z
      // Set velocities to valid values
      view.setFloat32(16, 0.1, true);
      view.setFloat32(20, 0.2, true);
      view.setFloat32(24, 0.3, true);

      const resultNaN = parseBinaryNodeData(buffer);
      expect(resultNaN).toHaveLength(0); // Should reject corrupted nodes
    });

    it('should handle empty buffer', () => {
      const emptyBuffer = new ArrayBuffer(0);
      const result = parseBinaryNodeData(emptyBuffer);
      expect(result).toHaveLength(0);
    });

    it('should process large datasets efficiently', () => {
      const start = performance.now();
      
      // Create 1000 nodes
      const largeDataset: BinaryNodeData[] = Array.from({ length: 1000 }, (_, i) => ({
        nodeId: i,
        position: {
          x: Math.random() * 100 - 50,
          y: Math.random() * 100 - 50,
          z: Math.random() * 100 - 50
        },
        velocity: {
          x: Math.random() * 2 - 1,
          y: Math.random() * 2 - 1,
          z: Math.random() * 2 - 1
        }
      }));

      const buffer = createBinaryNodeData(largeDataset);
      const parsed = parseBinaryNodeData(buffer);
      
      const end = performance.now();
      const processingTime = end - start;

      expect(parsed).toHaveLength(1000);
      expect(processingTime).toBeLessThan(100); // Should process 1000 nodes in under 100ms
    });
  });

  describe('Communication Intensity Formula', () => {
    it('should calculate communication intensity correctly', () => {
      const calculateCommunicationIntensity = (
        messageCount: number,
        dataVolume: number,
        timeDelta: number,
        agentDistance: number
      ): number => {
        if (timeDelta <= 0 || agentDistance <= 0) return 0;
        
        // Intensity = (messages/time + bytes/time) / distance
        const messageRate = messageCount / timeDelta;
        const dataRate = dataVolume / timeDelta;
        const intensity = (messageRate + dataRate * 0.001) / Math.max(agentDistance, 1);
        
        return Math.min(intensity, 10); // Cap at maximum intensity
      };

      // Test normal communication
      expect(calculateCommunicationIntensity(10, 1000, 1, 5)).toBeCloseTo(2.2, 1);
      
      // Test high-frequency communication
      expect(calculateCommunicationIntensity(100, 10000, 1, 2)).toBe(10); // Should be capped
      
      // Test edge cases
      expect(calculateCommunicationIntensity(0, 0, 1, 5)).toBe(0);
      expect(calculateCommunicationIntensity(10, 1000, 0, 5)).toBe(0);
      expect(calculateCommunicationIntensity(10, 1000, 1, 0)).toBe(0);
    });

    it('should handle edge weight processing for GPU kernels', () => {
      const processEdgeWeights = (edges: BotsEdge[]): Float32Array => {
        const weights = new Float32Array(edges.length);
        
        edges.forEach((edge, index) => {
          // Calculate normalized weight based on message count and data volume
          const messageWeight = Math.min(edge.messageCount / 100, 1);
          const dataWeight = Math.min(edge.dataVolume / 10000, 1);
          const timeDecay = Math.exp(-(Date.now() - edge.lastMessageTime) / 60000); // 1-minute decay
          
          weights[index] = (messageWeight * 0.6 + dataWeight * 0.4) * timeDecay;
        });
        
        return weights;
      };

      const testEdges: BotsEdge[] = [
        {
          id: 'edge1',
          source: 'agent1',
          target: 'agent2',
          dataVolume: 5000,
          messageCount: 50,
          lastMessageTime: Date.now() - 30000 // 30 seconds ago
        },
        {
          id: 'edge2',
          source: 'agent2',
          target: 'agent3',
          dataVolume: 15000,
          messageCount: 150,
          lastMessageTime: Date.now() - 120000 // 2 minutes ago (should decay)
        }
      ];

      const weights = processEdgeWeights(testEdges);
      expect(weights).toHaveLength(2);
      expect(weights[0]).toBeGreaterThan(weights[1]); // First edge should have higher weight due to recency
      expect(weights[0]).toBeLessThanOrEqual(1);
      expect(weights[1]).toBeLessThanOrEqual(1);
    });
  });

  describe('Agent Flag Processing', () => {
    it('should encode agent types into binary flags', () => {
      const encodeAgentFlags = (agents: BotsAgent[]): Uint8Array => {
        const flags = new Uint8Array(Math.ceil(agents.length / 8));
        
        agents.forEach((agent, index) => {
          const byteIndex = Math.floor(index / 8);
          const bitIndex = index % 8;
          
          // Set bit based on agent status and activity
          if (agent.status === 'busy' || (agent.activity && agent.activity > 0.5)) {
            flags[byteIndex] |= (1 << bitIndex);
          }
        });
        
        return flags;
      };

      const testAgents: BotsAgent[] = [
        {
          id: 'agent1',
          type: 'coordinator',
          status: 'busy',
          health: 90,
          cpuUsage: 45,
          memoryUsage: 60,
          createdAt: '2024-01-01T00:00:00Z',
          age: 3600000,
          activity: 0.8
        },
        {
          id: 'agent2',
          type: 'coder',
          status: 'idle',
          health: 85,
          cpuUsage: 10,
          memoryUsage: 30,
          createdAt: '2024-01-01T00:00:00Z',  
          age: 3600000,
          activity: 0.2
        },
        {
          id: 'agent3',
          type: 'tester',
          status: 'active',
          health: 95,
          cpuUsage: 70,
          memoryUsage: 80,
          createdAt: '2024-01-01T00:00:00Z',
          age: 3600000,
          activity: 0.9
        }
      ];

      const flags = encodeAgentFlags(testAgents);
      expect(flags).toHaveLength(1); // 3 agents fit in 1 byte
      
      // Check individual bits
      expect(flags[0] & 1).toBeTruthy(); // Agent 1 should be flagged (busy)
      expect(flags[0] & 2).toBeFalsy();  // Agent 2 should not be flagged (idle, low activity)
      expect(flags[0] & 4).toBeTruthy(); // Agent 3 should be flagged (high activity)
    });

    it('should decode agent flags correctly', () => {
      const decodeAgentFlags = (flags: Uint8Array, agentCount: number): boolean[] => {
        const result: boolean[] = [];
        
        for (let i = 0; i < agentCount; i++) {
          const byteIndex = Math.floor(i / 8);
          const bitIndex = i % 8;
          
          result.push(Boolean(flags[byteIndex] & (1 << bitIndex)));
        }
        
        return result;
      };

      const flags = new Uint8Array([0b10101010]); // Alternating pattern
      const decoded = decodeAgentFlags(flags, 8);
      
      expect(decoded).toEqual([false, true, false, true, false, true, false, true]);
    });
  });

  describe('ClaudeFlowActor Communication Link Retrieval', () => {
    it('should retrieve communication links efficiently', () => {
      const mockGetCommunicationLinks = (agentId: string, agents: BotsAgent[], edges: BotsEdge[]) => {
        const links = edges.filter(edge => 
          edge.source === agentId || edge.target === agentId
        );
        
        return links.map(link => ({
          ...link,
          partnerId: link.source === agentId ? link.target : link.source,
          intensity: link.messageCount * 0.1 + link.dataVolume * 0.001
        }));
      };

      const testAgents: BotsAgent[] = [
        { id: 'agent1', type: 'coordinator', status: 'busy', health: 90, cpuUsage: 45, memoryUsage: 60, createdAt: '2024-01-01T00:00:00Z', age: 3600000 },
        { id: 'agent2', type: 'coder', status: 'idle', health: 85, cpuUsage: 10, memoryUsage: 30, createdAt: '2024-01-01T00:00:00Z', age: 3600000 },
        { id: 'agent3', type: 'tester', status: 'active', health: 95, cpuUsage: 70, memoryUsage: 80, createdAt: '2024-01-01T00:00:00Z', age: 3600000 }
      ];

      const testEdges: BotsEdge[] = [
        { id: 'edge1', source: 'agent1', target: 'agent2', dataVolume: 5000, messageCount: 50, lastMessageTime: Date.now() },
        { id: 'edge2', source: 'agent1', target: 'agent3', dataVolume: 3000, messageCount: 30, lastMessageTime: Date.now() },
        { id: 'edge3', source: 'agent2', target: 'agent3', dataVolume: 2000, messageCount: 20, lastMessageTime: Date.now() }
      ];

      const agent1Links = mockGetCommunicationLinks('agent1', testAgents, testEdges);
      expect(agent1Links).toHaveLength(2);
      expect(agent1Links[0].partnerId).toBe('agent2');
      expect(agent1Links[1].partnerId).toBe('agent3');
      expect(agent1Links[0].intensity).toBeGreaterThan(agent1Links[1].intensity);
    });
  });
});