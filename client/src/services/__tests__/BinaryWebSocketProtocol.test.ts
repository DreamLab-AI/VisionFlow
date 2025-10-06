/**
 * Test suite for node ID truncation bug fix
 * Verifies that V2 protocol correctly handles node IDs > 16383
 */

import {
  BinaryWebSocketProtocol,
  PROTOCOL_V1,
  PROTOCOL_V2,
  AGENT_POSITION_SIZE_V1,
  AGENT_POSITION_SIZE_V2,
  AGENT_STATE_SIZE_V1,
  AGENT_STATE_SIZE_V2,
  type AgentPositionUpdate,
  type AgentStateData,
} from '../BinaryWebSocketProtocol';

describe('BinaryWebSocketProtocol - Node ID Truncation Fix', () => {
  let protocol: BinaryWebSocketProtocol;

  beforeEach(() => {
    protocol = BinaryWebSocketProtocol.getInstance();
  });

  describe('V1 Legacy Protocol (u16 IDs)', () => {
    it('should decode V1 format with small IDs', () => {
      const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V1);
      const view = new DataView(payload);

      // Write V1 format: u16 ID
      view.setUint16(0, 100, true);
      view.setFloat32(2, 1.0, true);
      view.setFloat32(6, 2.0, true);
      view.setFloat32(10, 3.0, true);
      view.setUint32(14, Date.now(), true);
      view.setUint8(18, 0);

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(1);
      expect(updates[0].agentId).toBe(100);
      expect(updates[0].position.x).toBe(1.0);
    });

    it('should handle V1 format with maximum safe ID (16383)', () => {
      const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V1);
      const view = new DataView(payload);

      view.setUint16(0, 16383, true); // Maximum 14-bit ID
      view.setFloat32(2, 1.0, true);
      view.setFloat32(6, 2.0, true);
      view.setFloat32(10, 3.0, true);
      view.setUint32(14, Date.now(), true);
      view.setUint8(18, 0);

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(1);
      expect(updates[0].agentId).toBe(16383);
    });
  });

  describe('V2 Protocol (u32 IDs)', () => {
    it('should encode V2 format with large IDs', () => {
      protocol.setUserInteracting(true);

      const updates: AgentPositionUpdate[] = [
        {
          agentId: 20000, // Larger than V1 max
          position: { x: 1.0, y: 2.0, z: 3.0 },
          timestamp: Date.now(),
          flags: 0,
        },
      ];

      const encoded = protocol.encodePositionUpdates(updates);

      expect(encoded).not.toBeNull();
      expect(encoded!.byteLength).toBe(4 + AGENT_POSITION_SIZE_V2); // Header + 1 update
    });

    it('should decode V2 format with large IDs', () => {
      const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V2);
      const view = new DataView(payload);

      // Write V2 format: u32 ID
      view.setUint32(0, 50000, true); // Large ID
      view.setFloat32(4, 1.0, true);
      view.setFloat32(8, 2.0, true);
      view.setFloat32(12, 3.0, true);
      view.setUint32(16, Date.now(), true);
      view.setUint8(20, 0);

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(1);
      expect(updates[0].agentId).toBe(50000);
      expect(updates[0].position.x).toBe(1.0);
    });

    it('should handle very large node IDs', () => {
      const largeIds = [16384, 20000, 50000, 100000, 1000000];

      for (const nodeId of largeIds) {
        const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V2);
        const view = new DataView(payload);

        view.setUint32(0, nodeId, true);
        view.setFloat32(4, 1.0, true);
        view.setFloat32(8, 2.0, true);
        view.setFloat32(12, 3.0, true);
        view.setUint32(16, Date.now(), true);
        view.setUint8(20, 0);

        const updates = protocol.decodePositionUpdates(payload);

        expect(updates).toHaveLength(1);
        expect(updates[0].agentId).toBe(nodeId);
      }
    });

    it('should decode multiple V2 updates correctly', () => {
      const nodeIds = [100, 20000, 50000];
      const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V2 * nodeIds.length);
      const view = new DataView(payload);

      nodeIds.forEach((nodeId, i) => {
        const offset = i * AGENT_POSITION_SIZE_V2;
        view.setUint32(offset, nodeId, true);
        view.setFloat32(offset + 4, i + 1.0, true);
        view.setFloat32(offset + 8, i + 2.0, true);
        view.setFloat32(offset + 12, i + 3.0, true);
        view.setUint32(offset + 16, Date.now(), true);
        view.setUint8(offset + 20, 0);
      });

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(3);
      expect(updates[0].agentId).toBe(100);
      expect(updates[1].agentId).toBe(20000);
      expect(updates[2].agentId).toBe(50000);
    });
  });

  describe('Agent State Data', () => {
    it('should encode V2 agent state with large IDs', () => {
      const agents: AgentStateData[] = [
        {
          agentId: 50000,
          position: { x: 1.0, y: 2.0, z: 3.0 },
          velocity: { x: 0.1, y: 0.2, z: 0.3 },
          health: 100.0,
          cpuUsage: 50.0,
          memoryUsage: 60.0,
          workload: 70.0,
          tokens: 1000,
          flags: 0,
        },
      ];

      const encoded = protocol.encodeAgentState(agents);
      const payload = encoded.slice(4); // Skip header

      expect(payload.byteLength).toBe(AGENT_STATE_SIZE_V2);
    });

    it('should decode V2 agent state with large IDs', () => {
      const payload = new ArrayBuffer(AGENT_STATE_SIZE_V2);
      const view = new DataView(payload);

      view.setUint32(0, 100000, true); // Very large ID
      view.setFloat32(4, 1.0, true);
      view.setFloat32(8, 2.0, true);
      view.setFloat32(12, 3.0, true);
      view.setFloat32(16, 0.1, true);
      view.setFloat32(20, 0.2, true);
      view.setFloat32(24, 0.3, true);
      view.setFloat32(28, 100.0, true);
      view.setFloat32(32, 50.0, true);
      view.setFloat32(36, 60.0, true);
      view.setFloat32(40, 70.0, true);
      view.setUint32(44, 1000, true);
      view.setUint8(48, 0);

      const agents = protocol.decodeAgentState(payload);

      expect(agents).toHaveLength(1);
      expect(agents[0].agentId).toBe(100000);
      expect(agents[0].position.x).toBe(1.0);
      expect(agents[0].health).toBe(100.0);
    });

    it('should decode V1 agent state for backward compatibility', () => {
      const payload = new ArrayBuffer(AGENT_STATE_SIZE_V1);
      const view = new DataView(payload);

      view.setUint16(0, 100, true); // V1 uses u16
      view.setFloat32(2, 1.0, true);
      view.setFloat32(6, 2.0, true);
      view.setFloat32(10, 3.0, true);
      view.setFloat32(14, 0.1, true);
      view.setFloat32(18, 0.2, true);
      view.setFloat32(22, 0.3, true);
      view.setFloat32(26, 100.0, true);
      view.setFloat32(30, 50.0, true);
      view.setFloat32(34, 60.0, true);
      view.setFloat32(38, 70.0, true);
      view.setUint32(42, 1000, true);
      view.setUint8(46, 0);

      const agents = protocol.decodeAgentState(payload);

      expect(agents).toHaveLength(1);
      expect(agents[0].agentId).toBe(100);
    });
  });

  describe('No Collision Tests', () => {
    it('should have no collisions with V2 for different large IDs', () => {
      const nodeIds = [16384, 20000, 50000, 100000, 500000];
      const payloadSize = AGENT_POSITION_SIZE_V2 * nodeIds.length;
      const payload = new ArrayBuffer(payloadSize);
      const view = new DataView(payload);

      nodeIds.forEach((nodeId, i) => {
        const offset = i * AGENT_POSITION_SIZE_V2;
        view.setUint32(offset, nodeId, true);
        view.setFloat32(offset + 4, i + 1.0, true);
        view.setFloat32(offset + 8, i + 2.0, true);
        view.setFloat32(offset + 12, i + 3.0, true);
        view.setUint32(offset + 16, Date.now(), true);
        view.setUint8(offset + 20, 0);
      });

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(nodeIds.length);

      // Verify no collisions
      const decodedIds = updates.map(u => u.agentId);
      const uniqueIds = new Set(decodedIds);
      expect(uniqueIds.size).toBe(nodeIds.length);

      // Verify each ID matches
      nodeIds.forEach((nodeId, i) => {
        expect(updates[i].agentId).toBe(nodeId);
      });
    });

    it('should demonstrate V1 would have collisions', () => {
      // IDs that would collide in V1 (differ by 16384)
      const id1 = 100;
      const id2 = 16384 + 100; // Would collide with id1 in V1

      // In V1, both would truncate to 100 (collision)
      // In V2, they remain distinct
      const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V2 * 2);
      const view = new DataView(payload);

      view.setUint32(0, id1, true);
      view.setFloat32(4, 1.0, true);
      view.setFloat32(8, 2.0, true);
      view.setFloat32(12, 3.0, true);
      view.setUint32(16, Date.now(), true);
      view.setUint8(20, 0);

      view.setUint32(21, id2, true);
      view.setFloat32(25, 4.0, true);
      view.setFloat32(29, 5.0, true);
      view.setFloat32(33, 6.0, true);
      view.setUint32(37, Date.now(), true);
      view.setUint8(41, 0);

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(2);
      expect(updates[0].agentId).toBe(id1);
      expect(updates[1].agentId).toBe(id2);
      expect(updates[0].agentId).not.toBe(updates[1].agentId);
    });
  });

  describe('Protocol Version Detection', () => {
    it('should auto-detect V1 based on payload size', () => {
      const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V1);
      const view = new DataView(payload);

      view.setUint16(0, 100, true);
      view.setFloat32(2, 1.0, true);
      view.setFloat32(6, 2.0, true);
      view.setFloat32(10, 3.0, true);
      view.setUint32(14, Date.now(), true);
      view.setUint8(18, 0);

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(1);
      expect(updates[0].agentId).toBe(100);
    });

    it('should auto-detect V2 based on payload size', () => {
      const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V2);
      const view = new DataView(payload);

      view.setUint32(0, 50000, true);
      view.setFloat32(4, 1.0, true);
      view.setFloat32(8, 2.0, true);
      view.setFloat32(12, 3.0, true);
      view.setUint32(16, Date.now(), true);
      view.setUint8(20, 0);

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(1);
      expect(updates[0].agentId).toBe(50000);
    });

    it('should handle invalid payload sizes gracefully', () => {
      const payload = new ArrayBuffer(17); // Invalid size

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(0);
    });
  });

  describe('Maximum Node ID Support', () => {
    it('should support maximum 30-bit node ID', () => {
      const maxId = 0x3FFFFFFF; // 1,073,741,823 (30 bits)
      const payload = new ArrayBuffer(AGENT_POSITION_SIZE_V2);
      const view = new DataView(payload);

      view.setUint32(0, maxId, true);
      view.setFloat32(4, 1.0, true);
      view.setFloat32(8, 2.0, true);
      view.setFloat32(12, 3.0, true);
      view.setUint32(16, Date.now(), true);
      view.setUint8(20, 0);

      const updates = protocol.decodePositionUpdates(payload);

      expect(updates).toHaveLength(1);
      expect(updates[0].agentId).toBe(maxId);
    });
  });

  describe('Performance and Bandwidth', () => {
    it('should calculate correct V2 bandwidth', () => {
      const agentCount = 100;
      const updateRateHz = 60;

      const bandwidth = protocol.calculateBandwidth(agentCount, updateRateHz);

      // V2 uses 49 bytes per agent (AGENT_STATE_SIZE_V2)
      const expectedFullState = agentCount * 49 * updateRateHz + 4 * updateRateHz;

      expect(bandwidth.fullState).toBe(expectedFullState);
    });
  });
});
