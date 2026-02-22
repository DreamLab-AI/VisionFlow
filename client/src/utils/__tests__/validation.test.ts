import { describe, it, expect } from 'vitest';
import { vi } from 'vitest';

vi.mock('../loggerConfig', () => ({
  createLogger: () => ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  }),
  createErrorMetadata: vi.fn(),
}));

import {
  validateVec3,
  validateVelocity,
  validateNodeData,
  validateNodePositions,
  sanitizeNodeData,
  validateAndSanitizeBatch,
  createValidationMiddleware,
  validateWebSocketMessage,
} from '../validation';

describe('validation utilities', () => {
  // --- validateVec3 ---

  describe('validateVec3', () => {
    it('should accept valid coordinates', () => {
      const result = validateVec3({ x: 1, y: 2, z: 3 }, 'pos');
      expect(result.valid).toBe(true);
      expect(result.errors).toBeUndefined();
    });

    it('should reject NaN values by default', () => {
      const result = validateVec3({ x: NaN, y: 0, z: 0 }, 'pos');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('pos.x is NaN');
    });

    it('should allow NaN when configured', () => {
      const result = validateVec3({ x: NaN, y: 0, z: 0 }, 'pos', { allowNaN: true, allowInfinity: true });
      // NaN is allowed, but it might still fail bounds check
      // NaN comparison: NaN < x is always false, so bounds check passes
      expect(result.valid).toBe(true);
    });

    it('should reject Infinity by default', () => {
      const result = validateVec3({ x: Infinity, y: 0, z: 0 }, 'pos');
      expect(result.valid).toBe(false);
      expect(result.errors).toEqual(expect.arrayContaining([expect.stringContaining('not finite')]));
    });

    it('should reject out-of-bounds coordinates', () => {
      const result = validateVec3({ x: 20000, y: 0, z: 0 }, 'pos', { maxCoordinate: 10000 });
      expect(result.valid).toBe(false);
      expect(result.errors).toEqual(expect.arrayContaining([expect.stringContaining('out of bounds')]));
    });

    it('should accept coordinates at boundary values', () => {
      const result = validateVec3({ x: 10000, y: -10000, z: 0 }, 'pos');
      expect(result.valid).toBe(true);
    });
  });

  // --- validateVelocity ---

  describe('validateVelocity', () => {
    it('should accept velocity within limits', () => {
      const result = validateVelocity({ x: 10, y: 10, z: 10 });
      expect(result.valid).toBe(true);
    });

    it('should reject velocity exceeding maxVelocity', () => {
      const result = validateVelocity({ x: 800, y: 800, z: 800 }, { maxVelocity: 1000 });
      // magnitude = sqrt(800^2 * 3) = ~1385
      expect(result.valid).toBe(false);
    });

    it('should propagate vec3 validation errors', () => {
      const result = validateVelocity({ x: NaN, y: 0, z: 0 });
      expect(result.valid).toBe(false);
    });
  });

  // --- validateNodeData ---

  describe('validateNodeData', () => {
    const validNode = {
      nodeId: 1,
      position: { x: 0, y: 0, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      ssspDistance: Infinity,
      ssspParent: -1,
    };

    it('should accept valid node data', () => {
      const result = validateNodeData(validNode);
      expect(result.valid).toBe(true);
    });

    it('should reject negative node ID', () => {
      const result = validateNodeData({ ...validNode, nodeId: -1 });
      expect(result.valid).toBe(false);
      expect(result.errors).toEqual(expect.arrayContaining([expect.stringContaining('Invalid node ID')]));
    });

    it('should validate both position and velocity', () => {
      const result = validateNodeData({
        ...validNode,
        position: { x: NaN, y: 0, z: 0 },
        velocity: { x: 0, y: NaN, z: 0 },
      });
      expect(result.valid).toBe(false);
      expect(result.errors!.length).toBeGreaterThanOrEqual(2);
    });
  });

  // --- validateNodePositions ---

  describe('validateNodePositions', () => {
    const makeNode = (id: number) => ({
      nodeId: id,
      position: { x: 0, y: 0, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      ssspDistance: Infinity,
      ssspParent: -1,
    });

    it('should accept valid node list', () => {
      const nodes = [makeNode(0), makeNode(1), makeNode(2)];
      const result = validateNodePositions(nodes);
      expect(result.valid).toBe(true);
    });

    it('should reject too many nodes', () => {
      const nodes = Array.from({ length: 10001 }, (_, i) => makeNode(i));
      const result = validateNodePositions(nodes);
      expect(result.valid).toBe(false);
      expect(result.errors).toEqual(expect.arrayContaining([expect.stringContaining('Too many nodes')]));
    });

    it('should detect duplicate node IDs', () => {
      const nodes = [makeNode(1), makeNode(1)];
      const result = validateNodePositions(nodes);
      expect(result.valid).toBe(false);
      expect(result.errors).toEqual(expect.arrayContaining([expect.stringContaining('Duplicate')]));
    });

    it('should accept empty node list', () => {
      const result = validateNodePositions([]);
      expect(result.valid).toBe(true);
    });
  });

  // --- sanitizeNodeData ---

  describe('sanitizeNodeData', () => {
    it('should clamp out-of-bounds coordinates', () => {
      const node = {
        nodeId: 1,
        position: { x: 99999, y: -99999, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        ssspDistance: Infinity,
        ssspParent: -1,
      };
      const sanitized = sanitizeNodeData(node);
      expect(sanitized.position.x).toBe(10000);
      expect(sanitized.position.y).toBe(-10000);
    });

    it('should replace NaN with 0', () => {
      const node = {
        nodeId: 1,
        position: { x: NaN, y: 0, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        ssspDistance: Infinity,
        ssspParent: -1,
      };
      const sanitized = sanitizeNodeData(node);
      expect(sanitized.position.x).toBe(0);
    });

    it('should clamp negative nodeId to 0', () => {
      const node = {
        nodeId: -5,
        position: { x: 0, y: 0, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        ssspDistance: Infinity,
        ssspParent: -1,
      };
      const sanitized = sanitizeNodeData(node);
      expect(sanitized.nodeId).toBe(0);
    });

    it('should scale down excessive velocity', () => {
      const node = {
        nodeId: 1,
        position: { x: 0, y: 0, z: 0 },
        velocity: { x: 1000, y: 1000, z: 1000 },
        ssspDistance: Infinity,
        ssspParent: -1,
      };
      const sanitized = sanitizeNodeData(node);
      const speed = Math.sqrt(
        sanitized.velocity.x ** 2 + sanitized.velocity.y ** 2 + sanitized.velocity.z ** 2
      );
      // Allow for floating-point rounding (1e-10 tolerance)
      expect(speed).toBeLessThanOrEqual(1000 + 1e-6);
    });
  });

  // --- validateAndSanitizeBatch ---

  describe('validateAndSanitizeBatch', () => {
    it('should pass through valid nodes', () => {
      const nodes = [
        { nodeId: 0, position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 }, ssspDistance: Infinity, ssspParent: -1 },
      ];
      const { valid, invalid } = validateAndSanitizeBatch(nodes);
      expect(valid).toHaveLength(1);
      expect(invalid).toHaveLength(0);
    });

    it('should sanitize invalid nodes and re-validate', () => {
      const nodes = [
        { nodeId: 1, position: { x: NaN, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 }, ssspDistance: Infinity, ssspParent: -1 },
      ];
      const { valid, invalid } = validateAndSanitizeBatch(nodes);
      expect(valid).toHaveLength(1);
      expect(valid[0].position.x).toBe(0); // sanitized
      expect(invalid).toHaveLength(0);
    });
  });

  // --- createValidationMiddleware ---

  describe('createValidationMiddleware', () => {
    it('should return a function that filters nodes', () => {
      const middleware = createValidationMiddleware();
      const nodes = [
        { nodeId: 0, position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 }, ssspDistance: Infinity, ssspParent: -1 },
      ];
      const result = middleware(nodes);
      expect(result).toHaveLength(1);
    });
  });

  // --- validateWebSocketMessage ---

  describe('validateWebSocketMessage', () => {
    it('should reject null', () => {
      expect(validateWebSocketMessage(null)).toBe(false);
    });

    it('should reject non-object', () => {
      expect(validateWebSocketMessage('string')).toBe(false);
    });

    it('should reject message without type', () => {
      expect(validateWebSocketMessage({ data: [] })).toBe(false);
    });

    it('should reject message with non-string type', () => {
      expect(validateWebSocketMessage({ type: 123 })).toBe(false);
    });

    it('should validate node_position_update messages', () => {
      expect(validateWebSocketMessage({ type: 'node_position_update', data: [1, 2] })).toBe(true);
      expect(validateWebSocketMessage({ type: 'node_position_update', data: [] })).toBe(false);
    });

    it('should validate settings_update messages', () => {
      expect(validateWebSocketMessage({ type: 'settings_update', data: { key: 'val' } })).toBe(true);
      expect(validateWebSocketMessage({ type: 'settings_update', data: null })).toBeFalsy();
    });

    it('should validate error messages', () => {
      expect(validateWebSocketMessage({ type: 'error', message: 'oops' })).toBe(true);
      expect(validateWebSocketMessage({ type: 'error', message: 123 })).toBe(false);
    });

    it('should accept unknown message types', () => {
      expect(validateWebSocketMessage({ type: 'custom_event' })).toBe(true);
    });
  });
});
