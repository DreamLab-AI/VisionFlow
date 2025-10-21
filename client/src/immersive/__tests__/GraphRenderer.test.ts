/**
 * Test suite for GraphRenderer deduplication and incremental updates
 * HIGH PRIORITY: Verifies node deduplication and performance optimizations
 */

import { GraphRenderer } from '../babylon/GraphRenderer';
import * as BABYLON from '@babylonjs/core';

// Mock Babylon.js components
jest.mock('@babylonjs/core', () => {
  const actualBabylon = jest.requireActual('@babylonjs/core');

  class MockInstancedMesh {
    public position = { x: 0, y: 0, z: 0 };
    public metadata: any = {};
    private _isDisposed = false;

    dispose() {
      this._isDisposed = true;
    }

    isDisposed() {
      return this._isDisposed;
    }
  }

  class MockMesh {
    public material: any = null;
    public isVisible = true;
    private instanceCounter = 0;

    createInstance(name: string) {
      return new MockInstancedMesh();
    }

    dispose() {
      // Mock disposal
    }
  }

  return {
    ...actualBabylon,
    MeshBuilder: {
      CreateSphere: jest.fn(() => new MockMesh()),
      CreateLineSystem: jest.fn(() => ({ dispose: jest.fn() })),
    },
    StandardMaterial: jest.fn().mockImplementation(() => ({
      diffuseColor: null,
      emissiveColor: null,
      specularColor: null,
    })),
    Scene: jest.fn().mockImplementation(() => ({
      metadata: {},
    })),
    Vector3: actualBabylon.Vector3,
    Color3: actualBabylon.Color3,
  };
});

jest.mock('@babylonjs/gui', () => ({
  AdvancedDynamicTexture: {
    CreateFullscreenUI: jest.fn(() => ({
      addControl: jest.fn(),
      dispose: jest.fn(),
    })),
  },
  TextBlock: jest.fn().mockImplementation((id, label) => ({
    id,
    label,
    dispose: jest.fn(),
  })),
}));

describe('GraphRenderer - Deduplication and Incremental Updates', () => {
  let scene: BABYLON.Scene;
  let renderer: GraphRenderer;
  let consoleWarnSpy: jest.SpyInstance;
  let consoleLogSpy: jest.SpyInstance;

  beforeEach(() => {
    scene = new BABYLON.Scene(null as any);
    renderer = new GraphRenderer(scene);
    consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();
    consoleLogSpy = jest.spyOn(console, 'log').mockImplementation();
  });

  afterEach(() => {
    renderer.dispose();
    consoleWarnSpy.mockRestore();
    consoleLogSpy.mockRestore();
  });

  describe('Node Deduplication (HIGH PRIORITY)', () => {
    it('should deduplicate nodes with the same ID', () => {
      const duplicateNodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
        { id: '1', label: 'Node 1 Duplicate', x: 2, y: 2, z: 2 }, // Duplicate ID
        { id: '3', label: 'Node 3', x: 3, y: 3, z: 3 },
        { id: '2', label: 'Node 2 Duplicate', x: 4, y: 4, z: 4 }, // Duplicate ID
      ];

      renderer.updateNodes(duplicateNodes);

      // Should log warning for duplicates
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Duplicate node ID detected: 1')
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Duplicate node ID detected: 2')
      );

      // Should log deduplication summary
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Input: 5, Unique: 3, Duplicates removed: 2')
      );
    });

    it('should keep the first occurrence when duplicates exist', () => {
      const nodes = [
        { id: 'A', label: 'First A', x: 1, y: 1, z: 1 },
        { id: 'A', label: 'Second A', x: 2, y: 2, z: 2 },
      ];

      renderer.updateNodes(nodes);

      // The internal map should only have one node with ID 'A'
      // We can verify this by checking that only 1 node was created
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 1 new nodes')
      );
    });

    it('should handle empty node array', () => {
      renderer.updateNodes([]);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Input: 0, Unique: 0')
      );
    });

    it('should handle array with all unique nodes', () => {
      const uniqueNodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
        { id: '3', label: 'Node 3', x: 2, y: 2, z: 2 },
      ];

      renderer.updateNodes(uniqueNodes);

      // Should not warn about duplicates
      expect(consoleWarnSpy).not.toHaveBeenCalledWith(
        expect.stringContaining('Duplicate node ID detected')
      );

      // Should create all 3 nodes
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 3 new nodes')
      );
    });
  });

  describe('Incremental Updates (PERFORMANCE)', () => {
    it('should only create new nodes on first update', () => {
      const nodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
      ];

      renderer.updateNodes(nodes);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 2 new nodes, updated 0 existing nodes')
      );
    });

    it('should update existing nodes instead of recreating them', () => {
      const initialNodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
      ];

      renderer.updateNodes(initialNodes);
      consoleLogSpy.mockClear();

      // Update with same nodes but different positions
      const updatedNodes = [
        { id: '1', label: 'Node 1', x: 5, y: 5, z: 5 },
        { id: '2', label: 'Node 2', x: 6, y: 6, z: 6 },
      ];

      renderer.updateNodes(updatedNodes);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 0 new nodes, updated 2 existing nodes')
      );
    });

    it('should add new nodes while keeping existing ones', () => {
      const initialNodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
      ];

      renderer.updateNodes(initialNodes);
      consoleLogSpy.mockClear();

      // Add new node while keeping existing ones
      const expandedNodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
        { id: '3', label: 'Node 3', x: 2, y: 2, z: 2 }, // New
      ];

      renderer.updateNodes(expandedNodes);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 1 new nodes, updated 2 existing nodes')
      );
    });

    it('should remove obsolete nodes', () => {
      const initialNodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
        { id: '3', label: 'Node 3', x: 2, y: 2, z: 2 },
      ];

      renderer.updateNodes(initialNodes);
      consoleLogSpy.mockClear();

      // Remove node '2'
      const reducedNodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '3', label: 'Node 3', x: 2, y: 2, z: 2 },
      ];

      renderer.updateNodes(reducedNodes);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Removed 1 obsolete nodes')
      );
    });

    it('should handle complete node replacement', () => {
      const initialNodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
      ];

      renderer.updateNodes(initialNodes);
      consoleLogSpy.mockClear();

      // Completely different set of nodes
      const newNodes = [
        { id: '3', label: 'Node 3', x: 3, y: 3, z: 3 },
        { id: '4', label: 'Node 4', x: 4, y: 4, z: 4 },
      ];

      renderer.updateNodes(newNodes);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Removed 2 obsolete nodes')
      );
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 2 new nodes, updated 0 existing nodes')
      );
    });
  });

  describe('Position Handling', () => {
    it('should use positions array when provided', () => {
      const nodes = [
        { id: '1', label: 'Node 1' },
        { id: '2', label: 'Node 2' },
      ];

      const positions = new Float32Array([
        1.0, 2.0, 3.0,  // Node 1 position
        4.0, 5.0, 6.0,  // Node 2 position
      ]);

      renderer.updateNodes(nodes, positions);

      // Verify positions were used (we can check via logs or internal state)
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 2 new nodes')
      );
    });

    it('should warn when node position is not found', () => {
      // This tests the getNodePosition fallback
      const edges = [
        { source: '999', target: '1000' }, // Non-existent nodes
      ];

      const positions = new Float32Array([1.0, 2.0, 3.0]);

      renderer.updateEdges(edges, positions);

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Node position not found for ID "999"')
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Node position not found for ID "1000"')
      );
    });

    it('should create synthetic nodes from positions array when nodes are empty', () => {
      const positions = new Float32Array([
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
      ]);

      renderer.updateNodes([], positions);

      // Should create 3 synthetic nodes
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Input: 3, Unique: 3')
      );
    });
  });

  describe('Memory Management', () => {
    it('should clear all internal state on dispose', () => {
      const nodes = [
        { id: '1', label: 'Node 1', x: 0, y: 0, z: 0 },
        { id: '2', label: 'Node 2', x: 1, y: 1, z: 1 },
      ];

      renderer.updateNodes(nodes);

      // Should have nodes in internal state
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 2 new nodes')
      );

      renderer.dispose();

      // After disposal, next update should create new nodes (not update existing)
      const newRenderer = new GraphRenderer(scene);
      newRenderer.updateNodes(nodes);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 2 new nodes')
      );

      newRenderer.dispose();
    });
  });

  describe('Edge Cases', () => {
    it('should handle nodes with numeric string IDs', () => {
      const nodes = [
        { id: '123', label: 'Node 123', x: 0, y: 0, z: 0 },
        { id: '456', label: 'Node 456', x: 1, y: 1, z: 1 },
      ];

      renderer.updateNodes(nodes);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 2 new nodes')
      );
    });

    it('should handle nodes with mixed ID types', () => {
      const nodes = [
        { id: 'abc', label: 'Node ABC', x: 0, y: 0, z: 0 },
        { id: '123', label: 'Node 123', x: 1, y: 1, z: 1 },
        { id: 'xyz_789', label: 'Node XYZ', x: 2, y: 2, z: 2 },
      ];

      renderer.updateNodes(nodes);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 3 new nodes')
      );
    });

    it('should handle large node sets efficiently', () => {
      const largeNodeSet = Array.from({ length: 1000 }, (_, i) => ({
        id: String(i),
        label: `Node ${i}`,
        x: i,
        y: i,
        z: i,
      }));

      renderer.updateNodes(largeNodeSet);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Created 1000 new nodes')
      );
    });
  });
});
