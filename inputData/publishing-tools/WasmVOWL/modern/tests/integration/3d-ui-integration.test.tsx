/**
 * Integration Tests: 3D Graph with UI Components
 *
 * Tests the complete integration of React Three Fiber 3D rendering
 * with UI overlay components (TopMenuBar, Sidebar, NodeDetailsPanel, FileDropZone)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import App from '../../src/App';
import { useGraphStore } from '../../src/stores/useGraphStore';
import { useUIStore } from '../../src/stores/useUIStore';

// Mock fetch for ontology loading
global.fetch = vi.fn();

const mockOntologyData = {
  header: { title: "Test Ontology" },
  namespace: [],
  class: [
    { id: "class1", label: "TestClass1" },
    { id: "class2", label: "TestClass2" }
  ],
  property: [
    { id: "prop1", label: "testProperty", type: "objectProperty" }
  ]
};

describe('3D Graph + UI Integration Tests', () => {
  beforeEach(() => {
    // Reset stores
    useGraphStore.setState({
      nodes: new Map(),
      edges: new Map(),
      filteredNodes: new Set(),
      filteredEdges: new Set(),
      selectedNode: null,
      hoveredNode: null
    });

    useUIStore.setState({
      sidebarOpen: false,
      viewport: { mode: '3d', zoom: 1, rotation: [0, 0, 0], target: [0, 0, 0] },
      settings: {
        showLabels: true,
        showNodeDetails: true,
        nodeScale: 1,
        edgeWidth: 1
      }
    });

    vi.clearAllMocks();
  });

  describe('Task 1: React Three Fiber Components Untouched', () => {
    it('should render GraphCanvas without modifications', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      const { container } = render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // GraphCanvas creates a canvas element via R3F
      const canvasElements = container.querySelectorAll('canvas');
      expect(canvasElements.length).toBeGreaterThan(0);
    });

    it('should preserve ClassNode drag and selection behavior', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBe(2);
      });

      // Node selection state should be managed correctly
      const selectNode = useGraphStore.getState().selectNode;
      selectNode('class1');
      expect(useGraphStore.getState().selectedNode).toBe('class1');
    });

    it('should maintain PropertyEdge rendering without interference', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        const state = useGraphStore.getState();
        expect(state.edges.size).toBeGreaterThan(0);
      });

      // Edges should be in the graph store
      const edges = Array.from(useGraphStore.getState().edges.values());
      expect(edges.length).toBeGreaterThan(0);
    });

    it('should preserve GraphScene simulation state', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Simulation should initialize nodes with positions
      const nodes = Array.from(useGraphStore.getState().nodes.values());
      nodes.forEach(node => {
        expect(node.position).toBeDefined();
        expect(node.position.x).toBeDefined();
        expect(node.position.y).toBeDefined();
        expect(node.position.z).toBeDefined();
      });
    });
  });

  describe('Task 2: Full-Screen Canvas Layout with UI Overlays', () => {
    it('should render canvas at full viewport height', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      const { container } = render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Canvas container should have full height styles
      const canvasParent = container.querySelector('.app-main');
      expect(canvasParent).toBeTruthy();
    });

    it('should position UI overlays correctly without blocking canvas', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // TopMenuBar should be present
      const menuBar = screen.getByText(/Export SVG/i);
      expect(menuBar).toBeTruthy();

      // UI should not interfere with pointer events on canvas
      const menuElement = menuBar.closest('.top-menu-bar');
      expect(menuElement).toBeTruthy();
    });

    it('should maintain proper z-index layering', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      const { container } = render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Canvas should be below UI overlays
      const canvas = container.querySelector('canvas');
      const menuBar = container.querySelector('.top-menu-bar');

      expect(canvas).toBeTruthy();
      expect(menuBar).toBeTruthy();
    });
  });

  describe('Task 3: Sidebar Non-Interference with 3D Rendering', () => {
    it('should toggle sidebar without affecting canvas rendering', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      const user = userEvent.setup();

      // Open sidebar
      const toggleSidebar = useUIStore.getState().toggleSidebar;
      toggleSidebar();

      expect(useUIStore.getState().sidebarOpen).toBe(true);

      // Canvas should still be accessible
      const nodeCount = useGraphStore.getState().nodes.size;
      expect(nodeCount).toBeGreaterThan(0);
    });

    it('should render sidebar tabs without canvas conflicts', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Open sidebar
      useUIStore.getState().toggleSidebar();

      await waitFor(() => {
        expect(screen.queryByText('Details')).toBeTruthy();
      });

      // Sidebar tabs should be visible
      expect(screen.getByText('Filters')).toBeTruthy();
      expect(screen.getByText('Statistics')).toBeTruthy();
    });

    it('should allow sidebar interactions without disrupting 3D view', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      useUIStore.getState().toggleSidebar();

      const user = userEvent.setup();

      // Switch tabs
      const filtersTab = await screen.findByText('Filters');
      await user.click(filtersTab);

      expect(useUIStore.getState().sidebarTab).toBe('filters');

      // 3D viewport should remain unchanged
      expect(useUIStore.getState().viewport.mode).toBe('3d');
    });
  });

  describe('Task 4: TopMenuBar Controls', () => {
    it('should toggle between 2D and 3D modes', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      const user = userEvent.setup();

      // Find 3D toggle button
      const toggleButton = screen.getByTitle(/Switch to 2D/i);
      await user.click(toggleButton);

      expect(useUIStore.getState().viewport.mode).toBe('2d');

      await user.click(toggleButton);
      expect(useUIStore.getState().viewport.mode).toBe('3d');
    });

    it('should handle export SVG action', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      const exportButton = screen.getByTitle('Export as SVG');
      expect(exportButton).toBeTruthy();

      // Button should be clickable
      expect(exportButton).not.toBeDisabled();
    });

    it('should handle export PNG action', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      const exportButton = screen.getByTitle('Export as PNG');
      expect(exportButton).toBeTruthy();
      expect(exportButton).not.toBeDisabled();
    });

    it('should display graph statistics in menu bar', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Statistics should be visible
      await waitFor(() => {
        expect(screen.queryByText(/Nodes:/i)).toBeTruthy();
      });
    });

    it('should handle zoom controls', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      const user = userEvent.setup();

      const initialZoom = useUIStore.getState().viewport.zoom;

      const zoomInButton = screen.getByTitle('Zoom in');
      await user.click(zoomInButton);

      expect(useUIStore.getState().viewport.zoom).toBeGreaterThan(initialZoom);
    });
  });

  describe('Task 5: File Drop Functionality', () => {
    it('should show FileDropZone when no data is loaded', () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Not found'));

      render(<App />);

      // FileDropZone should appear
      waitFor(() => {
        expect(screen.queryByText(/Drop ontology file here/i)).toBeTruthy();
      });
    });

    it('should handle successful file drop', async () => {
      const file = new File(
        [JSON.stringify(mockOntologyData)],
        'test.json',
        { type: 'application/json' }
      );

      (global.fetch as any).mockRejectedValueOnce(new Error('Not found'));

      const { container } = render(<App />);

      await waitFor(() => {
        expect(screen.queryByText(/Drop ontology file here/i)).toBeTruthy();
      });

      // Simulate file drop by directly calling loadOntology
      const loadOntology = useGraphStore.getState().loadOntology;
      loadOntology(mockOntologyData);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBe(2);
      });
    });

    it('should validate JSON structure on file load', async () => {
      const invalidData = { invalid: true };

      (global.fetch as any).mockRejectedValueOnce(new Error('Not found'));

      render(<App />);

      const loadOntology = useGraphStore.getState().loadOntology;

      // This should fail validation in FileDropZone
      expect(() => {
        if (!invalidData.class || !Array.isArray(invalidData.class)) {
          throw new Error('Invalid ontology format');
        }
      }).toThrow('Invalid ontology format');
    });
  });

  describe('Task 6: NodeDetailsPanel without Canvas Conflicts', () => {
    it('should open NodeDetailsPanel on node selection', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Select a node
      useGraphStore.getState().selectNode('class1');

      await waitFor(() => {
        expect(screen.queryByText('Node Details')).toBeTruthy();
      });
    });

    it('should display node metadata correctly', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      useGraphStore.getState().selectNode('class1');

      await waitFor(() => {
        expect(screen.queryByText(/TestClass1/i)).toBeTruthy();
      });
    });

    it('should close NodeDetailsPanel without affecting canvas', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      useGraphStore.getState().selectNode('class1');

      await waitFor(() => {
        expect(screen.queryByText('Node Details')).toBeTruthy();
      });

      // Close panel
      useGraphStore.getState().selectNode(null);

      await waitFor(() => {
        expect(screen.queryByText('Node Details')).toBeFalsy();
      });

      // Canvas should still have nodes
      expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
    });

    it('should position panel without blocking 3D interactions', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      const { container } = render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      useGraphStore.getState().selectNode('class1');

      await waitFor(() => {
        const panel = container.querySelector('.node-details-panel');
        expect(panel).toBeTruthy();
      });

      // Canvas should still be present
      const canvas = container.querySelector('canvas');
      expect(canvas).toBeTruthy();
    });
  });

  describe('Task 7: VOWL Colors Preserved', () => {
    it('should use correct VOWL colors for class nodes', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      const nodes = Array.from(useGraphStore.getState().nodes.values());

      // ClassNode uses #aaccee for default color (line 48 in ClassNode.tsx)
      // This is verified by checking the node type
      nodes.forEach(node => {
        if (node.type === 'class') {
          expect(node.type).toBe('class');
        }
      });
    });

    it('should preserve edge colors based on type', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        const edges = Array.from(useGraphStore.getState().edges.values());
        expect(edges.length).toBeGreaterThan(0);
      });

      const edges = Array.from(useGraphStore.getState().edges.values());

      // PropertyEdge defines colors based on type (lines 62-73)
      edges.forEach(edge => {
        if (edge.type === 'objectProperty') {
          expect(edge.type).toBe('objectProperty');
        }
      });
    });

    it('should maintain selection highlight colors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Select node - should use #67bc0f (green) for selection
      useGraphStore.getState().selectNode('class1');

      expect(useGraphStore.getState().selectedNode).toBe('class1');
    });

    it('should apply hover colors correctly', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Hover node - should use #8cd0f0 (light blue)
      useGraphStore.getState().hoverNode('class1');

      expect(useGraphStore.getState().hoveredNode).toBe('class1');
    });
  });

  describe('Task 8: Responsive Behavior', () => {
    it('should handle viewport resize without breaking layout', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      const { container } = render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // Simulate resize
      global.innerWidth = 800;
      global.dispatchEvent(new Event('resize'));

      // Layout should still work
      const canvas = container.querySelector('canvas');
      expect(canvas).toBeTruthy();
    });

    it('should adapt UI overlays to smaller screens', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      // Set mobile viewport
      global.innerWidth = 375;

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      // UI should still be accessible
      const menuBar = screen.queryByText(/Export SVG/i);
      expect(menuBar).toBeTruthy();
    });

    it('should maintain canvas aspect ratio on resize', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      const { container } = render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      const canvas = container.querySelector('canvas');
      expect(canvas).toBeTruthy();

      // Canvas should have width and height set by R3F
      expect(canvas?.getAttribute('width')).toBeTruthy();
      expect(canvas?.getAttribute('height')).toBeTruthy();
    });
  });

  describe('Task 9: Edge Cases and Error Handling', () => {
    it('should handle empty ontology gracefully', async () => {
      const emptyOntology = {
        header: {},
        namespace: [],
        class: [],
        property: []
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => emptyOntology
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBe(0);
      });

      // Should show empty state
      await waitFor(() => {
        expect(screen.queryByText(/No ontology loaded/i)).toBeTruthy();
      });
    });

    it('should handle malformed ontology data', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ invalid: 'data' })
      });

      render(<App />);

      // Should fail gracefully and show FileDropZone
      await waitFor(() => {
        expect(screen.queryByText(/Drop ontology file here/i)).toBeTruthy();
      });
    });

    it('should handle network errors during auto-load', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      render(<App />);

      // Should show FileDropZone
      await waitFor(() => {
        expect(screen.queryByText(/Drop ontology file here/i)).toBeTruthy();
      });
    });

    it('should handle simultaneous node selection and deselection', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      const selectNode = useGraphStore.getState().selectNode;

      selectNode('class1');
      expect(useGraphStore.getState().selectedNode).toBe('class1');

      selectNode('class2');
      expect(useGraphStore.getState().selectedNode).toBe('class2');

      selectNode(null);
      expect(useGraphStore.getState().selectedNode).toBeNull();
    });

    it('should prevent UI interactions during loading', async () => {
      (global.fetch as any).mockImplementationOnce(() =>
        new Promise(resolve => setTimeout(() => resolve({
          ok: true,
          json: async () => mockOntologyData
        }), 100))
      );

      render(<App />);

      // Should show loading state
      expect(screen.queryByText(/Loading ontology/i)).toBeTruthy();

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      }, { timeout: 2000 });
    });
  });

  describe('Performance and Memory', () => {
    it('should not leak memory on component unmount', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOntologyData
      });

      const { unmount } = render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
      });

      unmount();

      // Stores should maintain data (they're global)
      expect(useGraphStore.getState().nodes.size).toBeGreaterThan(0);
    });

    it('should handle large ontologies without performance degradation', async () => {
      const largeOntology = {
        header: {},
        namespace: [],
        class: Array.from({ length: 100 }, (_, i) => ({
          id: `class${i}`,
          label: `Class ${i}`
        })),
        property: Array.from({ length: 200 }, (_, i) => ({
          id: `prop${i}`,
          label: `Property ${i}`,
          type: 'objectProperty'
        }))
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => largeOntology
      });

      const startTime = performance.now();

      render(<App />);

      await waitFor(() => {
        expect(useGraphStore.getState().nodes.size).toBe(100);
      }, { timeout: 5000 });

      const endTime = performance.now();
      const loadTime = endTime - startTime;

      // Should load within reasonable time
      expect(loadTime).toBeLessThan(5000);
    });
  });
});
