/**
 * AgentActionVisualization Tests
 *
 * Tests for the AgentActionVisualization component covering:
 * - Conditional rendering based on settings
 * - VR mode detection and adaptation
 * - maxConnections scaling for VR vs desktop
 * - Opacity scaling at high connection counts
 * - forceVrMode prop override behavior
 *
 * Uses unit test approach with React hooks mocked to avoid React 19 compatibility issues.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Store captured hook options for verification (module-level globals)
interface MockSettingsState {
  settings: Record<string, unknown>;
}
interface MockVisualizationState {
  connections: Array<Record<string, unknown>>;
  activeCount: number;
  enabled: boolean;
}
const mockState: {
  capturedVisualizationOptions: Record<string, unknown> | null;
  quest3State: { isQuest3Detected: boolean };
  settingsState: MockSettingsState;
  visualizationState: MockVisualizationState;
} = {
  capturedVisualizationOptions: null,
  quest3State: { isQuest3Detected: false },
  settingsState: { settings: { agents: { visualization: { show_action_connections: true } } } },
  visualizationState: { connections: [], activeCount: 0, enabled: true },
};

// Mock React hooks to avoid need for React rendering context
vi.mock('react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react')>();
  return {
    ...actual,
    useMemo: (factory: () => unknown) => factory(),
  };
});

// Mock dependencies - all mocks must use factory functions that don't reference external variables
vi.mock('@/hooks/useQuest3Integration', () => ({
  useQuest3Integration: vi.fn(() => mockState.quest3State),
}));

vi.mock('@/store/settingsStore', () => ({
  useSettingsStore: vi.fn(() => mockState.settingsState),
}));

vi.mock('@/features/visualisation/hooks/useAgentActionVisualization', () => ({
  useAgentActionVisualization: vi.fn((options: Record<string, unknown>) => {
    mockState.capturedVisualizationOptions = options;
    return {
      ...mockState.visualizationState,
      addActions: vi.fn(),
      updatePositions: vi.fn(),
    };
  }),
}));

vi.mock('@/features/visualisation/components/ActionConnectionsLayer', () => ({
  ActionConnectionsLayer: (props: Record<string, unknown>) => ({ type: 'ActionConnectionsLayer', props }),
  ActionConnectionsStats: (props: Record<string, unknown>) => ({ type: 'ActionConnectionsStats', props }),
}));

import { AgentActionVisualization, AgentActionVisualizationXR } from '@/features/visualisation/components/AgentActionVisualization';
import { useQuest3Integration } from '@/hooks/useQuest3Integration';

/**
 * Helper to extract ActionConnectionsLayer props from rendered output
 */
interface ReactLikeElement {
  props?: { children?: ReactLikeElement | ReactLikeElement[] | boolean | null; [key: string]: unknown };
  type?: unknown;
}
function getLayerProps(result: ReactLikeElement | null): Record<string, unknown> | null {
  if (!result) return null;

  // Result is a React fragment with children
  const children = result.props?.children;
  if (!children) return null;

  // First child should be ActionConnectionsLayer
  if (Array.isArray(children)) {
    const layer = children[0] as ReactLikeElement | undefined;
    return (layer?.props as Record<string, unknown>) ?? null;
  }

  return ((children as ReactLikeElement)?.props as Record<string, unknown>) ?? null;
}

/**
 * Helper to check if ActionConnectionsStats is rendered and get its props
 */
function getStatsProps(result: ReactLikeElement | null): Record<string, unknown> | null {
  if (!result) return null;

  const children = result.props?.children;
  if (!Array.isArray(children) || children.length < 2) return null;

  // Second child is ActionConnectionsStats (conditionally rendered)
  const stats = children[1];
  if (stats === false || stats === null || stats === undefined) return null;

  return ((stats as ReactLikeElement)?.props as Record<string, unknown>) ?? null;
}

/**
 * Helper to reset all mocks and state between tests
 */
function resetMocks() {
  vi.clearAllMocks();
  mockState.capturedVisualizationOptions = null;
  mockState.quest3State = { isQuest3Detected: false };
  mockState.settingsState = { settings: { agents: { visualization: { show_action_connections: true } } } } as MockSettingsState;
  mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };
}

describe('AgentActionVisualization', () => {
  beforeEach(() => {
    resetMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Conditional Rendering', () => {
    it('renders ActionConnectionsLayer when enabled in settings', () => {
      mockState.settingsState = {
        settings: {
          agents: {
            visualization: {
              show_action_connections: true,
            },
          },
        },
      };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});

      expect(result).not.toBeNull();
      const layerProps = getLayerProps(result);
      expect(layerProps).not.toBeNull();
    });

    it('does not render when settings.agents.visualization.show_action_connections is false', () => {
      mockState.settingsState = {
        settings: {
          agents: {
            visualization: {
              show_action_connections: false,
            },
          },
        },
      };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: false };

      const result = AgentActionVisualization({});

      // Component returns null when disabled
      expect(result).toBeNull();
    });

    it('defaults to enabled when agents.visualization is undefined', () => {
      mockState.settingsState = { settings: {} };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});

      // Should still render when settings don't explicitly disable
      expect(result).not.toBeNull();
    });

    it('renders when settings.agents is undefined (defaults enabled)', () => {
      mockState.settingsState = { settings: { agents: undefined } };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});

      // Default enabled=true, should render
      expect(result).not.toBeNull();
    });
  });

  describe('VR Mode Detection', () => {
    it('detects VR mode via useQuest3Integration when isQuest3Detected is true', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      // Verify vrMode is passed to hook and layer
      expect(mockState.capturedVisualizationOptions.vrMode).toBe(true);
      expect(layerProps.vrMode).toBe(true);
    });

    it('uses desktop mode when isQuest3Detected is false', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(mockState.capturedVisualizationOptions.vrMode).toBe(false);
      expect(layerProps.vrMode).toBe(false);
    });

    it('calls useQuest3Integration with enableAutoStart: false', () => {
      AgentActionVisualization({});

      expect(useQuest3Integration).toHaveBeenCalledWith({ enableAutoStart: false });
    });
  });

  describe('maxConnections Adaptation', () => {
    it('passes maxConnections=50 for desktop mode (default)', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ maxConnections: 50 });

      expect(mockState.capturedVisualizationOptions.maxConnections).toBe(50);
    });

    it('caps maxConnections at 25 for VR mode when provided value is higher', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ maxConnections: 50 });

      // VR mode: Math.min(maxConnections, 25) = Math.min(50, 25) = 25
      expect(mockState.capturedVisualizationOptions.maxConnections).toBe(25);
    });

    it('uses provided maxConnections when less than 25 in VR mode', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ maxConnections: 15 });

      // VR mode: Math.min(15, 25) = 15
      expect(mockState.capturedVisualizationOptions.maxConnections).toBe(15);
    });

    it('uses default maxConnections of 50 when not specified', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({});

      expect(mockState.capturedVisualizationOptions.maxConnections).toBe(50);
    });

    it('uses maxConnections=25 in VR with default prop', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({});

      // Default is 50, but VR caps at 25
      expect(mockState.capturedVisualizationOptions.maxConnections).toBe(25);
    });
  });

  describe('Opacity Scaling at High Load', () => {
    it('renders with opacity 1.0 when activeCount <= 30', () => {
      mockState.visualizationState = { connections: [], activeCount: 25, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.opacity).toBe(1);
    });

    it('renders with opacity 0.8 when activeCount > 30', () => {
      mockState.visualizationState = { connections: [], activeCount: 35, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.opacity).toBe(0.8);
    });

    it('renders with opacity 0.6 when activeCount > 40', () => {
      mockState.visualizationState = { connections: [], activeCount: 45, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.opacity).toBe(0.6);
    });

    it('renders with opacity 0.8 at boundary activeCount = 31', () => {
      mockState.visualizationState = { connections: [], activeCount: 31, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.opacity).toBe(0.8);
    });

    it('renders with opacity 0.6 at boundary activeCount = 41', () => {
      mockState.visualizationState = { connections: [], activeCount: 41, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.opacity).toBe(0.6);
    });

    it('renders with opacity 1.0 at boundary activeCount = 30', () => {
      mockState.visualizationState = { connections: [], activeCount: 30, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.opacity).toBe(1);
    });

    it('renders with opacity 0.8 at boundary activeCount = 40', () => {
      mockState.visualizationState = { connections: [], activeCount: 40, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      // 40 is > 30 but not > 40, so opacity = 0.8
      expect(layerProps.opacity).toBe(0.8);
    });

    it('handles activeCount = 0 with full opacity', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.opacity).toBe(1);
    });
  });

  describe('forceVrMode Prop', () => {
    it('overrides auto-detection when forceVrMode=true', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({ forceVrMode: true });
      const layerProps = getLayerProps(result);

      expect(mockState.capturedVisualizationOptions.vrMode).toBe(true);
      expect(layerProps.vrMode).toBe(true);
    });

    it('overrides auto-detection when forceVrMode=false', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({ forceVrMode: false });
      const layerProps = getLayerProps(result);

      expect(mockState.capturedVisualizationOptions.vrMode).toBe(false);
      expect(layerProps.vrMode).toBe(false);
    });

    it('caps maxConnections at 25 when forceVrMode=true regardless of detection', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ forceVrMode: true, maxConnections: 50 });

      expect(mockState.capturedVisualizationOptions.maxConnections).toBe(25);
    });

    it('allows maxConnections=50 when forceVrMode=false despite Quest detection', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ forceVrMode: false, maxConnections: 50 });

      expect(mockState.capturedVisualizationOptions.maxConnections).toBe(50);
    });

    it('uses undefined forceVrMode to fall back to auto-detection', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({});

      expect(mockState.capturedVisualizationOptions.vrMode).toBe(true);
    });
  });

  describe('AgentActionVisualizationXR Variant', () => {
    it('always sets forceVrMode=true via JSX rendering', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      // AgentActionVisualizationXR returns JSX which calls AgentActionVisualization
      // We verify the JSX props passed to the inner component
      const result = AgentActionVisualizationXR({});

      // The XR component wraps AgentActionVisualization with forceVrMode=true and maxConnections=20
      // When rendered as JSX, it passes these props
      expect(result.props.forceVrMode).toBe(true);
      expect(result.props.maxConnections).toBe(20);
    });

    it('uses maxConnections=20 for XR variant', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualizationXR({});

      // XR variant hardcodes maxConnections=20
      expect(result.props.maxConnections).toBe(20);
    });

    it('passes through other props like showStats', () => {
      mockState.visualizationState = { connections: [{ id: '1' }], activeCount: 1, enabled: true };

      const result = AgentActionVisualizationXR({ showStats: true });

      // showStats should be passed through to the inner component
      expect(result.props.showStats).toBe(true);
      // forceVrMode and maxConnections should still be set
      expect(result.props.forceVrMode).toBe(true);
      expect(result.props.maxConnections).toBe(20);
    });

    it('merges provided props with XR defaults', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualizationXR({
        debug: true,
        baseDuration: 800,
      });

      expect(result.props.debug).toBe(true);
      expect(result.props.baseDuration).toBe(800);
      expect(result.props.forceVrMode).toBe(true);
      expect(result.props.maxConnections).toBe(20);
    });
  });

  describe('Stats Overlay', () => {
    it('renders ActionConnectionsStats when showStats=true', () => {
      mockState.visualizationState = { connections: [{ id: '1' }, { id: '2' }], activeCount: 2, enabled: true };

      const result = AgentActionVisualization({ showStats: true });
      const statsProps = getStatsProps(result);

      expect(statsProps).not.toBeNull();
    });

    it('does not render ActionConnectionsStats when showStats=false (default)', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});
      const statsProps = getStatsProps(result);

      // showStats defaults to false
      expect(statsProps).toBeNull();
    });

    it('passes connections to ActionConnectionsStats', () => {
      const mockConnections = [{ id: 'conn-1' }, { id: 'conn-2' }];
      mockState.visualizationState = { connections: mockConnections, activeCount: 2, enabled: true };

      const result = AgentActionVisualization({ showStats: true });
      const statsProps = getStatsProps(result);

      expect(statsProps.connections).toEqual(mockConnections);
    });
  });

  describe('Line Width Configuration', () => {
    it('passes lineWidth=2 for desktop mode', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.lineWidth).toBe(2);
    });

    it('passes lineWidth=1 for VR mode', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.lineWidth).toBe(1);
    });

    it('uses lineWidth=1 when forceVrMode=true', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({ forceVrMode: true });
      const layerProps = getLayerProps(result);

      expect(layerProps.lineWidth).toBe(1);
    });
  });

  describe('Hook Configuration', () => {
    it('passes enabled flag to useAgentActionVisualization', () => {
      mockState.settingsState = {
        settings: {
          agents: {
            visualization: {
              show_action_connections: true,
            },
          },
        },
      };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({});

      expect(mockState.capturedVisualizationOptions.enabled).toBe(true);
    });

    it('passes enabled=false when settings disable visualization', () => {
      mockState.settingsState = {
        settings: {
          agents: {
            visualization: {
              show_action_connections: false,
            },
          },
        },
      };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: false };

      AgentActionVisualization({});

      expect(mockState.capturedVisualizationOptions.enabled).toBe(false);
    });

    it('passes debug flag to useAgentActionVisualization', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ debug: true });

      expect(mockState.capturedVisualizationOptions.debug).toBe(true);
    });

    it('passes baseDuration to useAgentActionVisualization', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ baseDuration: 750 });

      expect(mockState.capturedVisualizationOptions.baseDuration).toBe(750);
    });

    it('adjusts baseDuration minimum to 400 in VR mode', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ baseDuration: 300 });

      // VR mode: Math.max(baseDuration, 400) = Math.max(300, 400) = 400
      expect(mockState.capturedVisualizationOptions.baseDuration).toBe(400);
    });

    it('uses provided baseDuration in VR mode when >= 400', () => {
      mockState.quest3State = { isQuest3Detected: true };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({ baseDuration: 600 });

      expect(mockState.capturedVisualizationOptions.baseDuration).toBe(600);
    });

    it('uses default baseDuration of 500 when not specified', () => {
      mockState.quest3State = { isQuest3Detected: false };
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({});

      expect(mockState.capturedVisualizationOptions.baseDuration).toBe(500);
    });

    it('uses default debug=false when not specified', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      AgentActionVisualization({});

      expect(mockState.capturedVisualizationOptions.debug).toBe(false);
    });
  });

  describe('Connections Passthrough', () => {
    it('passes connections from hook to ActionConnectionsLayer', () => {
      const mockConnections = [
        { id: 'conn-1', actionType: 0 },
        { id: 'conn-2', actionType: 1 },
      ];
      mockState.visualizationState = { connections: mockConnections, activeCount: 2, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.connections).toEqual(mockConnections);
    });

    it('handles empty connections array', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      const result = AgentActionVisualization({});
      const layerProps = getLayerProps(result);

      expect(layerProps.connections).toEqual([]);
    });
  });

  describe('Props Interface', () => {
    it('accepts all documented props without error', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      // Test all props as documented in AgentActionVisualizationProps
      expect(() => {
        AgentActionVisualization({
          showStats: true,
          forceVrMode: true,
          maxConnections: 30,
          baseDuration: 750,
          debug: true,
        });
      }).not.toThrow();
    });

    it('works with no props (all defaults)', () => {
      mockState.visualizationState = { connections: [], activeCount: 0, enabled: true };

      expect(() => {
        AgentActionVisualization({});
      }).not.toThrow();
    });
  });
});
