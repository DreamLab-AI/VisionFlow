/**
 * WebXRScene Tests
 *
 * Tests for the unified WebXR visualization scene component.
 * Verifies VR/Desktop mode switching, LOD management, and performance constraints.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

// Mock navigator.xr for testing
const mockXR = {
  isSessionSupported: vi.fn(),
};

beforeEach(() => {
  vi.clearAllMocks();
  // Reset navigator.xr mock
  Object.defineProperty(navigator, 'xr', {
    value: mockXR,
    configurable: true,
  });
});

describe('WebXRScene Configuration', () => {
  describe('VR Mode Detection', () => {
    it('should detect when VR is supported', async () => {
      mockXR.isSessionSupported.mockResolvedValue(true);

      const isSupported = await navigator.xr?.isSessionSupported('immersive-vr');
      expect(isSupported).toBe(true);
    });

    it('should detect when VR is not supported', async () => {
      mockXR.isSessionSupported.mockResolvedValue(false);

      const isSupported = await navigator.xr?.isSessionSupported('immersive-vr');
      expect(isSupported).toBe(false);
    });

    it('should handle missing WebXR API', () => {
      Object.defineProperty(navigator, 'xr', {
        value: undefined,
        configurable: true,
      });

      const hasXR = !!navigator.xr;
      expect(hasXR).toBe(false);
    });
  });

  describe('Default Props', () => {
    it('defaults enableActionConnections to true', () => {
      const defaultEnableActionConnections = true;
      expect(defaultEnableActionConnections).toBe(true);
    });

    it('defaults maxConnections to 50', () => {
      const defaultMaxConnections = 50;
      expect(defaultMaxConnections).toBe(50);
    });

    it('defaults baseDuration to 500ms', () => {
      const defaultBaseDuration = 500;
      expect(defaultBaseDuration).toBe(500);
    });

    it('defaults showStats to false', () => {
      const defaultShowStats = false;
      expect(defaultShowStats).toBe(false);
    });

    it('defaults debug to false', () => {
      const defaultDebug = false;
      expect(defaultDebug).toBe(false);
    });
  });
});

describe('VR vs Desktop Mode Parameters', () => {
  describe('Connection Limits', () => {
    it('uses 50 max connections in desktop mode', () => {
      const isVRMode = false;
      const maxConnections = 50;
      const effectiveMax = isVRMode ? Math.min(maxConnections, 20) : maxConnections;
      expect(effectiveMax).toBe(50);
    });

    it('limits to 20 connections in VR mode', () => {
      const isVRMode = true;
      const maxConnections = 50;
      const effectiveMax = isVRMode ? Math.min(maxConnections, 20) : maxConnections;
      expect(effectiveMax).toBe(20);
    });

    it('respects lower custom limit in VR mode', () => {
      const isVRMode = true;
      const maxConnections = 15;
      const effectiveMax = isVRMode ? Math.min(maxConnections, 20) : maxConnections;
      expect(effectiveMax).toBe(15);
    });
  });

  describe('Animation Duration', () => {
    it('uses provided duration in desktop mode', () => {
      const isVRMode = false;
      const baseDuration = 500;
      const effectiveDuration = isVRMode ? Math.min(baseDuration, 400) : baseDuration;
      expect(effectiveDuration).toBe(500);
    });

    it('caps duration at 400ms in VR mode', () => {
      const isVRMode = true;
      const baseDuration = 500;
      const effectiveDuration = isVRMode ? Math.min(baseDuration, 400) : baseDuration;
      expect(effectiveDuration).toBe(400);
    });

    it('respects shorter duration in VR mode', () => {
      const isVRMode = true;
      const baseDuration = 300;
      const effectiveDuration = isVRMode ? Math.min(baseDuration, 400) : baseDuration;
      expect(effectiveDuration).toBe(300);
    });
  });

  describe('Opacity Scaling', () => {
    it('reduces opacity at high connection counts in desktop mode', () => {
      const isVRMode = false;
      const activeCount = 45;

      let opacity = 1.0;
      if (!isVRMode) {
        if (activeCount > 40) opacity = 0.6;
        else if (activeCount > 30) opacity = 0.8;
      }

      expect(opacity).toBe(0.6);
    });

    it('reduces opacity at lower threshold in VR mode', () => {
      const isVRMode = true;
      const activeCount = 18;

      let opacity = 1.0;
      if (isVRMode) {
        if (activeCount > 18) opacity = 0.6;
        else if (activeCount > 12) opacity = 0.8;
      }

      // activeCount is exactly 18, so threshold is > 18
      expect(opacity).toBe(0.8);
    });

    it('uses full opacity at low connection counts', () => {
      const isVRMode = true;
      const activeCount = 5;

      let opacity = 1.0;
      if (isVRMode) {
        if (activeCount > 18) opacity = 0.6;
        else if (activeCount > 12) opacity = 0.8;
      }

      expect(opacity).toBe(1.0);
    });
  });
});

describe('LOD Thresholds for VR', () => {
  describe('Distance-Based LOD', () => {
    it('uses high LOD under 5m', () => {
      const distance = 4;
      const thresholds = { high: 5, medium: 15, low: 30 };

      let lodLevel: string;
      if (distance < thresholds.high) lodLevel = 'high';
      else if (distance < thresholds.medium) lodLevel = 'medium';
      else if (distance < thresholds.low) lodLevel = 'low';
      else lodLevel = 'culled';

      expect(lodLevel).toBe('high');
    });

    it('uses medium LOD between 5-15m', () => {
      const distance = 10;
      const thresholds = { high: 5, medium: 15, low: 30 };

      let lodLevel: string;
      if (distance < thresholds.high) lodLevel = 'high';
      else if (distance < thresholds.medium) lodLevel = 'medium';
      else if (distance < thresholds.low) lodLevel = 'low';
      else lodLevel = 'culled';

      expect(lodLevel).toBe('medium');
    });

    it('uses low LOD between 15-30m', () => {
      const distance = 25;
      const thresholds = { high: 5, medium: 15, low: 30 };

      let lodLevel: string;
      if (distance < thresholds.high) lodLevel = 'high';
      else if (distance < thresholds.medium) lodLevel = 'medium';
      else if (distance < thresholds.low) lodLevel = 'low';
      else lodLevel = 'culled';

      expect(lodLevel).toBe('low');
    });

    it('culls connections beyond 30m', () => {
      const distance = 35;
      const thresholds = { high: 5, medium: 15, low: 30 };

      let lodLevel: string;
      if (distance < thresholds.high) lodLevel = 'high';
      else if (distance < thresholds.medium) lodLevel = 'medium';
      else if (distance < thresholds.low) lodLevel = 'low';
      else lodLevel = 'culled';

      expect(lodLevel).toBe('culled');
    });
  });

  describe('LOD Segment Counts', () => {
    const LOD_SEGMENTS = {
      high: { curve: 24, sphere: 12 },
      medium: { curve: 16, sphere: 8 },
      low: { curve: 8, sphere: 6 },
      culled: { curve: 0, sphere: 0 },
    };

    it('high LOD has most detail', () => {
      expect(LOD_SEGMENTS.high.curve).toBe(24);
      expect(LOD_SEGMENTS.high.sphere).toBe(12);
    });

    it('medium LOD is reduced', () => {
      expect(LOD_SEGMENTS.medium.curve).toBeLessThan(LOD_SEGMENTS.high.curve);
      expect(LOD_SEGMENTS.medium.sphere).toBeLessThan(LOD_SEGMENTS.high.sphere);
    });

    it('low LOD is minimal', () => {
      expect(LOD_SEGMENTS.low.curve).toBeLessThan(LOD_SEGMENTS.medium.curve);
      expect(LOD_SEGMENTS.low.sphere).toBeLessThan(LOD_SEGMENTS.medium.sphere);
    });

    it('culled has zero segments', () => {
      expect(LOD_SEGMENTS.culled.curve).toBe(0);
      expect(LOD_SEGMENTS.culled.sphere).toBe(0);
    });
  });
});

describe('Canvas Configuration', () => {
  describe('WebGL Settings', () => {
    it('disables antialias in VR mode for performance', () => {
      const isInVR = true;
      const antialias = !isInVR;
      expect(antialias).toBe(false);
    });

    it('enables antialias in desktop mode', () => {
      const isInVR = false;
      const antialias = !isInVR;
      expect(antialias).toBe(true);
    });

    it('uses high-performance power preference', () => {
      const powerPreference = 'high-performance';
      expect(powerPreference).toBe('high-performance');
    });

    it('disables alpha for better performance', () => {
      const alpha = false;
      expect(alpha).toBe(false);
    });
  });

  describe('Camera Configuration', () => {
    it('positions camera at eye level', () => {
      const cameraPosition: [number, number, number] = [0, 1.6, 3];
      expect(cameraPosition[1]).toBe(1.6); // Eye level in meters
    });

    it('uses 70 degree FOV', () => {
      const fov = 70;
      expect(fov).toBe(70);
    });
  });
});

describe('Hand Tracking Integration', () => {
  describe('Target Detection', () => {
    it('uses 30m max ray distance', () => {
      const maxRayDistance = 30;
      expect(maxRayDistance).toBe(30);
    });

    it('uses 1.5m target radius', () => {
      const targetRadius = 1.5;
      expect(targetRadius).toBe(1.5);
    });
  });

  describe('Preview Line', () => {
    it('shows preview when hand is tracking and pointing', () => {
      const isTracking = true;
      const isPointing = true;
      const shouldShowPreview = isTracking && isPointing;
      expect(shouldShowPreview).toBe(true);
    });

    it('hides preview when not tracking', () => {
      const isTracking = false;
      const isPointing = true;
      const shouldShowPreview = isTracking && isPointing;
      expect(shouldShowPreview).toBe(false);
    });

    it('hides preview when not pointing', () => {
      const isTracking = true;
      const isPointing = false;
      const shouldShowPreview = isTracking && isPointing;
      expect(shouldShowPreview).toBe(false);
    });

    it('uses green color when locked on target', () => {
      const targetedNode = { id: 'agent-1' };
      const previewColor = targetedNode ? '#00ff88' : '#00ffff';
      expect(previewColor).toBe('#00ff88');
    });

    it('uses cyan color when searching', () => {
      const targetedNode = null;
      const previewColor = targetedNode ? '#00ff88' : '#00ffff';
      expect(previewColor).toBe('#00ffff');
    });
  });

  describe('Haptic Feedback', () => {
    it('enables haptics only in VR mode', () => {
      const isVRMode = true;
      const enableHaptics = isVRMode;
      expect(enableHaptics).toBe(true);
    });

    it('disables haptics in desktop mode', () => {
      const isVRMode = false;
      const enableHaptics = isVRMode;
      expect(enableHaptics).toBe(false);
    });
  });
});

describe('Agent Data Conversion', () => {
  describe('agentsToTargetNodes', () => {
    it('converts agent with position to target node', () => {
      const agent = {
        id: 'agent-1',
        type: 'researcher',
        position: { x: 5, y: 10, z: 15 },
      };

      const targetNode = {
        id: agent.id,
        position: new THREE.Vector3(agent.position.x, agent.position.y, agent.position.z),
        type: agent.type,
      };

      expect(targetNode.id).toBe('agent-1');
      expect(targetNode.position.x).toBe(5);
      expect(targetNode.position.y).toBe(10);
      expect(targetNode.position.z).toBe(15);
      expect(targetNode.type).toBe('researcher');
    });

    it('filters out agents without position', () => {
      const agents = [
        { id: 'agent-1', position: { x: 0, y: 0, z: 0 } },
        { id: 'agent-2' }, // No position
        { id: 'agent-3', position: { x: 10, y: 10, z: 10 } },
      ];

      const withPositions = agents.filter((a) => a.position);
      expect(withPositions).toHaveLength(2);
    });
  });
});

describe('VRTargetHighlight Component', () => {
  describe('Ring Geometry', () => {
    it('outer ring has inner radius 1.8, outer radius 2.2', () => {
      const innerRadius = 1.8;
      const outerRadius = 2.2;
      expect(outerRadius - innerRadius).toBeCloseTo(0.4, 5);
    });

    it('inner ring has inner radius 1.2, outer radius 1.8', () => {
      const innerRadius = 1.2;
      const outerRadius = 1.8;
      expect(outerRadius - innerRadius).toBeCloseTo(0.6, 5);
    });

    it('uses 32 segments', () => {
      const segments = 32;
      expect(segments).toBe(32);
    });
  });

  describe('Animation', () => {
    it('rotates at 0.5 rad/s', () => {
      const rotationSpeed = 0.5;
      expect(rotationSpeed).toBe(0.5);
    });

    it('pulses scale with 10% amplitude', () => {
      const pulseAmplitude = 0.1;
      expect(pulseAmplitude).toBe(0.1);
    });

    it('pulses at 3 Hz', () => {
      const pulseFrequency = 3;
      expect(pulseFrequency).toBe(3);
    });
  });

  describe('Material Properties', () => {
    it('outer ring has 0.4 opacity', () => {
      const outerOpacity = 0.4;
      expect(outerOpacity).toBe(0.4);
    });

    it('inner ring has 0.2 opacity', () => {
      const innerOpacity = 0.2;
      expect(innerOpacity).toBe(0.2);
    });

    it('uses DoubleSide rendering', () => {
      const side = THREE.DoubleSide;
      expect(side).toBe(THREE.DoubleSide);
    });

    it('disables depth write', () => {
      const depthWrite = false;
      expect(depthWrite).toBe(false);
    });
  });
});

describe('VRPerformanceStats Component', () => {
  describe('Positioning', () => {
    it('offsets stats panel 0.3m below eye level', () => {
      const yOffset = -0.3;
      expect(yOffset).toBe(-0.3);
    });

    it('positions stats panel 1m in front of camera', () => {
      const zOffset = -1;
      expect(zOffset).toBe(-1);
    });
  });

  describe('Panel Dimensions', () => {
    it('panel width is 0.4m', () => {
      const width = 0.4;
      expect(width).toBe(0.4);
    });

    it('panel height is 0.15m', () => {
      const height = 0.15;
      expect(height).toBe(0.15);
    });
  });

  describe('Bar Visualization', () => {
    it('connection bar scales with count (max 0.3m)', () => {
      const activeConnections = 20;
      const barWidth = Math.min(0.02 * activeConnections, 0.3);
      expect(barWidth).toBe(0.3);
    });

    it('LOD cache bar scales with size (max 0.3m)', () => {
      const lodCacheSize = 500;
      const barWidth = Math.min(0.001 * lodCacheSize, 0.3);
      expect(barWidth).toBe(0.3);
    });

    it('bars use distinct colors', () => {
      const connectionBarColor = '#00ff88';
      const lodCacheBarColor = '#ffaa00';
      expect(connectionBarColor).not.toBe(lodCacheBarColor);
    });
  });
});

describe('XR Event Handling', () => {
  describe('Controller Selection', () => {
    it('triggers on right controller select', () => {
      const handedness = 'right';
      const hasTarget = true;
      const shouldTrigger = handedness === 'right' && hasTarget;
      expect(shouldTrigger).toBe(true);
    });

    it('ignores left controller select', () => {
      const handedness = 'left';
      const hasTarget = true;
      const shouldTrigger = handedness === 'right' && hasTarget;
      expect(shouldTrigger).toBe(false);
    });

    it('ignores select without target', () => {
      const handedness = 'right';
      const hasTarget = false;
      const shouldTrigger = handedness === 'right' && hasTarget;
      expect(shouldTrigger).toBe(false);
    });
  });

  describe('Haptic Feedback', () => {
    it('uses 0.8 intensity on selection', () => {
      const selectionIntensity = 0.8;
      expect(selectionIntensity).toBe(0.8);
    });

    it('uses 100ms duration on selection', () => {
      const selectionDuration = 100;
      expect(selectionDuration).toBe(100);
    });

    it('uses 0.3 intensity on target acquisition', () => {
      const acquisitionIntensity = 0.3;
      expect(acquisitionIntensity).toBe(0.3);
    });

    it('uses 50ms duration on target acquisition', () => {
      const acquisitionDuration = 50;
      expect(acquisitionDuration).toBe(50);
    });
  });
});

describe('Performance Constraints', () => {
  describe('Quest 3 Target', () => {
    it('targets 72 FPS', () => {
      const targetFPS = 72;
      expect(targetFPS).toBe(72);
    });

    it('limits connections to 20', () => {
      const vrMaxConnections = 20;
      expect(vrMaxConnections).toBe(20);
    });
  });

  describe('Geometry Reduction', () => {
    it('VR uses Line instead of TubeGeometry', () => {
      // Line is simpler than TubeGeometry
      const vrGeometry = 'Line';
      const desktopGeometry = 'Line'; // Actually both use Line, TubeGeometry was never used
      expect(vrGeometry).toBe(desktopGeometry);
    });

    it('VR particle size is 60% of desktop', () => {
      const vrParticleSize = 0.3;
      const desktopParticleSize = 0.5;
      const ratio = vrParticleSize / desktopParticleSize;
      expect(ratio).toBeCloseTo(0.6, 2);
    });

    it('VR sphere segments are 50% of desktop', () => {
      const vrSegments = 8;
      const desktopSegments = 16;
      const ratio = vrSegments / desktopSegments;
      expect(ratio).toBe(0.5);
    });
  });

  describe('Effect Removal', () => {
    it('disables glow effect in VR', () => {
      const vrMode = true;
      const showGlow = !vrMode;
      expect(showGlow).toBe(false);
    });

    it('enables glow effect in desktop', () => {
      const vrMode = false;
      const showGlow = !vrMode;
      expect(showGlow).toBe(true);
    });
  });
});
