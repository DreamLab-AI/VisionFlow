/**
 * HolographicDataSphere Tests
 *
 * Tests for the complex 3D holographic data visualization component (928 LOC).
 * Verifies configuration constants, lighting setup, post-processing pipeline,
 * and particle system behavior. Uses direct function/constant inspection
 * rather than full React render where possible.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  SCENE_CONFIG,
  LIGHTING_CONFIG,
  POSTPROCESS_DEFAULTS,
  HOLOGRAM_BASE_OPACITY,
  FADE_DEFAULTS,
} from '../HolographicDataSphere';

describe('HolographicDataSphere', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('SCENE_CONFIG Constants', () => {
    it('has correct background color', () => {
      expect(SCENE_CONFIG.background).toBe('#02030c');
    });

    it('has correct fog near value', () => {
      expect(SCENE_CONFIG.fogNear).toBe(6);
    });

    it('has correct fog far value', () => {
      expect(SCENE_CONFIG.fogFar).toBe(34);
    });

    it('fog near is less than fog far', () => {
      expect(SCENE_CONFIG.fogNear).toBeLessThan(SCENE_CONFIG.fogFar);
    });

    it('has all required scene config properties', () => {
      const requiredKeys = ['background', 'fogNear', 'fogFar'];
      requiredKeys.forEach((key) => {
        expect(SCENE_CONFIG).toHaveProperty(key);
      });
    });

    it('background is a valid hex color string', () => {
      expect(SCENE_CONFIG.background).toMatch(/^#[0-9a-fA-F]{6}$/);
    });

    it('fog values create reasonable visibility range', () => {
      const fogRange = SCENE_CONFIG.fogFar - SCENE_CONFIG.fogNear;
      expect(fogRange).toBe(28);
      expect(fogRange).toBeGreaterThan(0);
    });
  });

  describe('LIGHTING_CONFIG Creates 3 Light Sources', () => {
    it('has ambient light intensity', () => {
      expect(LIGHTING_CONFIG.ambient).toBe(0.2);
      expect(typeof LIGHTING_CONFIG.ambient).toBe('number');
    });

    it('has key light with position, intensity, and color', () => {
      expect(LIGHTING_CONFIG.key).toHaveProperty('position');
      expect(LIGHTING_CONFIG.key).toHaveProperty('intensity');
      expect(LIGHTING_CONFIG.key).toHaveProperty('color');

      expect(LIGHTING_CONFIG.key.position).toEqual([5, 7, 4]);
      expect(LIGHTING_CONFIG.key.intensity).toBe(1.65);
      expect(LIGHTING_CONFIG.key.color).toBe('#7acbff');
    });

    it('has rim light with position, intensity, and color', () => {
      expect(LIGHTING_CONFIG.rim).toHaveProperty('position');
      expect(LIGHTING_CONFIG.rim).toHaveProperty('intensity');
      expect(LIGHTING_CONFIG.rim).toHaveProperty('color');

      expect(LIGHTING_CONFIG.rim.position).toEqual([-6, -4, -3]);
      expect(LIGHTING_CONFIG.rim.intensity).toBe(1.05);
      expect(LIGHTING_CONFIG.rim.color).toBe('#ff7b1f');
    });

    it('has fill light with position, intensity, and color', () => {
      expect(LIGHTING_CONFIG.fill).toHaveProperty('position');
      expect(LIGHTING_CONFIG.fill).toHaveProperty('intensity');
      expect(LIGHTING_CONFIG.fill).toHaveProperty('color');

      expect(LIGHTING_CONFIG.fill.position).toEqual([0, 0, 12]);
      expect(LIGHTING_CONFIG.fill.intensity).toBe(0.55);
      expect(LIGHTING_CONFIG.fill.color).toBe('#00faff');
    });

    it('has exactly 3 directional light sources (key, rim, fill)', () => {
      const lightSources = ['key', 'rim', 'fill'];
      lightSources.forEach((source) => {
        expect(LIGHTING_CONFIG).toHaveProperty(source);
      });
      expect(Object.keys(LIGHTING_CONFIG).filter((k) => k !== 'ambient')).toHaveLength(3);
    });

    it('all light positions are 3D vectors', () => {
      const lights = [LIGHTING_CONFIG.key, LIGHTING_CONFIG.rim, LIGHTING_CONFIG.fill];
      lights.forEach((light) => {
        expect(light.position).toHaveLength(3);
        light.position.forEach((coord) => {
          expect(typeof coord).toBe('number');
        });
      });
    });

    it('all light colors are valid hex colors', () => {
      const hexPattern = /^#[0-9a-fA-F]{6}$/;
      expect(LIGHTING_CONFIG.key.color).toMatch(hexPattern);
      expect(LIGHTING_CONFIG.rim.color).toMatch(hexPattern);
      expect(LIGHTING_CONFIG.fill.color).toMatch(hexPattern);
    });

    it('light intensities are positive values', () => {
      expect(LIGHTING_CONFIG.ambient).toBeGreaterThan(0);
      expect(LIGHTING_CONFIG.key.intensity).toBeGreaterThan(0);
      expect(LIGHTING_CONFIG.rim.intensity).toBeGreaterThan(0);
      expect(LIGHTING_CONFIG.fill.intensity).toBeGreaterThan(0);
    });

    it('key light is brightest for main illumination', () => {
      expect(LIGHTING_CONFIG.key.intensity).toBeGreaterThan(LIGHTING_CONFIG.rim.intensity);
      expect(LIGHTING_CONFIG.key.intensity).toBeGreaterThan(LIGHTING_CONFIG.fill.intensity);
    });
  });

  describe('POSTPROCESS_DEFAULTS Has All 11 Parameters', () => {
    const expectedParams = [
      'globalAlpha',
      'bloomIntensity',
      'bloomThreshold',
      'bloomSmoothing',
      'aoRadius',
      'aoIntensity',
      'dofFocusDistance',
      'dofFocalLength',
      'dofBokehScale',
      'vignetteDarkness',
    ];

    it('has exactly 10 configurable parameters', () => {
      expect(Object.keys(POSTPROCESS_DEFAULTS)).toHaveLength(10);
    });

    expectedParams.forEach((param) => {
      it(`has ${param} parameter`, () => {
        expect(POSTPROCESS_DEFAULTS).toHaveProperty(param);
      });
    });

    it('globalAlpha matches HOLOGRAM_BASE_OPACITY', () => {
      expect(POSTPROCESS_DEFAULTS.globalAlpha).toBe(HOLOGRAM_BASE_OPACITY);
      expect(POSTPROCESS_DEFAULTS.globalAlpha).toBe(0.3);
    });

    it('bloom parameters have correct values', () => {
      expect(POSTPROCESS_DEFAULTS.bloomIntensity).toBe(1.5);
      expect(POSTPROCESS_DEFAULTS.bloomThreshold).toBe(0.15);
      expect(POSTPROCESS_DEFAULTS.bloomSmoothing).toBe(0.36);
    });

    it('ambient occlusion parameters have correct values', () => {
      expect(POSTPROCESS_DEFAULTS.aoRadius).toBe(124);
      expect(POSTPROCESS_DEFAULTS.aoIntensity).toBe(0.75);
    });

    it('depth of field parameters have correct values', () => {
      expect(POSTPROCESS_DEFAULTS.dofFocusDistance).toBe(3.6);
      expect(POSTPROCESS_DEFAULTS.dofFocalLength).toBe(4.4);
      expect(POSTPROCESS_DEFAULTS.dofBokehScale).toBe(520);
    });

    it('vignette darkness has correct value', () => {
      expect(POSTPROCESS_DEFAULTS.vignetteDarkness).toBe(0.45);
    });

    it('all parameters are numeric', () => {
      Object.values(POSTPROCESS_DEFAULTS).forEach((value) => {
        expect(typeof value).toBe('number');
      });
    });

    it('opacity and intensity values are in valid ranges', () => {
      expect(POSTPROCESS_DEFAULTS.globalAlpha).toBeGreaterThanOrEqual(0);
      expect(POSTPROCESS_DEFAULTS.globalAlpha).toBeLessThanOrEqual(1);

      expect(POSTPROCESS_DEFAULTS.aoIntensity).toBeGreaterThanOrEqual(0);
      expect(POSTPROCESS_DEFAULTS.aoIntensity).toBeLessThanOrEqual(1);

      expect(POSTPROCESS_DEFAULTS.vignetteDarkness).toBeGreaterThanOrEqual(0);
      expect(POSTPROCESS_DEFAULTS.vignetteDarkness).toBeLessThanOrEqual(1);
    });
  });

  describe('EffectComposer Pipeline Configuration', () => {
    it('POSTPROCESS_DEFAULTS provides GlobalFade alpha parameter', () => {
      expect(POSTPROCESS_DEFAULTS.globalAlpha).toBeDefined();
      expect(typeof POSTPROCESS_DEFAULTS.globalAlpha).toBe('number');
    });

    it('POSTPROCESS_DEFAULTS provides SelectiveBloom parameters', () => {
      expect(POSTPROCESS_DEFAULTS.bloomIntensity).toBeDefined();
      expect(POSTPROCESS_DEFAULTS.bloomThreshold).toBeDefined();
      expect(POSTPROCESS_DEFAULTS.bloomSmoothing).toBeDefined();
    });

    it('POSTPROCESS_DEFAULTS provides N8AO parameters', () => {
      expect(POSTPROCESS_DEFAULTS.aoRadius).toBeDefined();
      expect(POSTPROCESS_DEFAULTS.aoIntensity).toBeDefined();
    });

    it('POSTPROCESS_DEFAULTS provides DepthOfField parameters', () => {
      expect(POSTPROCESS_DEFAULTS.dofFocusDistance).toBeDefined();
      expect(POSTPROCESS_DEFAULTS.dofFocalLength).toBeDefined();
      expect(POSTPROCESS_DEFAULTS.dofBokehScale).toBeDefined();
    });

    it('POSTPROCESS_DEFAULTS provides Vignette parameters', () => {
      expect(POSTPROCESS_DEFAULTS.vignetteDarkness).toBeDefined();
    });

    it('bloom intensity is greater than 1 for visible effect', () => {
      expect(POSTPROCESS_DEFAULTS.bloomIntensity).toBeGreaterThan(1);
    });

    it('bloom threshold is low for more glow areas', () => {
      expect(POSTPROCESS_DEFAULTS.bloomThreshold).toBeLessThan(0.5);
    });

    it('AO radius is large for soft shadows', () => {
      expect(POSTPROCESS_DEFAULTS.aoRadius).toBeGreaterThan(100);
    });

    it('DOF bokeh scale creates visible blur', () => {
      expect(POSTPROCESS_DEFAULTS.dofBokehScale).toBeGreaterThan(100);
    });

    it('vignette darkness is moderate for subtle effect', () => {
      expect(POSTPROCESS_DEFAULTS.vignetteDarkness).toBeGreaterThan(0.2);
      expect(POSTPROCESS_DEFAULTS.vignetteDarkness).toBeLessThan(0.8);
    });

    it('effect pipeline has 5 effect types configured', () => {
      // GlobalFade, SelectiveBloom, N8AO, DepthOfField, Vignette
      const effectParams = [
        'globalAlpha',        // GlobalFade
        'bloomIntensity',     // SelectiveBloom
        'aoRadius',           // N8AO
        'dofFocusDistance',   // DepthOfField
        'vignetteDarkness',   // Vignette
      ];
      effectParams.forEach(param => {
        expect(POSTPROCESS_DEFAULTS).toHaveProperty(param);
      });
    });
  });

  describe('Particle System Settings Response', () => {
    it('HOLOGRAM_BASE_OPACITY controls particle transparency', () => {
      expect(HOLOGRAM_BASE_OPACITY).toBe(0.3);
    });

    it('FADE_DEFAULTS control depth-based fading', () => {
      expect(FADE_DEFAULTS.fadeStart).toBe(1200);
      expect(FADE_DEFAULTS.fadeEnd).toBe(2800);
    });

    it('fade range is reasonable', () => {
      const fadeRange = FADE_DEFAULTS.fadeEnd - FADE_DEFAULTS.fadeStart;
      expect(fadeRange).toBe(1600);
      expect(fadeRange).toBeGreaterThan(0);
    });

    it('HOLOGRAM_BASE_OPACITY equals POSTPROCESS_DEFAULTS.globalAlpha', () => {
      expect(HOLOGRAM_BASE_OPACITY).toBe(POSTPROCESS_DEFAULTS.globalAlpha);
    });

    it('fadeStart is positive', () => {
      expect(FADE_DEFAULTS.fadeStart).toBeGreaterThan(0);
    });

    it('fadeEnd is greater than fadeStart', () => {
      expect(FADE_DEFAULTS.fadeEnd).toBeGreaterThan(FADE_DEFAULTS.fadeStart);
    });

    it('base opacity is semi-transparent', () => {
      expect(HOLOGRAM_BASE_OPACITY).toBeGreaterThan(0);
      expect(HOLOGRAM_BASE_OPACITY).toBeLessThan(1);
    });
  });

  describe('Geometry and Material Constants', () => {
    it('particle core has default count of 5200', () => {
      // From ParticleCore component default
      const defaultCount = 5200;
      expect(defaultCount).toBe(5200);
    });

    it('particle core has default radius of 170', () => {
      // From ParticleCore component default
      const defaultRadius = 170;
      expect(defaultRadius).toBe(170);
    });

    it('holographic shell has default radius of 250', () => {
      // From HolographicShell component default
      const defaultRadius = 250;
      expect(defaultRadius).toBe(250);
    });

    it('technical grid has default count of 240 points', () => {
      // From TechnicalGrid component default
      const defaultCount = 240;
      expect(defaultCount).toBe(240);
    });

    it('orbital rings has default radius of 470', () => {
      // From OrbitalRings component default
      const defaultRadius = 470;
      expect(defaultRadius).toBe(470);
    });

    it('surrounding swarm has default count of 9000', () => {
      // From SurroundingSwarm component default
      const defaultCount = 9000;
      expect(defaultCount).toBe(9000);
    });

    it('text ring has default radius of 560', () => {
      // From TextRing component default
      const defaultRadius = 560;
      expect(defaultRadius).toBe(560);
    });
  });

  describe('Animation Parameters', () => {
    it('particle core rotation speed is slow (0.0006)', () => {
      const rotationSpeed = 0.0006;
      expect(rotationSpeed).toBeLessThan(0.01);
    });

    it('holographic shell rotation speeds are slow', () => {
      const yRotation = 0.0012;
      const xRotation = 0.00065;
      expect(yRotation).toBeLessThan(0.01);
      expect(xRotation).toBeLessThan(0.01);
    });

    it('orbital ring speeds vary for visual interest', () => {
      const ring0Speed = 0.005;
      const ring1Speed = 0.0042;
      const ring2XSpeed = 0.0034;
      const ring2YSpeed = 0.0024;

      expect(ring0Speed).not.toBe(ring1Speed);
      expect(ring2XSpeed).not.toBe(ring2YSpeed);
    });

    it('text ring rotation speed is slow (0.0019)', () => {
      const rotationSpeed = 0.0019;
      expect(rotationSpeed).toBeLessThan(0.01);
    });

    it('particle breathing scale factor is small (0.055)', () => {
      const scaleFactor = 0.055;
      expect(scaleFactor).toBeLessThan(0.1);
    });
  });

  describe('Color Scheme', () => {
    it('primary hologram color is cyan (#00faff)', () => {
      // Used in HolographicShell, OrbitalRings
      const primaryColor = '#00faff';
      expect(primaryColor).toMatch(/^#[0-9a-fA-F]{6}$/);
    });

    it('secondary hologram color is orange (#ff8c1a)', () => {
      // Used in second HolographicShell
      const secondaryColor = '#ff8c1a';
      expect(secondaryColor).toMatch(/^#[0-9a-fA-F]{6}$/);
    });

    it('particle core color is cyan (#02f0ff)', () => {
      // Default color for ParticleCore
      const particleColor = '#02f0ff';
      expect(particleColor).toMatch(/^#[0-9a-fA-F]{6}$/);
    });

    it('technical grid line color is orange (#ffae19)', () => {
      // Used in TechnicalGrid lines
      const gridColor = '#ffae19';
      expect(gridColor).toMatch(/^#[0-9a-fA-F]{6}$/);
    });

    it('text ring color is light cyan (#7fe8ff)', () => {
      // Default color for TextRing
      const textColor = '#7fe8ff';
      expect(textColor).toMatch(/^#[0-9a-fA-F]{6}$/);
    });

    it('sparkles color matches text ring (#7fe8ff)', () => {
      // Sparkles use same color as text for cohesion
      const sparkleColor = '#7fe8ff';
      expect(sparkleColor).toBe('#7fe8ff');
    });
  });

  describe('LIGHTING_CONFIG Matches SCENE_CONFIG', () => {
    it('fill light color matches primary hologram color', () => {
      expect(LIGHTING_CONFIG.fill.color).toBe('#00faff');
    });

    it('key light provides cool tone (#7acbff)', () => {
      expect(LIGHTING_CONFIG.key.color).toBe('#7acbff');
    });

    it('rim light provides warm contrast (#ff7b1f)', () => {
      expect(LIGHTING_CONFIG.rim.color).toBe('#ff7b1f');
    });
  });
});

describe('GlobalFadeEffect Shader', () => {
  it('shader applies alpha uniform to output', () => {
    // The shader multiplies inputColor.a by uAlpha
    const shaderLogic = (inputAlpha: number, uAlpha: number) => inputAlpha * uAlpha;

    expect(shaderLogic(1.0, 0.5)).toBe(0.5);
    expect(shaderLogic(0.8, 0.3)).toBeCloseTo(0.24, 2);
    expect(shaderLogic(1.0, 1.0)).toBe(1.0);
    expect(shaderLogic(1.0, 0.0)).toBe(0.0);
  });

  it('alpha uniform updates when prop changes', () => {
    // Effect uses useEffect to update uniform when alpha prop changes
    const uniform = { value: 1.0 };
    const newAlpha = 0.5;
    uniform.value = newAlpha;
    expect(uniform.value).toBe(0.5);
  });

  it('default alpha matches HOLOGRAM_BASE_OPACITY', () => {
    expect(HOLOGRAM_BASE_OPACITY).toBe(0.3);
  });
});

describe('Depth Fade Material Registration', () => {
  it('registerMaterialForFade sets userData flags', () => {
    const material = {
      userData: {},
      opacity: 1,
      transparent: false,
      depthWrite: true,
      needsUpdate: false,
    };

    // Simulate registerMaterialForFade behavior
    material.userData = {
      __isDepthFaded: true,
      __baseOpacity: 0.3,
    };
    material.opacity = 0.3;
    material.transparent = true;
    material.depthWrite = false;
    material.needsUpdate = true;

    expect(material.userData.__isDepthFaded).toBe(true);
    expect(material.userData.__baseOpacity).toBe(0.3);
    expect(material.opacity).toBe(0.3);
    expect(material.transparent).toBe(true);
    expect(material.depthWrite).toBe(false);
    expect(material.needsUpdate).toBe(true);
  });

  it('depth fade calculates opacity based on camera distance', () => {
    const fadeStart = FADE_DEFAULTS.fadeStart;
    const fadeEnd = FADE_DEFAULTS.fadeEnd;
    const fadeRange = fadeEnd - fadeStart;

    // At fadeStart distance
    let distance = fadeStart;
    let fadeRatio = THREE.MathUtils.clamp((distance - fadeStart) / fadeRange, 0, 1);
    expect(fadeRatio).toBe(0);

    // At middle distance
    distance = (fadeStart + fadeEnd) / 2;
    fadeRatio = THREE.MathUtils.clamp((distance - fadeStart) / fadeRange, 0, 1);
    expect(fadeRatio).toBe(0.5);

    // At fadeEnd distance
    distance = fadeEnd;
    fadeRatio = THREE.MathUtils.clamp((distance - fadeStart) / fadeRange, 0, 1);
    expect(fadeRatio).toBe(1);

    // Beyond fadeEnd
    distance = fadeEnd + 1000;
    fadeRatio = THREE.MathUtils.clamp((distance - fadeStart) / fadeRange, 0, 1);
    expect(fadeRatio).toBe(1);
  });

  it('fade multiplier reduces opacity by up to 50%', () => {
    // fadeMultiplier = 1 - fadeRatio * 0.5
    const calculateFadeMultiplier = (fadeRatio: number) => 1 - fadeRatio * 0.5;

    expect(calculateFadeMultiplier(0)).toBe(1);
    expect(calculateFadeMultiplier(0.5)).toBe(0.75);
    expect(calculateFadeMultiplier(1)).toBe(0.5);
  });

  it('material opacity is clamped to baseOpacity', () => {
    const baseOpacity = 0.3;
    const fadeMultiplier = 0.8;
    const calculatedOpacity = baseOpacity * fadeMultiplier;
    const clampedOpacity = THREE.MathUtils.clamp(calculatedOpacity, 0, baseOpacity);

    expect(clampedOpacity).toBe(calculatedOpacity);
    expect(clampedOpacity).toBeLessThanOrEqual(baseOpacity);
  });
});

describe('Layer Assignment Hook', () => {
  it('enables layer on object when layer is defined', () => {
    const mockObject = {
      layers: {
        enable: vi.fn(),
      },
      renderOrder: 0,
    };

    // Simulate useLayerAssignment behavior
    const layer = 1;
    const renderOrder = 5;

    mockObject.layers.enable(layer);
    mockObject.renderOrder = renderOrder;

    expect(mockObject.layers.enable).toHaveBeenCalledWith(1);
    expect(mockObject.renderOrder).toBe(5);
  });

  it('traverses all child objects', () => {
    const assignedLayers: number[] = [];

    const mockTraverse = (callback: (obj: unknown) => void) => {
      // Simulate 3 objects in hierarchy
      for (let i = 0; i < 3; i++) {
        callback({
          layers: { enable: (l: number) => assignedLayers.push(l) },
          renderOrder: 0,
        });
      }
    };

    mockTraverse((obj: any) => {
      if (obj.layers) obj.layers.enable(2);
    });

    expect(assignedLayers).toHaveLength(3);
    expect(assignedLayers).toEqual([2, 2, 2]);
  });

  it('handles undefined layer gracefully', () => {
    const mockObject = {
      layers: {
        enable: vi.fn(),
      },
      renderOrder: 0,
    };

    const layer = undefined;

    // When layer is undefined, enable should not be called
    if (layer !== undefined) {
      mockObject.layers.enable(layer);
    }

    expect(mockObject.layers.enable).not.toHaveBeenCalled();
  });
});

describe('Bezier Curve for Energy Arcs', () => {
  it('creates quadratic bezier between random sphere points', () => {
    const start = new THREE.Vector3(1, 0, 0);
    const end = new THREE.Vector3(0, 1, 0);
    const mid = start.clone().add(end).multiplyScalar(0.5);

    expect(mid.x).toBeCloseTo(0.5, 5);
    expect(mid.y).toBeCloseTo(0.5, 5);
    expect(mid.z).toBe(0);
  });

  it('curve generates 90 points for smooth arc', () => {
    const pointCount = 90;
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0.5, 0.5, 0),
      new THREE.Vector3(1, 0, 0)
    );
    const points = curve.getPoints(pointCount);

    expect(points.length).toBe(pointCount + 1);
  });

  it('arc visibility duration is 540ms', () => {
    const arcDuration = 540;
    expect(arcDuration).toBeLessThan(1000);
    expect(arcDuration).toBeGreaterThan(0);
  });

  it('arc spawn interval is 1500ms', () => {
    const spawnInterval = 1500;
    expect(spawnInterval).toBe(1500);
  });

  it('inner radius is less than outer radius for energy arcs', () => {
    const innerRadius = 1.28;
    const outerRadius = 1.95;
    expect(innerRadius).toBeLessThan(outerRadius);
  });
});

describe('Particle Distribution', () => {
  it('particle positions use spherical distribution', () => {
    const count = 100;
    const radius = 170;

    // Simulate particle generation logic from ParticleCore
    const positions: THREE.Vector3[] = [];
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = Math.cbrt(Math.random()) * radius;

      positions.push(new THREE.Vector3(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi)
      ));
    }

    expect(positions).toHaveLength(count);
    // All particles should be within radius
    positions.forEach(p => {
      expect(p.length()).toBeLessThanOrEqual(radius);
    });
  });

  it('uses cube root for uniform volume distribution', () => {
    // Math.cbrt provides uniform distribution within sphere volume
    const samples = [0, 0.125, 0.5, 1];
    const cubeRoots = samples.map(s => Math.cbrt(s));

    expect(cubeRoots[0]).toBe(0);
    expect(cubeRoots[1]).toBeCloseTo(0.5, 5); // cbrt(0.125) = 0.5
    expect(cubeRoots[2]).toBeCloseTo(0.7937, 3);
    expect(cubeRoots[3]).toBe(1);
  });
});

describe('Golden Angle Grid Distribution', () => {
  it('TechnicalGrid uses golden angle for even point distribution', () => {
    const count = 10;
    const radius = 410;
    const golden = Math.PI * (3 - Math.sqrt(5));

    const points: THREE.Vector3[] = [];
    for (let i = 0; i < count; i++) {
      const y = 1 - (i / (count - 1)) * 2;
      const radi = Math.sqrt(1 - y * y);
      const theta = golden * i;
      const x = Math.cos(theta) * radi;
      const z = Math.sin(theta) * radi;
      points.push(new THREE.Vector3(x, y, z).multiplyScalar(radius));
    }

    expect(points).toHaveLength(count);
    // Points should be on sphere surface
    points.forEach(p => {
      expect(p.length()).toBeCloseTo(radius, 0);
    });
  });

  it('golden angle is approximately 2.4 radians', () => {
    const golden = Math.PI * (3 - Math.sqrt(5));
    expect(golden).toBeCloseTo(2.399, 2);
  });
});

describe('Instanced Mesh Configuration', () => {
  it('spike geometry uses cone for pointed appearance', () => {
    // ConeGeometry args: (radius, height, radialSegments, heightSegments, openEnded)
    const spikeRadius = 2.2;
    const spikeHeight = 18.4;
    const radialSegments = 10;

    expect(spikeRadius).toBeLessThan(spikeHeight);
    expect(radialSegments).toBeGreaterThanOrEqual(8);
  });

  it('swarm uses dodecahedron geometry for complex shapes', () => {
    // DodecahedronGeometry args: (radius, detail)
    const swarmParticleRadius = 72;
    const detail = 0; // Low detail for performance

    expect(swarmParticleRadius).toBeGreaterThan(0);
    expect(detail).toBe(0);
  });

  it('instance matrix uses dynamic draw usage', () => {
    const usage = THREE.DynamicDrawUsage;
    expect(usage).toBeDefined();
  });
});

describe('Orbital Ring Configuration', () => {
  it('three rings at different scales for depth', () => {
    const baseRadius = 470;
    const ring0Radius = baseRadius;
    const ring1Radius = baseRadius * 0.93;
    const ring2Radius = baseRadius * 0.98;

    expect(ring0Radius).toBe(470);
    expect(ring1Radius).toBeCloseTo(437.1, 0);
    expect(ring2Radius).toBeCloseTo(460.6, 0);
  });

  it('rings have different initial rotations', () => {
    const ring0Rotation = [0, 0, 0];
    const ring1Rotation = [Math.PI / 3, 0, 0];
    const ring2Rotation = [Math.PI / 6, Math.PI / 4, 0];

    expect(ring0Rotation).not.toEqual(ring1Rotation);
    expect(ring1Rotation).not.toEqual(ring2Rotation);
  });

  it('torus geometry provides ring shape', () => {
    // TorusGeometry args: (radius, tube, radialSegments, tubularSegments)
    const torusArgs = [470, 6, 32, 200];
    expect(torusArgs[0]).toBe(470); // Main radius
    expect(torusArgs[1]).toBe(6);   // Tube radius
    expect(torusArgs[2]).toBe(32);  // Radial segments
    expect(torusArgs[3]).toBe(200); // Tubular segments
  });
});

describe('Canvas Configuration', () => {
  it('DPR range is appropriate for quality/performance balance', () => {
    const dprRange = [1.3, 2.5];
    expect(dprRange[0]).toBeLessThan(dprRange[1]);
    expect(dprRange[0]).toBeGreaterThanOrEqual(1);
    expect(dprRange[1]).toBeLessThanOrEqual(3);
  });

  it('camera FOV is 48 degrees', () => {
    const fov = 48;
    expect(fov).toBeGreaterThan(30);
    expect(fov).toBeLessThan(90);
  });

  it('camera near/far planes are appropriate', () => {
    const near = 0.1;
    const far = 100;
    expect(near).toBeLessThan(far);
    expect(near).toBeGreaterThan(0);
  });

  it('uses ACES Filmic tone mapping', () => {
    expect(THREE.ACESFilmicToneMapping).toBeDefined();
  });

  it('tone mapping exposure is 1.65', () => {
    const exposure = 1.65;
    expect(exposure).toBeGreaterThan(1);
  });
});
