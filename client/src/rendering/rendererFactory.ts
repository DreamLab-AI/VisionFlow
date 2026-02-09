import * as THREE from 'three';

/**
 * Whether the active renderer is WebGPU.
 * Components check this to adjust material properties for WebGPU compatibility
 * (e.g. disabling transmission which crashes the transparent pass).
 */
export let isWebGPURenderer = false;

/**
 * Renderer factory for R3F <Canvas gl={rendererFactory}>.
 * R3F calls: await glConfig(defaultProps) where defaultProps = { canvas, antialias, ... }
 * Tries WebGPURenderer first, falls back to WebGLRenderer.
 */
export async function createGemRenderer(defaultProps: Record<string, any>) {
  const canvas = defaultProps.canvas as HTMLCanvasElement;

  try {
    const threeWebGPU = await import('three/webgpu') as any;
    const WebGPURenderer = threeWebGPU.WebGPURenderer;

    if (typeof WebGPURenderer !== 'function') {
      throw new Error('WebGPURenderer export not found');
    }

    const renderer = new WebGPURenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      forceWebGL: false,
    });

    await renderer.init();

    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Expose renderer type for components to check
    (renderer as any).__isWebGPURenderer = true;
    isWebGPURenderer = true;

    console.log('[GemRenderer] WebGPU renderer initialized');
    return renderer;
  } catch (err) {
    console.warn('[GemRenderer] WebGPU unavailable, falling back to WebGL:', err);

    const renderer = new THREE.WebGLRenderer({
      ...defaultProps,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });

    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    isWebGPURenderer = false;

    console.log('[GemRenderer] WebGL fallback renderer initialized');
    return renderer;
  }
}
