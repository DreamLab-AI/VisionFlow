import * as THREE from 'three';
import type { RendererCapabilities } from '../features/settings/config/settings';

/**
 * Whether the active renderer is a true WebGPU backend (not WebGPURenderer
 * falling back to its internal WebGL2 backend).
 *
 * Components check this to adjust material properties for WebGPU compatibility
 * (e.g. disabling transmission which crashes the transparent pass).
 *
 * On browsers without `navigator.gpu` (Firefox, older Safari), we skip
 * WebGPURenderer entirely and use WebGLRenderer — this avoids the hybrid
 * WebGPURenderer+WebGLBackend path that causes oversized render targets,
 * PMREM blowups, and incorrect material codepaths.
 */
export let isWebGPURenderer = false;

/**
 * Runtime renderer capabilities — populated after renderer init.
 * Read by the settings panel to display active rendering features.
 */
export let rendererCapabilities: RendererCapabilities = {
  backend: 'webgl',
  tslMaterialsActive: false,
  nodeBasedBloom: false,
  gpuAdapterName: 'unknown',
  maxTextureSize: 0,
  pixelRatio: 1,
};

/**
 * Detect XR headset user agents (Quest 3, Oculus Browser, etc.)
 * for pixel ratio capping and WebGPU init timeout.
 */
function isXRHeadsetBrowser(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  return /Quest|OculusBrowser|Pico|VR/i.test(ua);
}

/**
 * Resolve a max pixel ratio appropriate for the device.
 * XR headsets get capped to 1.0 to avoid GPU memory blowup on
 * the stereoscopic render targets (each eye = full resolution).
 */
function getMaxPixelRatio(): number {
  return isXRHeadsetBrowser() ? 1.0 : 2.0;
}

/**
 * Renderer factory for R3F <Canvas gl={rendererFactory}>.
 * R3F calls: await glConfig(defaultProps) where defaultProps = { canvas, antialias, ... }
 *
 * Strategy:
 *   1. Check `navigator.gpu` — if absent, go straight to WebGLRenderer.
 *   2. Create WebGPURenderer with forceWebGL: false.
 *   3. After init(), verify the backend is actually WebGPU (not internal WebGL2 fallback).
 *   4. If the backend fell back to WebGL2, discard and use clean WebGLRenderer instead.
 *   5. Timeout guard: if WebGPU init takes >5s, fall back to WebGL (Quest 3 sometimes hangs).
 */
export async function createGemRenderer(defaultProps: Record<string, any>) {
  const canvas = defaultProps.canvas as HTMLCanvasElement;
  const maxDPR = getMaxPixelRatio();

  // Gate 1: browser must expose the WebGPU API
  if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
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

      // Timeout guard: Quest 3's Oculus Browser can hang during WebGPU adapter
      // negotiation. Cap init to 5 seconds then fall back to WebGL.
      const initTimeout = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('WebGPU init timed out (5s)')), 5000)
      );
      await Promise.race([renderer.init(), initTimeout]);

      // Gate 2: verify the backend is actually WebGPU, not the internal WebGL2 fallback.
      // Three.js r182 WebGPURenderer.init() silently falls back to WebGLBackend when
      // the GPU adapter request fails. Check the backend class name.
      const backendName = renderer.backend?.constructor?.name ?? '';
      if (backendName === 'WebGLBackend') {
        console.warn('[GemRenderer] WebGPURenderer fell back to WebGLBackend — using clean WebGLRenderer instead');
        renderer.dispose();
        throw new Error('WebGPU backend unavailable (got WebGLBackend)');
      }

      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.2;
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxDPR));

      // Expose renderer type for components to check
      (renderer as any).__isWebGPURenderer = true;
      isWebGPURenderer = true;

      // Guard against drawIndexed(Infinity) crashes in the WebGPU backend.
      // Some objects (e.g. InstancedMesh during async init, or three.js internal
      // passes) can transiently have invalid draw parameters. Rather than crashing
      // the entire render loop, we catch the TypeError and skip the object.
      const _origRenderObject = renderer.renderObject.bind(renderer);
      const _warnedObjects = new WeakSet<object>();
      renderer.renderObject = function (object: any, ...rest: any[]) {
        try {
          return _origRenderObject(object, ...rest);
        } catch (err: any) {
          if (!_warnedObjects.has(object)) {
            _warnedObjects.add(object);
            console.warn(
              '[GemRenderer] renderObject skipped:',
              object?.name || object?.type || object?.uuid,
              err?.message,
            );
          }
        }
      };

      // Populate renderer capabilities for settings panel
      const adapterInfo = renderer.backend?.adapter?.info ?? renderer.backend?.adapter ?? {};
      rendererCapabilities = {
        backend: 'webgpu',
        tslMaterialsActive: true,  // TSL upgrade runs asynchronously per-material
        nodeBasedBloom: true,
        gpuAdapterName: (adapterInfo as any)?.description
          || (adapterInfo as any)?.device
          || backendName
          || 'WebGPU',
        maxTextureSize: 16384,  // WebGPU minimum guaranteed
        pixelRatio: Math.min(window.devicePixelRatio, maxDPR),
      };

      console.log('[GemRenderer] WebGPU renderer initialized (backend:', backendName + ')');
      return renderer;
    } catch (err) {
      console.warn('[GemRenderer] WebGPU unavailable, falling back to WebGL:', err);
    }
  } else {
    console.log('[GemRenderer] navigator.gpu not available — using WebGL directly');
  }

  // WebGL fallback — clean path, no hybrid renderer quirks
  const renderer = new THREE.WebGLRenderer({
    ...defaultProps,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });

  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxDPR));

  isWebGPURenderer = false;

  // Populate renderer capabilities for WebGL fallback
  const gl2 = renderer.getContext();
  const isXR = isXRHeadsetBrowser();
  rendererCapabilities = {
    backend: 'webgl',
    tslMaterialsActive: false,
    nodeBasedBloom: false,
    gpuAdapterName: (gl2 as any)?.getParameter?.((gl2 as any)?.RENDERER) || (isXR ? 'WebGL (XR)' : 'WebGL'),
    maxTextureSize: (gl2 as any)?.getParameter?.((gl2 as any)?.MAX_TEXTURE_SIZE) || 4096,
    pixelRatio: Math.min(window.devicePixelRatio, maxDPR),
  };

  if (isXR) {
    console.log('[GemRenderer] XR headset detected — pixel ratio capped to', maxDPR);
  }

  console.log('[GemRenderer] WebGL renderer initialized');
  return renderer;
}
