import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { useSettingsStore } from '../store/settingsStore';

interface GemPostProcessingProps {
  enabled?: boolean;
}

/**
 * Unified post-processing component supporting both WebGPU (node-based) and WebGL (EffectComposer).
 *
 * Rendering ownership:
 *   R3F v9 skips its default gl.render() call whenever any useFrame subscriber
 *   has priority > 0 (see @react-three/fiber loop.ts: `if (!state.internal.priority ...)`).
 *   We register at priority 1 when post-processing is enabled, which means this
 *   component becomes the sole renderer -- no double-render occurs.
 *
 * WebGPU path uses:
 *   - PostProcessing from three/webgpu
 *   - pass() from three/tsl
 *   - bloom() from three/examples/jsm/tsl/display/BloomNode.js
 *
 * WebGL path uses:
 *   - EffectComposer + UnrealBloomPass from three/examples/jsm/postprocessing
 */
export const GemPostProcessing: React.FC<GemPostProcessingProps> = ({ enabled = true }) => {
  const { gl, scene, camera } = useThree();
  const settings = useSettingsStore(state => state.settings);
  const composerRef = useRef<any>(null);
  const postProcessingRef = useRef<any>(null);
  const bloomNodeRef = useRef<any>(null);
  const disposeRef = useRef<(() => void) | null>(null);
  const isWebGPU = (gl as any).__isWebGPURenderer === true;

  const glowSettings = settings?.visualisation?.glow;
  const bloomSettings = settings?.visualisation?.bloom;

  const effectEnabled = glowSettings?.enabled || bloomSettings?.enabled;
  const isEnabledWebGL = enabled && !isWebGPU && effectEnabled;
  const isEnabledWebGPU = enabled && isWebGPU && effectEnabled;

  // Extract primitive values to avoid stale closures and unnecessary effect deps.
  // Object references (glowSettings, bloomSettings) change on every settings update
  // even when the underlying values haven't changed -- primitives are stable.
  const activeSource = !bloomSettings?.enabled && glowSettings?.enabled ? glowSettings : bloomSettings;
  const bloomStrength = (activeSource as any)?.strength ?? (activeSource as any)?.intensity ?? 0.8;
  const bloomRadius = (activeSource as any)?.radius ?? 0.4;
  const bloomThreshold = (activeSource as any)?.threshold ?? 0.4;
  const glowEnabled = glowSettings?.enabled !== false;

  const getEffectParams = useCallback(() => ({
    strength: bloomStrength,
    radius: bloomRadius,
    threshold: bloomThreshold,
    enabled: glowEnabled,
  }), [bloomStrength, bloomRadius, bloomThreshold, glowEnabled]);

  // Stable params object for effects that need the values directly
  const effectParams = useMemo(() => ({
    strength: bloomStrength,
    radius: bloomRadius,
    threshold: bloomThreshold,
  }), [bloomStrength, bloomRadius, bloomThreshold]);

  // WebGPU node-based post-processing path
  useEffect(() => {
    if (!isEnabledWebGPU) {
      // Clean up WebGPU resources if switching away
      if (disposeRef.current) {
        disposeRef.current();
        disposeRef.current = null;
      }
      postProcessingRef.current = null;
      bloomNodeRef.current = null;
      return;
    }

    let disposed = false;

    (async () => {
      try {
        const { PostProcessing } = await import('three/webgpu');
        const { pass } = await import('three/tsl');
        const { bloom } = await import('three/examples/jsm/tsl/display/BloomNode.js');

        if (disposed) return;

        const { strength, radius, threshold } = effectParams;

        // Build the node graph:
        //   scenePass -> bloom -> scenePassColor + bloomResult
        const scenePass = pass(scene, camera);
        const scenePassColor = scenePass.getTextureNode('output');
        const bloomPass = bloom(scenePassColor, strength, radius, threshold);

        // Compose: original scene + additive bloom
        const outputNode = scenePassColor.add(bloomPass);

        const postProcessing = new PostProcessing(gl as any, outputNode);

        postProcessingRef.current = postProcessing;
        bloomNodeRef.current = bloomPass;

        disposeRef.current = () => {
          postProcessing.dispose();
          if (bloomPass.dispose) bloomPass.dispose();
          if (scenePass.dispose) scenePass.dispose();
        };
      } catch (err) {
        console.warn('[GemPostProcessing] Failed to init WebGPU bloom:', err);
      }
    })();

    return () => {
      disposed = true;
      if (disposeRef.current) {
        disposeRef.current();
        disposeRef.current = null;
      }
      postProcessingRef.current = null;
      bloomNodeRef.current = null;
    };
  }, [isEnabledWebGPU, gl, scene, camera, effectParams]);

  // Update WebGPU bloom uniforms when settings change without full rebuild
  useEffect(() => {
    if (!isEnabledWebGPU || !bloomNodeRef.current) return;
    const bloomNode = bloomNodeRef.current;
    if (bloomNode.strength) bloomNode.strength.value = bloomStrength;
    if (bloomNode.radius) bloomNode.radius.value = bloomRadius;
    if (bloomNode.threshold) bloomNode.threshold.value = bloomThreshold;
  }, [isEnabledWebGPU, bloomStrength, bloomRadius, bloomThreshold]);

  // WebGL EffectComposer path
  useEffect(() => {
    if (!isEnabledWebGL) {
      composerRef.current = null;
      return;
    }

    let disposed = false;
    (async () => {
      try {
        const { EffectComposer } = await import('three/examples/jsm/postprocessing/EffectComposer.js');
        const { RenderPass } = await import('three/examples/jsm/postprocessing/RenderPass.js');
        const { UnrealBloomPass } = await import('three/examples/jsm/postprocessing/UnrealBloomPass.js');
        const THREE = await import('three');

        if (disposed) return;

        const { strength, threshold, radius } = effectParams;

        const composer = new EffectComposer(gl as any);
        composer.addPass(new RenderPass(scene, camera));

        const bloomPass = new UnrealBloomPass(
          new THREE.Vector2(window.innerWidth, window.innerHeight),
          strength,
          radius,
          threshold
        );
        composer.addPass(bloomPass);

        composerRef.current = composer;
      } catch (err) {
        console.warn('[GemPostProcessing] Failed to init WebGL bloom:', err);
      }
    })();

    return () => {
      disposed = true;
      if (composerRef.current) {
        composerRef.current.dispose?.();
        composerRef.current = null;
      }
    };
  }, [isEnabledWebGL, gl, scene, camera, effectParams]);

  // Render loop: delegate to whichever pipeline is active.
  //
  // Priority 1 tells R3F to skip its default gl.render() call (R3F v9 increments
  // internal.priority for any subscriber with priority > 0, and only calls
  // gl.render when internal.priority === 0). This prevents double-rendering.
  //
  // During async initialization, the post-processing refs are null while the
  // dynamic imports resolve. We fall back to gl.render() in that window to
  // avoid black frames.
  useFrame(({ gl: renderer, scene: s, camera: cam }) => {
    if (postProcessingRef.current) {
      postProcessingRef.current.render();
    } else if (composerRef.current) {
      composerRef.current.render();
    } else {
      // Fallback: post-processing not yet initialized (async import in flight)
      // or effects disabled. Render the scene directly so there are no black frames.
      renderer.render(s, cam);
    }
  }, (isEnabledWebGPU || isEnabledWebGL) ? 1 : undefined);

  return null;
};

export default GemPostProcessing;
