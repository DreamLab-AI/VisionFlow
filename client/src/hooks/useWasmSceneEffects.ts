/**
 * useWasmSceneEffects
 *
 * React hook that lazy-loads the WASM scene effects module and provides
 * particle and atmosphere systems for Three.js rendering.
 *
 * - Loads WASM once on first mount (shared across all consumers)
 * - Returns stable refs that update in-place (no re-renders)
 * - Handles cleanup on unmount
 * - Falls back gracefully if WASM is unavailable
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  initSceneEffects,
  ParticleFieldBridge,
  AtmosphereFieldBridge,
  WispFieldBridge,
  type SceneEffectsAPI,
} from '../wasm/scene-effects-bridge';

export interface WasmSceneEffectsState {
  /** Whether the WASM module has loaded successfully. */
  ready: boolean;
  /** Whether WASM loading failed (effects should be hidden). */
  failed: boolean;
  /** The particle field instance (null until ready). */
  particles: ParticleFieldBridge | null;
  /** The atmosphere field instance (null until ready). */
  atmosphere: AtmosphereFieldBridge | null;
  /** The energy wisps instance (null until ready). */
  wisps: WispFieldBridge | null;
  /**
   * Call each frame with delta time and camera position.
   * No-op if WASM is not loaded.
   */
  update: (dt: number, cameraX: number, cameraY: number, cameraZ: number) => void;
}

interface UseWasmSceneEffectsOptions {
  /** Number of particles (default 256, max 512). */
  particleCount?: number;
  /** Number of energy wisps (default 48, max 128). */
  wispCount?: number;
  /** Atmosphere texture width (default 128). */
  atmosphereWidth?: number;
  /** Atmosphere texture height (default 128). */
  atmosphereHeight?: number;
  /** Whether effects are enabled (false skips WASM loading). */
  enabled?: boolean;
}

export function useWasmSceneEffects(
  options: UseWasmSceneEffectsOptions = {},
): WasmSceneEffectsState {
  const {
    particleCount = 256,
    wispCount = 48,
    atmosphereWidth = 128,
    atmosphereHeight = 128,
    enabled = true,
  } = options;

  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);

  const particlesRef = useRef<ParticleFieldBridge | null>(null);
  const atmosphereRef = useRef<AtmosphereFieldBridge | null>(null);
  const wispsRef = useRef<WispFieldBridge | null>(null);
  const apiRef = useRef<SceneEffectsAPI | null>(null);
  const mountedRef = useRef(true);

  // Stable update function that reads from refs (no re-renders).
  const update = useCallback(
    (dt: number, cameraX: number, cameraY: number, cameraZ: number) => {
      particlesRef.current?.update(dt, cameraX, cameraY, cameraZ);
      atmosphereRef.current?.update(dt);
      wispsRef.current?.update(dt, cameraX, cameraY, cameraZ);
    },
    [],
  );

  useEffect(() => {
    mountedRef.current = true;

    if (!enabled) {
      return;
    }

    let disposed = false;

    (async () => {
      try {
        const api = await initSceneEffects();

        if (disposed || !mountedRef.current) return;

        apiRef.current = api;

        const particles = api.createParticleField(particleCount);
        const atmosphere = api.createAtmosphereField(atmosphereWidth, atmosphereHeight);
        const wisps = api.createWispField(wispCount);

        if (disposed || !mountedRef.current) {
          particles.dispose();
          atmosphere.dispose();
          wisps.dispose();
          return;
        }

        particlesRef.current = particles;
        atmosphereRef.current = atmosphere;
        wispsRef.current = wisps;
        setReady(true);
      } catch {
        if (!disposed && mountedRef.current) {
          setFailed(true);
        }
      }
    })();

    return () => {
      disposed = true;
      mountedRef.current = false;

      if (particlesRef.current) {
        particlesRef.current.dispose();
        particlesRef.current = null;
      }
      if (atmosphereRef.current) {
        atmosphereRef.current.dispose();
        atmosphereRef.current = null;
      }
      if (wispsRef.current) {
        wispsRef.current.dispose();
        wispsRef.current = null;
      }
      apiRef.current = null;
      setReady(false);
    };
  }, [enabled, particleCount, wispCount, atmosphereWidth, atmosphereHeight]);

  return {
    ready,
    failed,
    particles: particlesRef.current,
    atmosphere: atmosphereRef.current,
    wisps: wispsRef.current,
    update,
  };
}
