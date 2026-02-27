import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useSettingsStore } from '../../../store/settingsStore';

interface FpsMonitorState {
  samples: number[];
  lastDegradeTime: number;
  lastRestoreTime: number;
  degradeLevel: number; // 0 = full quality, 1 = light degrade, 2 = medium, 3 = heavy
}

const SAMPLE_SIZE = 60; // ~1 second at 60fps
const DEGRADE_SUSTAIN_MS = 2000;
const RESTORE_SUSTAIN_MS = 3000;
const MAX_DEGRADE_LEVEL = 3;

/**
 * Monitors real-time FPS inside the R3F render loop and automatically
 * degrades / restores visual quality settings to maintain a minimum
 * frame-rate when `qualityGates.autoAdjust` is enabled.
 *
 * Degradation levels:
 *   0 - Full quality (defaults restored)
 *   1 - Reduced particles + atmosphere resolution
 *   2 - Wisps disabled, minimum particles
 *   3 - Scene effects disabled entirely
 */
export function useFpsMonitor() {
  const stateRef = useRef<FpsMonitorState>({
    samples: [],
    lastDegradeTime: 0,
    lastRestoreTime: 0,
    degradeLevel: 0,
  });

  useFrame((_rootState, delta) => {
    // Read settings outside of React subscription to avoid re-renders
    const settings = useSettingsStore.getState().settings;
    const autoAdjust = settings?.qualityGates?.autoAdjust;

    if (!autoAdjust) {
      // When the feature is toggled off, reset internal tracking state
      // but leave the current visual settings alone (user controls those).
      if (stateRef.current.degradeLevel > 0) {
        stateRef.current = {
          samples: [],
          lastDegradeTime: 0,
          lastRestoreTime: 0,
          degradeLevel: 0,
        };
      }
      return;
    }

    const minFps = settings?.qualityGates?.minFpsThreshold ?? 30;
    const now = performance.now();
    const fps = delta > 0 ? 1 / delta : 60;
    const state = stateRef.current;

    // Rolling FPS average (ring-buffer style via shift)
    state.samples.push(fps);
    if (state.samples.length > SAMPLE_SIZE) {
      state.samples.shift();
    }

    // Wait until we have a meaningful window before making decisions
    if (state.samples.length < SAMPLE_SIZE / 2) return;

    const avgFps =
      state.samples.reduce((sum, v) => sum + v, 0) / state.samples.length;

    // --- Degrade path: sustained low FPS ---
    if (avgFps < minFps && state.degradeLevel < MAX_DEGRADE_LEVEL) {
      if (state.lastDegradeTime === 0) {
        state.lastDegradeTime = now;
      } else if (now - state.lastDegradeTime > DEGRADE_SUSTAIN_MS) {
        state.degradeLevel++;
        state.lastDegradeTime = now;
        state.lastRestoreTime = 0;
        applyDegradeLevel(state.degradeLevel);
      }
    } else if (avgFps >= minFps) {
      // FPS recovered above threshold -- reset degrade timer
      state.lastDegradeTime = 0;
    }

    // --- Restore path: sustained high FPS (with hysteresis band of +10) ---
    if (avgFps > minFps + 10 && state.degradeLevel > 0) {
      if (state.lastRestoreTime === 0) {
        state.lastRestoreTime = now;
      } else if (now - state.lastRestoreTime > RESTORE_SUSTAIN_MS) {
        state.degradeLevel--;
        state.lastRestoreTime = now;
        state.lastDegradeTime = 0;
        applyDegradeLevel(state.degradeLevel);
      }
    } else if (avgFps <= minFps + 10) {
      // Not yet high enough -- reset restore timer
      state.lastRestoreTime = 0;
    }
  });

  return stateRef;
}

/**
 * Applies a single atomic settings update for the given degradation level.
 * Uses `updateSettings` (immer-based) to batch all changes into one store
 * update, avoiding cascading re-renders from multiple `set()` calls.
 */
function applyDegradeLevel(level: number): void {
  const store = useSettingsStore.getState();

  switch (level) {
    case 0:
      // Full quality -- restore defaults
      store.updateSettings(draft => {
        if (!draft.visualisation) return;
        if (!draft.visualisation.sceneEffects) {
          draft.visualisation.sceneEffects = {};
        }
        draft.visualisation.sceneEffects.enabled = true;
        draft.visualisation.sceneEffects.particleCount = 256;
        draft.visualisation.sceneEffects.wispsEnabled = true;
        draft.visualisation.sceneEffects.atmosphereResolution = 128;
      });
      break;

    case 1:
      // Light degrade -- reduce particles, lower atmosphere
      store.updateSettings(draft => {
        if (!draft.visualisation) return;
        if (!draft.visualisation.sceneEffects) {
          draft.visualisation.sceneEffects = {};
        }
        draft.visualisation.sceneEffects.particleCount = 128;
        draft.visualisation.sceneEffects.atmosphereResolution = 64;
      });
      break;

    case 2:
      // Medium degrade -- disable wisps, minimum particles
      store.updateSettings(draft => {
        if (!draft.visualisation) return;
        if (!draft.visualisation.sceneEffects) {
          draft.visualisation.sceneEffects = {};
        }
        draft.visualisation.sceneEffects.wispsEnabled = false;
        draft.visualisation.sceneEffects.particleCount = 64;
      });
      break;

    case 3:
      // Heavy degrade -- disable scene effects entirely
      store.updateSettings(draft => {
        if (!draft.visualisation) return;
        if (!draft.visualisation.sceneEffects) {
          draft.visualisation.sceneEffects = {};
        }
        draft.visualisation.sceneEffects.enabled = false;
      });
      break;
  }
}
