import { useFrame as useR3FFrame } from '@react-three/fiber';
import { useSettingsStore } from '@/store/settingsStore';
import { useRef } from 'react';

/**
 * Optimized useFrame hook that respects performance settings and throttles updates
 * when performance debug is disabled or when FPS targets are not met.
 */
export const useOptimizedFrame = (
  callback: (state: any, delta: number) => void,
  priority?: number,
  throttleMs: number = 16 // Default to ~60fps
) => {
  const settings = useSettingsStore(state => state.settings);
  const lastUpdateRef = useRef(0);
  const isPerformanceDebugEnabled = settings?.system?.debug?.enablePerformanceDebug;
  
  useR3FFrame((state, delta) => {
    // If performance debug is disabled, throttle the updates
    if (!isPerformanceDebugEnabled) {
      const now = Date.now();
      if (now - lastUpdateRef.current < throttleMs) {
        return;
      }
      lastUpdateRef.current = now;
    }
    
    // Call the original callback
    callback(state, delta);
  }, priority);
};

/**
 * High-performance useFrame for critical animations (always runs)
 */
export const useCriticalFrame = (
  callback: (state: any, delta: number) => void,
  priority?: number
) => {
  useR3FFrame(callback, priority);
};

/**
 * Low-priority useFrame for non-essential animations (heavily throttled when performance debug is off)
 */
export const useLowPriorityFrame = (
  callback: (state: any, delta: number) => void,
  priority?: number
) => {
  useOptimizedFrame(callback, priority, 33); // ~30fps throttling
};