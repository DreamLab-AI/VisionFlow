import { useFrame as useR3FFrame } from '@react-three/fiber';
import { useSettingsStore } from '@/store/settingsStore';
import { useRef } from 'react';


export const useOptimizedFrame = (
  callback: (state: any, delta: number) => void,
  priority?: number,
  throttleMs: number = 16 
) => {
  const settings = useSettingsStore(state => state.settings);
  const lastUpdateRef = useRef(0);
  const isPerformanceDebugEnabled = settings?.system?.debug?.enablePerformanceDebug;
  
  useR3FFrame((state, delta) => {
    
    if (!isPerformanceDebugEnabled) {
      const now = Date.now();
      if (now - lastUpdateRef.current < throttleMs) {
        return;
      }
      lastUpdateRef.current = now;
    }
    
    
    callback(state, delta);
  }, priority);
};


export const useCriticalFrame = (
  callback: (state: any, delta: number) => void,
  priority?: number
) => {
  useR3FFrame(callback, priority);
};


export const useLowPriorityFrame = (
  callback: (state: any, delta: number) => void,
  priority?: number
) => {
  useOptimizedFrame(callback, priority, 33); 
};