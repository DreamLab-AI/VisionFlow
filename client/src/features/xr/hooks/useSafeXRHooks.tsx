import { useEffect, useState, useCallback, useMemo } from 'react';
import { createLogger } from '../../../utils/logger';
import * as THREE from 'three';

const logger = createLogger('useSafeXRHooks');

interface SafeXRState {
  isAvailable: boolean;
  isSupported: boolean;
  isInCanvas: boolean;
  error: string | null;
}

/**
 * Safe wrapper for React Three Fiber context that doesn't throw errors
 * when used outside Canvas component.
 */
export const useSafeThreeContext = () => {
  const [context, setContext] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check if we're in a React Three Fiber context by checking for the global store
    try {
      // Try to import useThree dynamically only when needed
      const checkContext = () => {
        // Check if window.__r3f exists (React Three Fiber global)
        if (typeof window !== 'undefined' && (window as any).__r3f) {
          const store = (window as any).__r3f.roots?.values()?.next()?.value;
          if (store) {
            const state = store.getState();
            setContext(state);
            setError(null);
            return;
          }
        }
        setContext(null);
        setError('Not within Canvas context');
      };

      checkContext();
    } catch (err) {
      setContext(null);
      setError(err instanceof Error ? err.message : 'Failed to access Three context');
    }
  }, []);

  return { context, error };
};

/**
 * Hook for safely using React Three Fiber hooks within XR context.
 * Provides fallback implementations when useThree() is unavailable.
 */
export const useSafeThree = () => {
  const [safeState, setSafeState] = useState<SafeXRState>({
    isAvailable: false,
    isSupported: false,
    isInCanvas: false,
    error: null,
  });

  const { context, error } = useSafeThreeContext();

  useEffect(() => {
    if (context) {
      setSafeState({
        isAvailable: true,
        isSupported: true,
        isInCanvas: true,
        error: null,
      });
    } else {
      setSafeState({
        isAvailable: false,
        isSupported: false,
        isInCanvas: false,
        error: error || 'React Three Fiber context not available',
      });
    }
  }, [context, error]);

  // Fallback implementations
  const fallbackCamera = useMemo(() => {
    return new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  }, []);

  const fallbackScene = useMemo(() => {
    return new THREE.Scene();
  }, []);

  const fallbackRenderer = useMemo(() => {
    if (typeof window !== 'undefined') {
      try {
        return new THREE.WebGLRenderer({ antialias: true });
      } catch {
        return null;
      }
    }
    return null;
  }, []);

  return {
    ...safeState,
    camera: context?.camera || fallbackCamera,
    scene: context?.scene || fallbackScene,
    gl: context?.gl || fallbackRenderer,
    size: context?.size || { width: window.innerWidth, height: window.innerHeight },
    viewport: context?.viewport || { 
      width: window.innerWidth, 
      height: window.innerHeight,
      aspect: window.innerWidth / window.innerHeight,
      distance: 1,
      factor: 1,
      dpr: window.devicePixelRatio || 1
    },
  };
};

/**
 * Safe XR hook that provides fallback values
 */
export const useSafeXR = () => {
  return {
    isPresenting: false,
    session: null,
    controllers: [],
    player: null,
  };
};

/**
 * HOC to wrap components that need safe XR access
 */
export const withSafeXR = (Component: React.FC<any>, name: string) => {
  return (props: any) => {
    const { isInCanvas } = useSafeThree();
    
    // Only render XR components when inside Canvas
    if (!isInCanvas) {
      return null;
    }
    
    return <Component {...props} />;
  };
};

/**
 * Hook for async WebXR capability detection
 */
export const useAsyncXRCapability = () => {
  const [xrState, setXRState] = useState({
    isChecking: true,
    isSupported: false,
    supportedModes: [] as XRSessionMode[],
    error: null as string | null,
  });

  useEffect(() => {
    let mounted = true;

    const checkXRCapability = async () => {
      try {
        if (!('xr' in navigator)) {
          if (mounted) {
            setXRState({
              isChecking: false,
              isSupported: false,
              supportedModes: [],
              error: 'WebXR not available in this browser',
            });
          }
          return;
        }

        const modes: XRSessionMode[] = ['immersive-ar', 'immersive-vr', 'inline'];
        const supportedModes: XRSessionMode[] = [];

        // Check each mode asynchronously with timeout
        const checkMode = async (mode: XRSessionMode): Promise<boolean> => {
          try {
            const timeoutPromise = new Promise<boolean>((_, reject) => 
              setTimeout(() => reject(new Error('Timeout')), 2000)
            );
            
            const checkPromise = (navigator.xr as any).isSessionSupported(mode);
            
            return await Promise.race([checkPromise, timeoutPromise]);
          } catch (error) {
            logger.debug(`XR mode ${mode} not supported:`, error);
            return false;
          }
        };

        // Check all modes in parallel
        const results = await Promise.allSettled(
          modes.map(async (mode) => {
            const supported = await checkMode(mode);
            return { mode, supported };
          })
        );

        results.forEach((result) => {
          if (result.status === 'fulfilled' && result.value.supported) {
            supportedModes.push(result.value.mode);
          }
        });

        if (mounted) {
          setXRState({
            isChecking: false,
            isSupported: supportedModes.length > 0,
            supportedModes,
            error: null,
          });
        }
      } catch (error) {
        if (mounted) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to check XR support';
          setXRState({
            isChecking: false,
            isSupported: false,
            supportedModes: [],
            error: errorMessage,
          });
        }
      }
    };

    checkXRCapability();

    return () => {
      mounted = false;
    };
  }, []);

  return xrState;
};

/**
 * Safe XR session hook that handles errors gracefully
 */
export const useSafeXRSession = () => {
  const [sessionState, setSessionState] = useState({
    isActive: false,
    session: null as XRSession | null,
    mode: null as XRSessionMode | null,
    error: null as string | null,
  });

  const startSession = useCallback(async (mode: XRSessionMode, options?: XRSessionInit) => {
    try {
      if (!('xr' in navigator)) {
        throw new Error('WebXR not supported');
      }

      const session = await navigator.xr!.requestSession(mode, options);
      
      setSessionState({
        isActive: true,
        session,
        mode,
        error: null,
      });

      // Handle session end
      session.addEventListener('end', () => {
        setSessionState({
          isActive: false,
          session: null,
          mode: null,
          error: null,
        });
      });

      return session;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start XR session';
      logger.error('XR session start failed:', errorMessage);
      
      setSessionState({
        isActive: false,
        session: null,
        mode: null,
        error: errorMessage,
      });
      
      throw error;
    }
  }, []);

  const endSession = useCallback(async () => {
    try {
      if (sessionState.session) {
        await sessionState.session.end();
      }
    } catch (error) {
      logger.error('Failed to end XR session:', error);
    }
  }, [sessionState.session]);

  return {
    ...sessionState,
    startSession,
    endSession,
  };
};

/**
 * Conditional rendering hook for XR components
 */
export const useXRConditionalRender = (dependencies: any[] = []) => {
  const { isInCanvas, isAvailable } = useSafeThree();
  const { isSupported } = useAsyncXRCapability();

  const shouldRender = useMemo(() => {
    return isInCanvas && isAvailable && isSupported;
  }, [isInCanvas, isAvailable, isSupported, ...dependencies]);

  return {
    shouldRender,
    canUseXR: isSupported,
    isInThreeContext: isInCanvas,
  };
};

/**
 * Safe frame hook that works outside Canvas context
 */
export const useSafeFrame = (callback: (state: any, delta: number) => void, deps: any[] = []) => {
  const { isInCanvas } = useSafeThree();
  
  useEffect(() => {
    if (!isInCanvas) {
      // Fallback animation loop when outside Canvas
      let animationId: number;
      let lastTime = performance.now();

      const animate = (currentTime: number) => {
        const delta = (currentTime - lastTime) / 1000;
        lastTime = currentTime;

        try {
          callback({ clock: { elapsedTime: currentTime / 1000 } }, delta);
        } catch (error) {
          logger.error('Animation frame error:', error);
        }

        animationId = requestAnimationFrame(animate);
      };

      animationId = requestAnimationFrame(animate);

      return () => {
        cancelAnimationFrame(animationId);
      };
    }
  }, [isInCanvas, callback, ...deps]);
};

/**
 * Error boundary hook for XR components
 */
export const useXRErrorBoundary = () => {
  const [error, setError] = useState<string | null>(null);
  const [hasError, setHasError] = useState(false);

  const catchError = useCallback((error: Error | string) => {
    const message = typeof error === 'string' ? error : error.message;
    logger.error('XR Error caught:', message);
    setError(message);
    setHasError(true);
  }, []);

  const resetError = useCallback(() => {
    setError(null);
    setHasError(false);
  }, []);

  return {
    hasError,
    error,
    catchError,
    resetError,
  };
};

export default {
  useSafeThree,
  useSafeThreeContext,
  useAsyncXRCapability,
  useSafeXRSession,
  useXRConditionalRender,
  useSafeFrame,
  useXRErrorBoundary,
  useSafeXR,
  withSafeXR,
};