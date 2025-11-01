import { useEffect, useRef, useCallback } from 'react';
import { agentTelemetry } from './AgentTelemetry';


export function useTelemetry(componentName: string) {
  const renderStartTime = useRef<number>();
  const componentMountTime = useRef<number>();

  useEffect(() => {
    componentMountTime.current = performance.now();
    agentTelemetry.logAgentAction('react', 'component', 'mount', { componentName });

    return () => {
      const lifetime = componentMountTime.current ?
        performance.now() - componentMountTime.current : 0;

      agentTelemetry.logAgentAction('react', 'component', 'unmount', {
        componentName,
        lifetime
      });
    };
  }, [componentName]);

  const startRender = useCallback(() => {
    renderStartTime.current = performance.now();
  }, []);

  const endRender = useCallback(() => {
    if (renderStartTime.current) {
      const renderTime = performance.now() - renderStartTime.current;
      agentTelemetry.logRenderCycle(renderTime);
    }
  }, []);

  const logInteraction = useCallback((interactionType: string, metadata?: Record<string, any>) => {
    agentTelemetry.logUserInteraction(interactionType, componentName, metadata);
  }, [componentName]);

  const logError = useCallback((error: Error, context?: string) => {
    agentTelemetry.logAgentAction('react', 'error', 'component_error', {
      componentName,
      context,
      errorMessage: error.message,
      errorStack: error.stack
    });
  }, [componentName]);

  return {
    startRender,
    endRender,
    logInteraction,
    logError
  };
}


export function useThreeJSTelemetry(objectId: string) {
  const logPositionUpdate = useCallback((position: { x: number; y: number; z: number }, metadata?: Record<string, any>) => {
    agentTelemetry.logThreeJSOperation('position_update', objectId, position, undefined, metadata);
  }, [objectId]);

  const logMeshCreate = useCallback((position?: { x: number; y: number; z: number }, metadata?: Record<string, any>) => {
    agentTelemetry.logThreeJSOperation('mesh_create', objectId, position, undefined, metadata);
  }, [objectId]);

  const logAnimationFrame = useCallback((position?: { x: number; y: number; z: number }, rotation?: { x: number; y: number; z: number }) => {
    agentTelemetry.logThreeJSOperation('animation_frame', objectId, position, rotation);
  }, [objectId]);

  const logForceApplied = useCallback((force: { x: number; y: number; z: number }, position?: { x: number; y: number; z: number }) => {
    agentTelemetry.logThreeJSOperation('force_applied', objectId, position, undefined, { force });
  }, [objectId]);

  return {
    logPositionUpdate,
    logMeshCreate,
    logAnimationFrame,
    logForceApplied
  };
}


export function useWebSocketTelemetry() {
  const logMessage = useCallback((messageType: string, direction: 'incoming' | 'outgoing', data?: any) => {
    const size = data ? JSON.stringify(data).length : undefined;
    agentTelemetry.logWebSocketMessage(messageType, direction, data, size);
  }, []);

  return {
    logMessage
  };
}