import React, { useEffect, useCallback } from 'react';
import { webSocketService } from '../services/WebSocketService';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('BatchingExample');

/**
 * Example component demonstrating how to use the batching infrastructure
 * for sending position updates efficiently
 */
export const BatchingExample: React.FC = () => {
  // Example: Send position updates from mouse movement
  const handleMouseMove = useCallback((event: MouseEvent) => {
    // Example node updates - in real usage, these would come from your graph interaction
    const updates = [
      {
        nodeId: 1,
        position: { x: event.clientX, y: event.clientY, z: 0 },
        velocity: { x: 0, y: 0, z: 0 }
      },
      {
        nodeId: 2,
        position: { x: event.clientX + 50, y: event.clientY + 50, z: 0 },
        velocity: { x: 0, y: 0, z: 0 }
      }
    ];

    // Send updates - they will be automatically batched
    webSocketService.sendNodePositionUpdates(updates);
  }, []);

  // Example: Batch multiple updates programmatically
  const sendBatchedUpdates = useCallback(() => {
    const updates = [];
    
    // Generate 100 node updates
    for (let i = 0; i < 100; i++) {
      updates.push({
        nodeId: i,
        position: {
          x: Math.random() * 1000 - 500,
          y: Math.random() * 1000 - 500,
          z: Math.random() * 1000 - 500
        },
        velocity: {
          x: Math.random() * 10 - 5,
          y: Math.random() * 10 - 5,
          z: Math.random() * 10 - 5
        }
      });
    }

    // Send all updates - they will be batched automatically
    webSocketService.sendNodePositionUpdates(updates);
    
    logger.info(`Sent ${updates.length} updates for batching`);
  }, []);

  // Example: Force flush pending updates
  const flushUpdates = useCallback(async () => {
    try {
      await webSocketService.flushPositionUpdates();
      logger.info('Flushed all pending position updates');
    } catch (error) {
      logger.error('Failed to flush updates:', error);
    }
  }, []);

  // Example: Monitor queue metrics
  useEffect(() => {
    const interval = setInterval(() => {
      const metrics = webSocketService.getPositionQueueMetrics();
      if (metrics && metrics.currentQueueSize > 0) {
        logger.debug('Queue metrics:', metrics);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Example: Flush updates before navigation
  useEffect(() => {
    const handleBeforeUnload = async (event: BeforeUnloadEvent) => {
      // Flush any pending updates before leaving the page
      await webSocketService.flushPositionUpdates();
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, []);

  return (
    <div>
      <h2>Batching Example</h2>
      <p>This example demonstrates how to use the batching infrastructure.</p>
      
      <div style={{ marginTop: '20px' }}>
        <button onClick={sendBatchedUpdates}>
          Send 100 Batched Updates
        </button>
        
        <button onClick={flushUpdates} style={{ marginLeft: '10px' }}>
          Force Flush Queue
        </button>
      </div>
      
      <div 
        style={{ 
          marginTop: '20px',
          padding: '20px',
          border: '1px solid #ccc',
          minHeight: '200px'
        }}
        onMouseMove={handleMouseMove as any}
      >
        Move your mouse here to send position updates (batched)
      </div>
    </div>
  );
};

/**
 * Example: Using batching in a graph interaction handler
 */
export function useGraphInteractionBatching() {
  // Handle node drag with batching
  const handleNodeDrag = useCallback((nodeId: number, position: { x: number, y: number, z: number }) => {
    // Single update - will be batched with others
    webSocketService.sendNodePositionUpdates([{
      nodeId,
      position,
      velocity: { x: 0, y: 0, z: 0 } // Static during drag
    }]);
  }, []);

  // Handle multiple nodes selection and move
  const handleMultiNodeMove = useCallback((nodeUpdates: Array<{ nodeId: number, position: { x: number, y: number, z: number } }>) => {
    // Convert to the expected format
    const updates = nodeUpdates.map(update => ({
      ...update,
      velocity: { x: 0, y: 0, z: 0 }
    }));

    // Send all updates - automatically batched
    webSocketService.sendNodePositionUpdates(updates);
  }, []);

  // Physics simulation updates
  const handlePhysicsUpdate = useCallback((physicsData: Array<{ nodeId: number, position: { x: number, y: number, z: number }, velocity: { x: number, y: number, z: number } }>) => {
    // Physics updates include velocity - send with higher frequency
    webSocketService.sendNodePositionUpdates(physicsData);
  }, []);

  return {
    handleNodeDrag,
    handleMultiNodeMove,
    handlePhysicsUpdate
  };
}