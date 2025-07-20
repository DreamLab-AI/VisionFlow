/**
 * Graph Synchronization System
 * Provides camera sync, selection sync, zoom sync, and pan sync between dual graphs
 */

import { Camera, Vector3 } from 'three';
import { createLogger } from '../../../utils/logger';

const logger = createLogger('GraphSynchronization');

export interface SyncState {
  camera: {
    position: Vector3;
    target: Vector3;
    zoom: number;
  };
  selection: {
    selectedNodes: Set<string>;
    hoveredNode: string | null;
  };
  interaction: {
    isPanning: boolean;
    isZooming: boolean;
    lastUpdate: number;
  };
}

export interface SyncOptions {
  enableCameraSync: boolean;
  enableSelectionSync: boolean;
  enableZoomSync: boolean;
  enablePanSync: boolean;
  smoothTransitions: boolean;
  transitionDuration: number;
}

export class GraphSynchronization {
  private static instance: GraphSynchronization;
  private syncState: SyncState;
  private syncOptions: SyncOptions;
  private listeners: Map<string, Set<(state: SyncState) => void>> = new Map();
  private animationFrameId: number | null = null;

  private constructor() {
    this.syncState = {
      camera: {
        position: new Vector3(0, 0, 10),
        target: new Vector3(0, 0, 0),
        zoom: 1
      },
      selection: {
        selectedNodes: new Set(),
        hoveredNode: null
      },
      interaction: {
        isPanning: false,
        isZooming: false,
        lastUpdate: Date.now()
      }
    };

    this.syncOptions = {
      enableCameraSync: true,
      enableSelectionSync: true,
      enableZoomSync: true,
      enablePanSync: true,
      smoothTransitions: true,
      transitionDuration: 300
    };
  }

  public static getInstance(): GraphSynchronization {
    if (!GraphSynchronization.instance) {
      GraphSynchronization.instance = new GraphSynchronization();
    }
    return GraphSynchronization.instance;
  }

  /**
   * Update sync options
   */
  public updateSyncOptions(options: Partial<SyncOptions>): void {
    this.syncOptions = { ...this.syncOptions, ...options };
    logger.info('Sync options updated:', this.syncOptions);
  }

  /**
   * Get current sync options
   */
  public getSyncOptions(): SyncOptions {
    return { ...this.syncOptions };
  }

  /**
   * Sync camera state between graphs
   */
  public syncCamera(graphId: string, camera: Camera, target?: Vector3): void {
    if (!this.syncOptions.enableCameraSync) return;

    const newState = {
      ...this.syncState,
      camera: {
        position: camera.position.clone(),
        target: target ? target.clone() : this.syncState.camera.target,
        zoom: this.syncOptions.enableZoomSync ? camera.zoom : this.syncState.camera.zoom
      },
      interaction: {
        ...this.syncState.interaction,
        lastUpdate: Date.now()
      }
    };

    this.updateState(newState);
    this.notifyOtherGraphs(graphId, 'camera');
  }

  /**
   * Sync selection state between graphs
   */
  public syncSelection(graphId: string, selectedNodes: Set<string>, hoveredNode?: string | null): void {
    if (!this.syncOptions.enableSelectionSync) return;

    const newState = {
      ...this.syncState,
      selection: {
        selectedNodes: new Set(selectedNodes),
        hoveredNode: hoveredNode !== undefined ? hoveredNode : this.syncState.selection.hoveredNode
      },
      interaction: {
        ...this.syncState.interaction,
        lastUpdate: Date.now()
      }
    };

    this.updateState(newState);
    this.notifyOtherGraphs(graphId, 'selection');
  }

  /**
   * Sync pan state between graphs
   */
  public syncPan(graphId: string, delta: Vector3): void {
    if (!this.syncOptions.enablePanSync) return;

    const newState = {
      ...this.syncState,
      camera: {
        ...this.syncState.camera,
        position: this.syncState.camera.position.clone().add(delta),
        target: this.syncState.camera.target.clone().add(delta)
      },
      interaction: {
        ...this.syncState.interaction,
        isPanning: true,
        lastUpdate: Date.now()
      }
    };

    this.updateState(newState);
    this.notifyOtherGraphs(graphId, 'pan');
  }

  /**
   * Sync zoom state between graphs
   */
  public syncZoom(graphId: string, zoomFactor: number): void {
    if (!this.syncOptions.enableZoomSync) return;

    const newState = {
      ...this.syncState,
      camera: {
        ...this.syncState.camera,
        zoom: this.syncState.camera.zoom * zoomFactor
      },
      interaction: {
        ...this.syncState.interaction,
        isZooming: true,
        lastUpdate: Date.now()
      }
    };

    this.updateState(newState);
    this.notifyOtherGraphs(graphId, 'zoom');
  }

  /**
   * Subscribe to sync updates for a specific graph
   */
  public subscribe(graphId: string, callback: (state: SyncState) => void): () => void {
    if (!this.listeners.has(graphId)) {
      this.listeners.set(graphId, new Set());
    }
    
    this.listeners.get(graphId)!.add(callback);

    // Return unsubscribe function
    return () => {
      const graphListeners = this.listeners.get(graphId);
      if (graphListeners) {
        graphListeners.delete(callback);
        if (graphListeners.size === 0) {
          this.listeners.delete(graphId);
        }
      }
    };
  }

  /**
   * Get current sync state
   */
  public getState(): SyncState {
    return {
      camera: {
        position: this.syncState.camera.position.clone(),
        target: this.syncState.camera.target.clone(),
        zoom: this.syncState.camera.zoom
      },
      selection: {
        selectedNodes: new Set(this.syncState.selection.selectedNodes),
        hoveredNode: this.syncState.selection.hoveredNode
      },
      interaction: { ...this.syncState.interaction }
    };
  }

  /**
   * Update internal state
   */
  private updateState(newState: SyncState): void {
    this.syncState = newState;
  }

  /**
   * Notify all graphs except the sender
   */
  private notifyOtherGraphs(senderGraphId: string, syncType: string): void {
    this.listeners.forEach((callbacks, graphId) => {
      if (graphId !== senderGraphId) {
        callbacks.forEach(callback => {
          try {
            if (this.syncOptions.smoothTransitions) {
              this.smoothTransition(callback);
            } else {
              callback(this.getState());
            }
          } catch (error) {
            logger.error(`Error notifying graph ${graphId}:`, error);
          }
        });
      }
    });
  }

  /**
   * Apply smooth transitions for sync updates
   */
  private smoothTransition(callback: (state: SyncState) => void): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }

    const startTime = Date.now();
    const duration = this.syncOptions.transitionDuration;

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function for smooth transitions
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      
      callback(this.getState());

      if (progress < 1) {
        this.animationFrameId = requestAnimationFrame(animate);
      } else {
        this.animationFrameId = null;
      }
    };

    this.animationFrameId = requestAnimationFrame(animate);
  }

  /**
   * Reset interaction state (call after user interaction ends)
   */
  public resetInteractionState(): void {
    this.syncState.interaction.isPanning = false;
    this.syncState.interaction.isZooming = false;
  }

  /**
   * Cleanup resources
   */
  public dispose(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.listeners.clear();
    logger.info('Graph synchronization disposed');
  }
}

// Export singleton instance
export const graphSynchronization = GraphSynchronization.getInstance();