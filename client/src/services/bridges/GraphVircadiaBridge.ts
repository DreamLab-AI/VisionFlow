

import * as BABYLON from '@babylonjs/core';
import { ClientCore } from '../vircadia/VircadiaClientCore';
import { CollaborativeGraphSync } from '../vircadia/CollaborativeGraphSync';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('GraphVircadiaBridge');

export interface GraphNode {
  id: string;
  label: string;
  position: { x: number; y: number; z: number };
  type?: string;
  metadata?: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  type?: string;
}

export interface UserSelectionEvent {
  userId: string;
  username: string;
  nodeIds: string[];
}

export interface AnnotationEvent {
  id: string;
  userId: string;
  username: string;
  nodeId: string;
  text: string;
  position: { x: number; y: number; z: number };
}

export class GraphVircadiaBridge {
  private nodeEntityMap = new Map<string, string>(); 
  private localSelectionCallback?: (nodeIds: string[]) => void;
  private remoteSelectionCallback?: (event: UserSelectionEvent) => void;
  private annotationCallback?: (event: AnnotationEvent) => void;
  private isActive = false;

  constructor(
    private scene: BABYLON.Scene,
    private client: ClientCore,
    private collab: CollaborativeGraphSync
  ) {}

  
  async initialize(): Promise<void> {
    logger.info('Initializing GraphVircadiaBridge...');

    if (!this.client.Utilities.Connection.getConnectionInfo().isConnected) {
      throw new Error('Vircadia client must be connected before initializing bridge');
    }

    await this.collab.initialize();

    
    this.collab.on('user-selection', this.handleRemoteSelection.bind(this));

    
    this.collab.on('annotation-added', this.handleRemoteAnnotation.bind(this));
    this.collab.on('annotation-removed', this.handleAnnotationRemoved.bind(this));

    
    this.collab.on('filter-state-changed', this.handleFilterStateChanged.bind(this));

    this.isActive = true;
    logger.info('GraphVircadiaBridge initialized successfully');
  }

  
  syncGraphToVircadia(nodes: GraphNode[], edges: GraphEdge[]): void {
    if (!this.isActive) return;

    try {
      
      nodes.forEach(node => {
        this.syncNodeToEntity(node);
      });

      
      edges.forEach(edge => {
        this.syncEdgeToEntity(edge);
      });

      logger.debug(`Synced ${nodes.length} nodes and ${edges.length} edges to Vircadia`);
    } catch (error) {
      logger.error('Failed to sync graph to Vircadia:', error);
    }
  }

  
  private syncNodeToEntity(node: GraphNode): void {
    const entityId = `graph-node-${node.id}`;

    
    this.nodeEntityMap.set(node.id, entityId);

    
    
  }

  
  private syncEdgeToEntity(edge: GraphEdge): void {
    const sourceEntityId = this.nodeEntityMap.get(edge.source);
    const targetEntityId = this.nodeEntityMap.get(edge.target);

    if (sourceEntityId && targetEntityId) {
      
      
    }
  }

  
  broadcastLocalSelection(nodeIds: string[]): void {
    if (!this.isActive) return;

    try {
      this.collab.setLocalSelection(nodeIds);
      logger.debug(`Broadcasted selection of ${nodeIds.length} nodes`);
    } catch (error) {
      logger.error('Failed to broadcast selection:', error);
    }
  }

  
  async addAnnotation(
    nodeId: string,
    text: string,
    position: { x: number; y: number; z: number }
  ): Promise<string> {
    if (!this.isActive) {
      throw new Error('Bridge not active');
    }

    try {
      const annotationId = await this.collab.addAnnotation({
        nodeId,
        text,
        position
      });

      logger.info(`Added annotation ${annotationId} to node ${nodeId}`);
      return annotationId;
    } catch (error) {
      logger.error('Failed to add annotation:', error);
      throw error;
    }
  }

  
  async removeAnnotation(annotationId: string): Promise<void> {
    if (!this.isActive) return;

    try {
      await this.collab.removeAnnotation(annotationId);
      logger.info(`Removed annotation ${annotationId}`);
    } catch (error) {
      logger.error('Failed to remove annotation:', error);
    }
  }

  
  broadcastFilterState(filterState: {
    searchQuery?: string;
    categoryFilter?: string[];
    timeRange?: { start: number; end: number };
    customFilters?: Record<string, any>;
  }): void {
    if (!this.isActive) return;

    try {
      this.collab.setLocalFilterState(filterState);
      logger.debug('Broadcasted filter state');
    } catch (error) {
      logger.error('Failed to broadcast filter state:', error);
    }
  }

  
  private handleRemoteSelection(event: {
    agentId: string;
    username: string;
    nodeIds: string[];
  }): void {
    logger.debug(`Remote user ${event.username} selected ${event.nodeIds.length} nodes`);

    if (this.remoteSelectionCallback) {
      this.remoteSelectionCallback({
        userId: event.agentId,
        username: event.username,
        nodeIds: event.nodeIds
      });
    }
  }

  
  private handleRemoteAnnotation(annotation: {
    id: string;
    agentId: string;
    username: string;
    nodeId: string;
    text: string;
    position: { x: number; y: number; z: number };
  }): void {
    logger.info(`Remote annotation added by ${annotation.username} on node ${annotation.nodeId}`);

    if (this.annotationCallback) {
      this.annotationCallback({
        id: annotation.id,
        userId: annotation.agentId,
        username: annotation.username,
        nodeId: annotation.nodeId,
        text: annotation.text,
        position: annotation.position
      });
    }
  }

  
  private handleAnnotationRemoved(annotationId: string): void {
    logger.debug(`Annotation ${annotationId} removed`);
  }

  
  private handleFilterStateChanged(event: {
    agentId: string;
    username: string;
    filterState: any;
  }): void {
    logger.debug(`Remote user ${event.username} changed filter state`);
    
  }

  
  onLocalSelection(callback: (nodeIds: string[]) => void): void {
    this.localSelectionCallback = callback;
  }

  
  onRemoteSelection(callback: (event: UserSelectionEvent) => void): void {
    this.remoteSelectionCallback = callback;
  }

  
  onAnnotation(callback: (event: AnnotationEvent) => void): void {
    this.annotationCallback = callback;
  }

  
  getActiveUsers(): Array<{
    userId: string;
    username: string;
    selectedNodes: string[];
  }> {
    if (!this.isActive) return [];

    return this.collab.getActiveSelections().map(selection => ({
      userId: selection.agentId,
      username: selection.username,
      selectedNodes: selection.nodeIds
    }));
  }

  
  getAnnotations(): AnnotationEvent[] {
    if (!this.isActive) return [];

    return this.collab.getAnnotations().map(ann => ({
      id: ann.id,
      userId: ann.agentId,
      username: ann.username,
      nodeId: ann.nodeId,
      text: ann.text,
      position: ann.position
    }));
  }

  
  dispose(): void {
    this.isActive = false;
    this.nodeEntityMap.clear();
    this.localSelectionCallback = undefined;
    this.remoteSelectionCallback = undefined;
    this.annotationCallback = undefined;
    this.collab.dispose();
    logger.info('GraphVircadiaBridge disposed');
  }
}
