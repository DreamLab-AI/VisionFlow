

import { ClientCore } from '../vircadia/VircadiaClientCore';
import { EntitySyncManager } from '../vircadia/EntitySyncManager';
import { AvatarManager } from '../vircadia/AvatarManager';
import type { BotsAgent, BotsEdge } from '../../features/bots/types/BotsTypes';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('BotsVircadiaBridge');

export interface BridgeConfig {
  syncPositions: boolean;
  syncMetadata: boolean;
  syncEdges: boolean;
  updateInterval: number;
  enableAvatars: boolean;
}

export interface VircadiaEntity {
  id: string;
  type: string;
  position: { x: number; y: number; z: number };
  rotation?: { x: number; y: number; z: number; w: number };
  scale?: { x: number; y: number; z: number };
  metadata?: Record<string, any>;
}

export class BotsVircadiaBridge {
  private syncInterval: ReturnType<typeof setInterval> | null = null;
  private agentEntityMap = new Map<string, string>(); 
  private lastSyncedAgents = new Map<string, BotsAgent>();
  private isActive = false;

  private defaultConfig: BridgeConfig = {
    syncPositions: true,
    syncMetadata: true,
    syncEdges: true,
    updateInterval: 100, 
    enableAvatars: true
  };

  constructor(
    private client: ClientCore,
    private entitySync: EntitySyncManager,
    private avatars: AvatarManager | null,
    config?: Partial<BridgeConfig>
  ) {
    this.defaultConfig = { ...this.defaultConfig, ...config };
  }

  
  async initialize(): Promise<void> {
    logger.info('Initializing BotsVircadiaBridge...');

    if (!this.client.Utilities.Connection.getConnectionInfo().isConnected) {
      throw new Error('Vircadia client must be connected before initializing bridge');
    }

    // @ts-ignore - EntitySyncManager event methods may not be typed
    this.entitySync.on?.('entity-updated', this.handleVircadiaEntityUpdate.bind(this));
    // @ts-ignore - EntitySyncManager event methods may not be typed
    this.entitySync.on?.('entity-deleted', this.handleVircadiaEntityDeleted.bind(this));

    this.isActive = true;
    logger.info('BotsVircadiaBridge initialized successfully');
  }

  
  syncAgentsToVircadia(agents: BotsAgent[], edges: BotsEdge[]): void {
    if (!this.isActive) return;

    try {
      
      agents.forEach(agent => {
        this.syncAgentToEntity(agent);
      });

      
      this.cleanupStaleEntities(agents);

      
      if (this.defaultConfig.syncEdges) {
        this.syncEdgesToVircadia(edges);
      }

      logger.debug(`Synced ${agents.length} agents and ${edges.length} edges to Vircadia`);
    } catch (error) {
      logger.error('Failed to sync agents to Vircadia:', error);
    }
  }

  
  private syncAgentToEntity(agent: BotsAgent): void {
    
    const lastSynced = this.lastSyncedAgents.get(agent.id);
    if (lastSynced && this.isAgentUnchanged(agent, lastSynced)) {
      return; 
    }

    const entityId = this.agentEntityMap.get(agent.id) || `agent-${agent.id}`;

    
    const position = this.convertAgentPosition(agent.position || { x: 0, y: 0, z: 0 });

    
    const entityData: VircadiaEntity = {
      id: entityId,
      type: 'agent-avatar',
      position,
      scale: { x: 1, y: 1, z: 1 },
      metadata: this.defaultConfig.syncMetadata ? {
        agentName: agent.name,
        agentType: agent.type,
        health: agent.health,
        status: agent.status,
        capabilities: agent.capabilities,
        currentTask: agent.currentTask,
        tokenUsage: (agent as any).tokenUsage,
        isActive: agent.status === 'active',
        color: this.getAgentColor(agent)
      } : undefined
    };

    // @ts-ignore - EntitySyncManager method may not be typed
    this.entitySync.updateEntity?.(entityData);

    
    this.agentEntityMap.set(agent.id, entityId);
    this.lastSyncedAgents.set(agent.id, { ...agent });
  }

  
  private convertAgentPosition(position: { x: number; y: number; z: number }): { x: number; y: number; z: number } {
    
    
    return {
      x: position.x * 10, 
      y: position.z * 10, 
      z: position.y * 10
    };
  }

  
  private getAgentColor(agent: BotsAgent): string {
    const typeColors: Record<string, string> = {
      'researcher': '#4A90E2',
      'coder': '#50E3C2',
      'analyst': '#F5A623',
      'optimizer': '#D0021B',
      'coordinator': '#7ED321'
    };

    return typeColors[agent.type] || '#9013FE';
  }

  
  private isAgentUnchanged(agent: BotsAgent, lastSynced: BotsAgent): boolean {
    return (
      agent.position?.x === lastSynced.position?.x &&
      agent.position?.y === lastSynced.position?.y &&
      agent.position?.z === lastSynced.position?.z &&
      agent.health === lastSynced.health &&
      agent.status === lastSynced.status
    );
  }

  
  private cleanupStaleEntities(currentAgents: BotsAgent[]): void {
    const currentAgentIds = new Set(currentAgents.map(a => a.id));
    const staleAgentIds: string[] = [];

    this.agentEntityMap.forEach((entityId, agentId) => {
      if (!currentAgentIds.has(agentId)) {
        staleAgentIds.push(agentId);
        // @ts-ignore - EntitySyncManager method may not be typed
        this.entitySync.deleteEntity?.(entityId);
      }
    });

    staleAgentIds.forEach(agentId => {
      this.agentEntityMap.delete(agentId);
      this.lastSyncedAgents.delete(agentId);
    });

    if (staleAgentIds.length > 0) {
      logger.debug(`Cleaned up ${staleAgentIds.length} stale agent entities`);
    }
  }

  
  private syncEdgesToVircadia(edges: BotsEdge[]): void {
    
    edges.forEach(edge => {
      const entityId = `edge-${edge.source}-${edge.target}`;
      const sourceEntity = this.agentEntityMap.get(edge.source);
      const targetEntity = this.agentEntityMap.get(edge.target);

      if (sourceEntity && targetEntity) {
        // @ts-ignore - EntitySyncManager method may not be typed
        this.entitySync.updateEntity?.({
          id: entityId,
          type: 'communication-link',
          position: { x: 0, y: 0, z: 0 },
          metadata: {
            source: sourceEntity,
            target: targetEntity,
            type: (edge as any).type || 'default',
            active: true
          }
        });
      }
    });
  }

  
  private handleVircadiaEntityUpdate(entity: VircadiaEntity): void {
    
    if (entity.type !== 'agent-avatar') return;

    
    const agentId = Array.from(this.agentEntityMap.entries())
      .find(([_, entityId]) => entityId === entity.id)?.[0];

    if (!agentId) {
      logger.debug('Received update for unknown agent entity:', entity.id);
      return;
    }

    
    
    logger.debug(`Entity ${entity.id} updated by another user`);
  }

  
  private handleVircadiaEntityDeleted(entityId: string): void {
    
    const agentId = Array.from(this.agentEntityMap.entries())
      .find(([_, eid]) => eid === entityId)?.[0];

    if (agentId) {
      this.agentEntityMap.delete(agentId);
      this.lastSyncedAgents.delete(agentId);
      logger.debug(`Agent entity ${entityId} deleted`);
    }
  }

  
  startAutoSync(
    getAgentsCallback: () => { agents: BotsAgent[]; edges: BotsEdge[] }
  ): void {
    if (this.syncInterval) {
      this.stopAutoSync();
    }

    this.syncInterval = setInterval(() => {
      const { agents, edges } = getAgentsCallback();
      this.syncAgentsToVircadia(agents, edges);
    }, this.defaultConfig.updateInterval);

    logger.info(`Auto-sync started with ${this.defaultConfig.updateInterval}ms interval`);
  }

  
  stopAutoSync(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
      logger.info('Auto-sync stopped');
    }
  }

  
  dispose(): void {
    this.stopAutoSync();
    this.isActive = false;
    this.agentEntityMap.clear();
    this.lastSyncedAgents.clear();
    logger.info('BotsVircadiaBridge disposed');
  }
}
