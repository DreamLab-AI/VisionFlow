

import { unifiedApiClient } from '../../../services/api/UnifiedApiClient';
import { createLogger } from '../../../utils/loggerConfig';
import type { BotsAgent, BotsEdge } from '../types/botsTypes';

const logger = createLogger('ProgrammaticMonitor');

export interface BotsUpdatePayload {
  nodes: BotsAgent[];
  edges: BotsEdge[];
  timestamp?: string;
}


export async function sendBotsUpdate(data: BotsUpdatePayload): Promise<void> {
  try {
    
    const payload = {
      ...data,
      timestamp: data.timestamp || new Date().toISOString()
    };

    
    const response = await unifiedApiClient.post('/bots/update', payload);

    logger.debug('Sent bots update:', {
      nodeCount: data.nodes.length,
      edgeCount: data.edges.length,
      timestamp: payload.timestamp
    });

    return response;
  } catch (error) {
    logger.error('Failed to send bots update:', error);
    throw error;
  }
}


export function generateMockAgent(id: string, type: string, name: string): BotsAgent {
  return {
    id,
    type,
    status: 'active',
    name,
    cpuUsage: Math.random() * 100,
    memoryUsage: Math.random() * 100,
    health: 80 + Math.random() * 20,
    workload: Math.random(),
    createdAt: new Date().toISOString(),
    age: Math.floor(Math.random() * 3600)
  };
}


export function generateMockEdge(source: string, target: string): BotsEdge {
  return {
    id: `edge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    source,
    target,
    dataVolume: Math.random() * 5000,
    messageCount: Math.floor(Math.random() * 50) + 1,
    lastMessageTime: Date.now()
  };
}


export class ProgrammaticMonitor {
  private intervalId: number | null = null;
  private agents: Map<string, BotsAgent> = new Map();

  constructor() {
    
  }

  
  start(intervalMs: number = 2000): void {
    if (this.intervalId) {
      logger.warn('Monitor already running');
      return;
    }

    logger.info(`Starting programmatic monitor with ${intervalMs}ms interval`);

    this.intervalId = window.setInterval(() => {
      this.sendUpdate();
    }, intervalMs);

    
    this.sendUpdate();
  }

  
  stop(): void {
    if (this.intervalId) {
      window.clearInterval(this.intervalId);
      this.intervalId = null;
      logger.info('Programmatic monitor stopped');
    }
  }

  
  private updateAgentMetrics(agent: BotsAgent): void {
    
    agent.cpuUsage = Math.max(0, Math.min(100,
      agent.cpuUsage + (Math.random() - 0.5) * 20));

    
    if (agent.workload > 0.8) {
      agent.health = Math.max(0, agent.health - Math.random() * 2);
    } else {
      agent.health = Math.min(100, agent.health + Math.random());
    }

    
    agent.workload = Math.max(0, Math.min(1.0,
      agent.workload + (Math.random() - 0.5) * 0.2));

    
    agent.age = (agent.age || 0) + 2; 
  }

  
  private generateCommunications(): BotsEdge[] {
    const edges: BotsEdge[] = [];
    const agentIds = Array.from(this.agents.keys());

    if (agentIds.length >= 2) {
      
      const numComms = Math.floor(Math.random() * 3) + 1;

      for (let i = 0; i < numComms; i++) {
        const source = agentIds[Math.floor(Math.random() * agentIds.length)];
        const target = agentIds.find(id => id !== source) || agentIds[0];

        if (source !== target) {
          edges.push(generateMockEdge(source, target));
        }
      }
    }

    return edges;
  }

  
  private async sendUpdate(): Promise<void> {
    
    this.agents.forEach(agent => this.updateAgentMetrics(agent));

    
    const edges = this.generateCommunications();

    
    const payload: BotsUpdatePayload = {
      nodes: Array.from(this.agents.values()),
      edges
    };

    try {
      await sendBotsUpdate(payload);
    } catch (error) {
      logger.error('Failed to send update:', error);
    }
  }

  
  addAgent(agent: BotsAgent): void {
    this.agents.set(agent.id, agent);
  }

  
  removeAgent(agentId: string): void {
    this.agents.delete(agentId);
  }

  
  getAgents(): BotsAgent[] {
    return Array.from(this.agents.values());
  }
}

// Export singleton instance for easy use
export const programmaticMonitor = new ProgrammaticMonitor();