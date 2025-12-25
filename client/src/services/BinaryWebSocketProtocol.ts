

import { createLogger } from '../utils/loggerConfig';
import type { Vec3 } from '../types/binaryProtocol';

const logger = createLogger('BinaryWebSocketProtocol');

// Protocol versions
export const PROTOCOL_V1 = 1;  
export const PROTOCOL_V2 = 2;  
export const PROTOCOL_VERSION = PROTOCOL_V2;  

// Message types (1 byte header)
export enum MessageType {

  GRAPH_UPDATE = 0x01,


  VOICE_DATA = 0x02,


  POSITION_UPDATE = 0x10,
  AGENT_POSITIONS = 0x11,
  VELOCITY_UPDATE = 0x12,


  AGENT_STATE_FULL = 0x20,
  AGENT_STATE_DELTA = 0x21,
  AGENT_HEALTH = 0x22,


  CONTROL_BITS = 0x30,
  SSSP_DATA = 0x31,
  HANDSHAKE = 0x32,
  HEARTBEAT = 0x33,


  VOICE_CHUNK = 0x40,
  VOICE_START = 0x41,
  VOICE_END = 0x42,

  // Multi-user sync messages (Phase 6)
  SYNC_UPDATE = 0x50,        // Graph operation sync
  ANNOTATION_UPDATE = 0x51,  // Annotation sync
  SELECTION_UPDATE = 0x52,   // Selection sync
  USER_POSITION = 0x53,      // User cursor/avatar position
  VR_PRESENCE = 0x54,        // VR head + hand tracking


  ERROR = 0xFF
}

// Graph type flags for GRAPH_UPDATE messages
export enum GraphTypeFlag {
  KNOWLEDGE_GRAPH = 0x01,    
  ONTOLOGY = 0x02            
}

// Agent state flags (bit field)
export enum AgentStateFlags {
  ACTIVE = 1 << 0,           
  IDLE = 1 << 1,             
  ERROR = 1 << 2,            
  VOICE_ACTIVE = 1 << 3,     
  HIGH_PRIORITY = 1 << 4,    
  POSITION_CHANGED = 1 << 5,  
  METADATA_CHANGED = 1 << 6,  
  RESERVED = 1 << 7          
}

// Control bit flags
export enum ControlFlags {
  PAUSE_UPDATES = 1 << 0,    
  HIGH_FREQUENCY = 1 << 1,   
  LOW_BANDWIDTH = 1 << 2,    
  VOICE_ENABLED = 1 << 3,    
  DEBUG_MODE = 1 << 4,       
  FORCE_FULL_UPDATE = 1 << 5, 
  USER_INTERACTING = 1 << 6,  
  BACKGROUND_MODE = 1 << 7    
}

// Binary data structures


export interface AgentPositionUpdate {
  agentId: number;      
  position: Vec3;       
  timestamp: number;    
  flags: number;        
}


export interface AgentStateData {
  agentId: number;       
  position: Vec3;        
  velocity: Vec3;        
  health: number;        
  cpuUsage: number;      
  memoryUsage: number;   
  workload: number;      
  tokens: number;        
  flags: number;         
}


export interface SSSPData {
  nodeId: number;        
  distance: number;      
  parentId: number;      
  flags: number;         
}


export interface VoiceChunk {
  agentId: number;       
  chunkId: number;       
  format: number;        
  dataLength: number;    
  audioData: ArrayBuffer; 
}


export interface MessageHeader {
  type: MessageType;      
  version: number;        
  payloadLength: number;  
  graphTypeFlag?: GraphTypeFlag; 
}


export interface GraphUpdateHeader extends MessageHeader {
  graphTypeFlag: GraphTypeFlag; 
}

// Constants for binary layout (V1 - Legacy)
export const MESSAGE_HEADER_SIZE = 4;
export const GRAPH_UPDATE_HEADER_SIZE = 5; 
export const AGENT_POSITION_SIZE_V1 = 19;  
export const AGENT_STATE_SIZE_V1 = 47;     
export const SSSP_DATA_SIZE_V1 = 10;       

// Constants for binary layout (V2 - Fixed)
export const AGENT_POSITION_SIZE_V2 = 21;  
export const AGENT_STATE_SIZE_V2 = 49;     
export const SSSP_DATA_SIZE_V2 = 12;       

// Default to V2 sizes
export const AGENT_POSITION_SIZE = AGENT_POSITION_SIZE_V2;
export const AGENT_STATE_SIZE = AGENT_STATE_SIZE_V2;
export const SSSP_DATA_SIZE = SSSP_DATA_SIZE_V2;

export const VOICE_HEADER_SIZE = 7; 


export class BinaryWebSocketProtocol {
  private static instance: BinaryWebSocketProtocol;
  private lastPositionUpdate: number = 0;
  private positionUpdateThrottle: number = 16; 
  private metadataUpdateThrottle: number = 100; 
  private isUserInteracting: boolean = false;
  private pendingPositionUpdates: AgentPositionUpdate[] = [];
  private voiceEnabled: boolean = false;

  private constructor() {}

  public static getInstance(): BinaryWebSocketProtocol {
    if (!BinaryWebSocketProtocol.instance) {
      BinaryWebSocketProtocol.instance = new BinaryWebSocketProtocol();
    }
    return BinaryWebSocketProtocol.instance;
  }

  
  public createMessage(type: MessageType, payload: ArrayBuffer, graphTypeFlag?: GraphTypeFlag): ArrayBuffer {
    const isGraphUpdate = type === MessageType.GRAPH_UPDATE;
    const headerSize = isGraphUpdate ? GRAPH_UPDATE_HEADER_SIZE : MESSAGE_HEADER_SIZE;
    const totalSize = headerSize + payload.byteLength;
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    
    view.setUint8(0, type);
    view.setUint8(1, PROTOCOL_VERSION);
    view.setUint16(2, payload.byteLength, true);

    
    if (isGraphUpdate && graphTypeFlag !== undefined) {
      view.setUint8(4, graphTypeFlag);
    }

    
    new Uint8Array(buffer, headerSize).set(new Uint8Array(payload));

    return buffer;
  }

  
  public parseHeader(buffer: ArrayBuffer): MessageHeader | null {
    if (buffer.byteLength < MESSAGE_HEADER_SIZE) {
      logger.error('Buffer too small for message header');
      return null;
    }

    const view = new DataView(buffer);
    const type = view.getUint8(0) as MessageType;
    const header: MessageHeader = {
      type,
      version: view.getUint8(1),
      payloadLength: view.getUint16(2, true)
    };

    
    if (type === MessageType.GRAPH_UPDATE && buffer.byteLength >= GRAPH_UPDATE_HEADER_SIZE) {
      header.graphTypeFlag = view.getUint8(4) as GraphTypeFlag;
    }

    return header;
  }

  
  public extractPayload(buffer: ArrayBuffer, header?: MessageHeader): ArrayBuffer {
    const isGraphUpdate = header?.type === MessageType.GRAPH_UPDATE;
    const headerSize = isGraphUpdate ? GRAPH_UPDATE_HEADER_SIZE : MESSAGE_HEADER_SIZE;

    if (buffer.byteLength <= headerSize) {
      return new ArrayBuffer(0);
    }
    return buffer.slice(headerSize);
  }

  
  public encodePositionUpdates(updates: AgentPositionUpdate[]): ArrayBuffer | null {
    if (!this.isUserInteracting || updates.length === 0) {
      return null; 
    }

    
    const now = performance.now();
    if (now - this.lastPositionUpdate < this.positionUpdateThrottle) {
      
      this.pendingPositionUpdates.push(...updates);
      return null;
    }

    
    const allUpdates = [...this.pendingPositionUpdates, ...updates];
    this.pendingPositionUpdates = [];
    this.lastPositionUpdate = now;

    
    const payload = new ArrayBuffer(allUpdates.length * AGENT_POSITION_SIZE_V2);
    const view = new DataView(payload);

    allUpdates.forEach((update, index) => {
      const offset = index * AGENT_POSITION_SIZE_V2;

      
      view.setUint32(offset, update.agentId, true);
      view.setFloat32(offset + 4, update.position.x, true);
      view.setFloat32(offset + 8, update.position.y, true);
      view.setFloat32(offset + 12, update.position.z, true);
      view.setUint32(offset + 16, update.timestamp, true);
      view.setUint8(offset + 20, update.flags);
    });

    return this.createMessage(MessageType.POSITION_UPDATE, payload);
  }

  
  public decodePositionUpdates(payload: ArrayBuffer): AgentPositionUpdate[] {
    const updates: AgentPositionUpdate[] = [];
    const view = new DataView(payload);

    
    
    const isV2 = (payload.byteLength % AGENT_POSITION_SIZE_V2) === 0;
    const isV1 = (payload.byteLength % AGENT_POSITION_SIZE_V1) === 0;

    if (isV2) {
      const updateCount = payload.byteLength / AGENT_POSITION_SIZE_V2;
      for (let i = 0; i < updateCount; i++) {
        const offset = i * AGENT_POSITION_SIZE_V2;

        if (offset + AGENT_POSITION_SIZE_V2 > payload.byteLength) {
          logger.warn('Truncated V2 position update data');
          break;
        }

        updates.push({
          agentId: view.getUint32(offset, true),  
          position: {
            x: view.getFloat32(offset + 4, true),
            y: view.getFloat32(offset + 8, true),
            z: view.getFloat32(offset + 12, true)
          },
          timestamp: view.getUint32(offset + 16, true),
          flags: view.getUint8(offset + 20)
        });
      }
    } else if (isV1) {
      const updateCount = payload.byteLength / AGENT_POSITION_SIZE_V1;
      logger.warn('Decoding legacy V1 position updates (u16 IDs)');

      for (let i = 0; i < updateCount; i++) {
        const offset = i * AGENT_POSITION_SIZE_V1;

        if (offset + AGENT_POSITION_SIZE_V1 > payload.byteLength) {
          logger.warn('Truncated V1 position update data');
          break;
        }

        updates.push({
          agentId: view.getUint16(offset, true),  
          position: {
            x: view.getFloat32(offset + 2, true),
            y: view.getFloat32(offset + 6, true),
            z: view.getFloat32(offset + 10, true)
          },
          timestamp: view.getUint32(offset + 14, true),
          flags: view.getUint8(offset + 18)
        });
      }
    } else {
      logger.error(`Invalid position update payload size: ${payload.byteLength}`);
    }

    return updates;
  }

  
  public encodeAgentState(agents: AgentStateData[]): ArrayBuffer {
    const payload = new ArrayBuffer(agents.length * AGENT_STATE_SIZE_V2);
    const view = new DataView(payload);

    agents.forEach((agent, index) => {
      const offset = index * AGENT_STATE_SIZE_V2;

      
      view.setUint32(offset, agent.agentId, true);
      view.setFloat32(offset + 4, agent.position.x, true);
      view.setFloat32(offset + 8, agent.position.y, true);
      view.setFloat32(offset + 12, agent.position.z, true);
      view.setFloat32(offset + 16, agent.velocity.x, true);
      view.setFloat32(offset + 20, agent.velocity.y, true);
      view.setFloat32(offset + 24, agent.velocity.z, true);
      view.setFloat32(offset + 28, agent.health, true);
      view.setFloat32(offset + 32, agent.cpuUsage, true);
      view.setFloat32(offset + 36, agent.memoryUsage, true);
      view.setFloat32(offset + 40, agent.workload, true);
      view.setUint32(offset + 44, agent.tokens, true);
      view.setUint8(offset + 48, agent.flags);
    });

    return this.createMessage(MessageType.AGENT_STATE_FULL, payload);
  }

  
  public decodeAgentState(payload: ArrayBuffer): AgentStateData[] {
    const agents: AgentStateData[] = [];
    const view = new DataView(payload);

    
    const isV2 = (payload.byteLength % AGENT_STATE_SIZE_V2) === 0;
    const isV1 = (payload.byteLength % AGENT_STATE_SIZE_V1) === 0;

    if (isV2) {
      const agentCount = payload.byteLength / AGENT_STATE_SIZE_V2;
      for (let i = 0; i < agentCount; i++) {
        const offset = i * AGENT_STATE_SIZE_V2;

        if (offset + AGENT_STATE_SIZE_V2 > payload.byteLength) {
          logger.warn('Truncated V2 agent state data');
          break;
        }

        agents.push({
          agentId: view.getUint32(offset, true),  
          position: {
            x: view.getFloat32(offset + 4, true),
            y: view.getFloat32(offset + 8, true),
            z: view.getFloat32(offset + 12, true)
          },
          velocity: {
            x: view.getFloat32(offset + 16, true),
            y: view.getFloat32(offset + 20, true),
            z: view.getFloat32(offset + 24, true)
          },
          health: view.getFloat32(offset + 28, true),
          cpuUsage: view.getFloat32(offset + 32, true),
          memoryUsage: view.getFloat32(offset + 36, true),
          workload: view.getFloat32(offset + 40, true),
          tokens: view.getUint32(offset + 44, true),
          flags: view.getUint8(offset + 48)
        });
      }
    } else if (isV1) {
      const agentCount = payload.byteLength / AGENT_STATE_SIZE_V1;
      logger.warn('Decoding legacy V1 agent state (u16 IDs)');

      for (let i = 0; i < agentCount; i++) {
        const offset = i * AGENT_STATE_SIZE_V1;

        if (offset + AGENT_STATE_SIZE_V1 > payload.byteLength) {
          logger.warn('Truncated V1 agent state data');
          break;
        }

        agents.push({
          agentId: view.getUint16(offset, true),  
          position: {
            x: view.getFloat32(offset + 2, true),
            y: view.getFloat32(offset + 6, true),
            z: view.getFloat32(offset + 10, true)
          },
          velocity: {
            x: view.getFloat32(offset + 14, true),
            y: view.getFloat32(offset + 18, true),
            z: view.getFloat32(offset + 22, true)
          },
          health: view.getFloat32(offset + 26, true),
          cpuUsage: view.getFloat32(offset + 30, true),
          memoryUsage: view.getFloat32(offset + 34, true),
          workload: view.getFloat32(offset + 38, true),
          tokens: view.getUint32(offset + 42, true),
          flags: view.getUint8(offset + 46)
        });
      }
    } else {
      logger.error(`Invalid agent state payload size: ${payload.byteLength}`);
    }

    return agents;
  }

  
  public encodeSSSPData(nodes: SSSPData[]): ArrayBuffer {
    const payload = new ArrayBuffer(nodes.length * SSSP_DATA_SIZE);
    const view = new DataView(payload);

    nodes.forEach((node, index) => {
      const offset = index * SSSP_DATA_SIZE;

      view.setUint16(offset, node.nodeId, true);
      view.setFloat32(offset + 2, node.distance, true);
      view.setUint16(offset + 6, node.parentId, true);
      view.setUint16(offset + 8, node.flags, true);
    });

    return this.createMessage(MessageType.SSSP_DATA, payload);
  }

  
  public decodeSSSPData(payload: ArrayBuffer): SSSPData[] {
    const nodes: SSSPData[] = [];
    const view = new DataView(payload);
    const nodeCount = payload.byteLength / SSSP_DATA_SIZE;

    for (let i = 0; i < nodeCount; i++) {
      const offset = i * SSSP_DATA_SIZE;

      if (offset + SSSP_DATA_SIZE > payload.byteLength) {
        logger.warn('Truncated SSSP data');
        break;
      }

      nodes.push({
        nodeId: view.getUint16(offset, true),
        distance: view.getFloat32(offset + 2, true),
        parentId: view.getUint16(offset + 6, true),
        flags: view.getUint16(offset + 8, true)
      });
    }

    return nodes;
  }

  
  public encodeControlBits(flags: ControlFlags): ArrayBuffer {
    const payload = new ArrayBuffer(1);
    const view = new DataView(payload);
    view.setUint8(0, flags);
    return this.createMessage(MessageType.CONTROL_BITS, payload);
  }

  
  public decodeControlBits(payload: ArrayBuffer): ControlFlags {
    if (payload.byteLength < 1) {
      return 0 as ControlFlags;
    }
    const view = new DataView(payload);
    return view.getUint8(0) as ControlFlags;
  }

  
  public encodeVoiceChunk(chunk: VoiceChunk): ArrayBuffer {
    const totalSize = VOICE_HEADER_SIZE + chunk.audioData.byteLength;
    const payload = new ArrayBuffer(totalSize);
    const view = new DataView(payload);

    view.setUint16(0, chunk.agentId, true);
    view.setUint16(2, chunk.chunkId, true);
    view.setUint8(4, chunk.format);
    view.setUint16(5, chunk.dataLength, true);

    
    new Uint8Array(payload, VOICE_HEADER_SIZE).set(new Uint8Array(chunk.audioData));

    return this.createMessage(MessageType.VOICE_CHUNK, payload);
  }

  
  public decodeVoiceChunk(payload: ArrayBuffer): VoiceChunk | null {
    if (payload.byteLength < VOICE_HEADER_SIZE) {
      logger.error('Voice chunk payload too small');
      return null;
    }

    const view = new DataView(payload);
    const dataLength = view.getUint16(5, true);

    if (payload.byteLength < VOICE_HEADER_SIZE + dataLength) {
      logger.error('Voice chunk audio data truncated');
      return null;
    }

    return {
      agentId: view.getUint16(0, true),
      chunkId: view.getUint16(2, true),
      format: view.getUint8(4),
      dataLength: dataLength,
      audioData: payload.slice(VOICE_HEADER_SIZE, VOICE_HEADER_SIZE + dataLength)
    };
  }

  
  public setUserInteracting(interacting: boolean): void {
    this.isUserInteracting = interacting;
    logger.debug(`User interaction state: ${interacting}`);
  }

  
  public configureThrottling(positionMs: number, metadataMs: number): void {
    this.positionUpdateThrottle = positionMs;
    this.metadataUpdateThrottle = metadataMs;
    logger.info(`Throttling configured: position=${positionMs}ms, metadata=${metadataMs}ms`);
  }

  
  public setVoiceEnabled(enabled: boolean): void {
    this.voiceEnabled = enabled;
    logger.info(`Voice communication: ${enabled ? 'enabled' : 'disabled'}`);
  }

  
  public calculateBandwidth(agentCount: number, updateRateHz: number): {
    positionOnly: number;    
    fullState: number;       
    withVoice: number;       
  } {
    const positionBandwidth = agentCount * AGENT_POSITION_SIZE * updateRateHz;
    const stateBandwidth = agentCount * AGENT_STATE_SIZE * updateRateHz;
    const voiceBandwidth = this.voiceEnabled ? agentCount * 8000 : 0; 

    return {
      positionOnly: positionBandwidth + MESSAGE_HEADER_SIZE * updateRateHz,
      fullState: stateBandwidth + MESSAGE_HEADER_SIZE * updateRateHz,
      withVoice: stateBandwidth + voiceBandwidth + MESSAGE_HEADER_SIZE * updateRateHz
    };
  }

  
  public validateMessage(buffer: ArrayBuffer): boolean {
    const header = this.parseHeader(buffer);
    if (!header) return false;

    
    if (header.version !== PROTOCOL_VERSION) {
      logger.warn(`Protocol version mismatch: expected ${PROTOCOL_VERSION}, got ${header.version}`);
      return false;
    }

    
    const expectedSize = MESSAGE_HEADER_SIZE + header.payloadLength;
    if (buffer.byteLength !== expectedSize) {
      logger.warn(`Message size mismatch: expected ${expectedSize}, got ${buffer.byteLength}`);
      return false;
    }

    return true;
  }
}

// Export singleton instance
export const binaryProtocol = BinaryWebSocketProtocol.getInstance();

// Export utility functions for bandwidth analysis
export function estimateDataSize(agentCount: number): {
  perUpdate: number;
  perSecondAt10Hz: number;
  perSecondAt60Hz: number;
  comparison: string;
} {
  const perUpdate = agentCount * AGENT_STATE_SIZE + MESSAGE_HEADER_SIZE;
  const perSecond10Hz = perUpdate * 10;
  const perSecond60Hz = perUpdate * 60;

  
  const jsonEstimate = agentCount * 200; 
  const comparison = perUpdate < jsonEstimate
    ? `${Math.round((1 - perUpdate/jsonEstimate) * 100)}% smaller than JSON`
    : `${Math.round((perUpdate/jsonEstimate - 1) * 100)}% larger than JSON`;

  return {
    perUpdate,
    perSecondAt10Hz: perSecond10Hz,
    perSecondAt60Hz: perSecond60Hz,
    comparison
  };
}