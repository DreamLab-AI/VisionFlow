

import { createLogger } from '../utils/loggerConfig';
import type { Vec3 } from '../types/binaryProtocol';

const logger = createLogger('BinaryWebSocketProtocol');

// Protocol versions
export const PROTOCOL_V2 = 2;  // Legacy: uint16 payload length, uint16 SSSP IDs
export const PROTOCOL_V3 = 3;  // Analytics extension (48 bytes/node)
export const PROTOCOL_V4 = 4;  // CURRENT: uint32 payload length header (6 bytes), uint32 SSSP IDs (14 bytes/node)
export const PROTOCOL_VERSION = PROTOCOL_V4;  // Default to V4
export const SUPPORTED_PROTOCOLS = [PROTOCOL_V2, PROTOCOL_V3, PROTOCOL_V4];

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
  AGENT_ACTION = 0x23,        // Agent-to-data action for ephemeral connections


  CONTROL_BITS = 0x30,
  SSSP_DATA = 0x31,
  HANDSHAKE = 0x32,
  HEARTBEAT = 0x33,


  VOICE_CHUNK = 0x40,
  VOICE_START = 0x41,
  VOICE_END = 0x42,

  // Backpressure flow control (Phase 7)
  BROADCAST_ACK = 0x34,      // Client acknowledgement of position broadcast

  // Multi-user sync messages (Phase 6)
  SYNC_UPDATE = 0x50,        // Graph operation sync
  ANNOTATION_UPDATE = 0x51,  // Annotation sync
  SELECTION_UPDATE = 0x52,   // Selection sync
  USER_POSITION = 0x53,      // User cursor/avatar position
  VR_PRESENCE = 0x54,        // VR head + hand tracking


  ERROR = 0xFF
}

// Graph type flags for GRAPH_UPDATE messages
// Values must match server: src/utils/binary_protocol.rs GraphType enum
export enum GraphTypeFlag {
  KNOWLEDGE_GRAPH = 0x00,
  ONTOLOGY = 0x01
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

// Broadcast ACK data for backpressure flow control
export interface BroadcastAckData {
  sequenceId: number;     // 8 bytes - correlates with server broadcast sequence
  nodesReceived: number;  // 4 bytes - number of nodes client processed
  timestamp: number;      // 8 bytes - client receive timestamp (ms)
}

// Agent action types for ephemeral connection visualization
export enum AgentActionType {
  Query = 0,      // Agent querying data node (blue)
  Update = 1,     // Agent updating data node (yellow)
  Create = 2,     // Agent creating data node (green)
  Delete = 3,     // Agent deleting data node (red)
  Link = 4,       // Agent linking nodes (purple)
  Transform = 5,  // Agent transforming data (cyan)
}

// Agent action event for visualization
export interface AgentActionEvent {
  sourceAgentId: number;    // 4 bytes - ID of the acting agent
  targetNodeId: number;     // 4 bytes - ID of the target data node
  actionType: AgentActionType; // 1 byte
  timestamp: number;        // 4 bytes - Event timestamp (ms)
  durationMs: number;       // 2 bytes - Animation duration hint
  payload?: Uint8Array;     // Variable - Optional metadata
}

// Color mapping for action types (used by visualization layer)
export const AGENT_ACTION_COLORS: Record<AgentActionType, string> = {
  [AgentActionType.Query]: '#3b82f6',     // Blue
  [AgentActionType.Update]: '#eab308',    // Yellow
  [AgentActionType.Create]: '#22c55e',    // Green
  [AgentActionType.Delete]: '#ef4444',    // Red
  [AgentActionType.Link]: '#a855f7',      // Purple
  [AgentActionType.Transform]: '#06b6d4', // Cyan
};

// Wire format size for agent action header
export const AGENT_ACTION_HEADER_SIZE = 15;


export interface GraphUpdateHeader extends MessageHeader {
  graphTypeFlag: GraphTypeFlag; 
}

// Constants for binary layout
// V4 header: [1-byte type][1-byte version][4-byte payloadLength] = 6 bytes
export const MESSAGE_HEADER_SIZE = 6;
export const GRAPH_UPDATE_HEADER_SIZE = 7;  // MESSAGE_HEADER_SIZE + 1-byte graphTypeFlag
export const AGENT_POSITION_SIZE_V2 = 21;  // 4 (u32 id) + 12 (pos) + 4 (timestamp) + 1 (flags)
export const AGENT_STATE_SIZE_V2 = 49;     // Full agent state with u32 IDs
// V4 SSSP layout: 4 (u32 nodeId) + 4 (f32 distance) + 4 (u32 parentId) + 2 (u16 flags) = 14 bytes
export const SSSP_DATA_SIZE_V2 = 14;       // SSSP with u32 IDs

// Canonical sizes
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
  private static readonly MAX_PENDING_UPDATES = 1000;
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

    // V4 header: [1-byte type][1-byte version][4-byte payloadLength (uint32, LE)]
    view.setUint8(0, type);
    view.setUint8(1, PROTOCOL_VERSION);
    view.setUint32(2, payload.byteLength, true);

    if (isGraphUpdate && graphTypeFlag !== undefined) {
      view.setUint8(6, graphTypeFlag);
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
      payloadLength: view.getUint32(2, true)
    };

    if (type === MessageType.GRAPH_UPDATE && buffer.byteLength >= GRAPH_UPDATE_HEADER_SIZE) {
      header.graphTypeFlag = view.getUint8(6) as GraphTypeFlag;
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
      // Bound pending updates to prevent unbounded memory growth
      if (this.pendingPositionUpdates.length > BinaryWebSocketProtocol.MAX_PENDING_UPDATES) {
        this.pendingPositionUpdates.splice(0, this.pendingPositionUpdates.length - BinaryWebSocketProtocol.MAX_PENDING_UPDATES);
      }
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

    if (payload.byteLength < AGENT_POSITION_SIZE_V2) {
      if (payload.byteLength === 0) {
        return updates;
      }
      logger.error(`Position update payload too small: ${payload.byteLength}`);
      return updates;
    }

    // Version is in the message header (validated by parseHeader/validateMessage),
    // not embedded in the payload. Decode payload directly as V2+ data.
    const view = new DataView(payload);

    if ((payload.byteLength % AGENT_POSITION_SIZE_V2) !== 0) {
      logger.error(`Invalid position update payload size: ${payload.byteLength}`);
      return updates;
    }

    const updateCount = payload.byteLength / AGENT_POSITION_SIZE_V2;
    for (let i = 0; i < updateCount; i++) {
      const offset = i * AGENT_POSITION_SIZE_V2;

      if (offset + AGENT_POSITION_SIZE_V2 > payload.byteLength) {
        logger.warn('Truncated position update data');
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

    if (payload.byteLength < AGENT_STATE_SIZE_V2) {
      if (payload.byteLength === 0) {
        return agents;
      }
      logger.error(`Agent state payload too small: ${payload.byteLength}`);
      return agents;
    }

    // Version is in the message header (validated by parseHeader/validateMessage),
    // not embedded in the payload. Decode payload directly as V2+ data.
    const view = new DataView(payload);

    if ((payload.byteLength % AGENT_STATE_SIZE_V2) !== 0) {
      logger.error(`Invalid agent state payload size: ${payload.byteLength}`);
      return agents;
    }

    const agentCount = payload.byteLength / AGENT_STATE_SIZE_V2;
    for (let i = 0; i < agentCount; i++) {
      const offset = i * AGENT_STATE_SIZE_V2;

      if (offset + AGENT_STATE_SIZE_V2 > payload.byteLength) {
        logger.warn('Truncated agent state data');
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

    return agents;
  }

  
  public encodeSSSPData(nodes: SSSPData[]): ArrayBuffer {
    const payload = new ArrayBuffer(nodes.length * SSSP_DATA_SIZE);
    const view = new DataView(payload);

    nodes.forEach((node, index) => {
      const offset = index * SSSP_DATA_SIZE;

      // V4 layout: [u32 nodeId][f32 distance][u32 parentId][u16 flags] = 14 bytes
      view.setUint32(offset, node.nodeId, true);
      view.setFloat32(offset + 4, node.distance, true);
      view.setUint32(offset + 8, node.parentId, true);
      view.setUint16(offset + 12, node.flags, true);
    });

    return this.createMessage(MessageType.SSSP_DATA, payload);
  }

  
  public decodeSSSPData(payload: ArrayBuffer): SSSPData[] {
    const nodes: SSSPData[] = [];
    const view = new DataView(payload);
    const nodeCount = Math.floor(payload.byteLength / SSSP_DATA_SIZE);

    for (let i = 0; i < nodeCount; i++) {
      const offset = i * SSSP_DATA_SIZE;

      if (offset + SSSP_DATA_SIZE > payload.byteLength) {
        logger.warn('Truncated SSSP data');
        break;
      }

      // V4 layout: [u32 nodeId][f32 distance][u32 parentId][u16 flags] = 14 bytes
      nodes.push({
        nodeId: view.getUint32(offset, true),
        distance: view.getFloat32(offset + 4, true),
        parentId: view.getUint32(offset + 8, true),
        flags: view.getUint16(offset + 12, true)
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

  /**
   * Create a broadcast acknowledgement message for backpressure flow control.
   * This should be sent after client processes a position update broadcast.
   *
   * @param sequenceId - Server broadcast sequence number (from position update header)
   * @param nodesReceived - Number of nodes successfully processed
   * @returns Binary message ready to send over WebSocket
   */
  public createBroadcastAck(sequenceId: number, nodesReceived: number): ArrayBuffer {
    // Payload: 8 bytes sequenceId + 4 bytes nodesReceived + 8 bytes timestamp = 20 bytes
    const payload = new ArrayBuffer(20);
    const view = new DataView(payload);

    // Write sequenceId as BigInt64 (8 bytes, little-endian)
    const lowBits = sequenceId >>> 0;
    const highBits = Math.floor(sequenceId / 0x100000000) >>> 0;
    view.setUint32(0, lowBits, true);
    view.setUint32(4, highBits, true);

    // Write nodesReceived (4 bytes, little-endian)
    view.setUint32(8, nodesReceived, true);

    // Write timestamp (8 bytes, little-endian)
    const timestamp = Date.now();
    const tsLowBits = timestamp >>> 0;
    const tsHighBits = Math.floor(timestamp / 0x100000000) >>> 0;
    view.setUint32(12, tsLowBits, true);
    view.setUint32(16, tsHighBits, true);

    return this.createMessage(MessageType.BROADCAST_ACK, payload);
  }

  /**
   * Decode a broadcast acknowledgement from server (for server-sent acks if needed)
   */
  public decodeBroadcastAck(payload: ArrayBuffer): BroadcastAckData | null {
    if (payload.byteLength < 20) {
      logger.error('Broadcast ACK payload too small');
      return null;
    }

    const view = new DataView(payload);

    // Read sequenceId (8 bytes, little-endian)
    const lowBits = view.getUint32(0, true);
    const highBits = view.getUint32(4, true);
    const sequenceId = lowBits + highBits * 0x100000000;

    // Read nodesReceived (4 bytes)
    const nodesReceived = view.getUint32(8, true);

    // Read timestamp (8 bytes)
    const tsLowBits = view.getUint32(12, true);
    const tsHighBits = view.getUint32(16, true);
    const timestamp = tsLowBits + tsHighBits * 0x100000000;

    return { sequenceId, nodesReceived, timestamp };
  }

  /**
   * Decode an agent action event from binary payload.
   * Used to render ephemeral connections between agent and data nodes.
   *
   * @param payload - Binary payload (excluding message type byte)
   * @returns Decoded agent action event
   */
  public decodeAgentAction(payload: ArrayBuffer): AgentActionEvent | null {
    if (payload.byteLength < AGENT_ACTION_HEADER_SIZE) {
      logger.error('Agent action payload too small');
      return null;
    }

    const view = new DataView(payload);

    const sourceAgentId = view.getUint32(0, true);
    const targetNodeId = view.getUint32(4, true);
    const actionType = view.getUint8(8) as AgentActionType;
    const timestamp = view.getUint32(9, true);
    const durationMs = view.getUint16(13, true);

    const event: AgentActionEvent = {
      sourceAgentId,
      targetNodeId,
      actionType,
      timestamp,
      durationMs,
    };

    // Extract optional payload
    if (payload.byteLength > AGENT_ACTION_HEADER_SIZE) {
      event.payload = new Uint8Array(payload.slice(AGENT_ACTION_HEADER_SIZE));
    }

    return event;
  }

  /**
   * Decode a batch of agent action events.
   * Wire format: [count: u16][event1_len: u16][event1_data]...
   *
   * @param payload - Binary payload (excluding message type byte)
   * @returns Array of decoded agent action events
   */
  public decodeAgentActions(payload: ArrayBuffer): AgentActionEvent[] {
    const events: AgentActionEvent[] = [];

    if (payload.byteLength < 2) {
      logger.error('Agent actions batch payload too small');
      return events;
    }

    const view = new DataView(payload);
    const eventCount = view.getUint16(0, true);
    let offset = 2;

    for (let i = 0; i < eventCount; i++) {
      if (offset + 2 > payload.byteLength) {
        logger.warn('Truncated event length in agent actions batch');
        break;
      }

      const eventLen = view.getUint16(offset, true);
      offset += 2;

      if (offset + eventLen > payload.byteLength) {
        logger.warn('Truncated event data in agent actions batch');
        break;
      }

      const eventPayload = payload.slice(offset, offset + eventLen);
      const event = this.decodeAgentAction(eventPayload);
      if (event) {
        events.push(event);
      }
      offset += eventLen;
    }

    return events;
  }

  /**
   * Encode an agent action event for sending to server (if needed).
   * Primarily used for testing or client-initiated actions.
   *
   * @param event - Agent action event to encode
   * @returns Binary message ready to send
   */
  public encodeAgentAction(event: AgentActionEvent): ArrayBuffer {
    const payloadLen = event.payload?.length ?? 0;
    const payload = new ArrayBuffer(AGENT_ACTION_HEADER_SIZE + payloadLen);
    const view = new DataView(payload);

    view.setUint32(0, event.sourceAgentId, true);
    view.setUint32(4, event.targetNodeId, true);
    view.setUint8(8, event.actionType);
    view.setUint32(9, event.timestamp, true);
    view.setUint16(13, event.durationMs, true);

    // Copy optional payload
    if (event.payload && event.payload.length > 0) {
      new Uint8Array(payload, AGENT_ACTION_HEADER_SIZE).set(event.payload);
    }

    return this.createMessage(MessageType.AGENT_ACTION, payload);
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

    
    if (!SUPPORTED_PROTOCOLS.includes(header.version)) {
      logger.warn(`Unsupported protocol version: ${header.version}. Supported: ${SUPPORTED_PROTOCOLS.join(', ')}`);
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