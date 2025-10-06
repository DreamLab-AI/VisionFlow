/**
 * Binary WebSocket Protocol for Agent Data Streaming
 *
 * This protocol is designed for high-frequency, bandwidth-efficient streaming of:
 * - Agent position/velocity updates
 * - Agent metadata (status, health, resources)
 * - Control bits and state flags
 * - SSSP pathfinding data
 * - Voice/audio data chunks
 *
 * Architecture:
 * - Client-to-server: Position updates ONLY during user interactions
 * - Server-to-client: Full agent state streaming with smart throttling
 * - Bi-directional: Control messages and handshaking
 */

import { createLogger } from '../utils/loggerConfig';
import type { Vec3 } from '../types/binaryProtocol';

const logger = createLogger('BinaryWebSocketProtocol');

// Protocol versions
export const PROTOCOL_V1 = 1;  // Legacy 16-bit node IDs (34 bytes per node)
export const PROTOCOL_V2 = 2;  // Full 32-bit node IDs (38 bytes per node)
export const PROTOCOL_VERSION = PROTOCOL_V2;  // Default to V2

// Message types (1 byte header)
export enum MessageType {
  // Position/movement data
  POSITION_UPDATE = 0x01,     // Client -> Server: User moved nodes
  AGENT_POSITIONS = 0x02,     // Server -> Client: Agent position batch
  VELOCITY_UPDATE = 0x03,     // Bi-directional: Velocity changes

  // Agent metadata
  AGENT_STATE_FULL = 0x10,    // Server -> Client: Complete agent state
  AGENT_STATE_DELTA = 0x11,   // Server -> Client: Partial state updates
  AGENT_HEALTH = 0x12,        // Server -> Client: Health/resources only

  // Control and coordination
  CONTROL_BITS = 0x20,        // Bi-directional: Control flags
  SSSP_DATA = 0x21,          // Server -> Client: Pathfinding data
  HANDSHAKE = 0x22,          // Bi-directional: Protocol negotiation
  HEARTBEAT = 0x23,          // Bi-directional: Connection keepalive

  // Voice/audio streaming
  VOICE_CHUNK = 0x30,        // Bi-directional: Audio data chunks
  VOICE_START = 0x31,        // Client -> Server: Start voice transmission
  VOICE_END = 0x32,          // Client -> Server: End voice transmission

  // Error handling
  ERROR = 0xFF               // Bi-directional: Error messages
}

// Agent state flags (bit field)
export enum AgentStateFlags {
  ACTIVE = 1 << 0,           // Agent is actively processing
  IDLE = 1 << 1,             // Agent is idle
  ERROR = 1 << 2,            // Agent has errors
  VOICE_ACTIVE = 1 << 3,     // Agent is transmitting voice
  HIGH_PRIORITY = 1 << 4,    // Agent requires priority updates
  POSITION_CHANGED = 1 << 5,  // Position was updated this frame
  METADATA_CHANGED = 1 << 6,  // Metadata was updated this frame
  RESERVED = 1 << 7          // Reserved for future use
}

// Control bit flags
export enum ControlFlags {
  PAUSE_UPDATES = 1 << 0,    // Client requests pause in updates
  HIGH_FREQUENCY = 1 << 1,   // Client requests high-frequency updates
  LOW_BANDWIDTH = 1 << 2,    // Client is on limited bandwidth
  VOICE_ENABLED = 1 << 3,    // Voice communication enabled
  DEBUG_MODE = 1 << 4,       // Enable debug data in messages
  FORCE_FULL_UPDATE = 1 << 5, // Request complete state refresh
  USER_INTERACTING = 1 << 6,  // User is currently interacting
  BACKGROUND_MODE = 1 << 7    // Client is in background
}

// Binary data structures

/**
 * Agent Position Update (Client -> Server)
 * V1: 19 bytes per agent (u16 ID)
 * V2: 21 bytes per agent (u32 ID)
 */
export interface AgentPositionUpdate {
  agentId: number;      // V1: 2 bytes (uint16), V2: 4 bytes (uint32)
  position: Vec3;       // 12 bytes (3x float32)
  timestamp: number;    // 4 bytes (uint32) - milliseconds since epoch
  flags: number;        // 1 byte (interaction flags)
}

/**
 * Agent State Data (Server -> Client)
 * V1: 47 bytes per agent (full) with u16 ID
 * V2: 49 bytes per agent (full) with u32 ID
 */
export interface AgentStateData {
  agentId: number;       // V1: 2 bytes (uint16), V2: 4 bytes (uint32)
  position: Vec3;        // 12 bytes (3x float32)
  velocity: Vec3;        // 12 bytes (3x float32)
  health: number;        // 4 bytes (float32) - 0.0 to 100.0
  cpuUsage: number;      // 4 bytes (float32) - 0.0 to 100.0
  memoryUsage: number;   // 4 bytes (float32) - 0.0 to 100.0
  workload: number;      // 4 bytes (float32) - 0.0 to 100.0
  tokens: number;        // 4 bytes (uint32)
  flags: number;         // 1 byte (AgentStateFlags)
}

/**
 * SSSP Pathfinding Data
 * Size: 10 bytes per node
 */
export interface SSSPData {
  nodeId: number;        // 2 bytes (uint16)
  distance: number;      // 4 bytes (float32)
  parentId: number;      // 2 bytes (uint16)
  flags: number;         // 2 bytes (pathfinding flags)
}

/**
 * Voice Data Chunk
 * Variable size based on audio format
 */
export interface VoiceChunk {
  agentId: number;       // 2 bytes (uint16)
  chunkId: number;       // 2 bytes (uint16) - sequence number
  format: number;        // 1 byte (audio format)
  dataLength: number;    // 2 bytes (uint16)
  audioData: ArrayBuffer; // Variable length
}

/**
 * Message header for all binary messages
 * Size: 4 bytes
 */
export interface MessageHeader {
  type: MessageType;     // 1 byte
  version: number;       // 1 byte
  payloadLength: number; // 2 bytes (uint16) - excludes header
}

// Constants for binary layout (V1 - Legacy)
export const MESSAGE_HEADER_SIZE = 4;
export const AGENT_POSITION_SIZE_V1 = 19;  // u16 ID
export const AGENT_STATE_SIZE_V1 = 47;     // u16 ID
export const SSSP_DATA_SIZE_V1 = 10;       // u16 ID

// Constants for binary layout (V2 - Fixed)
export const AGENT_POSITION_SIZE_V2 = 21;  // u32 ID (+2 bytes)
export const AGENT_STATE_SIZE_V2 = 49;     // u32 ID (+2 bytes)
export const SSSP_DATA_SIZE_V2 = 12;       // u32 ID (+2 bytes)

// Default to V2 sizes
export const AGENT_POSITION_SIZE = AGENT_POSITION_SIZE_V2;
export const AGENT_STATE_SIZE = AGENT_STATE_SIZE_V2;
export const SSSP_DATA_SIZE = SSSP_DATA_SIZE_V2;

export const VOICE_HEADER_SIZE = 7; // Without audio data

/**
 * Binary WebSocket Protocol Handler
 */
export class BinaryWebSocketProtocol {
  private static instance: BinaryWebSocketProtocol;
  private lastPositionUpdate: number = 0;
  private positionUpdateThrottle: number = 16; // ~60fps max
  private metadataUpdateThrottle: number = 100; // 10fps for metadata
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

  /**
   * Create a binary message with header
   */
  public createMessage(type: MessageType, payload: ArrayBuffer): ArrayBuffer {
    const totalSize = MESSAGE_HEADER_SIZE + payload.byteLength;
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    // Write header
    view.setUint8(0, type);
    view.setUint8(1, PROTOCOL_VERSION);
    view.setUint16(2, payload.byteLength, true);

    // Copy payload
    new Uint8Array(buffer, MESSAGE_HEADER_SIZE).set(new Uint8Array(payload));

    return buffer;
  }

  /**
   * Parse message header from binary data
   */
  public parseHeader(buffer: ArrayBuffer): MessageHeader | null {
    if (buffer.byteLength < MESSAGE_HEADER_SIZE) {
      logger.error('Buffer too small for message header');
      return null;
    }

    const view = new DataView(buffer);
    return {
      type: view.getUint8(0) as MessageType,
      version: view.getUint8(1),
      payloadLength: view.getUint16(2, true)
    };
  }

  /**
   * Extract payload from binary message
   */
  public extractPayload(buffer: ArrayBuffer): ArrayBuffer {
    if (buffer.byteLength <= MESSAGE_HEADER_SIZE) {
      return new ArrayBuffer(0);
    }
    return buffer.slice(MESSAGE_HEADER_SIZE);
  }

  /**
   * Encode agent position updates for client->server transmission
   * Only sends updates during user interactions
   * Uses V2 protocol with u32 IDs
   */
  public encodePositionUpdates(updates: AgentPositionUpdate[]): ArrayBuffer | null {
    if (!this.isUserInteracting || updates.length === 0) {
      return null; // Only send during user interaction
    }

    // Throttle position updates
    const now = performance.now();
    if (now - this.lastPositionUpdate < this.positionUpdateThrottle) {
      // Add to pending updates
      this.pendingPositionUpdates.push(...updates);
      return null;
    }

    // Combine pending and new updates
    const allUpdates = [...this.pendingPositionUpdates, ...updates];
    this.pendingPositionUpdates = [];
    this.lastPositionUpdate = now;

    // Create payload with V2 format (u32 IDs)
    const payload = new ArrayBuffer(allUpdates.length * AGENT_POSITION_SIZE_V2);
    const view = new DataView(payload);

    allUpdates.forEach((update, index) => {
      const offset = index * AGENT_POSITION_SIZE_V2;

      // Write u32 ID (4 bytes) - FIXED for V2
      view.setUint32(offset, update.agentId, true);
      view.setFloat32(offset + 4, update.position.x, true);
      view.setFloat32(offset + 8, update.position.y, true);
      view.setFloat32(offset + 12, update.position.z, true);
      view.setUint32(offset + 16, update.timestamp, true);
      view.setUint8(offset + 20, update.flags);
    });

    return this.createMessage(MessageType.POSITION_UPDATE, payload);
  }

  /**
   * Decode agent position updates from server
   * Supports both V1 (u16) and V2 (u32) protocols
   */
  public decodePositionUpdates(payload: ArrayBuffer): AgentPositionUpdate[] {
    const updates: AgentPositionUpdate[] = [];
    const view = new DataView(payload);

    // Auto-detect protocol version based on payload size
    // Try V2 first (21 bytes), fallback to V1 (19 bytes)
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
          agentId: view.getUint32(offset, true),  // V2: u32
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
          agentId: view.getUint16(offset, true),  // V1: u16 (LEGACY)
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

  /**
   * Encode full agent state data for server->client transmission
   * Uses V2 protocol with u32 IDs
   */
  public encodeAgentState(agents: AgentStateData[]): ArrayBuffer {
    const payload = new ArrayBuffer(agents.length * AGENT_STATE_SIZE_V2);
    const view = new DataView(payload);

    agents.forEach((agent, index) => {
      const offset = index * AGENT_STATE_SIZE_V2;

      // Write u32 ID (4 bytes) - FIXED for V2
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

  /**
   * Decode agent state data from server
   * Supports both V1 (u16) and V2 (u32) protocols
   */
  public decodeAgentState(payload: ArrayBuffer): AgentStateData[] {
    const agents: AgentStateData[] = [];
    const view = new DataView(payload);

    // Auto-detect protocol version based on payload size
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
          agentId: view.getUint32(offset, true),  // V2: u32
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
          agentId: view.getUint16(offset, true),  // V1: u16 (LEGACY)
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

  /**
   * Encode SSSP pathfinding data
   */
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

  /**
   * Decode SSSP pathfinding data
   */
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

  /**
   * Encode control bits message
   */
  public encodeControlBits(flags: ControlFlags): ArrayBuffer {
    const payload = new ArrayBuffer(1);
    const view = new DataView(payload);
    view.setUint8(0, flags);
    return this.createMessage(MessageType.CONTROL_BITS, payload);
  }

  /**
   * Decode control bits message
   */
  public decodeControlBits(payload: ArrayBuffer): ControlFlags {
    if (payload.byteLength < 1) {
      return 0;
    }
    const view = new DataView(payload);
    return view.getUint8(0) as ControlFlags;
  }

  /**
   * Encode voice data chunk
   */
  public encodeVoiceChunk(chunk: VoiceChunk): ArrayBuffer {
    const totalSize = VOICE_HEADER_SIZE + chunk.audioData.byteLength;
    const payload = new ArrayBuffer(totalSize);
    const view = new DataView(payload);

    view.setUint16(0, chunk.agentId, true);
    view.setUint16(2, chunk.chunkId, true);
    view.setUint8(4, chunk.format);
    view.setUint16(5, chunk.dataLength, true);

    // Copy audio data
    new Uint8Array(payload, VOICE_HEADER_SIZE).set(new Uint8Array(chunk.audioData));

    return this.createMessage(MessageType.VOICE_CHUNK, payload);
  }

  /**
   * Decode voice data chunk
   */
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

  /**
   * Set user interaction state (controls position update sending)
   */
  public setUserInteracting(interacting: boolean): void {
    this.isUserInteracting = interacting;
    logger.debug(`User interaction state: ${interacting}`);
  }

  /**
   * Configure update throttling rates
   */
  public configureThrottling(positionMs: number, metadataMs: number): void {
    this.positionUpdateThrottle = positionMs;
    this.metadataUpdateThrottle = metadataMs;
    logger.info(`Throttling configured: position=${positionMs}ms, metadata=${metadataMs}ms`);
  }

  /**
   * Enable/disable voice communication
   */
  public setVoiceEnabled(enabled: boolean): void {
    this.voiceEnabled = enabled;
    logger.info(`Voice communication: ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Calculate bandwidth usage for different update scenarios
   */
  public calculateBandwidth(agentCount: number, updateRateHz: number): {
    positionOnly: number;    // bytes per second
    fullState: number;       // bytes per second
    withVoice: number;       // bytes per second (estimated)
  } {
    const positionBandwidth = agentCount * AGENT_POSITION_SIZE * updateRateHz;
    const stateBandwidth = agentCount * AGENT_STATE_SIZE * updateRateHz;
    const voiceBandwidth = this.voiceEnabled ? agentCount * 8000 : 0; // 8KB/s per agent estimate

    return {
      positionOnly: positionBandwidth + MESSAGE_HEADER_SIZE * updateRateHz,
      fullState: stateBandwidth + MESSAGE_HEADER_SIZE * updateRateHz,
      withVoice: stateBandwidth + voiceBandwidth + MESSAGE_HEADER_SIZE * updateRateHz
    };
  }

  /**
   * Validate message integrity
   */
  public validateMessage(buffer: ArrayBuffer): boolean {
    const header = this.parseHeader(buffer);
    if (!header) return false;

    // Check version compatibility
    if (header.version !== PROTOCOL_VERSION) {
      logger.warn(`Protocol version mismatch: expected ${PROTOCOL_VERSION}, got ${header.version}`);
      return false;
    }

    // Check payload length
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

  // Compare to typical REST API JSON payload (estimated)
  const jsonEstimate = agentCount * 200; // ~200 bytes per agent in JSON
  const comparison = perUpdate < jsonEstimate
    ? `${Math.round((1 - perUpdate/jsonEstimate) * 100)}% smaller than JSON`
    : `${Math.round((perUpdate/jsonEstimate - 1) * 100)}% larger than JSON`;

  return {
    perUpdate,
    perSecondAt10Hz,
    perSecondAt60Hz,
    comparison
  };
}