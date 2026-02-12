import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('binaryProtocol');


/**
 * Binary Protocol for WebSocket node data communication
 * Aligns with server-side src/utils/binary_protocol.rs
 *
 * Protocol Versions:
 * - V2: 36 bytes per node (basic pathfinding)
 * - V3: 48 bytes per node (adds cluster_id, anomaly_score, community_id)
 * - V4: Delta encoding (16 bytes per changed node) [client decoding pending]
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface BinaryNodeData {
  nodeId: number;
  position: Vec3;
  velocity: Vec3;
  ssspDistance: number;
  ssspParent: number;
  // V3 analytics fields (optional for backwards compatibility)
  clusterId?: number;
  anomalyScore?: number;
  communityId?: number;
}

// Protocol version constants (must match server)
export const PROTOCOL_V2 = 2;
export const PROTOCOL_V3 = 3;
export const PROTOCOL_V4 = 4;

// V2 wire format: 36 bytes per node
export const BINARY_NODE_SIZE_V2 = 36;
// V3 wire format: 48 bytes per node (V2 + 12 bytes analytics)
export const BINARY_NODE_SIZE_V3 = 48;
// Default to V3 (current server default)
export const BINARY_NODE_SIZE = BINARY_NODE_SIZE_V3;

// Field offsets (same for V2 and V3)
export const BINARY_NODE_ID_OFFSET = 0;
export const BINARY_POSITION_OFFSET = 4;
export const BINARY_VELOCITY_OFFSET = 16;
export const BINARY_SSSP_DISTANCE_OFFSET = 28;
export const BINARY_SSSP_PARENT_OFFSET = 32;
// V3 analytics offsets
export const BINARY_CLUSTER_ID_OFFSET = 36;
export const BINARY_ANOMALY_SCORE_OFFSET = 40;
export const BINARY_COMMUNITY_ID_OFFSET = 44;

// Node type flag constants (Protocol V2/V3 - must match server)
export const AGENT_NODE_FLAG = 0x80000000;
export const KNOWLEDGE_NODE_FLAG = 0x40000000;
export const NODE_ID_MASK = 0x3FFFFFFF;

// Ontology node type flags (bits 26-28)
export const ONTOLOGY_TYPE_MASK = 0x1C000000;
export const ONTOLOGY_CLASS_FLAG = 0x04000000;
export const ONTOLOGY_INDIVIDUAL_FLAG = 0x08000000;
export const ONTOLOGY_PROPERTY_FLAG = 0x10000000;

export enum NodeType {
  Knowledge = 'knowledge',
  Agent = 'agent',
  OntologyClass = 'ontology_class',
  OntologyIndividual = 'ontology_individual',
  OntologyProperty = 'ontology_property',
  Unknown = 'unknown'
}

export function getNodeType(nodeId: number): NodeType {
  if ((nodeId & AGENT_NODE_FLAG) !== 0) {
    return NodeType.Agent;
  } else if ((nodeId & KNOWLEDGE_NODE_FLAG) !== 0) {
    return NodeType.Knowledge;
  } else if ((nodeId & ONTOLOGY_TYPE_MASK) === ONTOLOGY_CLASS_FLAG) {
    return NodeType.OntologyClass;
  } else if ((nodeId & ONTOLOGY_TYPE_MASK) === ONTOLOGY_INDIVIDUAL_FLAG) {
    return NodeType.OntologyIndividual;
  } else if ((nodeId & ONTOLOGY_TYPE_MASK) === ONTOLOGY_PROPERTY_FLAG) {
    return NodeType.OntologyProperty;
  }
  return NodeType.Unknown;
}

export function getActualNodeId(nodeId: number): number {
  return nodeId & NODE_ID_MASK;
}

export function isAgentNode(nodeId: number): boolean {
  return (nodeId & AGENT_NODE_FLAG) !== 0;
}

export function isKnowledgeNode(nodeId: number): boolean {
  return (nodeId & KNOWLEDGE_NODE_FLAG) !== 0;
}

export function isOntologyNode(nodeId: number): boolean {
  return (nodeId & ONTOLOGY_TYPE_MASK) !== 0;
}

/**
 * Parse binary node data from server
 * Supports Protocol V2 (36 bytes) and V3 (48 bytes)
 */
export function parseBinaryNodeData(buffer: ArrayBuffer): BinaryNodeData[] {
  if (!buffer || buffer.byteLength === 0) {
    return [];
  }

  // Create a copy to avoid issues with detached buffers
  const safeBuffer = buffer.slice(0);
  const view = new DataView(safeBuffer);
  const nodes: BinaryNodeData[] = [];

  try {
    if (safeBuffer.byteLength < 1) {
      return [];
    }

    // Read protocol version byte
    const protocolVersion = view.getUint8(0);
    let offset = 1; // Skip version byte
    let nodeSize: number;
    let hasAnalytics: boolean;

    switch (protocolVersion) {
      case PROTOCOL_V2:
        nodeSize = BINARY_NODE_SIZE_V2;
        hasAnalytics = false;
        break;
      case PROTOCOL_V3:
        nodeSize = BINARY_NODE_SIZE_V3;
        hasAnalytics = true;
        break;
      case PROTOCOL_V4:
        // Delta encoding - not yet implemented on client
        logger.warn('Received Protocol V4 (delta encoding) frame — client decoding not yet implemented, skipping');
        return [];
      default:
        // Unknown version - try to detect format by size
        logger.warn(`Unknown protocol version: ${protocolVersion}, attempting auto-detection`);
        offset = 0; // No version byte - legacy format?
        nodeSize = BINARY_NODE_SIZE_V2;
        hasAnalytics = false;
    }

    const dataLength = safeBuffer.byteLength - offset;

    // Validate data length
    if (dataLength % nodeSize !== 0) {
      // Check if it might be the other version
      const otherSize = hasAnalytics ? BINARY_NODE_SIZE_V2 : BINARY_NODE_SIZE_V3;
      if (dataLength % otherSize === 0) {
        logger.warn(`Data size suggests ${hasAnalytics ? 'V2' : 'V3'} format, switching...`);
        nodeSize = otherSize;
        hasAnalytics = !hasAnalytics;
      } else {
        logger.warn(
          `Binary data length (${dataLength} bytes) is not a multiple of node size (${nodeSize}). ` +
          `Expected ${Math.floor(dataLength / nodeSize)} complete nodes.`
        );
      }
    }

    const completeNodes = Math.floor(dataLength / nodeSize);

    if (completeNodes === 0) {
      return [];
    }

    // Parse each node
    for (let i = 0; i < completeNodes; i++) {
      const nodeOffset = offset + (i * nodeSize);

      if (nodeOffset + nodeSize > safeBuffer.byteLength) {
        break;
      }

      // Node ID (4 bytes) - includes type flags in high bits
      const nodeId = view.getUint32(nodeOffset + BINARY_NODE_ID_OFFSET, true);

      // Position (12 bytes)
      const position: Vec3 = {
        x: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET, true),
        y: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET + 4, true),
        z: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET + 8, true)
      };

      // Velocity (12 bytes)
      const velocity: Vec3 = {
        x: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET, true),
        y: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET + 4, true),
        z: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET + 8, true)
      };

      // SSSP data (8 bytes)
      const ssspDistance = view.getFloat32(nodeOffset + BINARY_SSSP_DISTANCE_OFFSET, true);
      const ssspParent = view.getInt32(nodeOffset + BINARY_SSSP_PARENT_OFFSET, true);

      // Validate position and velocity (reject NaN/Inf)
      const isValid =
        !isNaN(position.x) && isFinite(position.x) &&
        !isNaN(position.y) && isFinite(position.y) &&
        !isNaN(position.z) && isFinite(position.z) &&
        !isNaN(velocity.x) && isFinite(velocity.x) &&
        !isNaN(velocity.y) && isFinite(velocity.y) &&
        !isNaN(velocity.z) && isFinite(velocity.z);

      if (isValid) {
        const node: BinaryNodeData = {
          nodeId,
          position,
          velocity,
          ssspDistance,
          ssspParent
        };

        // Parse V3 analytics fields if present
        if (hasAnalytics) {
          node.clusterId = view.getUint32(nodeOffset + BINARY_CLUSTER_ID_OFFSET, true);
          node.anomalyScore = view.getFloat32(nodeOffset + BINARY_ANOMALY_SCORE_OFFSET, true);
          node.communityId = view.getUint32(nodeOffset + BINARY_COMMUNITY_ID_OFFSET, true);
        }

        nodes.push(node);
      } else {
        // Only log first few corrupted nodes to avoid spam
        if (i < 3) {
          logger.warn(
            `Skipping corrupted node at index ${i}: id=${nodeId}, ` +
            `pos=[${position.x}, ${position.y}, ${position.z}]`
          );
        }
      }
    }
  } catch (error) {
    logger.error('Error parsing binary data:', error);
  }

  return nodes;
}

/**
 * Create binary node data for sending to server
 * Uses Protocol V3 format (48 bytes per node)
 */
export function createBinaryNodeData(nodes: BinaryNodeData[]): ArrayBuffer {
  // 1 byte version header + nodes * 48 bytes
  const buffer = new ArrayBuffer(1 + nodes.length * BINARY_NODE_SIZE_V3);
  const view = new DataView(buffer);

  // Write version header
  view.setUint8(0, PROTOCOL_V3);

  nodes.forEach((node, i) => {
    const offset = 1 + (i * BINARY_NODE_SIZE_V3);

    // Node ID
    view.setUint32(offset + BINARY_NODE_ID_OFFSET, node.nodeId, true);

    // Position
    view.setFloat32(offset + BINARY_POSITION_OFFSET, node.position.x, true);
    view.setFloat32(offset + BINARY_POSITION_OFFSET + 4, node.position.y, true);
    view.setFloat32(offset + BINARY_POSITION_OFFSET + 8, node.position.z, true);

    // Velocity
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET, node.velocity.x, true);
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET + 4, node.velocity.y, true);
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET + 8, node.velocity.z, true);

    // SSSP data
    view.setFloat32(offset + BINARY_SSSP_DISTANCE_OFFSET, node.ssspDistance ?? Infinity, true);
    view.setInt32(offset + BINARY_SSSP_PARENT_OFFSET, node.ssspParent ?? -1, true);

    // V3 analytics data
    view.setUint32(offset + BINARY_CLUSTER_ID_OFFSET, node.clusterId ?? 0, true);
    view.setFloat32(offset + BINARY_ANOMALY_SCORE_OFFSET, node.anomalyScore ?? 0, true);
    view.setUint32(offset + BINARY_COMMUNITY_ID_OFFSET, node.communityId ?? 0, true);
  });

  return buffer;
}

/**
 * Message type constants (must match server)
 */
export enum MessageType {
  BinaryPositions = 0x00,
  VoiceData = 0x02,
  ControlFrame = 0x03,
  PositionDelta = 0x04,
  BroadcastAck = 0x34,
}
