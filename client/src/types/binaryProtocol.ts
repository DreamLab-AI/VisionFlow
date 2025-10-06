/**
 * Binary protocol types for WebSocket communication
 * 
 * This aligns with the server's binary protocol format (src/utils/binary_protocol.rs)
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
  ssspDistance: number;  // Shortest path distance from source
  ssspParent: number;    // Parent node for path reconstruction
}

/**
 * Node binary format Protocol V2 (Updated to match server):
 * - Node ID: 4 bytes (uint32) - FIXED: No truncation, supports 1B nodes
 * - Position: 12 bytes (3 float32 values)
 * - Velocity: 12 bytes (3 float32 values)
 * - SSSP Distance: 4 bytes (float32)
 * - SSSP Parent: 4 bytes (int32)
 * Total: 36 bytes per node (NOT 38 - that was a documentation error!)
 *
 * Calculation: 4 + 12 + 12 + 4 + 4 = 36 bytes
 *
 * SSSP data is included for real-time path visualization
 *
 * FIXED: V1 had a bug that truncated node IDs > 16383 causing collisions
 * V2 uses full u32 IDs with flags at bits 31/30 instead of 15/14
 */
export const BINARY_NODE_SIZE = 36;
export const BINARY_NODE_ID_OFFSET = 0;
export const BINARY_POSITION_OFFSET = 4;  // After uint32 node ID
export const BINARY_VELOCITY_OFFSET = 16; // After position (4 + 12)
export const BINARY_SSSP_DISTANCE_OFFSET = 28; // After velocity (16 + 12)
export const BINARY_SSSP_PARENT_OFFSET = 32;   // After distance (28 + 4)

// Node type flag constants (Protocol V2 - must match server)
export const AGENT_NODE_FLAG = 0x80000000;     // Bit 31 indicates agent node
export const KNOWLEDGE_NODE_FLAG = 0x40000000; // Bit 14 indicates knowledge graph node
export const NODE_ID_MASK = 0x3FFFFFFF;        // Mask to extract actual node ID (bits 0-29)

export enum NodeType {
  Knowledge = 'knowledge',
  Agent = 'agent',
  Unknown = 'unknown'
}

export function getNodeType(nodeId: number): NodeType {
  if ((nodeId & AGENT_NODE_FLAG) !== 0) {
    return NodeType.Agent;
  } else if ((nodeId & KNOWLEDGE_NODE_FLAG) !== 0) {
    return NodeType.Knowledge;
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

/**
 * Parse binary data buffer into an array of BinaryNodeData objects
 * Supports both V1 (34 bytes, u16 IDs) and V2 (38 bytes, u32 IDs) protocols
 * Server sends: [1 byte version][node_data][node_data]...
 */
export function parseBinaryNodeData(buffer: ArrayBuffer): BinaryNodeData[] {
  if (!buffer || buffer.byteLength === 0) {
    return [];
  }

  // Make a copy of the buffer to avoid any issues with shared references
  const safeBuffer = buffer.slice(0);
  const view = new DataView(safeBuffer);
  const nodes: BinaryNodeData[] = [];

  try {
    // Check for protocol version byte (first byte)
    let offset = 0;
    let nodeSize = BINARY_NODE_SIZE; // Default to V2 (38 bytes)

    if (safeBuffer.byteLength > 0) {
      const firstByte = view.getUint8(0);

      // Check if first byte is a protocol version (1 or 2)
      if (firstByte === 1 || firstByte === 2) {
        const protocolVersion = firstByte;
        offset = 1; // Skip version byte

        if (protocolVersion === 1) {
          nodeSize = 34; // V1 legacy format
          console.warn('Received V1 protocol data (34 bytes/node). V2 (38 bytes/node) is recommended.');
        }
      }
    }

    const dataLength = safeBuffer.byteLength - offset;

    // Check if remaining data length is a multiple of the expected size
    if (dataLength % nodeSize !== 0) {
      console.warn(`Binary data length (${dataLength} bytes after version byte) is not a multiple of ${nodeSize}. This may indicate compressed data.`);
      console.warn(`First few bytes: ${new Uint8Array(safeBuffer.slice(0, Math.min(16, safeBuffer.byteLength))).join(', ')}`);

      // Check for zlib header (0x78 followed by compression level byte)
      const header = new Uint8Array(safeBuffer.slice(0, Math.min(4, safeBuffer.byteLength)));
      if (header[0] === 0x78 && (header[1] === 0x01 || header[1] === 0x5E || header[1] === 0x9C || header[1] === 0xDA)) {
        console.error("Data appears to be zlib compressed but decompression failed or wasn't attempted");
      }
    }

    // Calculate how many complete nodes we can process
    const completeNodes = Math.floor(dataLength / nodeSize);

    if (completeNodes === 0) {
      console.warn(`Received binary data with insufficient length: ${dataLength} bytes (needed at least ${nodeSize} bytes per node)`);
      return [];
    }
    
    // Parse nodes based on detected protocol
    const isV1 = nodeSize === 34;

    for (let i = 0; i < completeNodes; i++) {
      const nodeOffset = offset + (i * nodeSize);

      // Bounds check to prevent errors on corrupted data
      if (nodeOffset + nodeSize > safeBuffer.byteLength) {
        break;
      }

      let nodeId: number;
      let position: Vec3;
      let velocity: Vec3;
      let ssspDistance: number;
      let ssspParent: number;

      if (isV1) {
        // V1 format: u16 ID (2 bytes)
        nodeId = view.getUint16(nodeOffset, true);
        position = {
          x: view.getFloat32(nodeOffset + 2, true),
          y: view.getFloat32(nodeOffset + 6, true),
          z: view.getFloat32(nodeOffset + 10, true)
        };
        velocity = {
          x: view.getFloat32(nodeOffset + 14, true),
          y: view.getFloat32(nodeOffset + 18, true),
          z: view.getFloat32(nodeOffset + 22, true)
        };
        ssspDistance = view.getFloat32(nodeOffset + 26, true);
        ssspParent = view.getInt32(nodeOffset + 30, true);

        // Convert V1 flags (bits 15/14) to V2 flags (bits 31/30)
        const isAgent = (nodeId & 0x8000) !== 0;
        const isKnowledge = (nodeId & 0x4000) !== 0;
        const actualId = nodeId & 0x3FFF;

        if (isAgent) {
          nodeId = actualId | 0x80000000;
        } else if (isKnowledge) {
          nodeId = actualId | 0x40000000;
        } else {
          nodeId = actualId;
        }
      } else {
        // V2 format: u32 ID (4 bytes)
        nodeId = view.getUint32(nodeOffset + BINARY_NODE_ID_OFFSET, true);
        position = {
          x: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET, true),
          y: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET + 4, true),
          z: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET + 8, true)
        };
        velocity = {
          x: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET, true),
          y: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET + 4, true),
          z: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET + 8, true)
        };
        ssspDistance = view.getFloat32(nodeOffset + BINARY_SSSP_DISTANCE_OFFSET, true);
        ssspParent = view.getInt32(nodeOffset + BINARY_SSSP_PARENT_OFFSET, true);
      }

      // Basic validation to detect corrupted data
      const isValid =
        !isNaN(position.x) && isFinite(position.x) &&
        !isNaN(position.y) && isFinite(position.y) &&
        !isNaN(position.z) && isFinite(position.z) &&
        !isNaN(velocity.x) && isFinite(velocity.x) &&
        !isNaN(velocity.y) && isFinite(velocity.y) &&
        !isNaN(velocity.z) && isFinite(velocity.z);

      if (isValid) {
        nodes.push({ nodeId, position, velocity, ssspDistance, ssspParent });
      } else {
        console.warn(`Skipping corrupted node data at offset ${offset} (nodeId: ${nodeId})`);
      }
    }
  } catch (error) {
    console.error('Error parsing binary data:', error);
    // Return any nodes we've successfully parsed
  }

  return nodes;
}

/**
 * Create a binary buffer from an array of BinaryNodeData objects
 */
export function createBinaryNodeData(nodes: BinaryNodeData[]): ArrayBuffer {
  const buffer = new ArrayBuffer(nodes.length * BINARY_NODE_SIZE);
  const view = new DataView(buffer);
  
  nodes.forEach((node, i) => {
    const offset = i * BINARY_NODE_SIZE;

    // Write node ID (uint32, 4 bytes) - V2 protocol
    view.setUint32(offset + BINARY_NODE_ID_OFFSET, node.nodeId, true);

    // Write position (3 float32 values, 12 bytes)
    view.setFloat32(offset + BINARY_POSITION_OFFSET, node.position.x, true);
    view.setFloat32(offset + BINARY_POSITION_OFFSET + 4, node.position.y, true);
    view.setFloat32(offset + BINARY_POSITION_OFFSET + 8, node.position.z, true);

    // Write velocity (3 float32 values, 12 bytes)
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET, node.velocity.x, true);
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET + 4, node.velocity.y, true);
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET + 8, node.velocity.z, true);

    // Write SSSP distance (float32, 4 bytes)
    view.setFloat32(offset + BINARY_SSSP_DISTANCE_OFFSET, node.ssspDistance || Infinity, true);

    // Write SSSP parent (int32, 4 bytes)
    view.setInt32(offset + BINARY_SSSP_PARENT_OFFSET, node.ssspParent || -1, true);
  });
  
  return buffer;
}