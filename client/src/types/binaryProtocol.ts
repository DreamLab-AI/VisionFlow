

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
}


export const BINARY_NODE_SIZE = 36;
export const BINARY_NODE_ID_OFFSET = 0;
export const BINARY_POSITION_OFFSET = 4;  
export const BINARY_VELOCITY_OFFSET = 16; 
export const BINARY_SSSP_DISTANCE_OFFSET = 28; 
export const BINARY_SSSP_PARENT_OFFSET = 32;   

// Node type flag constants (Protocol V2 - must match server)
export const AGENT_NODE_FLAG = 0x80000000;     
export const KNOWLEDGE_NODE_FLAG = 0x40000000; 
export const NODE_ID_MASK = 0x3FFFFFFF;        

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


export function parseBinaryNodeData(buffer: ArrayBuffer): BinaryNodeData[] {
  if (!buffer || buffer.byteLength === 0) {
    return [];
  }

  
  const safeBuffer = buffer.slice(0);
  const view = new DataView(safeBuffer);
  const nodes: BinaryNodeData[] = [];

  try {
    
    let offset = 0;
    const nodeSize = BINARY_NODE_SIZE; 

    if (safeBuffer.byteLength > 0) {
      const firstByte = view.getUint8(0);

      
      if (firstByte === 2) {
        offset = 1; 
      } else if (firstByte === 1) {
        console.error('PROTOCOL_V1 is no longer supported. Please upgrade to V2.');
        return [];
      }
    }

    const dataLength = safeBuffer.byteLength - offset;

    
    if (dataLength % nodeSize !== 0) {
      console.warn(`Binary data length (${dataLength} bytes after version byte) is not a multiple of ${nodeSize}. This may indicate compressed data.`);
      console.warn(`First few bytes: ${new Uint8Array(safeBuffer.slice(0, Math.min(16, safeBuffer.byteLength))).join(', ')}`);

      
      const header = new Uint8Array(safeBuffer.slice(0, Math.min(4, safeBuffer.byteLength)));
      if (header[0] === 0x78 && (header[1] === 0x01 || header[1] === 0x5E || header[1] === 0x9C || header[1] === 0xDA)) {
        console.error("Data appears to be zlib compressed but decompression failed or wasn't attempted");
      }
    }

    
    const completeNodes = Math.floor(dataLength / nodeSize);

    if (completeNodes === 0) {
      console.warn(`Received binary data with insufficient length: ${dataLength} bytes (needed at least ${nodeSize} bytes per node)`);
      return [];
    }
    
    
    for (let i = 0; i < completeNodes; i++) {
      const nodeOffset = offset + (i * nodeSize);

      
      if (nodeOffset + nodeSize > safeBuffer.byteLength) {
        break;
      }

      
      const nodeId = view.getUint32(nodeOffset + BINARY_NODE_ID_OFFSET, true);
      const position: Vec3 = {
        x: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET, true),
        y: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET + 4, true),
        z: view.getFloat32(nodeOffset + BINARY_POSITION_OFFSET + 8, true)
      };
      const velocity: Vec3 = {
        x: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET, true),
        y: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET + 4, true),
        z: view.getFloat32(nodeOffset + BINARY_VELOCITY_OFFSET + 8, true)
      };
      const ssspDistance = view.getFloat32(nodeOffset + BINARY_SSSP_DISTANCE_OFFSET, true);
      const ssspParent = view.getInt32(nodeOffset + BINARY_SSSP_PARENT_OFFSET, true);

      
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
    
  }

  return nodes;
}


export function createBinaryNodeData(nodes: BinaryNodeData[]): ArrayBuffer {
  const buffer = new ArrayBuffer(nodes.length * BINARY_NODE_SIZE);
  const view = new DataView(buffer);
  
  nodes.forEach((node, i) => {
    const offset = i * BINARY_NODE_SIZE;

    
    view.setUint32(offset + BINARY_NODE_ID_OFFSET, node.nodeId, true);

    
    view.setFloat32(offset + BINARY_POSITION_OFFSET, node.position.x, true);
    view.setFloat32(offset + BINARY_POSITION_OFFSET + 4, node.position.y, true);
    view.setFloat32(offset + BINARY_POSITION_OFFSET + 8, node.position.z, true);

    
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET, node.velocity.x, true);
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET + 4, node.velocity.y, true);
    view.setFloat32(offset + BINARY_VELOCITY_OFFSET + 8, node.velocity.z, true);

    
    view.setFloat32(offset + BINARY_SSSP_DISTANCE_OFFSET, node.ssspDistance || Infinity, true);

    
    view.setInt32(offset + BINARY_SSSP_PARENT_OFFSET, node.ssspParent || -1, true);
  });
  
  return buffer;
}