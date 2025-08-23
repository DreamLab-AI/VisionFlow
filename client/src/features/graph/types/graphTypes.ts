/**
 * Graph Type Definitions
 * Common types for graph data management
 */

export type GraphType = 'logseq' | 'visionflow';

export interface GraphNode {
  id: string;
  label: string;
  position: {
    x: number;
    y: number;
    z: number;
  };
  metadata?: Record<string, any>;
  graphType?: GraphType; // Optional graph type identifier
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  weight?: number;
  metadata?: Record<string, any>;
  graphType?: GraphType; // Optional graph type identifier
}

export interface TypedGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  graphType: GraphType;
  lastUpdate?: number;
}

// Message types for graph updates
export interface GraphUpdateMessage {
  type: 'node-update' | 'edge-update' | 'position-update' | 'bulk-update';
  graphType: GraphType;
  data: any;
  timestamp: number;
}

// Physics settings per graph type
export interface GraphPhysicsConfig {
  // Core physics parameters (updated ranges)
  spring_k: number; // 0.01-2.0
  repel_k: number; // 0.01-10.0
  max_velocity: number; // 0.1-50.0
  damping: number; // 0.5-0.99
  
  // New CUDA kernel parameters
  rest_length: number; // default: 50.0
  repulsion_cutoff: number; // default: 50.0
  repulsion_softening_epsilon: number; // default: 0.0001
  center_gravity_k: number; // default: 0.0
  grid_cell_size: number; // default: 50.0
  warmup_iterations: number; // default: 100
  cooling_rate: number; // default: 0.001
  feature_flags: number; // default: 7
  
  // Missing CUDA parameters
  boundary_extreme_multiplier: number; // 1.0-5.0, default: 2.0
  boundary_extreme_force_multiplier: number; // 1.0-20.0, default: 5.0
  boundary_velocity_damping: number; // 0.0-1.0, default: 0.5
  max_force: number; // 1-1000, default: 100
  seed: number; // default: 42
  iteration: number; // current iteration count, default: 0
  
  // Legacy parameters (maintained for compatibility)
  springStrength?: number;
  updateThreshold?: number;
  nodeRepulsion?: number;
  linkDistance?: number;
  gravityStrength?: number;
}

export interface GraphTypeConfig {
  logseq: {
    physics: GraphPhysicsConfig;
    rendering: {
      nodeSize: number;
      edgeWidth: number;
      labelSize: number;
    };
  };
  visionflow: {
    physics: GraphPhysicsConfig;
    rendering: {
      agentSize: number;
      connectionWidth: number;
      healthIndicator: boolean;
    };
  };
}

// Default configurations
export const DEFAULT_GRAPH_CONFIG: GraphTypeConfig = {
  logseq: {
    physics: {
      // Core physics parameters
      spring_k: 0.2,
      repel_k: 1.0,
      max_velocity: 5.0,
      damping: 0.95,
      
      // New CUDA kernel parameters
      rest_length: 50.0,
      repulsion_cutoff: 50.0,
      repulsion_softening_epsilon: 0.0001,
      center_gravity_k: 0.0,
      grid_cell_size: 50.0,
      warmup_iterations: 100,
      cooling_rate: 0.001,
      feature_flags: 7,
      
      // Missing CUDA parameters
      boundary_extreme_multiplier: 2.0,
      boundary_extreme_force_multiplier: 5.0,
      boundary_velocity_damping: 0.5,
      max_force: 100,
      seed: 42,
      iteration: 0,
      
      // Legacy parameters (for compatibility)
      springStrength: 0.2,
      updateThreshold: 0.05,
      nodeRepulsion: 10,
      linkDistance: 30
    },
    rendering: {
      nodeSize: 5,
      edgeWidth: 1,
      labelSize: 12
    }
  },
  visionflow: {
    physics: {
      // Core physics parameters
      spring_k: 0.3,
      repel_k: 2.0,
      max_velocity: 10.0,
      damping: 0.95,
      
      // New CUDA kernel parameters
      rest_length: 50.0,
      repulsion_cutoff: 50.0,
      repulsion_softening_epsilon: 0.0001,
      center_gravity_k: 0.1,
      grid_cell_size: 50.0,
      warmup_iterations: 100,
      cooling_rate: 0.001,
      feature_flags: 7,
      
      // Missing CUDA parameters
      boundary_extreme_multiplier: 2.5,
      boundary_extreme_force_multiplier: 6.0,
      boundary_velocity_damping: 0.6,
      max_force: 120,
      seed: 42,
      iteration: 0,
      
      // Legacy parameters (for compatibility)
      springStrength: 0.3,
      updateThreshold: 0.1,
      nodeRepulsion: 15,
      linkDistance: 20,
      gravityStrength: 0.1
    },
    rendering: {
      agentSize: 8,
      connectionWidth: 2,
      healthIndicator: true
    }
  }
};