// Minimal settings interface - matches Rust exactly
// No duplication, no confusion, just what's needed

// ============================================================================
// PHYSICS - Core simulation parameters
// ============================================================================
export interface PhysicsSettings {
  enabled: boolean;
  iterations: number;
  damping: number;
  springStrength: number;
  repulsionStrength: number;
  repulsionDistance: number;
  maxVelocity: number;
  boundsSize: number;
  enableBounds: boolean;
  massScale: number;
  boundaryDamping: number;
}

// ============================================================================
// VISUAL - What users see
// ============================================================================
export interface NodeSettings {
  baseColor: string;
  size: number;
  opacity: number;
  metalness: number;
  roughness: number;
  enableHologram: boolean;
}

export interface EdgeSettings {
  color: string;
  width: number;
  opacity: number;
  enableArrows: boolean;
  arrowSize: number;
}

export interface LabelSettings {
  enabled: boolean;
  fontSize: number;
  color: string;
  outlineColor: string;
  outlineWidth: number;
}

export interface RenderingSettings {
  backgroundColor: string;
  ambientLight: number;
  directionalLight: number;
  enableBloom: boolean;
  bloomStrength: number;
}

// ============================================================================
// GRAPH - Per-graph configuration
// ============================================================================
export interface GraphSettings {
  physics: PhysicsSettings;
  nodes: NodeSettings;
  edges: EdgeSettings;
  labels: LabelSettings;
}

// ============================================================================
// ROOT - Complete minimal settings
// ============================================================================
export interface MinimalSettings {
  activeGraph: string;
  graphs: Record<string, GraphSettings>;
  rendering: RenderingSettings;
  debug: boolean;
}

// ============================================================================
// UI CONTROL PANEL SECTIONS - Organized and intuitive
// ============================================================================
export const CONTROL_PANEL_SECTIONS = {
  appearance: {
    title: 'Appearance',
    icon: '🎨',
    fields: [
      // Nodes
      { id: 'nodeColor', label: 'Node Color', type: 'color', path: 'graphs.{activeGraph}.nodes.baseColor' },
      { id: 'nodeSize', label: 'Node Size', type: 'slider', min: 0.2, max: 3, path: 'graphs.{activeGraph}.nodes.size' },
      { id: 'nodeOpacity', label: 'Node Opacity', type: 'slider', min: 0, max: 1, path: 'graphs.{activeGraph}.nodes.opacity' },
      { id: 'nodeMetalness', label: 'Metalness', type: 'slider', min: 0, max: 1, path: 'graphs.{activeGraph}.nodes.metalness' },
      { id: 'nodeRoughness', label: 'Roughness', type: 'slider', min: 0, max: 1, path: 'graphs.{activeGraph}.nodes.roughness' },
      { id: 'enableHologram', label: 'Hologram Effect', type: 'toggle', path: 'graphs.{activeGraph}.nodes.enableHologram' },
      
      // Edges
      { id: 'edgeColor', label: 'Edge Color', type: 'color', path: 'graphs.{activeGraph}.edges.color' },
      { id: 'edgeWidth', label: 'Edge Width', type: 'slider', min: 0.01, max: 2, path: 'graphs.{activeGraph}.edges.width' },
      { id: 'edgeOpacity', label: 'Edge Opacity', type: 'slider', min: 0, max: 1, path: 'graphs.{activeGraph}.edges.opacity' },
      { id: 'enableArrows', label: 'Show Arrows', type: 'toggle', path: 'graphs.{activeGraph}.edges.enableArrows' },
      { id: 'arrowSize', label: 'Arrow Size', type: 'slider', min: 0.01, max: 1, path: 'graphs.{activeGraph}.edges.arrowSize' },
    ]
  },
  
  physics: {
    title: 'Physics',
    icon: '⚡',
    fields: [
      { id: 'physicsEnabled', label: 'Enable Physics', type: 'toggle', path: 'graphs.{activeGraph}.physics.enabled' },
      { id: 'iterations', label: 'Iterations', type: 'slider', min: 10, max: 500, step: 10, path: 'graphs.{activeGraph}.physics.iterations' },
      { id: 'damping', label: 'Damping', type: 'slider', min: 0, max: 1, path: 'graphs.{activeGraph}.physics.damping' },
      { id: 'springStrength', label: 'Spring Force', type: 'slider', min: 0, max: 1, path: 'graphs.{activeGraph}.physics.springStrength' },
      { id: 'repulsionStrength', label: 'Repulsion', type: 'slider', min: 0, max: 500, path: 'graphs.{activeGraph}.physics.repulsionStrength' },
      { id: 'repulsionDistance', label: 'Repulsion Range', type: 'slider', min: 10, max: 500, path: 'graphs.{activeGraph}.physics.repulsionDistance' },
      { id: 'maxVelocity', label: 'Max Speed', type: 'slider', min: 0.1, max: 20, path: 'graphs.{activeGraph}.physics.maxVelocity' },
      { id: 'enableBounds', label: 'Enable Bounds', type: 'toggle', path: 'graphs.{activeGraph}.physics.enableBounds' },
      { id: 'boundsSize', label: 'Bounds Size', type: 'slider', min: 100, max: 5000, path: 'graphs.{activeGraph}.physics.boundsSize' },
    ]
  },
  
  labels: {
    title: 'Labels',
    icon: '📝',
    fields: [
      { id: 'labelsEnabled', label: 'Show Labels', type: 'toggle', path: 'graphs.{activeGraph}.labels.enabled' },
      { id: 'fontSize', label: 'Font Size', type: 'slider', min: 0.1, max: 2, path: 'graphs.{activeGraph}.labels.fontSize' },
      { id: 'labelColor', label: 'Text Color', type: 'color', path: 'graphs.{activeGraph}.labels.color' },
      { id: 'outlineColor', label: 'Outline Color', type: 'color', path: 'graphs.{activeGraph}.labels.outlineColor' },
      { id: 'outlineWidth', label: 'Outline Width', type: 'slider', min: 0, max: 0.02, path: 'graphs.{activeGraph}.labels.outlineWidth' },
    ]
  },
  
  rendering: {
    title: 'Rendering',
    icon: '💡',
    fields: [
      { id: 'backgroundColor', label: 'Background', type: 'color', path: 'rendering.backgroundColor' },
      { id: 'ambientLight', label: 'Ambient Light', type: 'slider', min: 0, max: 3, path: 'rendering.ambientLight' },
      { id: 'directionalLight', label: 'Direct Light', type: 'slider', min: 0, max: 3, path: 'rendering.directionalLight' },
      { id: 'enableBloom', label: 'Enable Bloom', type: 'toggle', path: 'rendering.enableBloom' },
      { id: 'bloomStrength', label: 'Bloom Strength', type: 'slider', min: 0, max: 3, path: 'rendering.bloomStrength' },
    ]
  },
  
  graph: {
    title: 'Graph Selection',
    icon: '🔄',
    fields: [
      { id: 'activeGraph', label: 'Active Graph', type: 'select', options: ['logseq', 'visionflow'], path: 'activeGraph' },
      { id: 'debug', label: 'Debug Mode', type: 'toggle', path: 'debug' },
    ]
  }
};

// ============================================================================
// DEFAULT SETTINGS
// ============================================================================
export const defaultMinimalSettings: MinimalSettings = {
  activeGraph: 'logseq',
  graphs: {
    logseq: {
      physics: {
        enabled: true,
        iterations: 200,
        damping: 0.85,
        springStrength: 0.02,
        repulsionStrength: 15.0,
        repulsionDistance: 50.0,
        maxVelocity: 0.5,
        boundsSize: 150.0,
        enableBounds: true,
        massScale: 1.5,
        boundaryDamping: 0.9,
      },
      nodes: {
        baseColor: '#66d9ef',
        size: 1.2,
        opacity: 0.95,
        metalness: 0.85,
        roughness: 0.15,
        enableHologram: false,
      },
      edges: {
        color: '#56b6c2',
        width: 0.5,
        opacity: 0.25,
        enableArrows: false,
        arrowSize: 0.02,
      },
      labels: {
        enabled: true,
        fontSize: 0.5,
        color: '#f8f8f2',
        outlineColor: '#181c28',
        outlineWidth: 0.005,
      }
    },
    visionflow: {
      physics: {
        enabled: true,
        iterations: 150,
        damping: 0.85,
        springStrength: 0.5,
        repulsionStrength: 150.0,
        repulsionDistance: 200.0,
        maxVelocity: 15.0,
        boundsSize: 3000.0,
        enableBounds: false,
        massScale: 1.2,
        boundaryDamping: 0.6,
      },
      nodes: {
        baseColor: '#ff8800',
        size: 1.5,
        opacity: 0.9,
        metalness: 0.7,
        roughness: 0.3,
        enableHologram: true,
      },
      edges: {
        color: '#ffaa00',
        width: 0.15,
        opacity: 0.6,
        enableArrows: true,
        arrowSize: 0.7,
      },
      labels: {
        enabled: true,
        fontSize: 16.0,
        color: '#ffaa00',
        outlineColor: '#000000',
        outlineWidth: 2.5,
      }
    }
  },
  rendering: {
    backgroundColor: '#0a0e1a',
    ambientLight: 1.2,
    directionalLight: 1.5,
    enableBloom: true,
    bloomStrength: 1.5,
  },
  debug: false,
};

// ============================================================================
// SETTINGS API
// ============================================================================
export class MinimalSettingsAPI {
  private static readonly BASE_URL = '/api/settings';
  
  static async getSettings(): Promise<MinimalSettings> {
    const response = await fetch(this.BASE_URL);
    if (!response.ok) throw new Error('Failed to fetch settings');
    return response.json();
  }
  
  static async updatePhysics(physics: PhysicsSettings): Promise<void> {
    const response = await fetch(`${this.BASE_URL}/physics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(physics),
    });
    if (!response.ok) throw new Error('Failed to update physics');
  }
  
  static async updateSection(section: string, data: any): Promise<void> {
    const response = await fetch(this.BASE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [section]: data }),
    });
    if (!response.ok) throw new Error(`Failed to update ${section}`);
  }
  
  static async switchGraph(graphName: string): Promise<void> {
    const response = await fetch(`${this.BASE_URL}/graph/${graphName}`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error(`Failed to switch to graph ${graphName}`);
  }
}