

export interface ControlPanelProps {
  showStats: boolean;
  enableBloom: boolean;
  onOrbitControlsToggle?: (enabled: boolean) => void;
  botsData?: BotsData;
  graphData?: GraphData;
  otherGraphData?: GraphData;
}

export interface BotsData {
  nodeCount: number;
  edgeCount: number;
  tokenCount: number;
  mcpConnected: boolean;
  dataSource: string;
}

export interface GraphData {
  nodes: any[];
  edges: any[];
}

export interface TabConfig {
  id: string;
  label: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  description: string;
  buttonKey?: string;
  /** Tab only visible in advanced mode */
  isAdvanced?: boolean;
  /** Tab only visible to power users */
  isPowerUserOnly?: boolean;
}

export interface SettingField {
  key: string;
  label: string;
  type: 'slider' | 'toggle' | 'color' | 'nostr-button' | 'text' | 'select' | 'action-button';
  path?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  /** Action to trigger for action-button type */
  action?: string;
  /** Setting only visible in advanced mode */
  isAdvanced?: boolean;
  /** Setting only visible to power users */
  isPowerUserOnly?: boolean;
  /** Description tooltip */
  description?: string;
}

export interface SectionConfig {
  title: string;
  fields: SettingField[];
  /** Section only visible in advanced mode */
  isAdvanced?: boolean;
  /** Section only visible to power users */
  isPowerUserOnly?: boolean;
}

export interface SystemHealthStatus {
  websocket: 'connected' | 'connecting' | 'disconnected';
  metadata: 'loaded' | 'loading' | 'error' | 'none';
  nodes: number;
  edges: number;
  mcpSwarm: 'connected' | 'disconnected' | 'unknown';
}
