

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
}

export interface SettingField {
  key: string;
  label: string;
  type: 'slider' | 'toggle' | 'color' | 'nostr-button' | 'text' | 'select';
  path: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
}

export interface SectionConfig {
  title: string;
  fields: SettingField[];
}
