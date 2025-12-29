

import {
  BarChart3,
  Eye,
  Zap,
  TrendingUp,
  Gauge,
  Palette,
  Code,
  Glasses,
  Network,
  Wrench,
  Hand,
  Download,
  Lock,
  ShieldCheck,
  Database,
} from 'lucide-react';
import type { TabConfig } from './types';

export const TAB_CONFIGS: TabConfig[] = [

  { id: 'dashboard', label: 'Dashboard', icon: BarChart3, description: 'System overview and status', buttonKey: '1' },
  { id: 'visualization', label: 'Visualization', icon: Eye, description: 'Visual rendering settings', buttonKey: '2' },
  { id: 'physics', label: 'Physics', icon: Zap, description: 'Physics simulation controls', buttonKey: '3' },
  { id: 'analytics', label: 'Analytics', icon: TrendingUp, description: 'Graph analysis metrics', buttonKey: '4' },
  { id: 'performance', label: 'Performance', icon: Gauge, description: 'Performance optimization', buttonKey: '5' },
  { id: 'integrations', label: 'Visual Effects', icon: Palette, description: 'Visual effects and glow', buttonKey: '6' },
  { id: 'developer', label: 'Developer', icon: Code, description: 'Development tools', buttonKey: '7' },
  { id: 'quality-gates', label: 'Quality Gates', icon: ShieldCheck, description: 'Feature toggles and performance thresholds', buttonKey: '8' },


  { id: 'graph-analysis', label: 'Analysis', icon: Network, description: 'Advanced graph analysis', buttonKey: 'A' },
  { id: 'graph-visualisation', label: 'Visualisation', icon: Palette, description: 'Graph visualization features', buttonKey: 'B' },
  { id: 'graph-optimisation', label: 'Optimisation', icon: Wrench, description: 'Graph optimization tools', buttonKey: 'C' },
  { id: 'graph-interaction', label: 'Interaction', icon: Hand, description: 'Interactive graph features', buttonKey: 'D' },
  { id: 'graph-export', label: 'Export', icon: Download, description: 'Export and sharing options', buttonKey: 'E' },
  { id: 'xr', label: 'XR/AR', icon: Glasses, description: 'Extended reality settings', buttonKey: 'F' },
  { id: 'auth', label: 'Auth/Nostr', icon: Lock, description: 'Authentication & Nostr', buttonKey: 'G' },
  { id: 'solid', label: 'Solid Pod', icon: Database, description: 'Decentralized data storage', buttonKey: 'H' },
];

// Group tabs by row for rendering
export const TAB_ROWS = {
  primary: TAB_CONFIGS.slice(0, 8),
  secondary: TAB_CONFIGS.slice(8),
};
