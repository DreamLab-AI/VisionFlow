// Bots visualization components
export { BotsVisualization } from './BotsVisualizationFixed';
// Legacy exports for backward compatibility
// export { BotsVisualization as BotsVisualizationLegacy } from './BotsVisualization'; // Commented out - file doesn't exist
// export { BotsVisualizationEnhanced } from './BotsVisualizationEnhanced'; // Commented out - file doesn't exist
export { BotsControlPanel } from './BotsControlPanel';
export { BotsDebugInfo } from './BotsVisualizationDebugInfo';
export { multiAgentInitializationPrompt } from './multiAgentInitializationPrompt';

// UI Panels
export { SystemHealthPanel } from './SystemHealthPanel';
export { ActivityLogPanel } from './ActivityLogPanel';
export { AgentDetailPanel } from './AgentDetailPanel';

// Services
export { configurationMapper } from '../services/ConfigurationMapper';
export type { VisualizationConfig } from '../services/ConfigurationMapper';