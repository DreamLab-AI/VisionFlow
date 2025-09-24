import React, { useState, useRef, useEffect, useCallback } from 'react';
import GraphCanvas from '../features/graph/components/GraphCanvas';
import SimpleThreeTest from '../features/graph/components/SimpleThreeTest';
import GraphCanvasSimple from '../features/graph/components/GraphCanvasSimple';
import { IntegratedControlPanel } from '../features/visualisation/components/IntegratedControlPanel';
import { useSettingsStore } from '../store/settingsStore';
import { BotsDataProvider, useBotsData } from '../features/bots/contexts/BotsDataContext';
import { AuthGatedVoiceButton } from '../components/AuthGatedVoiceButton';
import { AuthGatedVoiceIndicator } from '../components/AuthGatedVoiceIndicator';
import { BrowserSupportWarning } from '../components/BrowserSupportWarning';
import { SpaceMouseStatus } from '../components/SpaceMouseStatus';
import { AudioInputService } from '../services/AudioInputService';
import { graphDataManager, type GraphData } from '../features/graph/managers/graphDataManager';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('MainLayout');

const MainLayoutContent: React.FC = () => {
  // Both visualizations now positioned at origin (0, 0, 0) for unified view
  const { settings } = useSettingsStore();
  const { botsData } = useBotsData();
  const showStats = settings?.system?.debug?.enablePerformanceDebug ?? false;
  const enableBloom = settings?.visualisation?.glow?.enabled ?? false;
  const [hasVoiceSupport, setHasVoiceSupport] = useState(true);
  
  // Graph data state for integrated features
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [otherGraphData, setOtherGraphData] = useState<GraphData | undefined>();

  useEffect(() => {
    const support = AudioInputService.getBrowserSupport();
    const isSupported = support.getUserMedia && support.isHttps && support.audioContext && support.mediaRecorder;
    setHasVoiceSupport(isSupported);
  }, []);
  
  // Subscribe to graph data updates for integrated features
  useEffect(() => {
    const unsubscribe = graphDataManager.onGraphDataChange((data: GraphData) => {
      setGraphData(data);
      // For demo purposes, we can simulate "other" graph data
      // In a real dual-graph setup, this would come from a second data source
      if (data.nodes.length > 0) {
        setOtherGraphData({
          nodes: data.nodes.slice(0, Math.floor(data.nodes.length / 2)),
          edges: data.edges.slice(0, Math.floor(data.edges.length / 2))
        });
      }
    });
    
    return unsubscribe;
  }, []);
  
  // Handle graph feature updates
  const handleGraphFeatureUpdate = useCallback((feature: string, data: any) => {
    logger.debug(`Graph feature update: ${feature}`, data);
    
    // Handle different feature updates
    switch (feature) {
      case 'synchronisation':
        logger.info('Graph synchronisation settings updated', data);
        break;
      case 'comparison':
        logger.info('Graph comparison analysis completed', data);
        break;
      case 'aiInsights':
        logger.info('AI insights generated', data);
        break;
      case 'animations':
        logger.info('Animation settings updated', data);
        break;
      case 'export':
        logger.info('Graph export initiated', data);
        // Here you could trigger actual export functionality
        break;
      case 'timeTravel':
        logger.info('Time travel navigation', data);
        break;
      case 'collaboration':
        logger.info('Collaboration session update', data);
        break;
      case 'vrMode':
      case 'arMode':
        logger.info('Immersive mode toggle', { feature, ...data });
        break;
      default:
        logger.debug('Unknown feature update', { feature, data });
    }
  }, []);

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: '#000022'
    }}>
      <GraphCanvas />

      {/* Integrated Control Panel - Now includes GraphFeatures functionality */}
      <IntegratedControlPanel
        showStats={showStats}
        enableBloom={enableBloom}
        onOrbitControlsToggle={() => {}}
        botsData={botsData ? {
          nodeCount: botsData.nodeCount,
          edgeCount: botsData.edgeCount,
          tokenCount: botsData.tokenCount,
          mcpConnected: botsData.mcpConnected,
          dataSource: botsData.dataSource
        } : undefined}
        graphData={graphData}
        otherGraphData={otherGraphData}
        onGraphFeatureUpdate={handleGraphFeatureUpdate}
      />

      {/* SpaceMouse Status Warning */}
      <SpaceMouseStatus />

      {/* Browser Support Warning - Only show when there's no voice support */}
      {!hasVoiceSupport && (
        <div className="fixed bottom-20 left-4 z-40 max-w-sm pointer-events-auto">
          <BrowserSupportWarning className="shadow-lg" />
        </div>
      )}

      {/* Voice Interaction Components - Only show when browser supports it */}
      {/* Voice button removed - now integrated into control center */}
    </div>
  );
};

const MainLayout: React.FC = () => {
  return (
    <BotsDataProvider>
      <MainLayoutContent />
    </BotsDataProvider>
  );
};

export default MainLayout;