import React, { useState, useRef, useEffect, useCallback } from 'react';
import GraphCanvasWrapper from '../features/graph/components/GraphCanvasWrapper';
import SimpleThreeTest from '../features/graph/components/SimpleThreeTest';
import GraphCanvasSimple from '../features/graph/components/GraphCanvasSimple';
import { IntegratedControlPanel } from '../features/visualisation/components/IntegratedControlPanel';
import { useSettingsStore } from '../store/settingsStore';
import { BotsDataProvider, useBotsData } from '../features/bots/contexts/BotsDataContext';
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

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: '#000022'
    }}>
      <GraphCanvasWrapper />

      {/* Integrated Control Panel - Now includes GraphFeatures functionality */}
      <IntegratedControlPanel
        showStats={showStats}
        enableBloom={enableBloom}
        onOrbitControlsToggle={() => {}}
        botsData={botsData}
        graphData={graphData}
        otherGraphData={otherGraphData}
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