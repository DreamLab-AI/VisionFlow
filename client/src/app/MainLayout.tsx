import React, { useState, useRef, useEffect } from 'react';
import GraphViewport from '../features/graph/components/GraphViewport';
import { IntegratedControlPanel } from '../features/visualisation/components/IntegratedControlPanel';
import { HologramVisualisation } from '../features/visualisation/components/HologramVisualisation';
import { useSettingsStore } from '../store/settingsStore';
import { BotsDataProvider, useBotsData } from '../features/bots/contexts/BotsDataContext';
import { useSelectiveSetting } from '@/hooks/useSelectiveSettingsStore';
import { AuthGatedVoiceButton } from '../components/AuthGatedVoiceButton';
import { AuthGatedVoiceIndicator } from '../components/AuthGatedVoiceIndicator';
import { BrowserSupportWarning } from '../components/BrowserSupportWarning';
import { SpaceMouseStatus } from '../components/SpaceMouseStatus';
import { AudioInputService } from '../services/AudioInputService';

const MainLayoutContent: React.FC = () => {
  // Both visualizations now positioned at origin (0, 0, 0) for unified view
  const { settings } = useSettingsStore();
  const { botsData } = useBotsData();
  const showStats = settings?.system?.debug?.enablePerformanceDebug ?? false;
  const enableBloom = settings?.visualisation?.bloom?.enabled ?? false;
  const [hasVoiceSupport, setHasVoiceSupport] = useState(true);

  useEffect(() => {
    const support = AudioInputService.getBrowserSupport();
    const isSupported = support.getUserMedia && support.isHttps && support.audioContext && support.mediaRecorder;
    setHasVoiceSupport(isSupported);
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
      <GraphViewport />

      {/* Integrated Control Panel - Simplified version for VisionFlow status and SpacePilot */}
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
      {hasVoiceSupport && (
        <div className="fixed bottom-4 left-4 z-50 flex flex-col gap-1 items-start pointer-events-auto">
          <AuthGatedVoiceButton size="md" variant="primary" />
          <AuthGatedVoiceIndicator
            className="max-w-xs text-xs"
            showTranscription={true}
            showStatus={false}
          />
        </div>
      )}
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