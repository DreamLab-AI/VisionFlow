

import React, { useState, useEffect } from 'react';
import { SpaceDriver } from '../../../services/SpaceDriverService';
import { Tabs, TabsContent } from '../../design-system/components/Tabs';
import { TooltipProvider } from '../../design-system/components/Tooltip';
import ErrorBoundary from '../../../components/ErrorBoundary';
// Import to trigger scrollbar-hiding CSS injection
import '../../design-system/components/ScrollArea';

// Control Panel Components
import { ControlPanelHeader } from './ControlPanel/ControlPanelHeader';
import { SystemInfo } from './ControlPanel/SystemInfo';
import { BotsStatusPanel } from './ControlPanel/BotsStatusPanel';
import { SpacePilotStatus } from './ControlPanel/SpacePilotStatus';
import { TabNavigation } from './ControlPanel/TabNavigation';
import { TAB_CONFIGS } from './ControlPanel/config';
import type { ControlPanelProps } from './ControlPanel/types';

// Tab Components
import {
  RestoredGraphAnalysisTab,
  RestoredGraphVisualisationTab,
  RestoredGraphOptimisationTab,
  RestoredGraphInteractionTab,
  RestoredGraphExportTab
} from './ControlPanel/RestoredGraphTabs';
import { SettingsTabContent } from './ControlPanel/SettingsTabContent';

export const IntegratedControlPanel: React.FC<ControlPanelProps> = ({
  showStats,
  enableBloom,
  onOrbitControlsToggle,
  botsData,
  graphData,
  otherGraphData
}) => {
  
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeTab, setActiveTab] = useState<string>('dashboard');

  
  const [webHidAvailable, setWebHidAvailable] = useState(false);
  const [spacePilotConnected, setSpacePilotConnected] = useState(false);
  const [spacePilotButtons, setSpacePilotButtons] = useState<string[]>([]);

  
  useEffect(() => {
    setWebHidAvailable('hid' in navigator);
  }, []);

  
  useEffect(() => {
    const handleConnect = () => {
      setSpacePilotConnected(true);
      onOrbitControlsToggle?.(false);
    };

    const handleDisconnect = () => {
      setSpacePilotConnected(false);
      setSpacePilotButtons([]);
      onOrbitControlsToggle?.(true);
    };

    const handleButtons = (event: any) => {
      const buttons = event.detail.buttons || [];
      setSpacePilotButtons(buttons);
    };

    SpaceDriver.addEventListener('connect', handleConnect);
    SpaceDriver.addEventListener('disconnect', handleDisconnect);
    SpaceDriver.addEventListener('buttons', handleButtons);

    return () => {
      SpaceDriver.removeEventListener('connect', handleConnect);
      SpaceDriver.removeEventListener('disconnect', handleDisconnect);
      SpaceDriver.removeEventListener('buttons', handleButtons);
    };
  }, [onOrbitControlsToggle]);

  const handleConnectSpacePilot = async () => {
    try {
      await SpaceDriver.scan();
    } catch (error) {
      
    }
  };

  
  const renderTabContent = (tabId: string) => {
    switch (tabId) {
      
      case 'graph-analysis':
        return (
          <RestoredGraphAnalysisTab
            graphId="current"
            graphData={graphData}
            otherGraphData={otherGraphData}
          />
        );

      case 'graph-visualisation':
        return <RestoredGraphVisualisationTab graphId="current" />;

      case 'graph-optimisation':
        return (
          <RestoredGraphOptimisationTab
            graphId="current"
            graphData={graphData}
          />
        );

      case 'graph-interaction':
        return <RestoredGraphInteractionTab graphId="current" />;

      case 'graph-export':
        return (
          <RestoredGraphExportTab
            graphId="current"
            graphData={graphData}
            onExport={(format, options) => {
              console.log('Export:', format, options);
            }}
          />
        );

      
      default:
        return <SettingsTabContent sectionId={tabId} />;
    }
  };

  
  if (!isExpanded) {
    return (
      <div
        onClick={() => setIsExpanded(true)}
        style={{
          position: 'fixed',
          top: '10px',
          left: '10px',
          width: '40px',
          height: '40px',
          background: 'rgba(0,0,0,0.8)',
          border: '1px solid rgba(255,255,255,0.3)',
          borderRadius: '4px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          zIndex: 1000
        }}
      >
        <div style={{
          width: '12px',
          height: '12px',
          background: '#f87171',
          borderRadius: '50%',
          boxShadow: '0 0 5px rgba(248,113,113,0.5)'
        }} />
      </div>
    );
  }

  
  return (
    <div style={{
      position: 'fixed',
      top: '10px',
      left: '10px',
      color: 'white',
      fontFamily: 'sans-serif',
      fontSize: '11px',
      background: 'rgba(0,0,0,0.92)',
      padding: '12px',
      borderRadius: '6px',
      border: '1px solid rgba(255,255,255,0.2)',
      width: '360px',
      maxWidth: '360px',
      maxHeight: '88vh',
      display: 'flex',
      flexDirection: 'column',
      backdropFilter: 'blur(12px)',
      boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
      zIndex: 1000,
      overflow: 'hidden'
    }}>
      {}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '8px',
        paddingBottom: '8px',
        borderBottom: '1px solid rgba(255,255,255,0.15)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <div style={{
            width: '6px',
            height: '6px',
            background: '#10b981',
            borderRadius: '50%',
            boxShadow: '0 0 6px rgba(16,185,129,0.6)'
          }} />
          <h2 style={{
            fontSize: '12px',
            fontWeight: 'bold',
            color: '#10b981',
            margin: 0
          }}>
            Control Center
          </h2>
        </div>
        <button
          onClick={() => setIsExpanded(false)}
          style={{
            background: 'rgba(255,255,255,0.1)',
            border: '1px solid rgba(255,255,255,0.2)',
            color: 'white',
            width: '20px',
            height: '20px',
            borderRadius: '3px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            fontSize: '14px',
            lineHeight: '1'
          }}
        >
          ×
        </button>
      </div>

      {}
      <div style={{
        display: 'flex',
        gap: '6px',
        marginBottom: '8px',
        fontSize: '10px'
      }}>
        <div style={{
          flex: 1,
          padding: '4px',
          background: 'rgba(16,185,129,0.1)',
          border: '1px solid rgba(16,185,129,0.3)',
          borderRadius: '3px',
          textAlign: 'center'
        }}>
          <div style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '1px', fontSize: '9px' }}>Stats</div>
          <div style={{ color: '#10b981', fontWeight: '600', fontSize: '10px' }}>{showStats ? 'ON' : 'OFF'}</div>
        </div>
        <div style={{
          flex: 1,
          padding: '4px',
          background: 'rgba(147,51,234,0.1)',
          border: '1px solid rgba(147,51,234,0.3)',
          borderRadius: '3px',
          textAlign: 'center'
        }}>
          <div style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '1px', fontSize: '9px' }}>Bloom</div>
          <div style={{ color: '#a78bfa', fontWeight: '600', fontSize: '10px' }}>{enableBloom ? 'ON' : 'OFF'}</div>
        </div>
      </div>

      {}
      <BotsStatusPanel botsData={botsData} />

      {}
      <SpacePilotStatus
        webHidAvailable={webHidAvailable}
        spacePilotConnected={spacePilotConnected}
        spacePilotButtons={spacePilotButtons}
        onConnect={handleConnectSpacePilot}
      />

      {}
      <div className="scroll-area" style={{
        flex: 1,
        overflow: 'auto',
        marginTop: '8px',
        minHeight: 0
      }}>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabNavigation
            tabs={TAB_CONFIGS}
            pressedButtons={spacePilotButtons}
            spacePilotConnected={spacePilotConnected}
            rowClassName="grid-cols-4"
          />
          <TabsContent value={activeTab}>
            {renderTabContent(activeTab)}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default IntegratedControlPanel;
