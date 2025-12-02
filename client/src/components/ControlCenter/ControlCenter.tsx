// frontend/src/components/ControlCenter/ControlCenter.tsx
// Main control center integrating all panels - Unified implementation with advanced mode

import React, { useState, useMemo } from 'react';
import { ControlPanelProvider, useControlPanelContext } from '../../features/settings/components/control-panel-context';
import { useSettingsStore } from '../../store/settingsStore';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../../features/design-system/components/Tabs';
import { SystemHealthIndicator } from '../../features/visualisation/components/ControlPanel/SystemHealthIndicator';
import { AdvancedModeToggle } from '../../features/visualisation/components/ControlPanel/AdvancedModeToggle';
import { UnifiedSettingsTabContent } from '../../features/visualisation/components/ControlPanel/UnifiedSettingsTabContent';
import { UNIFIED_TABS, filterTabs } from '../../features/visualisation/components/ControlPanel/unifiedSettingsConfig';
import { ConstraintPanel } from './ConstraintPanel';
import { ProfileManager } from './ProfileManager';
import { QualityGatePanel } from './QualityGatePanel';
import { X, Lock, Star } from 'lucide-react';
import './ControlCenter.css';

interface ControlCenterInnerProps {
  graphData?: { nodes: any[]; edges: any[] };
  mcpConnected?: boolean;
  websocketStatus?: 'connected' | 'connecting' | 'disconnected';
  metadataStatus?: 'loaded' | 'loading' | 'error' | 'none';
  onClose?: () => void;
}

const ControlCenterInner: React.FC<ControlCenterInnerProps> = ({
  graphData,
  mcpConnected = false,
  websocketStatus = 'disconnected',
  metadataStatus = 'none',
  onClose
}) => {
  const [activeTab, setActiveTab] = useState('graph');
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const { advancedMode } = useControlPanelContext();
  const isPowerUser = useSettingsStore(state => state.isPowerUser);

  // Filter visible tabs based on advanced mode and power user status
  const visibleTabs = useMemo(() => {
    return filterTabs(UNIFIED_TABS, advancedMode, isPowerUser);
  }, [advancedMode, isPowerUser]);

  // Calculate grid columns for tab layout
  const gridColumns = useMemo(() => {
    const count = visibleTabs.length;
    if (count <= 4) return count;
    if (count <= 6) return 3;
    if (count <= 9) return Math.ceil(count / 2);
    return 4;
  }, [visibleTabs.length]);

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  const handleError = (error: string) => {
    showNotification('error', error);
  };

  const handleSuccess = (message: string) => {
    showNotification('success', message);
  };

  return (
    <div className="control-center unified-control-center">
      {/* Header with close button */}
      <div className="control-center-header" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '12px'
      }}>
        <div>
          <h1 style={{ fontSize: '14px', margin: 0, color: 'white', fontWeight: '600' }}>
            VisionFlow Control Center
          </h1>
          <p style={{ fontSize: '10px', margin: '2px 0 0', color: 'rgba(255,255,255,0.6)' }}>
            {advancedMode ? 'Advanced settings and controls' : 'Essential settings and controls'}
          </p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            style={{
              width: '28px',
              height: '28px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '4px',
              cursor: 'pointer',
              color: 'rgba(255,255,255,0.6)',
              transition: 'all 0.2s'
            }}
          >
            <X size={14} />
          </button>
        )}
      </div>

      {/* Status Indicators Row - replaces bloom/stats telltales */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
        <div style={{ flex: 1 }}>
          <SystemHealthIndicator
            graphData={graphData}
            mcpConnected={mcpConnected}
            websocketStatus={websocketStatus}
            metadataStatus={metadataStatus}
          />
        </div>
        <div style={{ flex: 1 }}>
          <AdvancedModeToggle />
        </div>
      </div>

      {/* Notification */}
      {notification && (
        <div
          className={`notification ${notification.type}`}
          style={{
            padding: '8px 12px',
            marginBottom: '8px',
            borderRadius: '4px',
            fontSize: '11px',
            background: notification.type === 'error'
              ? 'rgba(239,68,68,0.15)'
              : 'rgba(34,197,94,0.15)',
            border: `1px solid ${notification.type === 'error' ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.3)'}`,
            color: notification.type === 'error' ? '#ef4444' : '#22c55e'
          }}
        >
          {notification.message}
        </div>
      )}

      {/* Unified Tab Navigation */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList style={{
          width: '100%',
          background: 'rgba(255,255,255,0.08)',
          border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: '4px',
          padding: '2px',
          marginBottom: '8px',
          display: 'grid',
          gridTemplateColumns: `repeat(${gridColumns}, 1fr)`,
          gap: '2px',
          height: 'auto',
          minHeight: 'auto'
        }}>
          {visibleTabs.map((tab) => {
            const IconComponent = tab.icon;
            const isAdvancedTab = tab.isAdvanced;
            const isPowerUserTab = tab.isPowerUserOnly;

            return (
              <TabsTrigger
                key={tab.id}
                value={tab.id}
                title={tab.description}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '2px',
                  padding: '6px 4px',
                  fontSize: '9px',
                  fontWeight: '500',
                  color: isAdvancedTab ? 'rgba(168,85,247,0.9)' : 'rgba(255,255,255,0.7)',
                  border: isPowerUserTab
                    ? '1px solid rgba(251,191,36,0.3)'
                    : '0',
                  borderRadius: '3px',
                  background: isPowerUserTab
                    ? 'rgba(251,191,36,0.05)'
                    : 'transparent',
                  cursor: 'pointer',
                  height: '100%',
                  transition: 'all 0.2s',
                  position: 'relative'
                }}
              >
                {/* Power user indicator */}
                {isPowerUserTab && (
                  <div style={{ position: 'absolute', top: '2px', right: '2px' }}>
                    <Star size={6} style={{ color: '#fbbf24', fill: '#fbbf24' }} />
                  </div>
                )}

                {/* Advanced mode indicator */}
                {isAdvancedTab && !isPowerUserTab && (
                  <div style={{ position: 'absolute', top: '2px', right: '2px' }}>
                    <Lock size={6} style={{ color: '#a855f7' }} />
                  </div>
                )}

                {IconComponent && <IconComponent size={14} />}
                <div style={{ textAlign: 'center', lineHeight: '1.1' }}>
                  {tab.buttonKey && (
                    <div style={{ opacity: 0.6, fontSize: '7px' }}>{tab.buttonKey}</div>
                  )}
                  <div style={{ fontSize: '9px' }}>{tab.label}</div>
                </div>
              </TabsTrigger>
            );
          })}
        </TabsList>

        {/* Tab Content - Using unified settings content for all unified tabs */}
        <div style={{
          background: 'rgba(0,0,0,0.2)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '4px',
          padding: '8px',
          maxHeight: '400px',
          overflowY: 'auto'
        }}>
          {/* Unified settings tabs */}
          {UNIFIED_TABS.map(tab => (
            <TabsContent key={tab.id} value={tab.id}>
              <UnifiedSettingsTabContent
                sectionId={tab.id}
                onError={handleError}
                onSuccess={handleSuccess}
              />
            </TabsContent>
          ))}

          {/* Legacy panels for constraints, profiles, quality-gates if needed */}
          <TabsContent value="constraints-legacy">
            <ConstraintPanel onError={handleError} onSuccess={handleSuccess} />
          </TabsContent>
          <TabsContent value="profiles-legacy">
            <ProfileManager onError={handleError} onSuccess={handleSuccess} />
          </TabsContent>
          <TabsContent value="quality-gates-legacy">
            <QualityGatePanel onError={handleError} onSuccess={handleSuccess} />
          </TabsContent>
        </div>
      </Tabs>

      {/* Footer with tab count info */}
      <div style={{
        marginTop: '8px',
        padding: '4px 8px',
        background: 'rgba(255,255,255,0.03)',
        borderRadius: '3px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '8px',
        color: 'rgba(255,255,255,0.4)'
      }}>
        <span>{visibleTabs.length} tabs visible</span>
        {!advancedMode && (
          <span>+{UNIFIED_TABS.filter(t => t.isAdvanced).length} advanced tabs hidden</span>
        )}
        {advancedMode && !isPowerUser && (
          <span>Connect Nostr for power user features</span>
        )}
      </div>
    </div>
  );
};

// Export wrapped component with provider
export interface ControlCenterProps {
  graphData?: { nodes: any[]; edges: any[] };
  mcpConnected?: boolean;
  websocketStatus?: 'connected' | 'connecting' | 'disconnected';
  metadataStatus?: 'loaded' | 'loading' | 'error' | 'none';
  onClose?: () => void;
}

export const ControlCenter: React.FC<ControlCenterProps> = (props) => {
  return (
    <ControlPanelProvider>
      <ControlCenterInner {...props} />
    </ControlPanelProvider>
  );
};
