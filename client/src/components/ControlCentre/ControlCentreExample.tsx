import React, { useState, useEffect } from 'react';
import { ControlCentre } from './ControlCentre';
import { useSettingsStore } from '../../store/settingsStore';

/**
 * Example component demonstrating Control Centre integration
 * This shows how to use the Control Centre with proper state management
 */
export const ControlCentreExample: React.FC = () => {
  const [showStats, setShowStats] = useState(true);
  const [enableBloom, setEnableBloom] = useState(false);
  
  const { settings, initialized } = useSettingsStore();

  // Update local state when settings change
  useEffect(() => {
    if (initialized && settings) {
      setEnableBloom(settings.visualisation?.bloom?.enabled || false);
    }
  }, [settings, initialized]);

  // Mock stats for demonstration
  const mockStats = {
    fps: 60,
    nodeCount: 150,
    edgeCount: 300,
    gpuMemory: '2.1 GB'
  };

  return (
    <div className="relative w-full h-screen bg-gradient-to-br from-gray-900 to-black">
      {/* Mock 3D Scene Background */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center text-white/50">
          <div className="text-6xl font-bold mb-4">3D Graph Scene</div>
          <div className="text-xl">Use the Control Centre to adjust settings</div>
          
          {showStats && (
            <div className="mt-8 p-4 bg-black/50 rounded-lg border border-white/20 inline-block">
              <div className="text-sm font-mono space-y-1">
                <div>FPS: {mockStats.fps}</div>
                <div>Nodes: {mockStats.nodeCount}</div>
                <div>Edges: {mockStats.edgeCount}</div>
                <div>GPU Memory: {mockStats.gpuMemory}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Control Centre Overlay */}
      <ControlCentre
        defaultExpanded={true}
        showStats={showStats}
        enableBloom={enableBloom}
        className="z-10"
      />

      {/* Settings Display (for debugging) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute bottom-4 right-4 max-w-sm bg-black/80 text-white p-4 rounded-lg border border-white/20 text-xs">
          <div className="font-bold mb-2">Current Settings (Debug)</div>
          <pre className="overflow-auto max-h-32">
            {JSON.stringify({
              background: {
                color: settings?.visualisation?.rendering?.backgroundColor,
                opacity: settings?.visualisation?.rendering?.backgroundOpacity,
                ambientLight: settings?.visualisation?.rendering?.ambientLightIntensity
              },
              bloom: {
                enabled: settings?.visualisation?.bloom?.enabled,
                strength: settings?.visualisation?.bloom?.environmentBloomStrength
              },
              physics: {
                enabled: settings?.visualisation?.graphs?.logseq?.physics?.enabled,
                springK: settings?.visualisation?.graphs?.logseq?.physics?.springK,
                repelK: settings?.visualisation?.graphs?.logseq?.physics?.repelK
              }
            }, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};