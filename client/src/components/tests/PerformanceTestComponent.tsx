/**
 * Performance Test Component
 * Demonstrates the performance improvements of selective hooks vs traditional hooks
 */

import React, { useEffect, useState, useRef } from 'react';
import { useSettingsStore } from '../../store/settingsStore';
import { 
  useSelectiveSetting, 
  useSelectiveSettings,
  useSettingSetter,
  useCacheManager 
} from '../../hooks/useSelectiveSettingsStore';

interface PerformanceMetrics {
  renderCount: number;
  lastRenderTime: number;
  averageRenderTime: number;
}

export const PerformanceTestComponent: React.FC = () => {
  const [testMode, setTestMode] = useState<'traditional' | 'selective'>('selective');
  const [metrics, setMetrics] = useState<PerformanceMetrics>({ 
    renderCount: 0, 
    lastRenderTime: 0, 
    averageRenderTime: 0 
  });
  const renderStartTime = useRef<number>(0);
  const renderTimes = useRef<number[]>([]);

  // Cache manager for testing
  const { getCacheStats, clearCache } = useCacheManager();

  // Performance tracking
  useEffect(() => {
    renderStartTime.current = performance.now();
  });

  useEffect(() => {
    const renderTime = performance.now() - renderStartTime.current;
    renderTimes.current.push(renderTime);
    
    // Keep only last 50 render times for average calculation
    if (renderTimes.current.length > 50) {
      renderTimes.current.shift();
    }

    const averageRenderTime = renderTimes.current.reduce((a, b) => a + b, 0) / renderTimes.current.length;
    
    setMetrics(prev => ({
      renderCount: prev.renderCount + 1,
      lastRenderTime: renderTime,
      averageRenderTime
    }));
  });

  return (
    <div style={{
      position: 'fixed',
      top: '10px',
      right: '10px',
      background: 'rgba(0, 0, 0, 0.9)',
      color: 'white',
      padding: '15px',
      borderRadius: '8px',
      fontFamily: 'monospace',
      fontSize: '12px',
      minWidth: '300px',
      zIndex: 1000
    }}>
      <h3 style={{ margin: '0 0 10px 0', color: '#4CAF50' }}>
        🚀 Selective Hooks Performance Test
      </h3>
      
      {/* Mode Selector */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ marginRight: '10px' }}>Test Mode:</label>
        <select 
          value={testMode}
          onChange={(e) => setTestMode(e.target.value as 'traditional' | 'selective')}
          style={{
            background: '#333',
            color: 'white',
            border: '1px solid #666',
            borderRadius: '4px',
            padding: '5px'
          }}
        >
          <option value="selective">Selective Hooks (Optimized)</option>
          <option value="traditional">Traditional Store (Baseline)</option>
        </select>
      </div>

      {/* Test Components */}
      {testMode === 'selective' ? (
        <SelectiveHookTest />
      ) : (
        <TraditionalHookTest />
      )}

      {/* Performance Metrics */}
      <div style={{
        marginTop: '15px',
        paddingTop: '15px',
        borderTop: '1px solid #444'
      }}>
        <div style={{ marginBottom: '5px', fontWeight: 'bold' }}>
          Performance Metrics:
        </div>
        <div>Render Count: {metrics.renderCount}</div>
        <div>Last Render: {metrics.lastRenderTime.toFixed(2)}ms</div>
        <div>Average Render: {metrics.averageRenderTime.toFixed(2)}ms</div>
        
        {testMode === 'selective' && (
          <div style={{ marginTop: '10px' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
              Cache Stats:
            </div>
            <div>Cache Size: {getCacheStats().responseCacheSize}</div>
            <div>Active Requests: {getCacheStats().activeRequests}</div>
            <div>Pending Debounces: {getCacheStats().pendingDebounces}</div>
            <button
              onClick={clearCache}
              style={{
                marginTop: '5px',
                background: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '5px 10px',
                cursor: 'pointer',
                fontSize: '11px'
              }}
            >
              Clear Cache
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Test component using selective hooks (optimized)
 */
const SelectiveHookTest: React.FC = () => {
  // Test single setting subscription
  const nodeOpacity = useSelectiveSetting<number>('visualisation.nodes.opacity', {
    enableCache: true,
    enableDeduplication: true
  });
  
  // Test multiple settings subscription
  const physicsSettings = useSelectiveSettings({
    springK: 'visualisation.graphs.logseq.physics.springK',
    repelK: 'visualisation.graphs.logseq.physics.repelK',
    damping: 'visualisation.graphs.logseq.physics.damping'
  }, {
    enableBatchLoading: true,
    enableCache: true
  });

  // Test setter with debouncing
  const { set, batchedSet } = useSettingSetter();

  const handleQuickUpdates = () => {
    // Simulate rapid updates that would normally flood the network
    for (let i = 0; i < 10; i++) {
      setTimeout(() => {
        set('visualisation.nodes.opacity', Math.random());
      }, i * 50);
    }
  };

  const handleBatchUpdate = () => {
    batchedSet({
      'visualisation.graphs.logseq.physics.springK': Math.random() * 2,
      'visualisation.graphs.logseq.physics.repelK': Math.random() * 100,
      'visualisation.graphs.logseq.physics.damping': Math.random()
    });
  };

  return (
    <div>
      <div style={{ color: '#4CAF50', fontWeight: 'bold', marginBottom: '10px' }}>
        ✅ Using Selective Hooks (Optimized)
      </div>
      
      <div style={{ marginBottom: '10px' }}>
        <div>Node Opacity: {nodeOpacity?.toFixed(3) || 'Loading...'}</div>
        <div>Spring K: {physicsSettings.springK?.toFixed(3) || 'Loading...'}</div>
        <div>Repel K: {physicsSettings.repelK?.toFixed(3) || 'Loading...'}</div>
        <div>Damping: {physicsSettings.damping?.toFixed(3) || 'Loading...'}</div>
      </div>

      <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
        <button
          onClick={handleQuickUpdates}
          style={{
            background: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '5px 8px',
            cursor: 'pointer',
            fontSize: '10px'
          }}
        >
          Rapid Updates
        </button>
        <button
          onClick={handleBatchUpdate}
          style={{
            background: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '5px 8px',
            cursor: 'pointer',
            fontSize: '10px'
          }}
        >
          Batch Update
        </button>
      </div>
    </div>
  );
};

/**
 * Test component using traditional store hooks (baseline)
 */
const TraditionalHookTest: React.FC = () => {
  // Traditional approach - subscribes to entire settings object
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);

  // Extract values manually (causes re-render on any settings change)
  const nodeOpacity = settings?.visualisation?.nodes?.opacity;
  const springK = settings?.visualisation?.graphs?.logseq?.physics?.springK;
  const repelK = settings?.visualisation?.graphs?.logseq?.physics?.repelK;
  const damping = settings?.visualisation?.graphs?.logseq?.physics?.damping;

  const handleQuickUpdates = () => {
    // Each update will cause multiple re-renders across all components
    for (let i = 0; i < 10; i++) {
      setTimeout(() => {
        updateSettings((draft) => {
          if (draft.visualisation?.nodes) {
            draft.visualisation.nodes.opacity = Math.random();
          }
        });
      }, i * 50);
    }
  };

  const handleBatchUpdate = () => {
    updateSettings((draft) => {
      if (draft.visualisation?.graphs?.logseq?.physics) {
        draft.visualisation.graphs.logseq.physics.springK = Math.random() * 2;
        draft.visualisation.graphs.logseq.physics.repelK = Math.random() * 100;
        draft.visualisation.graphs.logseq.physics.damping = Math.random();
      }
    });
  };

  return (
    <div>
      <div style={{ color: '#FF5722', fontWeight: 'bold', marginBottom: '10px' }}>
        ⚠️ Using Traditional Store (Baseline)
      </div>
      
      <div style={{ marginBottom: '10px' }}>
        <div>Node Opacity: {nodeOpacity?.toFixed(3) || 'Loading...'}</div>
        <div>Spring K: {springK?.toFixed(3) || 'Loading...'}</div>
        <div>Repel K: {repelK?.toFixed(3) || 'Loading...'}</div>
        <div>Damping: {damping?.toFixed(3) || 'Loading...'}</div>
      </div>

      <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
        <button
          onClick={handleQuickUpdates}
          style={{
            background: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '5px 8px',
            cursor: 'pointer',
            fontSize: '10px'
          }}
        >
          Rapid Updates
        </button>
        <button
          onClick={handleBatchUpdate}
          style={{
            background: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '5px 8px',
            cursor: 'pointer',
            fontSize: '10px'
          }}
        >
          Batch Update
        </button>
      </div>
    </div>
  );
};