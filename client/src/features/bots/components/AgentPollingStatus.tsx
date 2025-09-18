import React from 'react';
import { useBotsData } from '../contexts/BotsDataContext';
import { Badge } from '../../design-system/components/Badge';
import { Progress } from '../../design-system/components/Progress';
import { createLogger } from '../../../utils/logger';

const logger = createLogger('AgentPollingStatus');

export const AgentPollingStatus: React.FC = () => {
  const { pollingStatus, configurePolling } = useBotsData();
  
  if (!pollingStatus) {
    return null;
  }

  const { isPolling, activityLevel, lastUpdate, error } = pollingStatus;
  const timeSinceUpdate = Date.now() - lastUpdate;
  const isStale = timeSinceUpdate > 10000; // Data is stale after 10s
  
  return (
    <div style={{
      position: 'absolute',
      top: '10px',
      right: '10px',
      background: 'rgba(0, 0, 0, 0.8)',
      padding: '12px 16px',
      borderRadius: '8px',
      color: 'white',
      fontSize: '12px',
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
      minWidth: '200px',
      zIndex: 1000,
      backdropFilter: 'blur(10px)'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontWeight: 'bold', fontSize: '14px' }}>Agent Network Status</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <div
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isPolling ? (error ? '#E74C3C' : '#2ECC71') : '#95A5A6',
              animation: isPolling && !error ? 'pulse 2s infinite' : 'none'
            }}
          />
          <span style={{ fontSize: '11px', opacity: 0.8 }}>
            {isPolling ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Activity Level */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ opacity: 0.7 }}>Activity:</span>
        <Badge
          variant={activityLevel === 'active' ? 'default' : 'secondary'}
          style={{
            backgroundColor: activityLevel === 'active' ? '#F39C12' : '#95A5A6',
            fontSize: '10px',
            padding: '2px 6px'
          }}
        >
          {activityLevel === 'active' ? 'Active' : 'Idle'}
        </Badge>
        <span style={{ fontSize: '10px', opacity: 0.6 }}>
          {activityLevel === 'active' ? '(1s interval)' : '(5s interval)'}
        </span>
      </div>

      {/* Last Update */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ opacity: 0.7 }}>Last Update:</span>
        <span style={{ 
          color: isStale ? '#E74C3C' : '#2ECC71',
          fontSize: '11px'
        }}>
          {timeSinceUpdate < 1000 ? 'just now' :
           timeSinceUpdate < 60000 ? `${Math.floor(timeSinceUpdate / 1000)}s ago` :
           `${Math.floor(timeSinceUpdate / 60000)}m ago`}
        </span>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          backgroundColor: 'rgba(231, 76, 60, 0.2)',
          border: '1px solid #E74C3C',
          borderRadius: '4px',
          padding: '6px 8px',
          fontSize: '11px',
          color: '#E74C3C'
        }}>
          Error: {error.message}
        </div>
      )}

      {/* Polling Controls */}
      <div style={{ 
        marginTop: '4px',
        paddingTop: '8px',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        display: 'flex',
        gap: '8px'
      }}>
        <button
          onClick={() => configurePolling?.({ activePollingInterval: 500 })}
          style={{
            flex: 1,
            padding: '4px 8px',
            fontSize: '11px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: 'white',
            cursor: 'pointer'
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.2)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)'}
        >
          Fast (0.5s)
        </button>
        <button
          onClick={() => configurePolling?.({ activePollingInterval: 1000, idlePollingInterval: 5000 })}
          style={{
            flex: 1,
            padding: '4px 8px',
            fontSize: '11px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: 'white',
            cursor: 'pointer'
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.2)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)'}
        >
          Normal
        </button>
        <button
          onClick={() => configurePolling?.({ activePollingInterval: 2000, idlePollingInterval: 10000 })}
          style={{
            flex: 1,
            padding: '4px 8px',
            fontSize: '11px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: 'white',
            cursor: 'pointer'
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.2)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)'}
        >
          Slow
        </button>
      </div>
    </div>
  );
};