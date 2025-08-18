import React, { useEffect, useState } from 'react';
import { useSettingsStore } from '../../../store/settingsStore';

interface AutoBalanceIndicatorProps {
  style?: React.CSSProperties;
}

export const AutoBalanceIndicator: React.FC<AutoBalanceIndicatorProps> = ({ style }) => {
  const [isActive, setIsActive] = useState(false);
  const [lastNotificationTime, setLastNotificationTime] = useState(0);
  
  useEffect(() => {
    // Check if auto-balance is enabled
    const checkAutoBalance = () => {
      const settings = useSettingsStore.getState().settings;
      const autoBalanceEnabled = settings?.visualisation?.graphs?.logseq?.physics?.autoBalance;
      
      if (!autoBalanceEnabled) {
        setIsActive(false);
        return;
      }
      
      // Poll for recent notifications to determine if actively tuning
      fetch('/api/graph/auto-balance-notifications')
        .then(res => res.json())
        .then(data => {
          if (data.success && data.notifications && data.notifications.length > 0) {
            const latestNotification = data.notifications[data.notifications.length - 1];
            const now = Date.now();
            const timeSinceLastNotification = now - latestNotification.timestamp;
            
            // Consider active if there was a notification in the last 5 seconds
            setIsActive(timeSinceLastNotification < 5000);
            
            // If it's a "stable" notification, stop showing as active after 2 seconds
            if (latestNotification.severity === 'success' && timeSinceLastNotification < 2000) {
              setTimeout(() => setIsActive(false), 2000);
            }
          }
        })
        .catch(err => console.error('Failed to check auto-balance status:', err));
    };
    
    // Check immediately and then every 2 seconds
    checkAutoBalance();
    const interval = setInterval(checkAutoBalance, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  if (!isActive) return null;
  
  return (
    <div 
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        marginLeft: '8px',
        ...style
      }}
    >
      <style>
        {`
          @keyframes pulse-scale {
            0%, 100% {
              transform: scale(1);
              opacity: 1;
            }
            50% {
              transform: scale(1.2);
              opacity: 0.8;
            }
          }
          
          @keyframes rotate-smooth {
            from {
              transform: rotate(0deg);
            }
            to {
              transform: rotate(360deg);
            }
          }
          
          .auto-balance-icon {
            animation: pulse-scale 2s ease-in-out infinite, rotate-smooth 4s linear infinite;
            color: #F1C40F;
            font-size: 14px;
            display: inline-block;
            transform-origin: center;
          }
          
          .auto-balance-dot {
            width: 6px;
            height: 6px;
            background: #F1C40F;
            border-radius: 50%;
            animation: pulse-scale 1s ease-in-out infinite;
            margin: 0 2px;
          }
        `}
      </style>
      <span className="auto-balance-icon">⚖️</span>
      <div style={{ display: 'flex', marginLeft: '4px' }}>
        <div className="auto-balance-dot" style={{ animationDelay: '0s' }} />
        <div className="auto-balance-dot" style={{ animationDelay: '0.3s' }} />
        <div className="auto-balance-dot" style={{ animationDelay: '0.6s' }} />
      </div>
    </div>
  );
};