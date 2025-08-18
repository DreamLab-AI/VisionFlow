import { useEffect, useRef } from 'react';
import { toast } from '../utils/toast';
import { useSettingsStore } from '../store/settingsStore';

interface AutoBalanceNotification {
  message: string;
  timestamp: number;
  severity: 'info' | 'warning' | 'success';
}

export function useAutoBalanceNotifications() {
  const lastTimestampRef = useRef<number>(Date.now());
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  useEffect(() => {
    // Only poll if auto-balance is enabled
    const checkAndPoll = async () => {
      const settings = useSettingsStore.getState().settings;
      const autoBalanceEnabled = settings?.visualisation?.graphs?.logseq?.physics?.autoBalance;
      
      if (!autoBalanceEnabled) {
        // Stop polling if auto-balance is disabled
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        return;
      }
      
      // Start polling if not already polling
      if (!pollingIntervalRef.current) {
        pollingIntervalRef.current = setInterval(async () => {
          try {
            const response = await fetch(`/api/graph/auto-balance-notifications?since=${lastTimestampRef.current}`);
            if (response.ok) {
              const data = await response.json();
              if (data.success && data.notifications && data.notifications.length > 0) {
                // Process new notifications
                data.notifications.forEach((notification: AutoBalanceNotification) => {
                  // Show toast based on severity
                  switch (notification.severity) {
                    case 'success':
                      toast.success(notification.message, { duration: 5000 });
                      break;
                    case 'warning':
                      toast.warning(notification.message, { duration: 4000 });
                      break;
                    default:
                      toast.info(notification.message, { duration: 3000 });
                  }
                  
                  // Update last timestamp
                  if (notification.timestamp > lastTimestampRef.current) {
                    lastTimestampRef.current = notification.timestamp;
                  }
                });
              }
            }
          } catch (error) {
            console.error('Failed to fetch auto-balance notifications:', error);
          }
        }, 2000); // Poll every 2 seconds
      }
    };
    
    // Check settings and start polling if needed
    checkAndPoll();
    
    // Subscribe to settings changes
    const unsubscribe = useSettingsStore.subscribe((state) => {
      checkAndPoll();
    });
    
    // Cleanup on unmount
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      unsubscribe();
    };
  }, []);
}