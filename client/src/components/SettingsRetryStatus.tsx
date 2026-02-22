import React, { useState, useEffect } from 'react';
import { RefreshCw, AlertTriangle, X, CheckCircle } from 'lucide-react';
import { Button } from '../features/design-system/components';
import { SettingsRetryManager } from '../store/settingsRetryManager';
import { cn } from '../utils/classNameUtils';
import { useToast } from '../features/design-system/components/Toast';

interface SettingsRetryStatusProps {
  retryManager: SettingsRetryManager;
  className?: string;
}

export const SettingsRetryStatus: React.FC<SettingsRetryStatusProps> = ({
  retryManager,
  className
}) => {
  const [status, setStatus] = useState(retryManager.getRetryStatus());
  const [isRetrying, setIsRetrying] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const { toast } = useToast();
  
  useEffect(() => {
    
    const interval = setInterval(() => {
      setStatus(retryManager.getRetryStatus());
    }, 2000);
    
    
    const handleRetryFailed = (event: CustomEvent) => {
      const { path, error } = event.detail;
      toast({
        title: 'Settings Update Failed',
        description: `Failed to update ${path}: ${error}`,
        variant: 'destructive',
      });
    };
    
    window.addEventListener('settings-retry-failed', handleRetryFailed as EventListener);
    
    return () => {
      clearInterval(interval);
      window.removeEventListener('settings-retry-failed', handleRetryFailed as EventListener);
    };
  }, [retryManager, toast]);
  
  const handleRetryPath = async (path: string) => {
    setIsRetrying(path);
    try {
      const success = await retryManager.retryPath(path);
      if (success) {
        toast({
          title: 'Success',
          description: `Settings updated successfully`,
          variant: 'default',
        });
      } else {
        toast({
          title: 'Retry Failed',
          description: `Failed to update settings. Will retry automatically.`,
          variant: 'destructive',
        });
      }
    } finally {
      setIsRetrying(null);
    }
  };
  
  const handleClearQueue = () => {
    retryManager.clearRetryQueue();
    setStatus(retryManager.getRetryStatus());
    toast({
      title: 'Queue Cleared',
      description: 'Retry queue has been cleared',
      variant: 'default',
    });
  };
  
  if (status.queueSize === 0) return null;
  
  return (
    <div
      className={cn(
        'fixed bottom-4 left-4 max-w-md bg-background border rounded-lg shadow-lg p-4',
        className
      )}
    >
      {}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-yellow-600" />
          <h3 className="font-semibold text-sm">
            Pending Settings Updates ({status.queueSize})
          </h3>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          {showDetails ? 'Hide' : 'Show'} Details
        </button>
      </div>
      
      {}
      <p className="text-sm text-muted-foreground mb-3">
        {status.queueSize} setting{status.queueSize > 1 ? 's' : ''} failed to save and will retry automatically.
      </p>
      
      {}
      {showDetails && (
        <div className="space-y-2 mb-3 max-h-40 overflow-y-auto">
          {status.items.map((item) => (
            <div
              key={item.path}
              className="flex items-center justify-between gap-2 p-2 bg-muted/50 rounded text-xs"
            >
              <div className="flex-1 min-w-0">
                <p className="font-mono truncate">{item.path}</p>
                {item.error && (
                  <p className="text-destructive mt-1">{item.error}</p>
                )}
                <p className="text-muted-foreground mt-1">
                  Attempts: {item.attempts}/3
                </p>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handleRetryPath(item.path)}
                disabled={isRetrying === item.path}
                className="flex-shrink-0"
              >
                <RefreshCw 
                  className={cn(
                    'h-3 w-3',
                    isRetrying === item.path && 'animate-spin'
                  )} 
                />
              </Button>
            </div>
          ))}
        </div>
      )}
      
      {}
      <div className="flex gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={handleClearQueue}
          className="text-xs"
        >
          <X className="h-3 w-3 mr-1" />
          Clear Queue
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => window.location.reload()}
          className="text-xs"
        >
          <RefreshCw className="h-3 w-3 mr-1" />
          Reload Page
        </Button>
      </div>
    </div>
  );
};

// Global retry status component that can be added to app root
export const GlobalSettingsRetryStatus: React.FC = () => {
  const [retryManager] = useState(() => new SettingsRetryManager());
  
  useEffect(() => {
    
    const devWindow = window as unknown as Record<string, unknown>;
    devWindow.__settingsRetryManager = retryManager;

    return () => {
      retryManager.stopRetryProcessor();
      delete devWindow.__settingsRetryManager;
    };
  }, [retryManager]);
  
  return <SettingsRetryStatus retryManager={retryManager} />;
};