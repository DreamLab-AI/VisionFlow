import React, { useEffect, useState } from 'react';
import { AlertCircle, X, RefreshCw, Wifi, WifiOff, AlertTriangle, CheckCircle } from 'lucide-react';
import { Button } from '../features/design-system/components';
import { cn } from '../utils/classNameUtils';

export interface ErrorNotificationProps {
  type: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  detail?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  retry?: {
    onRetry: () => Promise<void>;
    maxRetries?: number;
  };
  dismissible?: boolean;
  autoClose?: number; 
  onClose?: () => void;
  className?: string;
}

export const ErrorNotification: React.FC<ErrorNotificationProps> = ({
  type = 'error',
  title,
  message,
  detail,
  action,
  retry,
  dismissible = true,
  autoClose,
  onClose,
  className
}) => {
  const [isVisible, setIsVisible] = useState(true);
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [showDetail, setShowDetail] = useState(false);

  useEffect(() => {
    if (autoClose && isVisible) {
      const timer = setTimeout(() => {
        handleClose();
      }, autoClose);
      return () => clearTimeout(timer);
    }
  }, [autoClose, isVisible]);

  const handleClose = () => {
    setIsVisible(false);
    onClose?.();
  };

  const handleRetry = async () => {
    if (!retry || isRetrying) return;
    
    const maxRetries = retry.maxRetries || 3;
    if (retryCount >= maxRetries) return;

    setIsRetrying(true);
    setRetryCount(prev => prev + 1);

    try {
      await retry.onRetry();
      handleClose(); 
    } catch (error) {
      
      console.error('Retry failed:', error);
    } finally {
      setIsRetrying(false);
    }
  };

  if (!isVisible) return null;

  const iconMap = {
    error: <AlertCircle className="h-5 w-5" />,
    warning: <AlertTriangle className="h-5 w-5" />,
    info: <AlertCircle className="h-5 w-5" />,
    success: <CheckCircle className="h-5 w-5" />
  };

  const colorMap = {
    error: 'bg-destructive/10 text-destructive border-destructive/20',
    warning: 'bg-yellow-500/10 text-yellow-700 border-yellow-500/20',
    info: 'bg-blue-500/10 text-blue-700 border-blue-500/20',
    success: 'bg-green-500/10 text-green-700 border-green-500/20'
  };

  const iconColorMap = {
    error: 'text-destructive',
    warning: 'text-yellow-600',
    info: 'text-blue-600',
    success: 'text-green-600'
  };

  return (
    <div
      className={cn(
        'relative rounded-lg border p-4 shadow-sm',
        colorMap[type],
        className
      )}
      role="alert"
      aria-live="assertive"
    >
      <div className="flex items-start gap-3">
        <div className={cn('flex-shrink-0 mt-0.5', iconColorMap[type])}>
          {iconMap[type]}
        </div>
        
        <div className="flex-1 space-y-2">
          <div className="flex items-center justify-between gap-2">
            <h3 className="font-semibold text-sm">{title}</h3>
            {dismissible && (
              <button
                onClick={handleClose}
                className="rounded-md p-1 hover:bg-black/5 transition-colors"
                aria-label="Close notification"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          
          <p className="text-sm opacity-90">{message}</p>
          
          {detail && (
            <div>
              <button
                onClick={() => setShowDetail(!showDetail)}
                className="text-xs underline opacity-70 hover:opacity-100"
              >
                {showDetail ? 'Hide' : 'Show'} details
              </button>
              {showDetail && (
                <pre className="mt-2 text-xs bg-black/5 rounded p-2 overflow-x-auto">
                  {detail}
                </pre>
              )}
            </div>
          )}
          
          <div className="flex items-center gap-2 mt-3">
            {retry && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleRetry}
                disabled={isRetrying || retryCount >= (retry.maxRetries || 3)}
                className="text-xs"
              >
                <RefreshCw className={cn('h-3 w-3 mr-1', isRetrying && 'animate-spin')} />
                {isRetrying ? 'Retrying...' : `Retry (${retryCount}/${retry.maxRetries || 3})`}
              </Button>
            )}
            
            {action && (
              <Button
                size="sm"
                variant="outline"
                onClick={action.onClick}
                className="text-xs"
              >
                {action.label}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Connection status notification component
export const ConnectionStatusNotification: React.FC<{
  isConnected: boolean;
  isReconnecting?: boolean;
  reconnectAttempts?: number;
  maxReconnectAttempts?: number;
  onRetry?: () => void;
}> = ({
  isConnected,
  isReconnecting,
  reconnectAttempts = 0,
  maxReconnectAttempts = 10,
  onRetry
}) => {
  if (isConnected) return null;

  return (
    <ErrorNotification
      type="warning"
      title={isReconnecting ? "Reconnecting..." : "Connection Lost"}
      message={
        isReconnecting
          ? `Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})`
          : "Unable to connect to the server"
      }
      action={
        !isReconnecting && onRetry
          ? { label: "Retry Now", onClick: onRetry }
          : undefined
      }
      dismissible={false}
      className="fixed bottom-4 right-4 max-w-md z-50"
    />
  );
};

// Settings sync error notification
export const SettingsSyncErrorNotification: React.FC<{
  failedPaths: string[];
  onRetry: () => Promise<void>;
  onDismiss: () => void;
}> = ({ failedPaths, onRetry, onDismiss }) => {
  const pathsDisplay = failedPaths.length > 3 
    ? `${failedPaths.slice(0, 3).join(', ')} and ${failedPaths.length - 3} more`
    : failedPaths.join(', ');

  return (
    <ErrorNotification
      type="error"
      title="Settings Sync Failed"
      message={`Failed to update settings: ${pathsDisplay}`}
      detail={failedPaths.join('\n')}
      retry={{ onRetry, maxRetries: 3 }}
      onClose={onDismiss}
      className="fixed top-20 right-4 max-w-md z-50"
    />
  );
};

// Validation error notification
export const ValidationErrorNotification: React.FC<{
  field: string;
  error: string;
  onDismiss: () => void;
}> = ({ field, error, onDismiss }) => {
  return (
    <ErrorNotification
      type="error"
      title="Validation Error"
      message={`${field}: ${error}`}
      onClose={onDismiss}
      autoClose={5000}
      className="fixed top-20 right-4 max-w-md z-50"
    />
  );
};