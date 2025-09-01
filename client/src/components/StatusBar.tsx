import React, { useState, useEffect } from 'react';
import { useSelectiveSetting } from '../hooks/useSelectiveSettingsStore';
import { 
  Wifi, 
  WifiOff, 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Zap,
  Users,
  Database,
  Settings,
  Eye,
  EyeOff
} from 'lucide-react';

/**
 * Status bar component props
 */
interface StatusBarProps {
  className?: string;
  position?: 'top' | 'bottom';
  showNetworkStatus?: boolean;
  showPerformanceStatus?: boolean;
  showSystemStatus?: boolean;
  showUserStatus?: boolean;
  collapsible?: boolean;
}

/**
 * Status bar component with selective settings access
 * Provides real-time system status information
 */
export function StatusBar({
  className = '',
  position = 'bottom',
  showNetworkStatus = true,
  showPerformanceStatus = true,
  showSystemStatus = true,
  showUserStatus = true,
  collapsible = true
}: StatusBarProps) {
  // Settings
  const statusBarEnabled = useSelectiveSetting<boolean>('ui.statusBar.enabled');
  const statusBarCompact = useSelectiveSetting<boolean>('ui.statusBar.compact');
  const theme = useSelectiveSetting<string>('ui.theme');
  const debugMode = useSelectiveSetting<boolean>('system.debug');
  
  // Network and performance settings
  const networkMonitoring = useSelectiveSetting<boolean>('system.network.monitoring');
  const performanceMonitoring = useSelectiveSetting<boolean>('system.performance.monitoring');
  const frameRate = useSelectiveSetting<number>('system.performance.rendering.frameRate');
  
  // Component state
  const [isCollapsed, setIsCollapsed] = useState(statusBarCompact || false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [connectionStatus, setConnectionStatus] = useState<'online' | 'offline' | 'unstable'>('online');
  const [systemLoad, setSystemLoad] = useState(0);
  const [activeUsers, setActiveUsers] = useState(1);
  const [dataSync, setDataSync] = useState<'synced' | 'syncing' | 'error'>('synced');
  const [latency, setLatency] = useState(0);
  
  // Mock latency calculation
  useEffect(() => {
    if (!networkMonitoring) return;
    
    const updateLatency = () => {
      setLatency(Math.random() * 100 + 20); // 20-120ms
    };
    
    updateLatency();
    const interval = setInterval(updateLatency, 5000);
    
    return () => clearInterval(interval);
  }, [networkMonitoring]);
  
  // Update current time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    
    return () => clearInterval(timer);
  }, []);
  
  // Monitor connection status
  useEffect(() => {
    const updateConnectionStatus = () => {
      if (!navigator.onLine) {
        setConnectionStatus('offline');
      } else if (latency && latency > 200) {
        setConnectionStatus('unstable');
      } else {
        setConnectionStatus('online');
      }
    };
    
    updateConnectionStatus();
    
    const interval = setInterval(updateConnectionStatus, 5000);
    
    window.addEventListener('online', updateConnectionStatus);
    window.addEventListener('offline', updateConnectionStatus);
    
    return () => {
      clearInterval(interval);
      window.removeEventListener('online', updateConnectionStatus);
      window.removeEventListener('offline', updateConnectionStatus);
    };
  }, [latency]);
  
  // Simulate system load and other metrics
  useEffect(() => {
    const updateSystemMetrics = () => {
      // Simulate system load (0-100%)
      setSystemLoad(Math.random() * 100);
      
      // Simulate active users (1-5)
      setActiveUsers(Math.floor(Math.random() * 5) + 1);
      
      // Simulate data sync status
      const syncStates: ('synced' | 'syncing' | 'error')[] = ['synced', 'syncing', 'error'];
      const weights = [0.8, 0.15, 0.05]; // 80% synced, 15% syncing, 5% error
      const random = Math.random();
      let currentWeight = 0;
      
      for (let i = 0; i < syncStates.length; i++) {
        currentWeight += weights[i];
        if (random <= currentWeight) {
          setDataSync(syncStates[i]);
          break;
        }
      }
    };
    
    updateSystemMetrics();
    const interval = setInterval(updateSystemMetrics, 10000); // Update every 10 seconds
    
    return () => clearInterval(interval);
  }, []);
  
  // Don't render if disabled
  if (!statusBarEnabled) {
    return null;
  }
  
  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'online':
        return <Wifi className="w-4 h-4 text-green-500" />;
      case 'unstable':
        return <Wifi className="w-4 h-4 text-yellow-500" />;
      case 'offline':
        return <WifiOff className="w-4 h-4 text-red-500" />;
    }
  };
  
  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'online':
        return latency ? `${Math.round(latency)}ms` : 'Online';
      case 'unstable':
        return 'Unstable';
      case 'offline':
        return 'Offline';
    }
  };
  
  const getSystemLoadColor = () => {
    if (systemLoad < 50) return 'text-green-500';
    if (systemLoad < 80) return 'text-yellow-500';
    return 'text-red-500';
  };
  
  const getDataSyncIcon = () => {
    switch (dataSync) {
      case 'synced':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'syncing':
        return <Clock className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-red-500" />;
    }
  };
  
  const getDataSyncText = () => {
    switch (dataSync) {
      case 'synced':
        return 'Synced';
      case 'syncing':
        return 'Syncing...';
      case 'error':
        return 'Sync Error';
    }
  };
  
  const positionClasses = {
    top: 'top-0 border-b',
    bottom: 'bottom-0 border-t'
  };
  
  const themeClasses = {
    light: 'bg-white/90 backdrop-blur-sm border-gray-200 text-gray-800',
    dark: 'bg-gray-900/90 backdrop-blur-sm border-gray-700 text-gray-200',
    system: theme === 'dark' ? 'bg-gray-900/90 backdrop-blur-sm border-gray-700 text-gray-200' : 'bg-white/90 backdrop-blur-sm border-gray-200 text-gray-800'
  };
  
  const currentThemeClass = themeClasses[theme as keyof typeof themeClasses] || themeClasses.system;
  
  return (
    <div
      className={`
        fixed left-0 right-0 z-40
        ${positionClasses[position]}
        ${currentThemeClass}
        ${isCollapsed ? 'h-8' : 'h-12'}
        transition-all duration-200
        ${className}
      `}
    >
      <div className="h-full px-4 flex items-center justify-between">
        {/* Left Section - System Status */}
        <div className="flex items-center gap-4">
          {/* Data Sync Status */}
          {showSystemStatus && (
            <div className="flex items-center gap-2 text-sm">
              {getDataSyncIcon()}
              {!isCollapsed && <span>{getDataSyncText()}</span>}
            </div>
          )}
          
          {/* Network Status */}
          {showNetworkStatus && networkMonitoring && (
            <div className="flex items-center gap-2 text-sm">
              {getConnectionIcon()}
              {!isCollapsed && <span>{getConnectionText()}</span>}
            </div>
          )}
          
          {/* Performance Status */}
          {showPerformanceStatus && performanceMonitoring && (
            <div className="flex items-center gap-2 text-sm">
              <Activity className={`w-4 h-4 ${getSystemLoadColor()}`} />
              {!isCollapsed && (
                <span className={getSystemLoadColor()}>
                  {Math.round(systemLoad)}%
                </span>
              )}
            </div>
          )}
        </div>
        
        {/* Center Section - Additional Info */}
        {!isCollapsed && (
          <div className="flex items-center gap-4 text-sm text-gray-500">
            {/* Active Users */}
            {showUserStatus && (
              <div className="flex items-center gap-1">
                <Users className="w-4 h-4" />
                <span>{activeUsers}</span>
              </div>
            )}
            
            {/* Frame Rate (if performance monitoring enabled) */}
            {showPerformanceStatus && frameRate && (
              <div className="flex items-center gap-1">
                <Zap className="w-4 h-4" />
                <span>{frameRate}fps</span>
              </div>
            )}
            
            {/* Debug Mode Indicator */}
            {debugMode && (
              <div className="flex items-center gap-1 text-orange-500">
                <Settings className="w-4 h-4" />
                <span>DEBUG</span>
              </div>
            )}
          </div>
        )}
        
        {/* Right Section - Time and Controls */}
        <div className="flex items-center gap-4">
          {/* Current Time */}
          <div className="text-sm font-mono">
            {currentTime.toLocaleTimeString()}
          </div>
          
          {/* Collapse Toggle */}
          {collapsible && (
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              title={isCollapsed ? 'Expand status bar' : 'Collapse status bar'}
            >
              {isCollapsed ? (
                <Eye className="w-4 h-4" />
              ) : (
                <EyeOff className="w-4 h-4" />
              )}
            </button>
          )}
        </div>
      </div>
      
      {/* Progress indicator for sync operations */}
      {dataSync === 'syncing' && (
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-500/20">
          <div className="h-full bg-blue-500 animate-pulse" style={{ width: '60%' }} />
        </div>
      )}
    </div>
  );
}

/**
 * Compact status bar for minimal UI
 */
export function CompactStatusBar({ className = '' }: { className?: string }) {
  return (
    <StatusBar
      className={className}
      position="bottom"
      collapsible={false}
      showUserStatus={false}
      showSystemStatus={false}
    />
  );
}

/**
 * Development status bar with debug information
 */
export function DevStatusBar({ className = '' }: { className?: string }) {
  const debugMode = useSelectiveSetting<boolean>('system.debug');
  
  if (!debugMode) {
    return null;
  }
  
  return (
    <StatusBar
      className={`border-orange-500/20 bg-orange-50/5 ${className}`}
      position="top"
      showNetworkStatus={true}
      showPerformanceStatus={true}
      showSystemStatus={true}
      showUserStatus={true}
    />
  );
}