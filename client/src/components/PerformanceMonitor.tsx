import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useSelectiveSetting } from '../hooks/useSelectiveSettingsStore';
import { Activity, TrendingUp, TrendingDown, Zap, Monitor, Memory, Wifi } from 'lucide-react';

/**
 * Performance metrics interface
 */
interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  memoryUsed: number;
  memoryLimit: number;
  networkLatency: number;
  activeConnections: number;
  renderTime: number;
  updateTime: number;
  drawCalls: number;
  timestamp: number;
}

/**
 * Performance monitoring component props
 */
interface PerformanceMonitorProps {
  className?: string;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  minimizable?: boolean;
  showNetworkStats?: boolean;
  showRenderStats?: boolean;
  showMemoryStats?: boolean;
  updateInterval?: number;
  historyLength?: number;
}

/**
 * Performance monitoring component with selective settings access
 */
export function PerformanceMonitor({
  className = '',
  position = 'top-right',
  minimizable = true,
  showNetworkStats = true,
  showRenderStats = true,
  showMemoryStats = true,
  updateInterval = 1000,
  historyLength = 60
}: PerformanceMonitorProps) {
  // Use selective settings hooks
  const performanceEnabled = useSelectiveSetting<boolean>('system.performance.monitoring');
  const metricsInterval = useSelectiveSetting<number>('system.performance.metricsInterval');
  const fpsCounter = useSelectiveSetting<boolean>('system.performance.showFPS');
  const memoryUsage = useSelectiveSetting<boolean>('system.performance.showMemory');
  const networkLatency = useSelectiveSetting<boolean>('system.performance.showNetworkLatency');
  const debugMode = useSelectiveSetting<boolean>('system.debug');
  
  // Component state
  const [isMinimized, setIsMinimized] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    frameTime: 0,
    memoryUsed: 0,
    memoryLimit: 0,
    networkLatency: 0,
    activeConnections: 0,
    renderTime: 0,
    updateTime: 0,
    drawCalls: 0,
    timestamp: Date.now()
  });
  
  // Performance history for trends
  const [metricsHistory, setMetricsHistory] = useState<PerformanceMetrics[]>([]);
  const performanceObserverRef = useRef<PerformanceObserver>();
  const intervalRef = useRef<NodeJS.Timeout>();
  const frameTimeRef = useRef<number>();
  const fpsCounterRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());
  
  // Determine if monitor should be visible
  useEffect(() => {
    setIsVisible(Boolean(performanceEnabled && (debugMode || fpsCounter || memoryUsage || networkLatency)));
  }, [performanceEnabled, debugMode, fpsCounter, memoryUsage, networkLatency]);
  
  // FPS and frame time calculation
  const calculateFrameMetrics = useCallback(() => {
    const now = performance.now();
    const deltaTime = now - lastFrameTimeRef.current;
    lastFrameTimeRef.current = now;
    
    // Simple FPS calculation
    fpsCounterRef.current++;
    const fps = Math.round(1000 / deltaTime);
    
    return {
      fps: Math.min(fps, 120), // Cap at 120 FPS for display
      frameTime: deltaTime
    };
  }, []);
  
  // Memory metrics calculation
  const calculateMemoryMetrics = useCallback(() => {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        memoryUsed: Math.round(memory.usedJSHeapSize / (1024 * 1024)),
        memoryLimit: Math.round(memory.jsHeapSizeLimit / (1024 * 1024))
      };
    }
    
    return {
      memoryUsed: 0,
      memoryLimit: 0
    };
  }, []);
  
  // Network metrics calculation (simplified)
  const calculateNetworkMetrics = useCallback(() => {
    // In a real implementation, you'd measure actual network performance
    // For now, return mock values
    return {
      networkLatency: Math.random() * 100 + 20, // 20-120ms
      activeConnections: navigator.onLine ? 1 : 0
    };
  }, []);
  
  // Render metrics calculation (simplified)
  const calculateRenderMetrics = useCallback(() => {
    // In a real implementation, you'd integrate with WebGL context
    return {
      renderTime: Math.random() * 5 + 1, // 1-6ms
      updateTime: Math.random() * 3 + 0.5, // 0.5-3.5ms
      drawCalls: Math.floor(Math.random() * 50) + 10 // 10-60 draw calls
    };
  }, []);
  
  // Collect all performance metrics
  const collectMetrics = useCallback((): PerformanceMetrics => {
    const frameMetrics = calculateFrameMetrics();
    const memoryMetrics = calculateMemoryMetrics();
    const networkMetrics = calculateNetworkMetrics();
    const renderMetrics = calculateRenderMetrics();
    
    return {
      ...frameMetrics,
      ...memoryMetrics,
      ...networkMetrics,
      ...renderMetrics,
      timestamp: Date.now()
    };
  }, [calculateFrameMetrics, calculateMemoryMetrics, calculateNetworkMetrics, calculateRenderMetrics]);
  
  // Update metrics periodically
  useEffect(() => {
    if (!isVisible) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      return;
    }
    
    const interval = metricsInterval || updateInterval;
    
    intervalRef.current = setInterval(() => {
      const newMetrics = collectMetrics();
      setMetrics(newMetrics);
      
      // Update history
      setMetricsHistory(prev => {
        const updated = [...prev, newMetrics];
        return updated.slice(-historyLength);
      });
    }, interval);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isVisible, metricsInterval, updateInterval, historyLength, collectMetrics]);
  
  // Performance observer for additional metrics
  useEffect(() => {
    if (!isVisible || !window.PerformanceObserver) {
      return;
    }
    
    try {
      performanceObserverRef.current = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => {
          // Process performance entries (navigation, resource, measure, etc.)
          if (entry.entryType === 'measure') {
            // Handle custom performance measurements
          }
        });
      });
      
      performanceObserverRef.current.observe({
        entryTypes: ['measure', 'navigation', 'resource']
      });
    } catch (error) {
      console.warn('PerformanceObserver not fully supported', error);
    }
    
    return () => {
      if (performanceObserverRef.current) {
        performanceObserverRef.current.disconnect();
      }
    };
  }, [isVisible]);
  
  // Calculate performance trends
  const getPerformanceTrend = useCallback((metric: keyof PerformanceMetrics): 'up' | 'down' | 'stable' => {
    if (metricsHistory.length < 5) return 'stable';
    
    const recent = metricsHistory.slice(-5);
    const values = recent.map(m => Number(m[metric]) || 0);
    const trend = values[values.length - 1] - values[0];
    
    if (Math.abs(trend) < 2) return 'stable';
    return trend > 0 ? 'up' : 'down';
  }, [metricsHistory]);
  
  // Get performance color based on metric and its value
  const getPerformanceColor = useCallback((metric: keyof PerformanceMetrics, value: number): string => {
    switch (metric) {
      case 'fps':
        return value >= 50 ? '#22c55e' : value >= 30 ? '#f59e0b' : '#ef4444';
      case 'frameTime':
        return value <= 16.67 ? '#22c55e' : value <= 33.33 ? '#f59e0b' : '#ef4444';
      case 'memoryUsed':
        const memoryPercent = metrics.memoryLimit ? (value / metrics.memoryLimit) * 100 : 0;
        return memoryPercent <= 70 ? '#22c55e' : memoryPercent <= 85 ? '#f59e0b' : '#ef4444';
      case 'networkLatency':
        return value <= 50 ? '#22c55e' : value <= 100 ? '#f59e0b' : '#ef4444';
      default:
        return '#64748b';
    }
  }, [metrics.memoryLimit]);
  
  // Don't render if not visible
  if (!isVisible) {
    return null;
  }
  
  const positionClasses = {
    'top-left': 'top-4 left-4',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-right': 'bottom-4 right-4'
  };
  
  return (
    <div
      className={`
        fixed ${positionClasses[position]} z-50
        bg-black/90 backdrop-blur-sm border border-gray-700
        rounded-lg shadow-2xl font-mono text-sm
        ${isMinimized ? 'p-2' : 'p-4'}
        transition-all duration-200
        ${className}
      `}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-green-400">
          <Activity className="w-4 h-4" />
          <span className="font-semibold">Performance</span>
        </div>
        
        {minimizable && (
          <button
            onClick={() => setIsMinimized(!isMinimized)}
            className="text-gray-400 hover:text-white transition-colors p-1 rounded"
            title={isMinimized ? 'Expand' : 'Minimize'}
          >
            {isMinimized ? '⬆' : '⬇'}
          </button>
        )}
      </div>
      
      {/* Minimized View */}
      {isMinimized ? (
        <div className="flex items-center gap-4 text-xs">
          {fpsCounter && (
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3" style={{ color: getPerformanceColor('fps', metrics.fps) }} />
              <span>{metrics.fps} FPS</span>
            </div>
          )}
          
          {memoryUsage && (
            <div className="flex items-center gap-1">
              <Memory className="w-3 h-3" style={{ color: getPerformanceColor('memoryUsed', metrics.memoryUsed) }} />
              <span>{metrics.memoryUsed}MB</span>
            </div>
          )}
          
          {networkLatency && (
            <div className="flex items-center gap-1">
              <Wifi className="w-3 h-3" style={{ color: getPerformanceColor('networkLatency', metrics.networkLatency) }} />
              <span>{Math.round(metrics.networkLatency)}ms</span>
            </div>
          )}
        </div>
      ) : (
        /* Expanded View */
        <div className="space-y-3 min-w-64">
          {/* FPS and Frame Time */}
          {fpsCounter && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-xs">FPS</span>
                  {getPerformanceTrend('fps') === 'up' && <TrendingUp className="w-3 h-3 text-green-400" />}
                  {getPerformanceTrend('fps') === 'down' && <TrendingDown className="w-3 h-3 text-red-400" />}
                </div>
                <div className="text-lg font-bold" style={{ color: getPerformanceColor('fps', metrics.fps) }}>
                  {metrics.fps}
                </div>
              </div>
              
              <div>
                <span className="text-gray-400 text-xs">Frame Time</span>
                <div className="text-lg font-bold" style={{ color: getPerformanceColor('frameTime', metrics.frameTime) }}>
                  {metrics.frameTime.toFixed(1)}ms
                </div>
              </div>
            </div>
          )}
          
          {/* Memory Usage */}
          {memoryUsage && showMemoryStats && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-gray-400 text-xs flex items-center gap-1">
                  <Memory className="w-3 h-3" />
                  Memory
                </span>
                <span className="text-xs">{metrics.memoryUsed}MB / {metrics.memoryLimit}MB</span>
              </div>
              
              <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                <div
                  className="h-2 rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min((metrics.memoryUsed / metrics.memoryLimit) * 100, 100)}%`,
                    backgroundColor: getPerformanceColor('memoryUsed', metrics.memoryUsed)
                  }}
                />
              </div>
            </div>
          )}
          
          {/* Network Stats */}
          {networkLatency && showNetworkStats && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-gray-400 text-xs flex items-center gap-1">
                  <Wifi className="w-3 h-3" />
                  Latency
                </span>
                <div className="text-sm font-bold" style={{ color: getPerformanceColor('networkLatency', metrics.networkLatency) }}>
                  {Math.round(metrics.networkLatency)}ms
                </div>
              </div>
              
              <div>
                <span className="text-gray-400 text-xs">Connections</span>
                <div className="text-sm font-bold text-blue-400">
                  {metrics.activeConnections}
                </div>
              </div>
            </div>
          )}
          
          {/* Render Stats */}
          {showRenderStats && (
            <div className="grid grid-cols-3 gap-2 text-xs pt-2 border-t border-gray-700">
              <div>
                <span className="text-gray-400">Render</span>
                <div className="font-bold text-yellow-400">{metrics.renderTime.toFixed(1)}ms</div>
              </div>
              
              <div>
                <span className="text-gray-400">Update</span>
                <div className="font-bold text-cyan-400">{metrics.updateTime.toFixed(1)}ms</div>
              </div>
              
              <div>
                <span className="text-gray-400">Draws</span>
                <div className="font-bold text-purple-400">{metrics.drawCalls}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}