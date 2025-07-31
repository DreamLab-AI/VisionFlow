/**
 * Floating Activity Panels
 * Real-time performance dashboard and activity monitoring components
 */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { HologramOverlay } from '../HologramVisualisation';
import {
  FloatingPanelData,
  FloatingPanelProps,
  ActivityData,
  PerformanceMetrics,
  CoordinationPattern,
  SystemAlert,
  DashboardMetrics,
  MessageFlowData
} from './types';

// Base Floating Panel Component
export const FloatingPanel: React.FC<FloatingPanelProps & { children: React.ReactNode }> = ({
  data,
  children,
  onClose,
  onPin,
  onResize,
  onMove
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [position, setPosition] = useState(data.position);
  const [size, setSize] = useState(data.size);
  const panelRef = useRef<HTMLDivElement>(null);
  const dragStartRef = useRef({ x: 0, y: 0 });

  const handleMouseDown = (e: React.MouseEvent, action: 'drag' | 'resize') => {
    e.preventDefault();
    if (action === 'drag') {
      setIsDragging(true);
      dragStartRef.current = {
        x: e.clientX - position.x,
        y: e.clientY - position.y
      };
    } else {
      setIsResizing(true);
      dragStartRef.current = { x: e.clientX, y: e.clientY };
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const newPosition = {
          x: e.clientX - dragStartRef.current.x,
          y: e.clientY - dragStartRef.current.y
        };
        setPosition(newPosition);
        onMove?.(newPosition);
      } else if (isResizing) {
        const deltaX = e.clientX - dragStartRef.current.x;
        const deltaY = e.clientY - dragStartRef.current.y;
        const newSize = {
          width: Math.max(200, size.width + deltaX),
          height: Math.max(150, size.height + deltaY)
        };
        setSize(newSize);
        onResize?.(newSize);
        dragStartRef.current = { x: e.clientX, y: e.clientY };
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setIsResizing(false);
    };

    if (isDragging || isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, isResizing, position, size, onMove, onResize]);

  return (
    <div
      ref={panelRef}
      className={`fixed z-50 ${data.pinned ? 'opacity-100' : 'opacity-90 hover:opacity-100'} 
                  transition-opacity duration-200`}
      style={{
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
        cursor: isDragging ? 'grabbing' : 'default'
      }}
    >
      <HologramOverlay className="w-full h-full">
        {/* Panel Header */}
        <div 
          className="flex items-center justify-between p-2 border-b border-cyan-500/30 cursor-grab active:cursor-grabbing"
          onMouseDown={(e) => handleMouseDown(e, 'drag')}
        >
          <h3 className="text-sm font-semibold text-cyan-300 capitalize">
            {data.type.replace('-', ' ')}
          </h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={onPin}
              className={`text-xs px-2 py-1 rounded ${
                data.pinned 
                  ? 'bg-cyan-500/30 text-cyan-300' 
                  : 'bg-transparent text-cyan-500 hover:bg-cyan-500/20'
              }`}
            >
              {data.pinned ? 'Unpin' : 'Pin'}
            </button>
            <button
              onClick={onClose}
              className="text-xs px-2 py-1 rounded bg-red-500/20 text-red-300 hover:bg-red-500/40"
            >
              ×
            </button>
          </div>
        </div>

        {/* Panel Content */}
        <div className="flex-1 overflow-auto p-2">
          {children}
        </div>

        {/* Resize Handle */}
        <div
          className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize opacity-50 hover:opacity-100"
          onMouseDown={(e) => handleMouseDown(e, 'resize')}
        >
          <div className="w-full h-full bg-gradient-to-br from-transparent to-cyan-500/50" />
        </div>
      </HologramOverlay>
    </div>
  );
};

// Performance Dashboard Panel
export const PerformanceDashboardPanel: React.FC<{
  metrics: DashboardMetrics;
  panelData: FloatingPanelData;
  onClose?: () => void;
  onPin?: () => void;
}> = ({ metrics, panelData, onClose, onPin }) => {
  return (
    <FloatingPanel data={panelData} onClose={onClose} onPin={onPin}>
      <div className="space-y-4">
        {/* System Overview */}
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="Active Agents"
            value={metrics.overview.activeAgents}
            total={metrics.overview.totalAgents}
            color="cyan"
          />
          <MetricCard
            label="Avg Response"
            value={`${Math.round(metrics.overview.averageResponseTime)}ms`}
            color="green"
          />
        </div>

        {/* Network Health */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-cyan-400">Network Health</h4>
          <div className="grid grid-cols-1 gap-2">
            <ProgressBar
              label="Connectivity"
              value={metrics.networkHealth.connectivity}
              color="blue"
            />
            <ProgressBar
              label="Throughput"
              value={metrics.networkHealth.throughput}
              color="green"
            />
            <ProgressBar
              label="Error Rate"
              value={1 - metrics.networkHealth.errorRate}
              color="red"
              inverted
            />
          </div>
        </div>

        {/* Top Performers */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-cyan-400">Top Performers</h4>
          <div className="space-y-1">
            {Array.from(metrics.agentPerformance.entries())
              .sort(([,a], [,b]) => b.successRate - a.successRate)
              .slice(0, 3)
              .map(([agentId, performance]) => (
                <div key={agentId} className="flex justify-between text-xs">
                  <span className="text-cyan-300 truncate">{agentId.split(':')[1]}</span>
                  <span className="text-green-400">{Math.round(performance.successRate * 100)}%</span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </FloatingPanel>
  );
};

// Coordination Activity Panel
export const CoordinationActivityPanel: React.FC<{
  patterns: CoordinationPattern[];
  panelData: FloatingPanelData;
  onClose?: () => void;
  onPin?: () => void;
}> = ({ patterns, panelData, onClose, onPin }) => {
  const activePatterns = patterns.filter(p => p.status === 'active');
  const formingPatterns = patterns.filter(p => p.status === 'forming');

  return (
    <FloatingPanel data={panelData} onClose={onClose} onPin={onPin}>
      <div className="space-y-4">
        {/* Active Patterns Summary */}
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="Active Patterns"
            value={activePatterns.length}
            color="cyan"
          />
          <MetricCard
            label="Forming"
            value={formingPatterns.length}
            color="yellow"
          />
        </div>

        {/* Pattern List */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-cyan-400">Current Patterns</h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {patterns.slice(0, 5).map((pattern) => (
              <div key={pattern.id} className="p-2 bg-cyan-900/20 rounded border border-cyan-500/30">
                <div className="flex justify-between items-start mb-1">
                  <span className="text-xs font-medium text-cyan-300 capitalize">
                    {pattern.type}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded ${getStatusColor(pattern.status)}`}>
                    {pattern.status}
                  </span>
                </div>
                <div className="text-xs text-cyan-400">
                  {pattern.participants.length} participants
                </div>
                <ProgressBar
                  label="Progress"
                  value={pattern.progress}
                  color="blue"
                  showPercentage
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </FloatingPanel>
  );
};

// Message Flow Activity Panel
export const MessageFlowPanel: React.FC<{
  messages: MessageFlowData[];
  panelData: FloatingPanelData;
  onClose?: () => void;
  onPin?: () => void;
}> = ({ messages, panelData, onClose, onPin }) => {
  const recentMessages = useMemo(() => 
    messages
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 10),
    [messages]
  );

  const messageStats = useMemo(() => {
    const now = Date.now();
    const lastMinute = messages.filter(m => now - m.timestamp.getTime() < 60000);
    const byType = lastMinute.reduce((acc, msg) => {
      acc[msg.type] = (acc[msg.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      total: lastMinute.length,
      byType,
      averageLatency: lastMinute.reduce((sum, msg) => sum + (msg.latency || 0), 0) / lastMinute.length || 0
    };
  }, [messages]);

  return (
    <FloatingPanel data={panelData} onClose={onClose} onPin={onPin}>
      <div className="space-y-4">
        {/* Message Statistics */}
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="Messages/min"
            value={messageStats.total}
            color="cyan"
          />
          <MetricCard
            label="Avg Latency"
            value={`${Math.round(messageStats.averageLatency)}ms`}
            color="green"
          />
        </div>

        {/* Message Types Breakdown */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-cyan-400">Message Types</h4>
          <div className="space-y-1">
            {Object.entries(messageStats.byType).map(([type, count]) => (
              <div key={type} className="flex justify-between text-xs">
                <span className="text-cyan-300 capitalize">{type}</span>
                <span className="text-cyan-400">{count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Messages */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-cyan-400">Recent Messages</h4>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {recentMessages.map((message) => (
              <div key={message.id} className="text-xs p-1 bg-cyan-900/10 rounded">
                <div className="flex justify-between">
                  <span className={`capitalize ${getPriorityColor(message.priority)}`}>
                    {message.type}
                  </span>
                  <span className="text-cyan-500">
                    {formatTime(message.timestamp)}
                  </span>
                </div>
                <div className="text-cyan-400 truncate">
                  {message.content.topic}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </FloatingPanel>
  );
};

// System Health Alert Panel
export const SystemHealthPanel: React.FC<{
  alerts: SystemAlert[];
  systemMetrics: ActivityData['systemMetrics'];
  panelData: FloatingPanelData;
  onClose?: () => void;
  onPin?: () => void;
}> = ({ alerts, systemMetrics, panelData, onClose, onPin }) => {
  const criticalAlerts = alerts.filter(a => a.level === 'critical' && !a.acknowledged);
  const warningAlerts = alerts.filter(a => a.level === 'warning' && !a.acknowledged);

  return (
    <FloatingPanel data={panelData} onClose={onClose} onPin={onPin}>
      <div className="space-y-4">
        {/* System Status */}
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="System Load"
            value={`${Math.round(systemMetrics.systemLoad * 100)}%`}
            color={systemMetrics.systemLoad > 0.8 ? 'red' : 'green'}
          />
          <MetricCard
            label="Avg Latency"
            value={`${Math.round(systemMetrics.averageLatency)}ms`}
            color={systemMetrics.averageLatency > 1000 ? 'red' : 'green'}
          />
        </div>

        {/* Alert Summary */}
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="Critical"
            value={criticalAlerts.length}
            color="red"
          />
          <MetricCard
            label="Warnings"
            value={warningAlerts.length}
            color="yellow"
          />
        </div>

        {/* Recent Alerts */}
        {alerts.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-cyan-400">Recent Alerts</h4>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {alerts.slice(0, 5).map((alert) => (
                <div key={alert.id} className={`text-xs p-2 rounded border ${getAlertColor(alert.level)}`}>
                  <div className="flex justify-between items-start">
                    <span className="font-medium capitalize">{alert.level}</span>
                    <span className="text-xs opacity-75">
                      {formatTime(alert.timestamp)}
                    </span>
                  </div>
                  <div className="mt-1 text-xs opacity-90">
                    {alert.message}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </FloatingPanel>
  );
};

// Helper Components
const MetricCard: React.FC<{
  label: string;
  value: string | number;
  total?: number;
  color: string;
}> = ({ label, value, total, color }) => (
  <div className={`p-2 bg-${color}-900/20 rounded border border-${color}-500/30`}>
    <div className={`text-xs text-${color}-400`}>{label}</div>
    <div className={`text-sm font-semibold text-${color}-300`}>
      {value}{total ? `/${total}` : ''}
    </div>
  </div>
);

const ProgressBar: React.FC<{
  label: string;
  value: number;
  color: string;
  showPercentage?: boolean;
  inverted?: boolean;
}> = ({ label, value, color, showPercentage, inverted }) => {
  const percentage = Math.round(value * 100);
  const displayValue = inverted ? 100 - percentage : percentage;
  
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-cyan-400">{label}</span>
        {showPercentage && <span className="text-cyan-300">{displayValue}%</span>}
      </div>
      <div className={`w-full h-2 bg-${color}-900/30 rounded-full overflow-hidden`}>
        <div
          className={`h-full bg-${color}-500 transition-all duration-300`}
          style={{ width: `${displayValue}%` }}
        />
      </div>
    </div>
  );
};

// Helper Functions
function getStatusColor(status: string): string {
  const colors = {
    forming: 'bg-yellow-500/20 text-yellow-300',
    active: 'bg-green-500/20 text-green-300',
    completing: 'bg-blue-500/20 text-blue-300',
    dissolved: 'bg-gray-500/20 text-gray-300'
  };
  return colors[status as keyof typeof colors] || 'bg-gray-500/20 text-gray-300';
}

function getPriorityColor(priority: string): string {
  const colors = {
    urgent: 'text-red-400',
    high: 'text-orange-400',
    normal: 'text-cyan-400',
    low: 'text-gray-400'
  };
  return colors[priority as keyof typeof colors] || 'text-cyan-400';
}

function getAlertColor(level: string): string {
  const colors = {
    critical: 'bg-red-900/30 border-red-500/50 text-red-300',
    error: 'bg-red-900/20 border-red-500/30 text-red-400',
    warning: 'bg-yellow-900/20 border-yellow-500/30 text-yellow-400',
    info: 'bg-blue-900/20 border-blue-500/30 text-blue-400'
  };
  return colors[level as keyof typeof colors] || 'bg-gray-900/20 border-gray-500/30 text-gray-400';
}

function formatTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  
  if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  return `${Math.floor(diff / 3600000)}h ago`;
}

export default {
  FloatingPanel,
  PerformanceDashboardPanel,
  CoordinationActivityPanel,
  MessageFlowPanel,
  SystemHealthPanel
};