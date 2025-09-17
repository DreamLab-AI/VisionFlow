import React, { useState, useMemo } from 'react';
import { Card } from '../../design-system/components/Card';
import { Tabs } from '../../design-system/components/Tabs';
import { Switch } from '../../design-system/components/Switch';
import { Label } from '../../design-system/components/Label';
import { Button } from '../../design-system/components/Button';
import { createLogger } from '../../../utils/logger';
import { useTelemetryData } from '../hooks/useTelemetryData';
import { AgentLifecyclePanel } from './AgentLifecyclePanel';
import { AgentPositionTracker } from './AgentPositionTracker';
import { MCPBridgeMonitor } from './MCPBridgeMonitor';
import { GPUMetricsPanel } from './GPUMetricsPanel';
import { TimelineView } from './TimelineView';
import { NetworkGraph } from './NetworkGraph';
import { PerformanceCharts } from './PerformanceCharts';
import { LogViewer } from './LogViewer';
import { AgentInspector } from './AgentInspector';
import { MessageFlowDebugger } from './MessageFlowDebugger';
import { ForceVectorVisualization } from './ForceVectorVisualization';

const logger = createLogger('TelemetryDashboard');

interface TelemetryDashboardProps {
  className?: string;
}

export const TelemetryDashboard: React.FC<TelemetryDashboardProps> = ({
  className = ''
}) => {
  const { data, isConnected, error, filters, setFilters, clearData, exportData } = useTelemetryData();
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Get unique agent IDs for filtering
  const agentIds = useMemo(() => {
    const ids = new Set<string>();
    data.lifecycleEvents.forEach(event => ids.add(event.agentId));
    Object.keys(data.agentPositions).forEach(id => ids.add(id));
    data.logEntries.forEach(entry => entry.agentId && ids.add(entry.agentId));
    return Array.from(ids).sort();
  }, [data]);

  const connectionStatus = isConnected ? 'Connected' : 'Disconnected';
  const connectionColor = isConnected ? 'text-green-600' : 'text-red-600';

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold">Hive Telemetry Dashboard</h1>
          <div className="flex items-center gap-2">
            <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
            <span className={`text-sm font-medium ${connectionColor}`}>
              {connectionStatus}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Label htmlFor="auto-refresh">Auto-refresh</Label>
            <Switch
              id="auto-refresh"
              checked={autoRefresh}
              onCheckedChange={setAutoRefresh}
            />
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => exportData('json')}
          >
            Export JSON
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => exportData('csv')}
          >
            Export CSV
          </Button>

          <Button
            variant="destructive"
            size="sm"
            onClick={clearData}
          >
            Clear Data
          </Button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mx-4 mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-center">
            <span className="text-red-600 mr-2">⚠️</span>
            <span className="text-red-800 text-sm">{error.message}</span>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="overview" className="h-full">
          <div className="px-4 pt-4">
            <Tabs.List>
              <Tabs.Trigger value="overview">Overview</Tabs.Trigger>
              <Tabs.Trigger value="lifecycle">Agent Lifecycle</Tabs.Trigger>
              <Tabs.Trigger value="network">Network & Communication</Tabs.Trigger>
              <Tabs.Trigger value="performance">Performance</Tabs.Trigger>
              <Tabs.Trigger value="debugging">Debugging Tools</Tabs.Trigger>
              <Tabs.Trigger value="logs">Logs & Analysis</Tabs.Trigger>
            </Tabs.List>
          </div>

          {/* Overview Tab */}
          <Tabs.Content value="overview" className="p-4 overflow-y-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              <AgentLifecyclePanel
                events={data.lifecycleEvents}
                className="lg:col-span-1 xl:col-span-2"
              />
              <MCPBridgeMonitor
                bridges={data.mcpBridgeStatus}
                className="lg:col-span-1"
              />
              <GPUMetricsPanel
                metrics={data.gpuMetrics}
                className="lg:col-span-1"
              />
              <PerformanceCharts
                metrics={data.performanceMetrics}
                className="lg:col-span-1 xl:col-span-2"
              />
            </div>
          </Tabs.Content>

          {/* Agent Lifecycle Tab */}
          <Tabs.Content value="lifecycle" className="p-4 overflow-y-auto">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
              <div className="xl:col-span-2">
                <TimelineView
                  events={data.lifecycleEvents}
                  filters={filters}
                  onFiltersChange={setFilters}
                />
              </div>
              <div className="space-y-4">
                <AgentPositionTracker
                  positions={data.agentPositions}
                  selectedAgentId={selectedAgentId}
                  onAgentSelect={setSelectedAgentId}
                />
                <AgentLifecyclePanel
                  events={data.lifecycleEvents}
                  compact
                />
              </div>
            </div>
          </Tabs.Content>

          {/* Network & Communication Tab */}
          <Tabs.Content value="network" className="p-4 overflow-y-auto">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <NetworkGraph
                communications={data.communications}
                agentPositions={data.agentPositions}
                className="xl:col-span-1"
              />
              <MessageFlowDebugger
                communications={data.communications}
                selectedAgentId={selectedAgentId}
                onAgentSelect={setSelectedAgentId}
                className="xl:col-span-1"
              />
            </div>
          </Tabs.Content>

          {/* Performance Tab */}
          <Tabs.Content value="performance" className="p-4 overflow-y-auto">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <PerformanceCharts
                metrics={data.performanceMetrics}
                className="xl:col-span-2"
              />
              <GPUMetricsPanel
                metrics={data.gpuMetrics}
                showHeatmap
                className="xl:col-span-1"
              />
              <ForceVectorVisualization
                positions={data.agentPositions}
                className="xl:col-span-1"
              />
            </div>
          </Tabs.Content>

          {/* Debugging Tools Tab */}
          <Tabs.Content value="debugging" className="p-4 overflow-y-auto">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
              <AgentInspector
                agentIds={agentIds}
                selectedAgentId={selectedAgentId}
                onAgentSelect={setSelectedAgentId}
                telemetryData={data}
                className="xl:col-span-1"
              />
              <MessageFlowDebugger
                communications={data.communications}
                selectedAgentId={selectedAgentId}
                onAgentSelect={setSelectedAgentId}
                className="xl:col-span-2"
              />
            </div>
          </Tabs.Content>

          {/* Logs & Analysis Tab */}
          <Tabs.Content value="logs" className="p-4 overflow-y-auto">
            <LogViewer
              logEntries={data.logEntries}
              filters={filters}
              onFiltersChange={setFilters}
              agentIds={agentIds}
              className="h-full"
            />
          </Tabs.Content>
        </Tabs>
      </div>
    </div>
  );
};

export default TelemetryDashboard;