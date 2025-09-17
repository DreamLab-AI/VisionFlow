import React, { useMemo } from 'react';
import { Card } from '../../design-system/components/Card';
import { Switch } from '../../design-system/components/Switch';
import { Label } from '../../design-system/components/Label';
import { createLogger } from '../../../utils/logger';
import { PerformanceMetrics } from '../types';

const logger = createLogger('PerformanceCharts');

interface PerformanceChartsProps {
  metrics: PerformanceMetrics[];
  className?: string;
}

export const PerformanceCharts: React.FC<PerformanceChartsProps> = ({
  metrics,
  className = ''
}) => {
  const [showIndividualAgents, setShowIndividualAgents] = React.useState(false);

  const chartData = useMemo(() => {
    if (metrics.length === 0) return null;

    const recent = metrics.slice(-20); // Last 20 data points
    const maxValues = {
      cpu: Math.max(...recent.map(m => m.overall.cpu), 100),
      memory: Math.max(...recent.map(m => m.overall.memory), 100),
      network: Math.max(...recent.map(m => m.overall.network), 100),
      gpu: Math.max(...recent.flatMap(m => m.overall.gpu ? [m.overall.gpu] : []), 100)
    };

    return {
      metrics: recent,
      maxValues,
      agentMetrics: showIndividualAgents ? recent[recent.length - 1]?.agents || [] : []
    };
  }, [metrics, showIndividualAgents]);

  if (!chartData) {
    return (
      <Card className={className}>
        <div className="p-4">
          <h3 className="text-lg font-semibold mb-4">Performance Charts</h3>
          <div className="text-center text-gray-500 py-8">
            No performance metrics available
          </div>
        </div>
      </Card>
    );
  }

  const generatePath = (data: number[], maxValue: number, width: number, height: number) => {
    if (data.length < 2) return '';

    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - (value / maxValue) * height;
      return `${x},${y}`;
    });

    return `M${points.join(' L')}`;
  };

  const getMetricColor = (metric: string) => {
    switch (metric) {
      case 'cpu': return '#ef4444';
      case 'memory': return '#3b82f6';
      case 'network': return '#10b981';
      case 'gpu': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  const formatValue = (value: number, metric: string) => {
    return `${value.toFixed(1)}%`;
  };

  const currentMetrics = chartData.metrics[chartData.metrics.length - 1];

  return (
    <Card className={className}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Performance Charts</h3>
          <div className="flex items-center gap-2">
            <Label htmlFor="show-agents" className="text-sm">Individual Agents</Label>
            <Switch
              id="show-agents"
              checked={showIndividualAgents}
              onCheckedChange={setShowIndividualAgents}
              size="sm"
            />
          </div>
        </div>

        {/* Current Values */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-3 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">
              {currentMetrics.overall.cpu.toFixed(1)}%
            </div>
            <div className="text-sm text-red-700">CPU</div>
          </div>
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {currentMetrics.overall.memory.toFixed(1)}%
            </div>
            <div className="text-sm text-blue-700">Memory</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {currentMetrics.overall.network.toFixed(1)}%
            </div>
            <div className="text-sm text-green-700">Network</div>
          </div>
          {currentMetrics.overall.gpu !== undefined && (
            <div className="text-center p-3 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {currentMetrics.overall.gpu.toFixed(1)}%
              </div>
              <div className="text-sm text-orange-700">GPU</div>
            </div>
          )}
        </div>

        {/* Performance Charts */}
        <div className="mb-6">
          <h4 className="font-medium mb-3">System Performance Over Time</h4>
          <div className="relative">
            <svg width="100%" height="200" viewBox="0 0 400 160" className="border rounded-lg bg-gray-50">
              {/* Grid lines */}
              <defs>
                <pattern id="grid" width="40" height="32" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 32" fill="none" stroke="#e5e7eb" strokeWidth="1" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />

              {/* Performance lines */}
              {/* CPU */}
              <path
                d={generatePath(
                  chartData.metrics.map(m => m.overall.cpu),
                  chartData.maxValues.cpu,
                  400,
                  160
                )}
                fill="none"
                stroke={getMetricColor('cpu')}
                strokeWidth="2"
                className="drop-shadow-sm"
              />

              {/* Memory */}
              <path
                d={generatePath(
                  chartData.metrics.map(m => m.overall.memory),
                  chartData.maxValues.memory,
                  400,
                  160
                )}
                fill="none"
                stroke={getMetricColor('memory')}
                strokeWidth="2"
                className="drop-shadow-sm"
              />

              {/* Network */}
              <path
                d={generatePath(
                  chartData.metrics.map(m => m.overall.network),
                  chartData.maxValues.network,
                  400,
                  160
                )}
                fill="none"
                stroke={getMetricColor('network')}
                strokeWidth="2"
                className="drop-shadow-sm"
              />

              {/* GPU (if available) */}
              {currentMetrics.overall.gpu !== undefined && (
                <path
                  d={generatePath(
                    chartData.metrics.map(m => m.overall.gpu || 0),
                    chartData.maxValues.gpu,
                    400,
                    160
                  )}
                  fill="none"
                  stroke={getMetricColor('gpu')}
                  strokeWidth="2"
                  strokeDasharray="5,5"
                  className="drop-shadow-sm"
                />
              )}

              {/* Current value indicators */}
              {chartData.metrics.map((metric, index) => {
                if (index !== chartData.metrics.length - 1) return null;
                const x = (index / (chartData.metrics.length - 1)) * 400;

                return (
                  <g key={index}>
                    <circle
                      cx={x}
                      cy={160 - (metric.overall.cpu / chartData.maxValues.cpu) * 160}
                      r="3"
                      fill={getMetricColor('cpu')}
                    />
                    <circle
                      cx={x}
                      cy={160 - (metric.overall.memory / chartData.maxValues.memory) * 160}
                      r="3"
                      fill={getMetricColor('memory')}
                    />
                    <circle
                      cx={x}
                      cy={160 - (metric.overall.network / chartData.maxValues.network) * 160}
                      r="3"
                      fill={getMetricColor('network')}
                    />
                    {metric.overall.gpu !== undefined && (
                      <circle
                        cx={x}
                        cy={160 - (metric.overall.gpu / chartData.maxValues.gpu) * 160}
                        r="3"
                        fill={getMetricColor('gpu')}
                      />
                    )}
                  </g>
                );
              })}
            </svg>

            {/* Legend */}
            <div className="flex items-center justify-center gap-4 mt-3 text-sm">
              <div className="flex items-center gap-1">
                <div className="w-3 h-0.5 bg-red-500"></div>
                <span>CPU</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-0.5 bg-blue-500"></div>
                <span>Memory</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-0.5 bg-green-500"></div>
                <span>Network</span>
              </div>
              {currentMetrics.overall.gpu !== undefined && (
                <div className="flex items-center gap-1">
                  <div className="w-3 h-0.5 bg-orange-500" style={{ borderTop: '2px dashed' }}></div>
                  <span>GPU</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Individual Agent Performance */}
        {showIndividualAgents && chartData.agentMetrics.length > 0 && (
          <div>
            <h4 className="font-medium mb-3">Individual Agent Performance</h4>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {chartData.agentMetrics.map(agent => (
                <div key={agent.agentId} className="p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">{agent.agentId}</span>
                    <div className="flex items-center gap-2">
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        agent.health > 80 ? 'bg-green-100 text-green-800' :
                        agent.health > 60 ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        Health: {agent.health.toFixed(1)}%
                      </span>
                      <span className="text-xs text-gray-600">
                        Tasks: {agent.taskCount}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-gray-600">CPU:</span>
                        <span className="font-mono">{agent.cpu.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div
                          className="h-1.5 bg-red-500 rounded-full"
                          style={{ width: `${Math.min(agent.cpu, 100)}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-gray-600">Memory:</span>
                        <span className="font-mono">{agent.memory.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div
                          className="h-1.5 bg-blue-500 rounded-full"
                          style={{ width: `${Math.min(agent.memory, 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Performance Summary */}
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h4 className="font-medium mb-3">Performance Summary</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-gray-600">Avg CPU:</div>
              <div className="font-mono">
                {(chartData.metrics.reduce((sum, m) => sum + m.overall.cpu, 0) / chartData.metrics.length).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-gray-600">Avg Memory:</div>
              <div className="font-mono">
                {(chartData.metrics.reduce((sum, m) => sum + m.overall.memory, 0) / chartData.metrics.length).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-gray-600">Avg Network:</div>
              <div className="font-mono">
                {(chartData.metrics.reduce((sum, m) => sum + m.overall.network, 0) / chartData.metrics.length).toFixed(1)}%
              </div>
            </div>
            {currentMetrics.overall.gpu !== undefined && (
              <div>
                <div className="text-gray-600">Avg GPU:</div>
                <div className="font-mono">
                  {(chartData.metrics.reduce((sum, m) => sum + (m.overall.gpu || 0), 0) / chartData.metrics.length).toFixed(1)}%
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};