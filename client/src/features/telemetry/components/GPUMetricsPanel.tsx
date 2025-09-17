import React, { useMemo } from 'react';
import { Card } from '../../design-system/components/Card';
import { createLogger } from '../../../utils/logger';
import { GPUMetrics } from '../types';

const logger = createLogger('GPUMetricsPanel');

interface GPUMetricsPanelProps {
  metrics: GPUMetrics[];
  className?: string;
  showHeatmap?: boolean;
}

export const GPUMetricsPanel: React.FC<GPUMetricsPanelProps> = ({
  metrics,
  className = '',
  showHeatmap = false
}) => {
  const currentMetrics = metrics[metrics.length - 1];

  const stats = useMemo(() => {
    if (metrics.length === 0) {
      return {
        avgUtilization: 0,
        peakUtilization: 0,
        avgMemoryUsage: 0,
        peakMemoryUsage: 0,
        avgTemperature: 0,
        maxTemperature: 0,
        totalComputeTasks: 0,
        activeAgents: new Set<string>()
      };
    }

    const recentMetrics = metrics.slice(-20); // Last 20 data points
    const avgUtilization = recentMetrics.reduce((sum, m) => sum + m.utilizationPercent, 0) / recentMetrics.length;
    const peakUtilization = Math.max(...metrics.map(m => m.utilizationPercent));

    const memoryUsagePercents = recentMetrics.map(m => (m.memoryUsed / m.memoryTotal) * 100);
    const avgMemoryUsage = memoryUsagePercents.reduce((sum, p) => sum + p, 0) / memoryUsagePercents.length;
    const peakMemoryUsage = Math.max(...memoryUsagePercents);

    const avgTemperature = recentMetrics.reduce((sum, m) => sum + m.temperature, 0) / recentMetrics.length;
    const maxTemperature = Math.max(...metrics.map(m => m.temperature));

    const totalComputeTasks = metrics.reduce((sum, m) => sum + m.computeTasks.length, 0);
    const activeAgents = new Set<string>();
    recentMetrics.forEach(m => m.computeTasks.forEach(task => activeAgents.add(task.agentId)));

    return {
      avgUtilization,
      peakUtilization,
      avgMemoryUsage,
      peakMemoryUsage,
      avgTemperature,
      maxTemperature,
      totalComputeTasks,
      activeAgents
    };
  }, [metrics]);

  const getUtilizationColor = (utilization: number) => {
    if (utilization < 30) return 'text-green-600 bg-green-50';
    if (utilization < 70) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getTemperatureColor = (temp: number) => {
    if (temp < 60) return 'text-green-600';
    if (temp < 80) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatMemory = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)}GB`;
  };

  // Generate heatmap data for compute tasks
  const heatmapData = useMemo(() => {
    if (!showHeatmap || !currentMetrics) return [];

    const heatmapSize = 20; // 20x20 grid
    const grid = Array(heatmapSize).fill(null).map(() => Array(heatmapSize).fill(0));

    currentMetrics.computeTasks.forEach((task, index) => {
      const x = Math.floor((index * 7) % heatmapSize); // Distribute tasks across grid
      const y = Math.floor((index * 11) % heatmapSize);
      grid[y][x] = Math.min(grid[y][x] + task.intensity, 1);
    });

    return grid;
  }, [currentMetrics, showHeatmap]);

  return (
    <Card className={className}>
      <div className="p-4">
        <h3 className="text-lg font-semibold mb-4">GPU Metrics</h3>

        {!currentMetrics ? (
          <div className="text-center text-gray-500 py-8">
            No GPU metrics available
          </div>
        ) : (
          <>
            {/* Current Status */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className={`text-center p-3 rounded-lg ${getUtilizationColor(currentMetrics.utilizationPercent)}`}>
                <div className="text-2xl font-bold">
                  {currentMetrics.utilizationPercent.toFixed(1)}%
                </div>
                <div className="text-sm">GPU Utilization</div>
              </div>
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {formatMemory(currentMetrics.memoryUsed)}
                </div>
                <div className="text-sm text-blue-700">
                  Memory Used
                </div>
              </div>
            </div>

            {/* Memory Usage Bar */}
            <div className="mb-6">
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>Memory Usage</span>
                <span>
                  {formatMemory(currentMetrics.memoryUsed)} / {formatMemory(currentMetrics.memoryTotal)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="h-3 bg-blue-500 rounded-full transition-all"
                  style={{
                    width: `${(currentMetrics.memoryUsed / currentMetrics.memoryTotal) * 100}%`
                  }}
                />
              </div>
            </div>

            {/* Temperature and Power */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className={`text-xl font-bold ${getTemperatureColor(currentMetrics.temperature)}`}>
                  {currentMetrics.temperature}°C
                </div>
                <div className="text-sm text-gray-700">Temperature</div>
              </div>
              <div className="text-center p-3 bg-orange-50 rounded-lg">
                <div className="text-xl font-bold text-orange-600">
                  {currentMetrics.powerConsumption.toFixed(0)}W
                </div>
                <div className="text-sm text-orange-700">Power Draw</div>
              </div>
            </div>

            {/* Compute Tasks */}
            <div className="mb-6">
              <h4 className="font-medium mb-3">Active Compute Tasks</h4>
              {currentMetrics.computeTasks.length === 0 ? (
                <div className="text-center text-gray-500 py-4">
                  No active compute tasks
                </div>
              ) : (
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {currentMetrics.computeTasks.map((task, index) => (
                    <div
                      key={`${task.taskId}-${index}`}
                      className="flex items-center justify-between p-2 bg-gray-50 rounded"
                    >
                      <div className="text-sm">
                        <div className="font-medium">Agent: {task.agentId}</div>
                        <div className="text-gray-600">Task: {task.taskId}</div>
                      </div>
                      <div className="text-right text-xs">
                        <div>Intensity: {(task.intensity * 100).toFixed(1)}%</div>
                        <div className="text-gray-500">
                          Duration: {(task.duration / 1000).toFixed(1)}s
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Heatmap */}
            {showHeatmap && heatmapData.length > 0 && (
              <div className="mb-6">
                <h4 className="font-medium mb-3">Compute Intensity Heatmap</h4>
                <div className="relative">
                  <svg width="200" height="200" className="mx-auto border rounded">
                    {heatmapData.map((row, y) =>
                      row.map((intensity, x) => (
                        <rect
                          key={`${x}-${y}`}
                          x={x * 10}
                          y={y * 10}
                          width="10"
                          height="10"
                          fill={`rgba(239, 68, 68, ${intensity})`}
                          stroke="#e5e7eb"
                          strokeWidth="0.5"
                        />
                      ))
                    )}
                  </svg>
                  <div className="flex justify-between items-center mt-2 text-xs text-gray-600">
                    <span>Low</span>
                    <span>Compute Intensity</span>
                    <span>High</span>
                  </div>
                </div>
              </div>
            )}

            {/* Statistics Summary */}
            <div className="pt-4 border-t border-gray-200">
              <h4 className="font-medium mb-3">Performance Summary</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-600">Avg Utilization:</div>
                  <div className="font-mono">{stats.avgUtilization.toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-gray-600">Peak Utilization:</div>
                  <div className="font-mono">{stats.peakUtilization.toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-gray-600">Avg Memory Usage:</div>
                  <div className="font-mono">{stats.avgMemoryUsage.toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-gray-600">Active Agents:</div>
                  <div className="font-mono">{stats.activeAgents.size}</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </Card>
  );
};