import React, { useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Badge } from '@/features/design-system/components/Badge';
import { Progress } from '@/features/design-system/components/Progress';
import { TrendingUp, TrendingDown, Minus, Activity, Clock, Users } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('MetricsDisplay');

interface MetricsDisplayProps {
  className?: string;
  compact?: boolean;
}

interface Metric {
  id: string;
  name: string;
  value: number | string;
  previousValue?: number;
  unit?: string;
  type: 'number' | 'percentage' | 'currency' | 'time' | 'text';
  trend?: 'up' | 'down' | 'neutral';
  status?: 'good' | 'warning' | 'error' | 'neutral';
  target?: number;
}

export const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ className, compact = false }) => {
  // Subscribe only to metrics-related settings
  const metricsSettings = useSelectiveSettings({
    displayMode: 'analytics.metrics.displayMode',
    showTrends: 'analytics.metrics.showTrends',
    showTargets: 'analytics.metrics.showTargets',
    refreshRate: 'analytics.metrics.refreshRateSeconds',
    colorCoding: 'analytics.metrics.colorCoding',
    precision: 'analytics.metrics.decimalPrecision'
  });
  
  // Mock metrics data - in real app this would come from store/API
  const metrics: Metric[] = useMemo(() => [
    {
      id: 'active_users',
      name: 'Active Users',
      value: 1247,
      previousValue: 1156,
      type: 'number',
      trend: 'up',
      status: 'good',
      target: 1500
    },
    {
      id: 'cpu_usage',
      name: 'CPU Usage',
      value: 67.8,
      previousValue: 72.1,
      unit: '%',
      type: 'percentage',
      trend: 'down',
      status: 'warning',
      target: 80
    },
    {
      id: 'memory_usage',
      name: 'Memory Usage',
      value: 45.2,
      previousValue: 43.8,
      unit: '%',
      type: 'percentage',
      trend: 'up',
      status: 'good',
      target: 70
    },
    {
      id: 'response_time',
      name: 'Avg Response Time',
      value: 245,
      previousValue: 267,
      unit: 'ms',
      type: 'time',
      trend: 'down',
      status: 'good',
      target: 200
    },
    {
      id: 'error_rate',
      name: 'Error Rate',
      value: 0.12,
      previousValue: 0.18,
      unit: '%',
      type: 'percentage',
      trend: 'down',
      status: 'good',
      target: 0.1
    },
    {
      id: 'throughput',
      name: 'Requests/sec',
      value: 1456,
      previousValue: 1389,
      type: 'number',
      trend: 'up',
      status: 'good',
      target: 2000
    }
  ], []);
  
  const formatValue = (metric: Metric): string => {
    const precision = metricsSettings.precision || 1;
    
    switch (metric.type) {
      case 'percentage':
        return `${Number(metric.value).toFixed(precision)}${metric.unit || '%'}`;
      case 'currency':
        return `$${Number(metric.value).toLocaleString()}`;
      case 'time':
        return `${metric.value}${metric.unit || 'ms'}`;
      case 'number':
        return `${Number(metric.value).toLocaleString()}`;
      default:
        return String(metric.value);
    }
  };
  
  const getTrendIcon = (trend?: Metric['trend']) => {
    switch (trend) {
      case 'up': return <TrendingUp size={16} className="text-green-600" />;
      case 'down': return <TrendingDown size={16} className="text-red-600" />;
      default: return <Minus size={16} className="text-gray-400" />;
    }
  };
  
  const getStatusColor = (status?: Metric['status']) => {
    switch (status) {
      case 'good': return 'bg-green-100 text-green-800';
      case 'warning': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };
  
  const getMetricIcon = (metric: Metric) => {
    switch (metric.id) {
      case 'active_users': return <Users size={16} />;
      case 'response_time': return <Clock size={16} />;
      default: return <Activity size={16} />;
    }
  };
  
  const calculateProgress = (metric: Metric): number => {
    if (!metric.target || typeof metric.value !== 'number') return 0;
    return Math.min(100, (metric.value / metric.target) * 100);
  };
  
  const calculateChange = (metric: Metric): number | null => {
    if (typeof metric.value !== 'number' || !metric.previousValue) return null;
    return ((metric.value - metric.previousValue) / metric.previousValue) * 100;
  };
  
  if (compact) {
    return (
      <div className={`grid grid-cols-2 lg:grid-cols-3 gap-2 ${className}`}>
        {metrics.slice(0, 6).map((metric) => {
          const change = calculateChange(metric);
          return (
            <div key={metric.id} className="border rounded p-3 text-center">
              <div className="flex items-center justify-center gap-1 mb-1">
                {getMetricIcon(metric)}
                {metricsSettings.showTrends && getTrendIcon(metric.trend)}
              </div>
              <div className="text-lg font-bold">{formatValue(metric)}</div>
              <div className="text-xs text-muted-foreground truncate">{metric.name}</div>
              {change !== null && metricsSettings.showTrends && (
                <div className={`text-xs ${
                  change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {change > 0 ? '+' : ''}{change.toFixed(1)}%
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity size={20} />
            System Metrics
          </div>
          <Badge variant="outline" className="text-xs">
            Updated {metricsSettings.refreshRate}s ago
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {metrics.map((metric) => {
            const change = calculateChange(metric);
            const progress = calculateProgress(metric);
            
            return (
              <div key={metric.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {metricsSettings.colorCoding && getMetricIcon(metric)}
                    <span className="text-sm font-medium text-muted-foreground">
                      {metric.name}
                    </span>
                  </div>
                  {metricsSettings.colorCoding && (
                    <Badge className={getStatusColor(metric.status)}>
                      {metric.status}
                    </Badge>
                  )}
                </div>
                
                <div className="flex items-center justify-between mb-2">
                  <span className="text-2xl font-bold">
                    {formatValue(metric)}
                  </span>
                  {metricsSettings.showTrends && (
                    <div className="flex items-center gap-1">
                      {getTrendIcon(metric.trend)}
                      {change !== null && (
                        <span className={`text-sm ${
                          change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : 'text-gray-600'
                        }`}>
                          {change > 0 ? '+' : ''}{change.toFixed(1)}%
                        </span>
                      )}
                    </div>
                  )}
                </div>
                
                {metricsSettings.showTargets && metric.target && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Target: {metric.target}{metric.unit}</span>
                      <span>{progress.toFixed(0)}%</span>
                    </div>
                    <Progress value={progress} className="w-full h-2" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
        
        {metricsSettings.displayMode === 'detailed' && (
          <div className="mt-6 border-t pt-4">
            <h3 className="text-sm font-medium mb-3">Performance Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Overall Health: </span>
                <Badge className="bg-green-100 text-green-800">Good</Badge>
              </div>
              <div>
                <span className="text-muted-foreground">Active Alerts: </span>
                <span className="font-medium">2</span>
              </div>
              <div>
                <span className="text-muted-foreground">Last Updated: </span>
                <span className="font-medium">{new Date().toLocaleTimeString()}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Next Refresh: </span>
                <span className="font-medium">{metricsSettings.refreshRate}s</span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default MetricsDisplay;