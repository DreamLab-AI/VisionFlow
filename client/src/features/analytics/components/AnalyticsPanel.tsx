import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Progress } from '@/features/design-system/components/Progress';
import { BarChart3, TrendingUp, Users, Activity, RefreshCw, Download } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('AnalyticsPanel');

interface AnalyticsPanelProps {
  className?: string;
}

interface AnalyticsMetric {
  id: string;
  name: string;
  value: number;
  change: number;
  changeType: 'increase' | 'decrease' | 'neutral';
  unit?: string;
}

interface AnalyticsChart {
  id: string;
  name: string;
  type: 'line' | 'bar' | 'pie';
  data: Array<{ label: string; value: number }>;
}

export const AnalyticsPanel: React.FC<AnalyticsPanelProps> = ({ className }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshing, setRefreshing] = useState(false);
  
  // Subscribe only to analytics-related settings
  const analyticsSettings = useSelectiveSettings({
    enabled: 'analytics.enabled',
    autoRefresh: 'analytics.autoRefresh',
    refreshInterval: 'analytics.refreshIntervalMinutes',
    retentionDays: 'analytics.dataRetentionDays',
    trackingLevel: 'analytics.trackingLevel',
    anonymizeData: 'analytics.anonymizeData',
    exportFormat: 'analytics.export.defaultFormat',
    realTimeEnabled: 'analytics.realTime.enabled'
  });
  
  // Mock analytics data - in real app this would come from store/API
  const metrics: AnalyticsMetric[] = useMemo(() => [
    {
      id: '1',
      name: 'Total Users',
      value: 2847,
      change: 12.5,
      changeType: 'increase',
      unit: 'users'
    },
    {
      id: '2',
      name: 'Active Sessions',
      value: 342,
      change: -3.2,
      changeType: 'decrease',
      unit: 'sessions'
    },
    {
      id: '3',
      name: 'Data Processing Rate',
      value: 87.3,
      change: 5.7,
      changeType: 'increase',
      unit: '%'
    },
    {
      id: '4',
      name: 'Error Rate',
      value: 0.12,
      change: -25.5,
      changeType: 'decrease',
      unit: '%'
    }
  ], []);
  
  const charts: AnalyticsChart[] = useMemo(() => [
    {
      id: '1',
      name: 'User Activity',
      type: 'line',
      data: [
        { label: 'Mon', value: 120 },
        { label: 'Tue', value: 150 },
        { label: 'Wed', value: 180 },
        { label: 'Thu', value: 165 },
        { label: 'Fri', value: 195 },
        { label: 'Sat', value: 145 },
        { label: 'Sun', value: 125 }
      ]
    },
    {
      id: '2',
      name: 'Feature Usage',
      type: 'bar',
      data: [
        { label: 'Graph View', value: 45 },
        { label: 'Data Import', value: 32 },
        { label: 'Analytics', value: 28 },
        { label: 'Export', value: 15 },
        { label: 'Settings', value: 12 }
      ]
    }
  ], []);
  
  const handleRefresh = async () => {
    setRefreshing(true);
    logger.info('Refreshing analytics data');
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setRefreshing(false);
  };
  
  const handleExport = () => {
    logger.info('Exporting analytics data', { format: analyticsSettings.exportFormat });
    // In real app, trigger export
  };
  
  const getChangeColor = (changeType: AnalyticsMetric['changeType']) => {
    switch (changeType) {
      case 'increase': return 'text-green-600';
      case 'decrease': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };
  
  const getChangeIcon = (changeType: AnalyticsMetric['changeType']) => {
    switch (changeType) {
      case 'increase': return <TrendingUp size={16} />;
      case 'decrease': return <TrendingUp size={16} className="rotate-180" />;
      default: return null;
    }
  };
  
  if (!analyticsSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 size={20} />
            Analytics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <BarChart3 size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">Analytics is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable analytics in settings to view metrics and insights
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 size={20} />
            Analytics Dashboard
            {analyticsSettings.realTimeEnabled && (
              <Badge className="bg-green-100 text-green-800">Live</Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button 
              size="sm" 
              variant="outline" 
              onClick={handleRefresh}
              disabled={refreshing}
            >
              <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
            </Button>
            <Button size="sm" variant="outline" onClick={handleExport}>
              <Download size={16} />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="charts">Charts</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="space-y-6">
            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {metrics.map((metric) => (
                <div key={metric.id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-muted-foreground">
                      {metric.name}
                    </span>
                    <div className={`flex items-center gap-1 ${getChangeColor(metric.changeType)}`}>
                      {getChangeIcon(metric.changeType)}
                      <span className="text-xs">
                        {Math.abs(metric.change)}%
                      </span>
                    </div>
                  </div>
                  <div className="text-2xl font-bold">
                    {metric.value.toLocaleString()}
                    {metric.unit && (
                      <span className="text-sm font-normal text-muted-foreground ml-1">
                        {metric.unit}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
            
            {/* System Health */}
            <div className="border rounded-lg p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Activity size={16} />
                System Health
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">CPU Usage</span>
                  <div className="flex items-center gap-2">
                    <Progress value={65} className="w-20" />
                    <span className="text-sm text-muted-foreground w-8">65%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Memory Usage</span>
                  <div className="flex items-center gap-2">
                    <Progress value={45} className="w-20" />
                    <span className="text-sm text-muted-foreground w-8">45%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Network I/O</span>
                  <div className="flex items-center gap-2">
                    <Progress value={32} className="w-20" />
                    <span className="text-sm text-muted-foreground w-8">32%</span>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="metrics" className="space-y-4">
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {metrics.map((metric) => (
                  <div key={metric.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{metric.name}</span>
                      <Badge className={getChangeColor(metric.changeType)}>
                        {metric.changeType} {Math.abs(metric.change)}%
                      </Badge>
                    </div>
                    <div className="text-xl font-bold mb-1">
                      {metric.value.toLocaleString()}{metric.unit && ` ${metric.unit}`}
                    </div>
                    <Progress value={Math.min(100, (metric.value / 3000) * 100)} className="w-full" />
                  </div>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="charts" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {charts.map((chart) => (
                <div key={chart.id} className="border rounded-lg p-4">
                  <h3 className="font-medium mb-4">{chart.name}</h3>
                  <div className="h-32 flex items-end justify-between gap-2">
                    {chart.data.map((point, index) => (
                      <div key={index} className="flex flex-col items-center gap-1">
                        <div 
                          className="bg-blue-500 w-6 rounded-t"
                          style={{ height: `${(point.value / Math.max(...chart.data.map(d => d.value))) * 100}%` }}
                        />
                        <span className="text-xs text-muted-foreground">
                          {point.label}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Auto Refresh</label>
                <Button
                  variant={analyticsSettings.autoRefresh ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {}}
                >
                  {analyticsSettings.autoRefresh ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Real-time Updates</label>
                <Button
                  variant={analyticsSettings.realTimeEnabled ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {}}
                >
                  {analyticsSettings.realTimeEnabled ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Data Anonymization</label>
                <Button
                  variant={analyticsSettings.anonymizeData ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {}}
                >
                  {analyticsSettings.anonymizeData ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Tracking Level</label>
                <Badge variant="outline">
                  {analyticsSettings.trackingLevel}
                </Badge>
              </div>
            </div>
            
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">Data Retention</h3>
              <p className="text-sm text-muted-foreground">
                Analytics data is retained for {analyticsSettings.retentionDays} days
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default AnalyticsPanel;