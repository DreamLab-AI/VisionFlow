import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Badge } from '@/features/design-system/components/Badge';
import { Button } from '@/features/design-system/components/Button';
import { Progress } from '@/features/design-system/components/Progress';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  Network, 
  Wifi,
  WifiOff,
  Zap,
  Database,
  BarChart3,
  AlertCircle,
  CheckCircle,
  Info,
  TrendingUp,
  TrendingDown,
  Gauge,
  Globe,
  Box,
  GitBranch
} from 'lucide-react';
import { useToast } from '@/features/design-system/components/Toast';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { cn } from '@/utils/classNameUtils';

interface SystemMetrics {
  cpu: number;
  memory: number;
  gpu: number;
  gpuMemory: number;
  fps: number;
  latency: number;
}

interface GraphStats {
  nodes: number;
  edges: number;
  clusters: number;
  activeConstraints: number;
}

interface ConnectionStatus {
  websocket: boolean;
  apiServer: boolean;
  gpuKernel: boolean;
  quest3: boolean;
}

interface ActivityLogItem {
  id: string;
  timestamp: Date;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  icon?: React.ReactNode;
}

export function DashboardPanel() {
  const { toast } = useToast();
  
  // State
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu: 0,
    memory: 0,
    gpu: 0,
    gpuMemory: 0,
    fps: 0,
    latency: 0,
  });
  
  const [graphStats, setGraphStats] = useState<GraphStats>({
    nodes: 177,
    edges: 234,
    clusters: 0,
    activeConstraints: 3,
  });
  
  const [connections, setConnections] = useState<ConnectionStatus>({
    websocket: false,
    apiServer: true,
    gpuKernel: true,
    quest3: false,
  });
  
  const [activityLog, setActivityLog] = useState<ActivityLogItem[]>([
    {
      id: '1',
      timestamp: new Date(),
      type: 'success',
      message: 'System initialized successfully',
      icon: <CheckCircle className="h-4 w-4" />,
    },
    {
      id: '2',
      timestamp: new Date(),
      type: 'info',
      message: 'GPU kernel mode: visual_analytics',
      icon: <Cpu className="h-4 w-4" />,
    },
  ]);

  // Fetch metrics periodically
  useEffect(() => {
    const fetchMetrics = async () => {
      // Simulate fetching metrics
      setMetrics({
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        gpu: Math.random() * 100,
        gpuMemory: Math.random() * 100,
        fps: 60 + Math.random() * 30,
        latency: 5 + Math.random() * 20,
      });
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  // Check connections
  useEffect(() => {
    const checkConnections = async () => {
      try {
        // Check API server
        const response = await fetch('/api/health');
        setConnections(prev => ({ ...prev, apiServer: response.ok }));
      } catch {
        setConnections(prev => ({ ...prev, apiServer: false }));
      }
      
      // Check WebSocket
      const ws = (window as any).webSocketService;
      if (ws) {
        setConnections(prev => ({ ...prev, websocket: ws.isConnected }));
      }
    };

    checkConnections();
    const interval = setInterval(checkConnections, 5000);
    return () => clearInterval(interval);
  }, []);

  const addActivityLog = (item: Omit<ActivityLogItem, 'id' | 'timestamp'>) => {
    setActivityLog(prev => [
      {
        ...item,
        id: Date.now().toString(),
        timestamp: new Date(),
      },
      ...prev.slice(0, 49), // Keep last 50 items
    ]);
  };

  const getMetricColor = (value: number, threshold: number = 80) => {
    if (value > threshold) return 'text-red-500';
    if (value > threshold * 0.7) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getConnectionIcon = (connected: boolean) => {
    return connected ? (
      <CheckCircle className="h-4 w-4 text-green-500" />
    ) : (
      <AlertCircle className="h-4 w-4 text-red-500" />
    );
  };

  return (
    <div className="h-full overflow-auto p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">System Dashboard</h2>
          <p className="text-sm text-muted-foreground">
            Real-time monitoring and control center
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Activity className="mr-2 h-4 w-4" />
            Export Metrics
          </Button>
          <Button variant="outline" size="sm">
            <Info className="mr-2 h-4 w-4" />
            Help
          </Button>
        </div>
      </div>

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Nodes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{graphStats.nodes}</div>
            <p className="text-xs text-muted-foreground">
              <TrendingUp className="inline h-3 w-3 mr-1" />
              Active in graph
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">FPS</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={cn("text-2xl font-bold", getMetricColor(metrics.fps, 30))}>
              {metrics.fps.toFixed(0)}
            </div>
            <p className="text-xs text-muted-foreground">
              <Gauge className="inline h-3 w-3 mr-1" />
              Render performance
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">GPU</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={cn("text-2xl font-bold", getMetricColor(metrics.gpu))}>
              {metrics.gpu.toFixed(0)}%
            </div>
            <Progress value={metrics.gpu} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Latency</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={cn("text-2xl font-bold", metrics.latency > 50 ? 'text-red-500' : 'text-green-500')}>
              {metrics.latency.toFixed(0)}ms
            </div>
            <p className="text-xs text-muted-foreground">
              <Network className="inline h-3 w-3 mr-1" />
              Network delay
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* System Resources */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Resources
            </CardTitle>
            <CardDescription>
              Real-time resource utilization
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  <span className="text-sm font-medium">CPU</span>
                </div>
                <span className={cn("text-sm font-bold", getMetricColor(metrics.cpu))}>
                  {metrics.cpu.toFixed(1)}%
                </span>
              </div>
              <Progress value={metrics.cpu} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4" />
                  <span className="text-sm font-medium">Memory</span>
                </div>
                <span className={cn("text-sm font-bold", getMetricColor(metrics.memory))}>
                  {metrics.memory.toFixed(1)}%
                </span>
              </div>
              <Progress value={metrics.memory} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  <span className="text-sm font-medium">GPU</span>
                </div>
                <span className={cn("text-sm font-bold", getMetricColor(metrics.gpu))}>
                  {metrics.gpu.toFixed(1)}%
                </span>
              </div>
              <Progress value={metrics.gpu} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4" />
                  <span className="text-sm font-medium">GPU Memory</span>
                </div>
                <span className={cn("text-sm font-bold", getMetricColor(metrics.gpuMemory))}>
                  {metrics.gpuMemory.toFixed(1)}%
                </span>
              </div>
              <Progress value={metrics.gpuMemory} />
            </div>
          </CardContent>
        </Card>

        {/* Connection Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5" />
              Connections
            </CardTitle>
            <CardDescription>
              Service connection status
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between p-2 rounded border">
              <div className="flex items-center gap-2">
                {connections.websocket ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />}
                <span className="text-sm">WebSocket</span>
              </div>
              {getConnectionIcon(connections.websocket)}
            </div>

            <div className="flex items-center justify-between p-2 rounded border">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4" />
                <span className="text-sm">API Server</span>
              </div>
              {getConnectionIcon(connections.apiServer)}
            </div>

            <div className="flex items-center justify-between p-2 rounded border">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                <span className="text-sm">GPU Kernel</span>
              </div>
              {getConnectionIcon(connections.gpuKernel)}
            </div>

            <div className="flex items-center justify-between p-2 rounded border">
              <div className="flex items-center gap-2">
                <Box className="h-4 w-4" />
                <span className="text-sm">Quest 3 AR</span>
              </div>
              {getConnectionIcon(connections.quest3)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Graph Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitBranch className="h-5 w-5" />
            Graph Statistics
          </CardTitle>
          <CardDescription>
            Current graph composition
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold">{graphStats.nodes}</div>
              <p className="text-xs text-muted-foreground">Nodes</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{graphStats.edges}</div>
              <p className="text-xs text-muted-foreground">Edges</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{graphStats.clusters}</div>
              <p className="text-xs text-muted-foreground">Clusters</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{graphStats.activeConstraints}</div>
              <p className="text-xs text-muted-foreground">Constraints</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Activity Log */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Activity Log
          </CardTitle>
          <CardDescription>
            Recent system events
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[200px]">
            <div className="space-y-2">
              {activityLog.map(item => (
                <div key={item.id} className="flex items-start gap-2 text-sm">
                  <div className={cn(
                    "mt-0.5",
                    item.type === 'error' && 'text-red-500',
                    item.type === 'warning' && 'text-yellow-500',
                    item.type === 'success' && 'text-green-500',
                    item.type === 'info' && 'text-blue-500'
                  )}>
                    {item.icon || <Info className="h-4 w-4" />}
                  </div>
                  <div className="flex-1">
                    <p>{item.message}</p>
                    <p className="text-xs text-muted-foreground">
                      {item.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}