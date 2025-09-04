// Dashboard Tab - System status overview and quick actions
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Badge } from '@/features/design-system/components/Badge';
import { Button } from '@/features/design-system/components/Button';
import { Progress } from '@/features/design-system/components/Progress';
import { 
  Cpu, 
  Memory, 
  Wifi, 
  Activity, 
  Users, 
  Eye,
  Zap,
  Server,
  Globe,
  Timer
} from 'lucide-react';
import { useSettingsStore } from '@/store/settingsStore';

interface DashboardTabProps {
  searchQuery?: string;
}

export const DashboardTab: React.FC<DashboardTabProps> = ({ searchQuery = '' }) => {
  const { settings, initialized } = useSettingsStore();
  
  // Mock system metrics - in real app these would come from actual system monitoring
  const systemMetrics = {
    gpu: {
      usage: 45,
      memory: 2.8,
      temperature: 72,
      enabled: settings?.visualisation?.physics?.enabled || false
    },
    cpu: {
      usage: 32,
      cores: 8,
      temperature: 58
    },
    memory: {
      used: 12.4,
      total: 32.0,
      usage: 38.7
    },
    network: {
      connections: 3,
      bandwidth: 125.6,
      status: 'connected'
    },
    graph: {
      nodes: settings?.visualisation?.graphs?.main?.nodeCount || 0,
      edges: settings?.visualisation?.graphs?.main?.edgeCount || 0,
      fps: 60
    }
  };
  
  const statusCards = [
    {
      title: 'GPU Status',
      icon: Zap,
      value: `${systemMetrics.gpu.usage}%`,
      detail: `${systemMetrics.gpu.memory}GB VRAM`,
      status: systemMetrics.gpu.enabled ? 'active' : 'inactive',
      color: systemMetrics.gpu.enabled ? 'green' : 'gray'
    },
    {
      title: 'CPU Usage',
      icon: Cpu,
      value: `${systemMetrics.cpu.usage}%`,
      detail: `${systemMetrics.cpu.cores} cores`,
      status: systemMetrics.cpu.usage > 80 ? 'high' : 'normal',
      color: systemMetrics.cpu.usage > 80 ? 'red' : 'blue'
    },
    {
      title: 'Memory',
      icon: Memory,
      value: `${systemMetrics.memory.usage}%`,
      detail: `${systemMetrics.memory.used}/${systemMetrics.memory.total}GB`,
      status: systemMetrics.memory.usage > 90 ? 'critical' : 'normal',
      color: systemMetrics.memory.usage > 90 ? 'red' : 'green'
    },
    {
      title: 'Network',
      icon: Wifi,
      value: systemMetrics.network.status,
      detail: `${systemMetrics.network.connections} connections`,
      status: systemMetrics.network.status === 'connected' ? 'online' : 'offline',
      color: systemMetrics.network.status === 'connected' ? 'green' : 'red'
    }
  ];
  
  const quickActions = [
    { label: 'Reset Physics', icon: Activity, action: 'resetPhysics' },
    { label: 'Clear Cache', icon: Server, action: 'clearCache' },
    { label: 'Export Data', icon: Globe, action: 'exportData' },
    { label: 'Performance Check', icon: Timer, action: 'performanceCheck' }
  ];
  
  const handleQuickAction = (action: string) => {
    console.log(`Quick action: ${action}`);
    // TODO: Implement actual quick actions
  };
  
  const filteredContent = !searchQuery || 
    'dashboard system status gpu cpu memory network quick actions'.toLowerCase().includes(searchQuery.toLowerCase());
  
  if (!filteredContent) {
    return (
      <div className="text-center text-muted-foreground py-8">
        No dashboard items match your search.
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* System Status Grid */}
      <div>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          System Status
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {statusCards.map((card) => {
            const Icon = card.icon;
            return (
              <Card key={card.title}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    {card.title}
                  </CardTitle>
                  <Icon className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{card.value}</div>
                  <p className="text-xs text-muted-foreground">
                    {card.detail}
                  </p>
                  <Badge 
                    variant={card.color === 'red' ? 'destructive' : 
                            card.color === 'green' ? 'default' : 
                            'secondary'}
                    className="mt-2"
                  >
                    {card.status}
                  </Badge>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
      
      {/* Graph Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="w-5 h-5" />
            Graph Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {systemMetrics.graph.nodes.toLocaleString()}
              </div>
              <p className="text-sm text-muted-foreground">Nodes</p>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">
                {systemMetrics.graph.edges.toLocaleString()}
              </div>
              <p className="text-sm text-muted-foreground">Edges</p>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600">
                {systemMetrics.graph.fps}
              </div>
              <p className="text-sm text-muted-foreground">FPS</p>
            </div>
          </div>
          
          {/* Performance Bars */}
          <div className="mt-4 space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Render Performance</span>
                <span>{systemMetrics.graph.fps}/60 FPS</span>
              </div>
              <Progress value={(systemMetrics.graph.fps / 60) * 100} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>GPU Utilization</span>
                <span>{systemMetrics.gpu.usage}%</span>
              </div>
              <Progress value={systemMetrics.gpu.usage} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Quick Actions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {quickActions.map((action) => {
              const Icon = action.icon;
              return (
                <Button
                  key={action.action}
                  variant="outline"
                  className="h-auto p-4 flex flex-col items-center gap-2"
                  onClick={() => handleQuickAction(action.action)}
                >
                  <Icon className="w-5 h-5" />
                  <span className="text-xs">{action.label}</span>
                </Button>
              );
            })}
          </div>
        </CardContent>
      </Card>
      
      {/* Active Connections */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="w-5 h-5" />
            Active Connections
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <div>
                  <div className="font-medium">WebSocket</div>
                  <div className="text-sm text-muted-foreground">Real-time updates</div>
                </div>
              </div>
              <Badge variant="outline">Connected</Badge>
            </div>
            
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div>
                  <div className="font-medium">REST API</div>
                  <div className="text-sm text-muted-foreground">Settings sync</div>
                </div>
              </div>
              <Badge variant="outline">Active</Badge>
            </div>
            
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                <div>
                  <div className="font-medium">GPU Compute</div>
                  <div className="text-sm text-muted-foreground">Physics engine</div>
                </div>
              </div>
              <Badge variant="outline">
                {systemMetrics.gpu.enabled ? 'Running' : 'Idle'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DashboardTab;