import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/features/design-system/components/Dialog';
import { Input } from '@/features/design-system/components/Input';
import { Textarea } from '@/features/design-system/components/Textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Bot, Plus, Play, Pause, Settings, Trash2, Edit } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('BotManager');

interface BotManagerProps {
  className?: string;
}

interface BotInstance {
  id: string;
  name: string;
  type: 'analyzer' | 'processor' | 'monitor' | 'custom';
  status: 'running' | 'stopped' | 'error' | 'paused';
  description: string;
  uptime: number;
  tasksCompleted: number;
  config: Record<string, any>;
  createdAt: Date;
  lastActivity: Date;
}

export const BotManager: React.FC<BotManagerProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [selectedBot, setSelectedBot] = useState<BotInstance | null>(null);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newBotData, setNewBotData] = useState({
    name: '',
    type: 'analyzer' as BotInstance['type'],
    description: ''
  });
  
  // Subscribe only to bot-related settings
  const botSettings = useSelectiveSettings({
    maxBots: 'bots.limits.maxInstances',
    autoStart: 'bots.autoStart.enabled',
    healthCheck: 'bots.healthCheck.enabled',
    healthCheckInterval: 'bots.healthCheck.intervalSeconds',
    logging: 'bots.logging.enabled',
    logLevel: 'bots.logging.level',
    retryAttempts: 'bots.retry.maxAttempts',
    retryDelay: 'bots.retry.delaySeconds'
  });
  
  // Mock bot instances - in real app this would come from store/API
  const botInstances: BotInstance[] = useMemo(() => [
    {
      id: '1',
      name: 'Data Analyzer Bot',
      type: 'analyzer',
      status: 'running',
      description: 'Analyzes incoming data streams and generates insights',
      uptime: 3600000, // 1 hour in ms
      tasksCompleted: 247,
      config: { threshold: 0.8, batchSize: 100 },
      createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      lastActivity: new Date(Date.now() - 2 * 60 * 1000)
    },
    {
      id: '2',
      name: 'Performance Monitor',
      type: 'monitor',
      status: 'running',
      description: 'Monitors system performance and alerts on issues',
      uptime: 86400000, // 24 hours in ms
      tasksCompleted: 1523,
      config: { cpuThreshold: 80, memoryThreshold: 90 },
      createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
      lastActivity: new Date(Date.now() - 30 * 1000)
    },
    {
      id: '3',
      name: 'Queue Processor',
      type: 'processor',
      status: 'error',
      description: 'Processes background tasks from the job queue',
      uptime: 1800000, // 30 minutes in ms
      tasksCompleted: 89,
      config: { concurrency: 5, timeout: 30000 },
      createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      lastActivity: new Date(Date.now() - 10 * 60 * 1000)
    }
  ], []);
  
  const getStatusColor = (status: BotInstance['status']) => {
    switch (status) {
      case 'running': return 'bg-green-100 text-green-800';
      case 'paused': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'stopped': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };
  
  const getTypeIcon = (type: BotInstance['type']) => {
    // All types use Bot icon for consistency
    return <Bot size={16} />;
  };
  
  const formatUptime = (uptimeMs: number) => {
    const hours = Math.floor(uptimeMs / (1000 * 60 * 60));
    const minutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
    return `${hours}h ${minutes}m`;
  };
  
  const handleBotAction = (botId: string, action: 'start' | 'stop' | 'pause' | 'delete') => {
    logger.info('Bot action triggered', { botId, action });
    // In real app, dispatch action to bot service
  };
  
  const handleCreateBot = () => {
    logger.info('Creating new bot', newBotData);
    // In real app, create bot instance
    setIsCreateDialogOpen(false);
    setNewBotData({ name: '', type: 'analyzer', description: '' });
  };
  
  const runningBots = botInstances.filter(bot => bot.status === 'running').length;
  const errorBots = botInstances.filter(bot => bot.status === 'error').length;
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot size={20} />
            Bot Manager
            <Badge variant="outline">
              {runningBots}/{botInstances.length} running
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
              <DialogTrigger asChild>
                <Button size="sm">
                  <Plus size={16} className="mr-1" />
                  New Bot
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Bot</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Name</label>
                    <Input
                      value={newBotData.name}
                      onChange={(e) => setNewBotData(prev => ({ ...prev, name: e.target.value }))}
                      placeholder="Enter bot name"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Type</label>
                    <Select 
                      value={newBotData.type} 
                      onValueChange={(value: BotInstance['type']) => 
                        setNewBotData(prev => ({ ...prev, type: value }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="analyzer">Data Analyzer</SelectItem>
                        <SelectItem value="processor">Task Processor</SelectItem>
                        <SelectItem value="monitor">System Monitor</SelectItem>
                        <SelectItem value="custom">Custom Bot</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Description</label>
                    <Textarea
                      value={newBotData.description}
                      onChange={(e) => setNewBotData(prev => ({ ...prev, description: e.target.value }))}
                      placeholder="Enter bot description"
                    />
                  </div>
                  <Button onClick={handleCreateBot} className="w-full">
                    Create Bot
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{runningBots}</div>
            <div className="text-sm text-muted-foreground">Running</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{errorBots}</div>
            <div className="text-sm text-muted-foreground">Errors</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{botInstances.length}</div>
            <div className="text-sm text-muted-foreground">Total Bots</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{botSettings.maxBots}</div>
            <div className="text-sm text-muted-foreground">Max Allowed</div>
          </div>
        </div>
        
        {/* Bot List */}
        <ScrollArea className="h-[400px]">
          <div className="space-y-3">
            {botInstances.map((bot) => (
              <div key={bot.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getTypeIcon(bot.type)}
                    <span className="font-medium">{bot.name}</span>
                    <Badge className={getStatusColor(bot.status)}>
                      {bot.status}
                    </Badge>
                    <Badge variant="outline">{bot.type}</Badge>
                  </div>
                  <div className="flex items-center gap-1">
                    {bot.status === 'running' ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleBotAction(bot.id, 'pause')}
                      >
                        <Pause size={14} />
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleBotAction(bot.id, 'start')}
                      >
                        <Play size={14} />
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setSelectedBot(bot)}
                    >
                      <Settings size={14} />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleBotAction(bot.id, 'delete')}
                    >
                      <Trash2 size={14} />
                    </Button>
                  </div>
                </div>
                
                <p className="text-sm text-muted-foreground mb-3">{bot.description}</p>
                
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Uptime: </span>
                    <span className="font-medium">{formatUptime(bot.uptime)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Tasks: </span>
                    <span className="font-medium">{bot.tasksCompleted.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Last Active: </span>
                    <span className="font-medium">{bot.lastActivity.toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
        
        {/* Settings Panel */}
        <div className="mt-6 border-t pt-4">
          <h3 className="font-medium mb-4">Bot Settings</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Auto-start Bots</span>
              <Button
                variant={botSettings.autoStart ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('bots.autoStart.enabled', !botSettings.autoStart)}
              >
                {botSettings.autoStart ? 'Enabled' : 'Disabled'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Health Monitoring</span>
              <Button
                variant={botSettings.healthCheck ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('bots.healthCheck.enabled', !botSettings.healthCheck)}
              >
                {botSettings.healthCheck ? 'Enabled' : 'Disabled'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Logging</span>
              <Button
                variant={botSettings.logging ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('bots.logging.enabled', !botSettings.logging)}
              >
                {botSettings.logging ? 'Enabled' : 'Disabled'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Log Level</span>
              <Badge variant="outline">{botSettings.logLevel}</Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default BotManager;