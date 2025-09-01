import React, { useMemo, useState } from 'react';
import { useSelectiveSetting, useSelectiveSettings } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Badge } from '@/features/design-system/components/Badge';
import { Button } from '@/features/design-system/components/Button';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { SearchInput } from '@/features/design-system/components/SearchInput';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Bot, Play, Pause, RotateCcw, Trash2, Activity, Clock, CheckCircle, XCircle } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('BotList');

interface BotListProps {
  className?: string;
  compact?: boolean;
}

interface Bot {
  id: string;
  name: string;
  type: 'analyzer' | 'processor' | 'monitor' | 'custom';
  status: 'running' | 'stopped' | 'error' | 'paused' | 'starting' | 'stopping';
  description: string;
  uptime: number;
  cpu: number;
  memory: number;
  tasksCompleted: number;
  tasksPerMinute: number;
  lastError?: string;
  priority: 'low' | 'medium' | 'high';
  tags: string[];
}

export const BotList: React.FC<BotListProps> = ({ className, compact = false }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  
  // Subscribe only to bot list settings
  const botListSettings = useSelectiveSettings({
    sortBy: 'bots.list.sortBy',
    sortOrder: 'bots.list.sortOrder',
    showInactive: 'bots.list.showInactive',
    groupByType: 'bots.list.groupByType',
    refreshInterval: 'bots.list.refreshIntervalSeconds',
    showMetrics: 'bots.list.showMetrics'
  });
  
  // Mock bot data - in real app this would come from store/API
  const bots: Bot[] = useMemo(() => [
    {
      id: '1',
      name: 'Data Stream Analyzer',
      type: 'analyzer',
      status: 'running',
      description: 'Analyzes real-time data streams for patterns and anomalies',
      uptime: 3600000,
      cpu: 23.5,
      memory: 45.2,
      tasksCompleted: 1247,
      tasksPerMinute: 15.3,
      priority: 'high',
      tags: ['production', 'critical']
    },
    {
      id: '2',
      name: 'Background Processor',
      type: 'processor',
      status: 'running',
      description: 'Processes queued background tasks and jobs',
      uptime: 7200000,
      cpu: 12.8,
      memory: 32.1,
      tasksCompleted: 2891,
      tasksPerMinute: 8.7,
      priority: 'medium',
      tags: ['background', 'queue']
    },
    {
      id: '3',
      name: 'System Health Monitor',
      type: 'monitor',
      status: 'error',
      description: 'Monitors system health and performance metrics',
      uptime: 1800000,
      cpu: 5.2,
      memory: 18.7,
      tasksCompleted: 156,
      tasksPerMinute: 2.1,
      lastError: 'Connection timeout to metrics endpoint',
      priority: 'high',
      tags: ['monitoring', 'health']
    },
    {
      id: '4',
      name: 'Data Cleanup Bot',
      type: 'custom',
      status: 'paused',
      description: 'Cleans up old data and optimizes storage',
      uptime: 900000,
      cpu: 8.1,
      memory: 25.4,
      tasksCompleted: 67,
      tasksPerMinute: 0.5,
      priority: 'low',
      tags: ['maintenance', 'cleanup']
    },
    {
      id: '5',
      name: 'API Response Analyzer',
      type: 'analyzer',
      status: 'stopped',
      description: 'Analyzes API response times and error rates',
      uptime: 0,
      cpu: 0,
      memory: 0,
      tasksCompleted: 523,
      tasksPerMinute: 0,
      priority: 'medium',
      tags: ['api', 'performance']
    }
  ], []);
  
  // Filter and sort bots
  const filteredBots = useMemo(() => {
    let filtered = bots.filter(bot => {
      const matchesSearch = bot.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           bot.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           bot.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesStatus = statusFilter === 'all' || bot.status === statusFilter;
      const matchesType = typeFilter === 'all' || bot.type === typeFilter;
      const showInactive = botListSettings.showInactive || bot.status !== 'stopped';
      
      return matchesSearch && matchesStatus && matchesType && showInactive;
    });
    
    // Sort bots
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (botListSettings.sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'status':
          comparison = a.status.localeCompare(b.status);
          break;
        case 'type':
          comparison = a.type.localeCompare(b.type);
          break;
        case 'uptime':
          comparison = a.uptime - b.uptime;
          break;
        case 'tasks':
          comparison = a.tasksCompleted - b.tasksCompleted;
          break;
        case 'priority':
          const priorityOrder = { high: 3, medium: 2, low: 1 };
          comparison = priorityOrder[b.priority] - priorityOrder[a.priority];
          break;
        default:
          comparison = 0;
      }
      
      return botListSettings.sortOrder === 'desc' ? -comparison : comparison;
    });
    
    return filtered;
  }, [bots, searchTerm, statusFilter, typeFilter, botListSettings]);
  
  const getStatusIcon = (status: Bot['status']) => {
    switch (status) {
      case 'running': return <CheckCircle size={16} className="text-green-600" />;
      case 'error': return <XCircle size={16} className="text-red-600" />;
      case 'paused': return <Pause size={16} className="text-yellow-600" />;
      case 'starting': return <RotateCcw size={16} className="text-blue-600 animate-spin" />;
      case 'stopping': return <RotateCcw size={16} className="text-gray-600 animate-spin" />;
      default: return <XCircle size={16} className="text-gray-400" />;
    }
  };
  
  const getStatusColor = (status: Bot['status']) => {
    switch (status) {
      case 'running': return 'bg-green-100 text-green-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'paused': return 'bg-yellow-100 text-yellow-800';
      case 'starting': return 'bg-blue-100 text-blue-800';
      case 'stopping': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };
  
  const getPriorityColor = (priority: Bot['priority']) => {
    switch (priority) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
    }
  };
  
  const formatUptime = (uptimeMs: number) => {
    if (uptimeMs === 0) return 'Not running';
    const hours = Math.floor(uptimeMs / (1000 * 60 * 60));
    const minutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
    return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
  };
  
  const handleBotAction = (botId: string, action: string) => {
    logger.info('Bot action', { botId, action });
    // In real app, dispatch action
  };
  
  if (compact) {
    return (
      <div className={className}>
        <div className="space-y-2">
          {filteredBots.slice(0, 5).map((bot) => (
            <div key={bot.id} className="flex items-center justify-between p-2 border rounded">
              <div className="flex items-center gap-2">
                {getStatusIcon(bot.status)}
                <span className="font-medium text-sm">{bot.name}</span>
                <Badge className={getStatusColor(bot.status)} size="sm">
                  {bot.status}
                </Badge>
              </div>
              <div className="flex items-center gap-1">
                {bot.status === 'running' && (
                  <Button size="sm" variant="ghost" onClick={() => handleBotAction(bot.id, 'pause')}>
                    <Pause size={12} />
                  </Button>
                )}
                {bot.status !== 'running' && (
                  <Button size="sm" variant="ghost" onClick={() => handleBotAction(bot.id, 'start')}>
                    <Play size={12} />
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
        {filteredBots.length > 5 && (
          <div className="text-center mt-2">
            <span className="text-xs text-muted-foreground">
              +{filteredBots.length - 5} more bots
            </span>
          </div>
        )}
      </div>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot size={20} />
            Bot List
            <Badge variant="outline">
              {filteredBots.length} bots
            </Badge>
          </div>
        </CardTitle>
        
        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <SearchInput
            value={searchTerm}
            onChange={setSearchTerm}
            placeholder="Search bots..."
          />
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger>
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="stopped">Stopped</SelectItem>
              <SelectItem value="error">Error</SelectItem>
              <SelectItem value="paused">Paused</SelectItem>
            </SelectContent>
          </Select>
          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger>
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="analyzer">Analyzer</SelectItem>
              <SelectItem value="processor">Processor</SelectItem>
              <SelectItem value="monitor">Monitor</SelectItem>
              <SelectItem value="custom">Custom</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      
      <CardContent>
        <ScrollArea className="h-[500px]">
          <div className="space-y-3">
            {filteredBots.map((bot) => (
              <div key={bot.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(bot.status)}
                    <span className="font-medium">{bot.name}</span>
                    <Badge className={getStatusColor(bot.status)}>
                      {bot.status}
                    </Badge>
                    <Badge className={getPriorityColor(bot.priority)}>
                      {bot.priority}
                    </Badge>
                    <Badge variant="outline">{bot.type}</Badge>
                  </div>
                  <div className="flex items-center gap-1">
                    {bot.status === 'running' ? (
                      <Button size="sm" variant="ghost" onClick={() => handleBotAction(bot.id, 'pause')}>
                        <Pause size={14} />
                      </Button>
                    ) : (
                      <Button size="sm" variant="ghost" onClick={() => handleBotAction(bot.id, 'start')}>
                        <Play size={14} />
                      </Button>
                    )}
                    <Button size="sm" variant="ghost" onClick={() => handleBotAction(bot.id, 'restart')}>
                      <RotateCcw size={14} />
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => handleBotAction(bot.id, 'delete')}>
                      <Trash2 size={14} />
                    </Button>
                  </div>
                </div>
                
                <p className="text-sm text-muted-foreground mb-3">{bot.description}</p>
                
                {/* Tags */}
                <div className="flex flex-wrap gap-1 mb-3">
                  {bot.tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
                
                {/* Metrics */}
                {botListSettings.showMetrics && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="flex items-center gap-1">
                      <Clock size={12} className="text-muted-foreground" />
                      <span className="text-muted-foreground">Uptime:</span>
                      <span className="font-medium">{formatUptime(bot.uptime)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Activity size={12} className="text-muted-foreground" />
                      <span className="text-muted-foreground">CPU:</span>
                      <span className="font-medium">{bot.cpu.toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Memory:</span>
                      <span className="font-medium ml-1">{bot.memory.toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Tasks:</span>
                      <span className="font-medium ml-1">{bot.tasksCompleted.toLocaleString()}</span>
                    </div>
                  </div>
                )}
                
                {bot.lastError && (
                  <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-800">
                    <XCircle size={12} className="inline mr-1" />
                    {bot.lastError}
                  </div>
                )}
              </div>
            ))}
          </div>
        </ScrollArea>
        
        {filteredBots.length === 0 && (
          <div className="text-center py-8">
            <Bot size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">No bots found</p>
            <p className="text-sm text-muted-foreground mt-2">
              Try adjusting your filters or search term
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default BotList;