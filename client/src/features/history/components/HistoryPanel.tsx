import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Input } from '@/features/design-system/components/Input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { History, RotateCcw, Trash2, Search, Filter, Calendar } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('HistoryPanel');

interface HistoryPanelProps {
  className?: string;
}

interface HistoryEntry {
  id: string;
  action: string;
  type: 'user' | 'system' | 'data' | 'setting';
  description: string;
  timestamp: Date;
  user?: string;
  metadata?: Record<string, any>;
  reversible: boolean;
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedTimeRange, setSelectedTimeRange] = useState<string>('all');
  
  // Subscribe only to history-related settings
  const historySettings = useSelectiveSettings({
    enabled: 'history.enabled',
    retentionDays: 'history.retentionDays',
    maxEntries: 'history.maxEntries',
    trackUserActions: 'history.trackUserActions',
    trackSystemEvents: 'history.trackSystemEvents',
    trackDataChanges: 'history.trackDataChanges',
    allowRevert: 'history.allowRevert',
    autoCleanup: 'history.autoCleanup'
  });
  
  // Mock history data - in real app this would come from store/API
  const historyEntries: HistoryEntry[] = useMemo(() => [
    {
      id: '1',
      action: 'Settings Updated',
      type: 'setting',
      description: 'Updated graph visualization settings - enabled bloom effect and increased node size',
      timestamp: new Date(Date.now() - 5 * 60 * 1000),
      user: 'John Smith',
      metadata: { section: 'visualization', changes: ['bloom', 'nodeSize'] },
      reversible: true
    },
    {
      id: '2',
      action: 'Data Import',
      type: 'data',
      description: 'Imported 1,247 records from user-analytics.csv',
      timestamp: new Date(Date.now() - 15 * 60 * 1000),
      user: 'Jane Doe',
      metadata: { fileName: 'user-analytics.csv', recordCount: 1247 },
      reversible: false
    },
    {
      id: '3',
      action: 'User Login',
      type: 'user',
      description: 'User authentication successful from IP 192.168.1.100',
      timestamp: new Date(Date.now() - 45 * 60 * 1000),
      user: 'System Admin',
      metadata: { ip: '192.168.1.100', method: 'oauth' },
      reversible: false
    },
    {
      id: '4',
      action: 'System Restart',
      type: 'system',
      description: 'Application restarted for maintenance update v2.1.3',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
      metadata: { version: 'v2.1.3', reason: 'maintenance' },
      reversible: false
    },
    {
      id: '5',
      action: 'Bot Created',
      type: 'system',
      description: 'Created new performance monitoring bot with 5-minute check interval',
      timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000),
      user: 'Admin User',
      metadata: { botType: 'monitor', interval: 300 },
      reversible: true
    },
    {
      id: '6',
      action: 'Data Export',
      type: 'data',
      description: 'Exported analytics data to CSV format (2.3MB)',
      timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000),
      user: 'Jane Doe',
      metadata: { format: 'csv', size: 2.3 },
      reversible: false
    }
  ], []);
  
  const timeRanges = useMemo(() => [
    { value: 'all', label: 'All Time' },
    { value: '1h', label: 'Last Hour' },
    { value: '1d', label: 'Last Day' },
    { value: '1w', label: 'Last Week' },
    { value: '1m', label: 'Last Month' }
  ], []);
  
  const typeOptions = useMemo(() => [
    { value: 'all', label: 'All Types' },
    { value: 'user', label: 'User Actions' },
    { value: 'system', label: 'System Events' },
    { value: 'data', label: 'Data Changes' },
    { value: 'setting', label: 'Settings' }
  ], []);
  
  // Filter history entries
  const filteredEntries = useMemo(() => {
    return historyEntries.filter(entry => {
      const matchesSearch = !searchTerm || 
        entry.action.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entry.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entry.user?.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesType = selectedType === 'all' || entry.type === selectedType;
      
      const matchesTimeRange = (() => {
        if (selectedTimeRange === 'all') return true;
        const now = Date.now();
        const entryTime = entry.timestamp.getTime();
        
        switch (selectedTimeRange) {
          case '1h': return now - entryTime <= 60 * 60 * 1000;
          case '1d': return now - entryTime <= 24 * 60 * 60 * 1000;
          case '1w': return now - entryTime <= 7 * 24 * 60 * 60 * 1000;
          case '1m': return now - entryTime <= 30 * 24 * 60 * 60 * 1000;
          default: return true;
        }
      })();
      
      return matchesSearch && matchesType && matchesTimeRange;
    });
  }, [historyEntries, searchTerm, selectedType, selectedTimeRange]);
  
  const getTypeIcon = (type: HistoryEntry['type']) => {
    switch (type) {
      case 'user': return '👤';
      case 'system': return '⚙️';
      case 'data': return '📊';
      case 'setting': return '🔧';
      default: return '📋';
    }
  };
  
  const getTypeColor = (type: HistoryEntry['type']) => {
    switch (type) {
      case 'user': return 'bg-blue-100 text-blue-800';
      case 'system': return 'bg-orange-100 text-orange-800';
      case 'data': return 'bg-green-100 text-green-800';
      case 'setting': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };
  
  const handleRevert = (entry: HistoryEntry) => {
    if (!entry.reversible) return;
    
    logger.info('Reverting history entry', { entryId: entry.id, action: entry.action });
    // In real app, trigger revert operation
  };
  
  const handleClearHistory = () => {
    logger.info('Clearing history');
    // In real app, clear history
  };
  
  const formatTimeAgo = (timestamp: Date) => {
    const now = Date.now();
    const diffMs = now - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };
  
  if (!historySettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History size={20} />
            History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <History size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">History tracking is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable history tracking in settings to monitor system activity
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
            <History size={20} />
            Activity History
            <Badge variant="outline">{filteredEntries.length} entries</Badge>
          </div>
          <Button 
            size="sm" 
            variant="outline" 
            onClick={handleClearHistory}
          >
            <Trash2 size={16} className="mr-1" />
            Clear History
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="timeline" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>
          
          <TabsContent value="timeline" className="space-y-4">
            {/* Filters */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="relative">
                <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search history..."
                  className="pl-10"
                />
              </div>
              <Select value={selectedType} onValueChange={setSelectedType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {typeOptions.map(option => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {timeRanges.map(range => (
                    <SelectItem key={range.value} value={range.value}>
                      {range.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            {/* History Timeline */}
            <ScrollArea className="h-[500px]">
              <div className="space-y-3">
                {filteredEntries.map((entry) => (
                  <div key={entry.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{getTypeIcon(entry.type)}</span>
                        <div>
                          <span className="font-medium">{entry.action}</span>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge className={getTypeColor(entry.type)}>
                              {entry.type}
                            </Badge>
                            {entry.reversible && historySettings.allowRevert && (
                              <Badge className="bg-yellow-100 text-yellow-800">
                                Reversible
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          {formatTimeAgo(entry.timestamp)}
                        </span>
                        {entry.reversible && historySettings.allowRevert && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleRevert(entry)}
                          >
                            <RotateCcw size={14} className="mr-1" />
                            Revert
                          </Button>
                        )}
                      </div>
                    </div>
                    
                    <p className="text-sm text-muted-foreground mb-2">
                      {entry.description}
                    </p>
                    
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Calendar size={12} />
                        {entry.timestamp.toLocaleString()}
                      </div>
                      {entry.user && (
                        <span>by {entry.user}</span>
                      )}
                    </div>
                    
                    {entry.metadata && Object.keys(entry.metadata).length > 0 && (
                      <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                        <span className="font-medium">Details: </span>
                        {Object.entries(entry.metadata).map(([key, value]) => (
                          <span key={key} className="mr-2">
                            {key}: {String(value)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
            
            {filteredEntries.length === 0 && (
              <div className="text-center py-8">
                <History size={48} className="mx-auto mb-4 text-gray-400" />
                <p className="text-muted-foreground">No history entries found</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Try adjusting your filters or time range
                </p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Track User Actions</span>
                <Button
                  variant={historySettings.trackUserActions ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('history.trackUserActions', !historySettings.trackUserActions)}
                >
                  {historySettings.trackUserActions ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Track System Events</span>
                <Button
                  variant={historySettings.trackSystemEvents ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('history.trackSystemEvents', !historySettings.trackSystemEvents)}
                >
                  {historySettings.trackSystemEvents ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Track Data Changes</span>
                <Button
                  variant={historySettings.trackDataChanges ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('history.trackDataChanges', !historySettings.trackDataChanges)}
                >
                  {historySettings.trackDataChanges ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Allow Revert</span>
                <Button
                  variant={historySettings.allowRevert ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('history.allowRevert', !historySettings.allowRevert)}
                >
                  {historySettings.allowRevert ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Auto Cleanup</span>
                <Button
                  variant={historySettings.autoCleanup ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('history.autoCleanup', !historySettings.autoCleanup)}
                >
                  {historySettings.autoCleanup ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
            </div>
            
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">Storage Settings</h3>
              <div className="text-sm text-muted-foreground space-y-1">
                <p>Maximum entries: {historySettings.maxEntries}</p>
                <p>Retention period: {historySettings.retentionDays} days</p>
                <p>Current entries: {historyEntries.length}</p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default HistoryPanel;