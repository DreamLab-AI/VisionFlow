import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { Progress } from '@/features/design-system/components/Progress';
import { Database, RefreshCw, Download, Upload, Search, Filter } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('DataPanel');

interface DataPanelProps {
  className?: string;
}

interface DataSource {
  id: string;
  name: string;
  type: 'database' | 'api' | 'file' | 'stream';
  status: 'connected' | 'disconnected' | 'error' | 'syncing';
  recordCount: number;
  lastSync: Date;
}

export const DataPanel: React.FC<DataPanelProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [activeTab, setActiveTab] = useState('sources');
  
  // Subscribe only to data-related settings
  const dataSettings = useSelectiveSettings({
    autoSync: 'data.autoSync.enabled',
    syncInterval: 'data.autoSync.intervalMinutes',
    maxRecords: 'data.limits.maxRecords',
    cacheEnabled: 'data.cache.enabled',
    compressionEnabled: 'data.compression.enabled',
    backupEnabled: 'data.backup.enabled',
    exportFormat: 'data.export.defaultFormat',
    importValidation: 'data.import.strictValidation'
  });
  
  // Mock data sources - in real app this would come from store/API
  const dataSources: DataSource[] = useMemo(() => [
    {
      id: '1',
      name: 'Primary Database',
      type: 'database',
      status: 'connected',
      recordCount: 150000,
      lastSync: new Date(Date.now() - 5 * 60 * 1000)
    },
    {
      id: '2',
      name: 'Analytics API',
      type: 'api',
      status: 'syncing',
      recordCount: 45000,
      lastSync: new Date(Date.now() - 2 * 60 * 1000)
    },
    {
      id: '3',
      name: 'Import Data.csv',
      type: 'file',
      status: 'connected',
      recordCount: 12500,
      lastSync: new Date(Date.now() - 30 * 60 * 1000)
    }
  ], []);
  
  const getStatusColor = (status: DataSource['status']) => {
    switch (status) {
      case 'connected': return 'bg-green-100 text-green-800';
      case 'syncing': return 'bg-blue-100 text-blue-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'disconnected': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };
  
  const getTypeIcon = (type: DataSource['type']) => {
    switch (type) {
      case 'database': return <Database size={16} />;
      case 'api': return <RefreshCw size={16} />;
      case 'file': return <Upload size={16} />;
      case 'stream': return <Search size={16} />;
      default: return <Database size={16} />;
    }
  };
  
  const handleSync = (sourceId: string) => {
    logger.info('Syncing data source', { sourceId });
    // In real app, trigger sync operation
  };
  
  const handleExport = () => {
    logger.info('Exporting data', { format: dataSettings.exportFormat });
    // In real app, trigger export
  };
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database size={20} />
            Data Management
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={handleExport}>
              <Download size={16} className="mr-1" />
              Export
            </Button>
            <Button size="sm" variant="outline">
              <Upload size={16} className="mr-1" />
              Import
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="sources">Data Sources</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
            <TabsTrigger value="operations">Operations</TabsTrigger>
          </TabsList>
          
          <TabsContent value="sources" className="space-y-4">
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {dataSources.map((source) => (
                  <div key={source.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getTypeIcon(source.type)}
                        <span className="font-medium">{source.name}</span>
                        <Badge className={getStatusColor(source.status)}>
                          {source.status}
                        </Badge>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleSync(source.id)}
                        disabled={source.status === 'syncing'}
                      >
                        <RefreshCw size={14} className={source.status === 'syncing' ? 'animate-spin' : ''} />
                      </Button>
                    </div>
                    <div className="text-sm text-muted-foreground space-y-1">
                      <div>Records: {source.recordCount.toLocaleString()}</div>
                      <div>Last sync: {source.lastSync.toLocaleTimeString()}</div>
                      {source.status === 'syncing' && (
                        <Progress value={65} className="w-full mt-2" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Auto Sync</label>
                <Button
                  variant={dataSettings.autoSync ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('data.autoSync.enabled', !dataSettings.autoSync)}
                >
                  {dataSettings.autoSync ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Cache Data</label>
                <Button
                  variant={dataSettings.cacheEnabled ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('data.cache.enabled', !dataSettings.cacheEnabled)}
                >
                  {dataSettings.cacheEnabled ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Compression</label>
                <Button
                  variant={dataSettings.compressionEnabled ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('data.compression.enabled', !dataSettings.compressionEnabled)}
                >
                  {dataSettings.compressionEnabled ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Auto Backup</label>
                <Button
                  variant={dataSettings.backupEnabled ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('data.backup.enabled', !dataSettings.backupEnabled)}
                >
                  {dataSettings.backupEnabled ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="operations" className="space-y-4">
            <div className="text-sm text-muted-foreground mb-4">
              Data operations and management tools
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Button variant="outline" className="h-20 flex-col">
                <Download size={20} className="mb-2" />
                <span>Export Data</span>
              </Button>
              <Button variant="outline" className="h-20 flex-col">
                <Upload size={20} className="mb-2" />
                <span>Import Data</span>
              </Button>
              <Button variant="outline" className="h-20 flex-col">
                <RefreshCw size={20} className="mb-2" />
                <span>Sync All</span>
              </Button>
              <Button variant="outline" className="h-20 flex-col">
                <Filter size={20} className="mb-2" />
                <span>Data Filters</span>
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default DataPanel;