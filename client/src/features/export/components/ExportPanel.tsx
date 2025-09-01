import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { Progress } from '@/features/design-system/components/Progress';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Download, FileText, Database, Image, Code, Calendar } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('ExportPanel');

interface ExportPanelProps {
  className?: string;
}

interface ExportJob {
  id: string;
  name: string;
  format: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  size?: number;
  createdAt: Date;
  completedAt?: Date;
  downloadUrl?: string;
  error?: string;
}

export const ExportPanel: React.FC<ExportPanelProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [selectedFormat, setSelectedFormat] = useState('json');
  const [selectedContent, setSelectedContent] = useState('all');
  const [isExporting, setIsExporting] = useState(false);
  
  // Subscribe only to export-related settings
  const exportSettings = useSelectiveSettings({
    enabled: 'export.enabled',
    defaultFormat: 'export.defaultFormat',
    includeMetadata: 'export.includeMetadata',
    compression: 'export.compression.enabled',
    compressionLevel: 'export.compression.level',
    maxFileSize: 'export.limits.maxFileSizeMB',
    retentionDays: 'export.cleanup.retentionDays',
    notifyOnComplete: 'export.notifications.onComplete',
    autoCleanup: 'export.cleanup.enabled'
  });
  
  // Mock export jobs - in real app this would come from store/API
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([
    {
      id: '1',
      name: 'User Data Export',
      format: 'csv',
      status: 'completed',
      progress: 100,
      size: 2.4, // MB
      createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 1.5 * 60 * 60 * 1000),
      downloadUrl: '/exports/user-data-export.csv'
    },
    {
      id: '2',
      name: 'Analytics Report',
      format: 'pdf',
      status: 'processing',
      progress: 65,
      createdAt: new Date(Date.now() - 15 * 60 * 1000)
    },
    {
      id: '3',
      name: 'System Configuration',
      format: 'json',
      status: 'error',
      progress: 0,
      createdAt: new Date(Date.now() - 5 * 60 * 1000),
      error: 'Insufficient permissions to access system configuration'
    }
  ]);
  
  const availableFormats = useMemo(() => [
    { value: 'json', label: 'JSON', icon: <Code size={16} />, description: 'Structured data format' },
    { value: 'csv', label: 'CSV', icon: <FileText size={16} />, description: 'Comma-separated values' },
    { value: 'xlsx', label: 'Excel', icon: <FileText size={16} />, description: 'Microsoft Excel format' },
    { value: 'pdf', label: 'PDF', icon: <FileText size={16} />, description: 'Portable document format' },
    { value: 'xml', label: 'XML', icon: <Code size={16} />, description: 'Extensible markup language' },
    { value: 'sql', label: 'SQL', icon: <Database size={16} />, description: 'Database dump' },
    { value: 'png', label: 'PNG', icon: <Image size={16} />, description: 'Visual export as image' }
  ], []);
  
  const contentOptions = useMemo(() => [
    { value: 'all', label: 'All Data' },
    { value: 'users', label: 'Users Only' },
    { value: 'analytics', label: 'Analytics Data' },
    { value: 'settings', label: 'Configuration' },
    { value: 'logs', label: 'System Logs' },
    { value: 'custom', label: 'Custom Selection' }
  ], []);
  
  const handleExport = async () => {
    setIsExporting(true);
    logger.info('Starting export', { format: selectedFormat, content: selectedContent });
    
    const newJob: ExportJob = {
      id: Date.now().toString(),
      name: `${contentOptions.find(c => c.value === selectedContent)?.label} Export`,
      format: selectedFormat,
      status: 'pending',
      progress: 0,
      createdAt: new Date()
    };
    
    setExportJobs(prev => [newJob, ...prev]);
    
    // Simulate export process
    setTimeout(() => {
      setExportJobs(prev => prev.map(job => 
        job.id === newJob.id ? { ...job, status: 'processing' as const } : job
      ));
      
      // Simulate progress updates
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress >= 100) {
          progress = 100;
          clearInterval(progressInterval);
          
          setExportJobs(prev => prev.map(job => 
            job.id === newJob.id ? {
              ...job,
              status: 'completed' as const,
              progress: 100,
              size: Math.random() * 10 + 1, // Random size between 1-11 MB
              completedAt: new Date(),
              downloadUrl: `/exports/${newJob.name.toLowerCase().replace(/\s+/g, '-')}.${selectedFormat}`
            } : job
          ));
          
          setIsExporting(false);
        } else {
          setExportJobs(prev => prev.map(job => 
            job.id === newJob.id ? { ...job, progress } : job
          ));
        }
      }, 500);
    }, 1000);
  };
  
  const handleDownload = (job: ExportJob) => {
    if (job.downloadUrl) {
      logger.info('Downloading export', { jobId: job.id, url: job.downloadUrl });
      // In real app, trigger download
    }
  };
  
  const getStatusColor = (status: ExportJob['status']) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-blue-100 text-blue-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };
  
  const formatFileSize = (sizeMB?: number) => {
    if (!sizeMB) return 'Unknown size';
    return sizeMB < 1 ? `${Math.round(sizeMB * 1000)}KB` : `${sizeMB.toFixed(1)}MB`;
  };
  
  if (!exportSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download size={20} />
            Export
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Download size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">Export is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable export functionality in settings to download data
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
            <Download size={20} />
            Export Panel
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="create" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="create">Create Export</TabsTrigger>
            <TabsTrigger value="history">Export History</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>
          
          <TabsContent value="create" className="space-y-6">
            {/* Export Configuration */}
            <div className="space-y-4">
              <h3 className="font-medium">Export Configuration</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Content to Export</label>
                  <Select value={selectedContent} onValueChange={setSelectedContent}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {contentOptions.map(option => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <label className="text-sm font-medium">Export Format</label>
                  <Select value={selectedFormat} onValueChange={setSelectedFormat}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {availableFormats.map(format => (
                        <SelectItem key={format.value} value={format.value}>
                          <div className="flex items-center gap-2">
                            {format.icon}
                            {format.label}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="p-3 bg-blue-50 border border-blue-200 rounded">
                <div className="flex items-center gap-2 mb-2">
                  {availableFormats.find(f => f.value === selectedFormat)?.icon}
                  <span className="text-sm font-medium">{selectedFormat.toUpperCase()} Export</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  {availableFormats.find(f => f.value === selectedFormat)?.description}
                </p>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="include-metadata"
                    checked={exportSettings.includeMetadata}
                    onChange={(e) => set('export.includeMetadata', e.target.checked)}
                  />
                  <label htmlFor="include-metadata" className="text-sm">Include metadata</label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="enable-compression"
                    checked={exportSettings.compression}
                    onChange={(e) => set('export.compression.enabled', e.target.checked)}
                  />
                  <label htmlFor="enable-compression" className="text-sm">Enable compression</label>
                </div>
              </div>
              
              <Button 
                onClick={handleExport} 
                disabled={isExporting}
                className="w-full"
              >
                <Download size={16} className="mr-2" />
                {isExporting ? 'Creating Export...' : 'Start Export'}
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="history" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">Recent Exports</h3>
              <Badge variant="outline">{exportJobs.length} jobs</Badge>
            </div>
            
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {exportJobs.map((job) => (
                  <div key={job.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{job.name}</span>
                        <Badge className={getStatusColor(job.status)}>
                          {job.status}
                        </Badge>
                        <Badge variant="outline">{job.format.toUpperCase()}</Badge>
                      </div>
                      {job.status === 'completed' && job.downloadUrl && (
                        <Button
                          size="sm"
                          onClick={() => handleDownload(job)}
                        >
                          <Download size={14} className="mr-1" />
                          Download
                        </Button>
                      )}
                    </div>
                    
                    {job.status === 'processing' && (
                      <div className="space-y-1">
                        <Progress value={job.progress} className="w-full" />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Processing...</span>
                          <span>{Math.round(job.progress)}%</span>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex items-center gap-4 text-sm text-muted-foreground mt-2">
                      <div className="flex items-center gap-1">
                        <Calendar size={12} />
                        Created: {job.createdAt.toLocaleString()}
                      </div>
                      {job.size && (
                        <span>Size: {formatFileSize(job.size)}</span>
                      )}
                      {job.completedAt && (
                        <span>Completed: {job.completedAt.toLocaleString()}</span>
                      )}
                    </div>
                    
                    {job.error && (
                      <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-800">
                        Error: {job.error}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Auto Cleanup</span>
                <Button
                  variant={exportSettings.autoCleanup ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('export.cleanup.enabled', !exportSettings.autoCleanup)}
                >
                  {exportSettings.autoCleanup ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Completion Notifications</span>
                <Button
                  variant={exportSettings.notifyOnComplete ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('export.notifications.onComplete', !exportSettings.notifyOnComplete)}
                >
                  {exportSettings.notifyOnComplete ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
            </div>
            
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">Export Limits</h3>
              <div className="text-sm text-muted-foreground space-y-1">
                <p>Maximum file size: {exportSettings.maxFileSize}MB</p>
                <p>Retention period: {exportSettings.retentionDays} days</p>
                <p>Compression level: {exportSettings.compressionLevel}</p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default ExportPanel;