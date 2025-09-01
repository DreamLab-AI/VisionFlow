import React, { useState, useCallback } from 'react';
import { useSelectiveSetting, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Progress } from '@/features/design-system/components/Progress';
import { Alert } from '@/features/design-system/components/Dialog'; // Using Dialog as base for Alert
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Upload, FileText, Database, Check, X, AlertTriangle } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('DataImporter');

interface DataImporterProps {
  className?: string;
}

interface ImportJob {
  id: string;
  fileName: string;
  fileSize: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  recordsProcessed: number;
  totalRecords: number;
  errors?: string[];
}

export const DataImporter: React.FC<DataImporterProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [importJobs, setImportJobs] = useState<ImportJob[]>([]);
  const [dragOver, setDragOver] = useState(false);
  
  // Subscribe only to import-related settings
  const fileFormat = useSelectiveSetting<string>('data.import.defaultFormat');
  const batchSize = useSelectiveSetting<number>('data.import.batchSize');
  const strictValidation = useSelectiveSetting<boolean>('data.import.strictValidation');
  const autoMapping = useSelectiveSetting<boolean>('data.import.autoMapping');
  const duplicateHandling = useSelectiveSetting<string>('data.import.duplicateHandling');
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  }, []);
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);
  
  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    handleFiles(files);
  }, []);
  
  const handleFiles = useCallback((files: File[]) => {
    const newJobs: ImportJob[] = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      fileName: file.name,
      fileSize: file.size,
      status: 'pending',
      progress: 0,
      recordsProcessed: 0,
      totalRecords: 0
    }));
    
    setImportJobs(prev => [...prev, ...newJobs]);
    
    // Start processing jobs
    newJobs.forEach(job => processImportJob(job.id));
  }, []);
  
  const processImportJob = useCallback(async (jobId: string) => {
    setImportJobs(prev => prev.map(job => 
      job.id === jobId ? { ...job, status: 'processing', totalRecords: 1000 } : job
    ));
    
    // Simulate processing with progress updates
    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 200));
      setImportJobs(prev => prev.map(job => 
        job.id === jobId ? { 
          ...job, 
          progress: i,
          recordsProcessed: Math.floor((i / 100) * 1000)
        } : job
      ));
    }
    
    // Complete the job
    setImportJobs(prev => prev.map(job => 
      job.id === jobId ? { 
        ...job, 
        status: 'completed',
        progress: 100,
        recordsProcessed: 1000
      } : job
    ));
    
    logger.info('Import job completed', { jobId });
  }, []);
  
  const getStatusIcon = (status: ImportJob['status']) => {
    switch (status) {
      case 'completed': return <Check size={16} className="text-green-600" />;
      case 'error': return <X size={16} className="text-red-600" />;
      case 'processing': return <Upload size={16} className="text-blue-600 animate-pulse" />;
      default: return <FileText size={16} className="text-gray-600" />;
    }
  };
  
  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload size={20} />
          Data Importer
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Drop Zone */}
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <Upload size={48} className="mx-auto mb-4 text-gray-400" />
          <p className="text-lg font-medium mb-2">Drop files here to import</p>
          <p className="text-sm text-muted-foreground mb-4">
            Supports CSV, JSON, Excel files up to 100MB
          </p>
          <input
            type="file"
            multiple
            accept=".csv,.json,.xlsx,.xls"
            onChange={handleFileInput}
            className="hidden"
            id="file-input"
          />
          <Button asChild>
            <label htmlFor="file-input" className="cursor-pointer">
              Select Files
            </label>
          </Button>
        </div>
        
        {/* Import Settings */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">File Format</label>
            <Select value={fileFormat} onValueChange={(value) => set('data.import.defaultFormat', value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="csv">CSV</SelectItem>
                <SelectItem value="json">JSON</SelectItem>
                <SelectItem value="excel">Excel</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Duplicate Handling</label>
            <Select value={duplicateHandling} onValueChange={(value) => set('data.import.duplicateHandling', value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="skip">Skip Duplicates</SelectItem>
                <SelectItem value="update">Update Existing</SelectItem>
                <SelectItem value="append">Append All</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <Button
            variant={strictValidation ? 'default' : 'outline'}
            size="sm"
            onClick={() => set('data.import.strictValidation', !strictValidation)}
          >
            {strictValidation ? 'Strict Validation' : 'Loose Validation'}
          </Button>
          <Button
            variant={autoMapping ? 'default' : 'outline'}
            size="sm"
            onClick={() => set('data.import.autoMapping', !autoMapping)}
          >
            {autoMapping ? 'Auto Mapping' : 'Manual Mapping'}
          </Button>
        </div>
        
        {/* Import Jobs */}
        {importJobs.length > 0 && (
          <div className="space-y-3">
            <h3 className="font-medium">Import Jobs</h3>
            <div className="space-y-2">
              {importJobs.map((job) => (
                <div key={job.id} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(job.status)}
                      <span className="font-medium text-sm">{job.fileName}</span>
                      <span className="text-xs text-muted-foreground">
                        {formatFileSize(job.fileSize)}
                      </span>
                    </div>
                    <span className="text-xs text-muted-foreground capitalize">
                      {job.status}
                    </span>
                  </div>
                  
                  {job.status === 'processing' && (
                    <div className="space-y-1">
                      <Progress value={job.progress} className="w-full" />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>{job.recordsProcessed} / {job.totalRecords} records</span>
                        <span>{job.progress}%</span>
                      </div>
                    </div>
                  )}
                  
                  {job.status === 'completed' && (
                    <div className="text-xs text-green-600">
                      Successfully imported {job.recordsProcessed.toLocaleString()} records
                    </div>
                  )}
                  
                  {job.errors && job.errors.length > 0 && (
                    <div className="text-xs text-red-600 mt-1">
                      <AlertTriangle size={12} className="inline mr-1" />
                      {job.errors[0]}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default DataImporter;