import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Card } from '../../design-system/components/Card';
import { Select } from '../../design-system/components/Select';
import { SearchInput } from '../../design-system/components/SearchInput';
import { Switch } from '../../design-system/components/Switch';
import { Label } from '../../design-system/components/Label';
import { Button } from '../../design-system/components/Button';
import { createLogger } from '../../../utils/logger';
import { LogEntry, TelemetryFilters } from '../types';

const logger = createLogger('LogViewer');

interface LogViewerProps {
  logEntries: LogEntry[];
  filters: TelemetryFilters;
  onFiltersChange: (filters: Partial<TelemetryFilters>) => void;
  agentIds: string[];
  className?: string;
}

export const LogViewer: React.FC<LogViewerProps> = ({
  logEntries,
  filters,
  onFiltersChange,
  agentIds,
  className = ''
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLevel, setSelectedLevel] = useState<LogEntry['level'] | 'all'>('all');
  const [selectedAgent, setSelectedAgent] = useState<string>('all');
  const [selectedSource, setSelectedSource] = useState<string>('all');
  const [autoScroll, setAutoScroll] = useState(true);
  const [showMetadata, setShowMetadata] = useState(false);

  const logContainerRef = useRef<HTMLDivElement>(null);

  // Get unique sources
  const sources = useMemo(() => {
    const sourceSet = new Set<string>();
    logEntries.forEach(entry => sourceSet.add(entry.source));
    return Array.from(sourceSet).sort();
  }, [logEntries]);

  // Filter and search logs
  const filteredLogs = useMemo(() => {
    let filtered = [...logEntries];

    // Apply level filter
    if (selectedLevel !== 'all') {
      filtered = filtered.filter(entry => entry.level === selectedLevel);
    }

    // Apply agent filter
    if (selectedAgent !== 'all') {
      filtered = filtered.filter(entry => entry.agentId === selectedAgent);
    }

    // Apply source filter
    if (selectedSource !== 'all') {
      filtered = filtered.filter(entry => entry.source === selectedSource);
    }

    // Apply search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(entry =>
        entry.message.toLowerCase().includes(query) ||
        entry.source.toLowerCase().includes(query) ||
        (entry.agentId && entry.agentId.toLowerCase().includes(query)) ||
        (entry.tags && entry.tags.some(tag => tag.toLowerCase().includes(query)))
      );
    }

    // Apply global filters
    if (filters.logLevels && filters.logLevels.length > 0) {
      filtered = filtered.filter(entry => filters.logLevels!.includes(entry.level));
    }

    if (filters.agentIds && filters.agentIds.length > 0) {
      filtered = filtered.filter(entry =>
        entry.agentId && filters.agentIds!.includes(entry.agentId)
      );
    }

    if (filters.dateRange) {
      filtered = filtered.filter(entry =>
        entry.timestamp >= filters.dateRange!.start &&
        entry.timestamp <= filters.dateRange!.end
      );
    }

    return filtered.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }, [logEntries, selectedLevel, selectedAgent, selectedSource, searchQuery, filters]);

  // Auto-scroll to top when new logs arrive
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = 0;
    }
  }, [filteredLogs, autoScroll]);

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'debug': return 'text-gray-600 bg-gray-50 border-gray-200';
      case 'info': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'warn': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'error': return 'text-red-600 bg-red-50 border-red-200';
      case 'critical': return 'text-purple-600 bg-purple-50 border-purple-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getLevelIcon = (level: LogEntry['level']) => {
    switch (level) {
      case 'debug': return '🐛';
      case 'info': return 'ℹ️';
      case 'warn': return '⚠️';
      case 'error': return '❌';
      case 'critical': return '🚨';
      default: return '📝';
    }
  };

  const exportLogs = (format: 'txt' | 'json' | 'csv') => {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const filename = `logs-${timestamp}.${format}`;

    let content: string;

    switch (format) {
      case 'txt':
        content = filteredLogs.map(entry =>
          `${entry.timestamp.toISOString()} [${entry.level.toUpperCase()}] ${entry.source}${entry.agentId ? ` (${entry.agentId})` : ''}: ${entry.message}`
        ).join('\n');
        break;
      case 'json':
        content = JSON.stringify(filteredLogs, null, 2);
        break;
      case 'csv':
        const headers = ['Timestamp', 'Level', 'Source', 'Agent ID', 'Message', 'Tags'];
        const rows = filteredLogs.map(entry => [
          entry.timestamp.toISOString(),
          entry.level,
          entry.source,
          entry.agentId || '',
          entry.message.replace(/"/g, '""'),
          (entry.tags || []).join(';')
        ]);
        content = [headers, ...rows].map(row =>
          row.map(cell => `"${cell}"`).join(',')
        ).join('\n');
        break;
      default:
        return;
    }

    const blob = new Blob([content], {
      type: format === 'json' ? 'application/json' : 'text/plain'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    logger.info(`Exported ${filteredLogs.length} log entries as ${format}`);
  };

  const clearFilters = () => {
    setSearchQuery('');
    setSelectedLevel('all');
    setSelectedAgent('all');
    setSelectedSource('all');
    onFiltersChange({
      logLevels: undefined,
      agentIds: undefined,
      sources: undefined,
      tags: undefined
    });
  };

  return (
    <Card className={className}>
      <div className="p-4 h-full flex flex-col">
        {/* Header and Controls */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Log Viewer</h3>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2">
              <Label htmlFor="auto-scroll" className="text-sm">Auto-scroll</Label>
              <Switch
                id="auto-scroll"
                checked={autoScroll}
                onCheckedChange={setAutoScroll}
                size="sm"
              />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="show-metadata" className="text-sm">Metadata</Label>
              <Switch
                id="show-metadata"
                checked={showMetadata}
                onCheckedChange={setShowMetadata}
                size="sm"
              />
            </div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 mb-4">
          <SearchInput
            placeholder="Search logs..."
            value={searchQuery}
            onChange={setSearchQuery}
            className="md:col-span-2"
          />

          <Select value={selectedLevel} onValueChange={(value: any) => setSelectedLevel(value)}>
            <Select.Trigger>
              <Select.Value placeholder="Log level" />
            </Select.Trigger>
            <Select.Content>
              <Select.Item value="all">All Levels</Select.Item>
              <Select.Item value="debug">Debug</Select.Item>
              <Select.Item value="info">Info</Select.Item>
              <Select.Item value="warn">Warning</Select.Item>
              <Select.Item value="error">Error</Select.Item>
              <Select.Item value="critical">Critical</Select.Item>
            </Select.Content>
          </Select>

          <Select value={selectedAgent} onValueChange={setSelectedAgent}>
            <Select.Trigger>
              <Select.Value placeholder="Agent" />
            </Select.Trigger>
            <Select.Content>
              <Select.Item value="all">All Agents</Select.Item>
              {agentIds.map(agentId => (
                <Select.Item key={agentId} value={agentId}>
                  {agentId}
                </Select.Item>
              ))}
            </Select.Content>
          </Select>

          <Select value={selectedSource} onValueChange={setSelectedSource}>
            <Select.Trigger>
              <Select.Value placeholder="Source" />
            </Select.Trigger>
            <Select.Content>
              <Select.Item value="all">All Sources</Select.Item>
              {sources.map(source => (
                <Select.Item key={source} value={source}>
                  {source}
                </Select.Item>
              ))}
            </Select.Content>
          </Select>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => exportLogs('txt')}>
              Export TXT
            </Button>
            <Button variant="outline" size="sm" onClick={() => exportLogs('json')}>
              Export JSON
            </Button>
            <Button variant="outline" size="sm" onClick={() => exportLogs('csv')}>
              Export CSV
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={clearFilters}>
              Clear Filters
            </Button>
            <span className="text-sm text-gray-500">
              {filteredLogs.length} / {logEntries.length} entries
            </span>
          </div>
        </div>

        {/* Log Entries */}
        <div
          ref={logContainerRef}
          className="flex-1 overflow-y-auto space-y-1 text-sm font-mono"
        >
          {filteredLogs.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              {logEntries.length === 0 ? 'No logs available' : 'No logs match the current filters'}
            </div>
          ) : (
            filteredLogs.map(entry => (
              <div
                key={entry.id}
                className={`p-3 rounded border ${getLevelColor(entry.level)}`}
              >
                <div className="flex items-start gap-2">
                  <span className="flex-shrink-0 text-lg">
                    {getLevelIcon(entry.level)}
                  </span>

                  <div className="flex-1 min-w-0">
                    {/* Header */}
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs text-gray-500">
                        {entry.timestamp.toLocaleString()}
                      </span>
                      <span className={`text-xs px-1.5 py-0.5 rounded-full uppercase font-semibold`}>
                        {entry.level}
                      </span>
                      <span className="text-xs text-gray-600 font-medium">
                        {entry.source}
                      </span>
                      {entry.agentId && (
                        <span className="text-xs text-blue-600 bg-blue-100 px-1.5 py-0.5 rounded">
                          {entry.agentId}
                        </span>
                      )}
                    </div>

                    {/* Message */}
                    <div className="break-words">
                      {entry.message}
                    </div>

                    {/* Tags */}
                    {entry.tags && entry.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {entry.tags.map(tag => (
                          <span
                            key={tag}
                            className="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded"
                          >
                            #{tag}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Metadata */}
                    {showMetadata && entry.metadata && Object.keys(entry.metadata).length > 0 && (
                      <details className="mt-2">
                        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                          Metadata
                        </summary>
                        <pre className="mt-1 p-2 bg-white bg-opacity-50 rounded text-xs overflow-x-auto">
                          {JSON.stringify(entry.metadata, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </Card>
  );
};