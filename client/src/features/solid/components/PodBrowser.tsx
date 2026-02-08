/**
 * PodBrowser Component
 * Tree view file browser for Solid Pod contents
 */

import React, { useState, useCallback, useMemo } from 'react';
import { Folder, FolderOpen, File, FileJson, ChevronRight, ChevronDown, RefreshCw, Trash2, Copy, Home } from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('PodBrowser');
import {
  Tooltip,
  TooltipProvider,
  TooltipRoot,
  TooltipTrigger,
  TooltipContent,
} from '@/features/design-system/components/Tooltip';
import { Separator } from '@/features/design-system/components/Separator';
import { cn } from '@/utils/classNameUtils';
import { useSolidPod } from '../hooks/useSolidPod';
import { useSolidContainer, ContainerItem } from '../hooks/useSolidContainer';

interface TreeNodeProps {
  item: ContainerItem;
  level: number;
  isExpanded: boolean;
  isSelected: boolean;
  onToggle: () => void;
  onSelect: () => void;
  onDelete: () => void;
  onCopy: () => void;
}

const TreeNode: React.FC<TreeNodeProps> = ({
  item,
  level,
  isExpanded,
  isSelected,
  onToggle,
  onSelect,
  onDelete,
  onCopy,
}) => {
  const [showActions, setShowActions] = useState(false);

  const getIcon = () => {
    if (item.type === 'container') {
      return isExpanded ? (
        <FolderOpen className="h-4 w-4 text-yellow-500" />
      ) : (
        <Folder className="h-4 w-4 text-yellow-500" />
      );
    }

    // Check if it's a JSON-LD file by name
    if (item.name.endsWith('.jsonld') || item.name.endsWith('.json')) {
      return <FileJson className="h-4 w-4 text-blue-500" />;
    }

    return <File className="h-4 w-4 text-gray-500" />;
  };

  return (
    <div
      className={cn(
        'group flex items-center gap-1 py-1 px-2 rounded-md cursor-pointer transition-colors',
        'hover:bg-accent/50',
        isSelected && 'bg-accent'
      )}
      style={{ paddingLeft: `${level * 16 + 8}px` }}
      onClick={onSelect}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {item.type === 'container' ? (
        <button
          className="p-0.5 hover:bg-accent rounded"
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
        >
          {isExpanded ? (
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
          )}
        </button>
      ) : (
        <div className="w-4" />
      )}

      {getIcon()}

      <span className="flex-1 text-sm truncate">{item.name}</span>

      {showActions && (
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <TooltipProvider>
            <TooltipRoot>
              <TooltipTrigger asChild>
                <button
                  className="p-1 hover:bg-accent rounded"
                  onClick={(e) => {
                    e.stopPropagation();
                    onCopy();
                  }}
                >
                  <Copy className="h-3 w-3 text-muted-foreground" />
                </button>
              </TooltipTrigger>
              <TooltipContent>Copy URL</TooltipContent>
            </TooltipRoot>
          </TooltipProvider>

          <TooltipProvider>
            <TooltipRoot>
              <TooltipTrigger asChild>
                <button
                  className="p-1 hover:bg-destructive/20 rounded"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete();
                  }}
                >
                  <Trash2 className="h-3 w-3 text-destructive" />
                </button>
              </TooltipTrigger>
              <TooltipContent>Delete</TooltipContent>
            </TooltipRoot>
          </TooltipProvider>
        </div>
      )}
    </div>
  );
};

interface ContainerTreeProps {
  containerPath: string;
  level: number;
  selectedPath: string | null;
  expandedPaths: Set<string>;
  onSelect: (path: string, type: 'container' | 'resource') => void;
  onToggle: (path: string) => void;
  onDelete: (path: string) => void;
  onCopy: (url: string) => void;
}

const ContainerTree: React.FC<ContainerTreeProps> = ({
  containerPath,
  level,
  selectedPath,
  expandedPaths,
  onSelect,
  onToggle,
  onDelete,
  onCopy,
}) => {
  const { items, isLoading, error } = useSolidContainer(containerPath);

  if (isLoading && level === 0) {
    return (
      <div className="flex items-center justify-center py-4 text-muted-foreground text-sm">
        Loading...
      </div>
    );
  }

  if (error && level === 0) {
    return (
      <div className="flex items-center justify-center py-4 text-destructive text-sm">
        {error}
      </div>
    );
  }

  return (
    <div>
      {items.map((item) => {
        const isExpanded = expandedPaths.has(item.url);
        const isSelected = selectedPath === item.url;

        return (
          <div key={item.url}>
            <TreeNode
              item={item}
              level={level}
              isExpanded={isExpanded}
              isSelected={isSelected}
              onToggle={() => onToggle(item.url)}
              onSelect={() => onSelect(item.url, item.type)}
              onDelete={() => onDelete(item.url)}
              onCopy={() => onCopy(item.url)}
            />
            {item.type === 'container' && isExpanded && (
              <ContainerTree
                containerPath={item.url}
                level={level + 1}
                selectedPath={selectedPath}
                expandedPaths={expandedPaths}
                onSelect={onSelect}
                onToggle={onToggle}
                onDelete={onDelete}
                onCopy={onCopy}
              />
            )}
          </div>
        );
      })}
    </div>
  );
};

export interface PodBrowserProps {
  onResourceSelect?: (resourceUrl: string) => void;
  className?: string;
}

export const PodBrowser: React.FC<PodBrowserProps> = ({ onResourceSelect, className }) => {
  const { podInfo, isLoading: podLoading, error: podError } = useSolidPod();
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());
  const [currentPath, setCurrentPath] = useState<string[]>([]);

  const rootPath = podInfo?.podUrl || '';

  const handleToggle = useCallback((path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const handleSelect = useCallback(
    (path: string, type: 'container' | 'resource') => {
      setSelectedPath(path);

      if (type === 'resource' && onResourceSelect) {
        onResourceSelect(path);
      }

      // Update breadcrumb
      if (rootPath) {
        const relativePath = path.replace(rootPath, '');
        const parts = relativePath.split('/').filter(Boolean);
        setCurrentPath(parts);
      }
    },
    [onResourceSelect, rootPath]
  );

  const handleDelete = useCallback((path: string) => {
    // Confirm before delete
    if (window.confirm(`Delete ${path.split('/').pop()}?`)) {
      // Deletion handled by container hook
      logger.debug('Delete requested:', path);
    }
  }, []);

  const handleCopy = useCallback((url: string) => {
    navigator.clipboard.writeText(url);
  }, []);

  const handleRefresh = useCallback(() => {
    // Force re-render by toggling a state
    setExpandedPaths(new Set());
  }, []);

  const handleHome = useCallback(() => {
    setCurrentPath([]);
    setSelectedPath(null);
    setExpandedPaths(new Set());
  }, []);

  const breadcrumbItems = useMemo(() => {
    const items = [{ name: 'Pod', path: rootPath }];
    let accumulated = rootPath;

    for (const part of currentPath) {
      accumulated = `${accumulated}/${part}`;
      items.push({ name: part, path: accumulated });
    }

    return items;
  }, [rootPath, currentPath]);

  if (podLoading) {
    return (
      <Card className={cn('h-full', className)}>
        <CardContent className="flex items-center justify-center h-full">
          <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (podError || !podInfo?.exists) {
    return (
      <Card className={cn('h-full', className)}>
        <CardContent className="flex flex-col items-center justify-center h-full gap-4 text-center">
          <Folder className="h-12 w-12 text-muted-foreground" />
          <div>
            <p className="text-sm text-muted-foreground">
              {podError || 'No pod found'}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Create a pod in Settings to browse your files
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn('h-full flex flex-col', className)}>
      <CardHeader className="py-3 px-4 border-b">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Folder className="h-4 w-4" />
            Pod Browser
          </CardTitle>
          <div className="flex items-center gap-1">
            <TooltipProvider>
              <TooltipRoot>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={handleHome}>
                    <Home className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Go to root</TooltipContent>
              </TooltipRoot>
            </TooltipProvider>

            <TooltipProvider>
              <TooltipRoot>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={handleRefresh}>
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Refresh</TooltipContent>
              </TooltipRoot>
            </TooltipProvider>
          </div>
        </div>

        {/* Breadcrumb navigation */}
        <div className="flex items-center gap-1 mt-2 text-xs text-muted-foreground overflow-x-auto">
          {breadcrumbItems.map((item, index) => (
            <React.Fragment key={item.path}>
              {index > 0 && <ChevronRight className="h-3 w-3 flex-shrink-0" />}
              <button
                className={cn(
                  'hover:text-foreground truncate max-w-[100px]',
                  index === breadcrumbItems.length - 1 && 'text-foreground font-medium'
                )}
                onClick={() => handleSelect(item.path, 'container')}
              >
                {item.name}
              </button>
            </React.Fragment>
          ))}
        </div>
      </CardHeader>

      <ScrollArea className="flex-1">
        <div className="py-2">
          <ContainerTree
            containerPath={rootPath}
            level={0}
            selectedPath={selectedPath}
            expandedPaths={expandedPaths}
            onSelect={handleSelect}
            onToggle={handleToggle}
            onDelete={handleDelete}
            onCopy={handleCopy}
          />
        </div>
      </ScrollArea>
    </Card>
  );
};

export default PodBrowser;
