import React, { useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardContent } from '@/features/design-system/components/Card';
import { Badge } from '@/features/design-system/components/Badge';
import { Button } from '@/features/design-system/components/Button';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { ExternalLink, FileText, Database, User, Settings } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('SearchResults');

interface SearchResultsProps {
  query: string;
  results: SearchResult[];
  onResultClick?: (result: SearchResult) => void;
  className?: string;
  compact?: boolean;
}

interface SearchResult {
  id: string;
  title: string;
  content: string;
  type: 'document' | 'data' | 'user' | 'setting';
  score: number;
  lastModified: Date;
  path?: string;
  highlights: string[];
}

export const SearchResults: React.FC<SearchResultsProps> = ({ 
  query, 
  results, 
  onResultClick, 
  className, 
  compact = false 
}) => {
  // Subscribe only to search display settings
  const searchSettings = useSelectiveSettings({
    highlightMatches: 'search.highlightMatches',
    showSnippets: 'search.results.showSnippets',
    showPath: 'search.results.showPath',
    showScore: 'search.results.showScore',
    groupByType: 'search.results.groupByType',
    sortBy: 'search.results.sortBy'
  });
  
  const sortedAndGroupedResults = useMemo(() => {
    let sorted = [...results];
    
    // Sort results
    switch (searchSettings.sortBy) {
      case 'relevance':
        sorted.sort((a, b) => b.score - a.score);
        break;
      case 'date':
        sorted.sort((a, b) => b.lastModified.getTime() - a.lastModified.getTime());
        break;
      case 'title':
        sorted.sort((a, b) => a.title.localeCompare(b.title));
        break;
    }
    
    // Group by type if enabled
    if (searchSettings.groupByType) {
      const grouped = sorted.reduce((acc, result) => {
        if (!acc[result.type]) acc[result.type] = [];
        acc[result.type].push(result);
        return acc;
      }, {} as Record<string, SearchResult[]>);
      
      return Object.entries(grouped).map(([type, items]) => ({ type, items }));
    }
    
    return [{ type: 'all', items: sorted }];
  }, [results, searchSettings]);
  
  const getTypeIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'document': return <FileText size={16} />;
      case 'data': return <Database size={16} />;
      case 'user': return <User size={16} />;
      case 'setting': return <Settings size={16} />;
      default: return <FileText size={16} />;
    }
  };
  
  const getTypeColor = (type: SearchResult['type']) => {
    switch (type) {
      case 'document': return 'bg-blue-100 text-blue-800';
      case 'data': return 'bg-green-100 text-green-800';
      case 'user': return 'bg-purple-100 text-purple-800';
      case 'setting': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };
  
  const highlightText = (text: string, query: string) => {
    if (!searchSettings.highlightMatches || !query) return text;
    
    const regex = new RegExp(`(${query})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) ? (
        <mark key={index} className="bg-yellow-200 px-1 rounded">{part}</mark>
      ) : part
    );
  };
  
  const handleResultClick = (result: SearchResult) => {
    logger.info('Search result clicked', { resultId: result.id, type: result.type });
    onResultClick?.(result);
  };
  
  if (results.length === 0) {
    return (
      <div className={`text-center py-8 ${className}`}>
        <FileText size={48} className="mx-auto mb-4 text-gray-400" />
        <p className="text-muted-foreground">No results found</p>
        {query && (
          <p className="text-sm text-muted-foreground mt-2">
            No matches for "{query}"
          </p>
        )}
      </div>
    );
  }
  
  if (compact) {
    return (
      <div className={`space-y-2 ${className}`}>
        {results.slice(0, 5).map((result) => (
          <div 
            key={result.id}
            className="flex items-center gap-2 p-2 border rounded cursor-pointer hover:bg-gray-50"
            onClick={() => handleResultClick(result)}
          >
            <div className="flex items-center gap-2 flex-1">
              {getTypeIcon(result.type)}
              <span className="font-medium text-sm truncate">{result.title}</span>
              <Badge className={getTypeColor(result.type)} size="sm">
                {result.type}
              </Badge>
            </div>
            {searchSettings.showScore && (
              <Badge variant="outline" className="text-xs">
                {Math.round(result.score * 100)}%
              </Badge>
            )}
            <ExternalLink size={12} className="text-muted-foreground" />
          </div>
        ))}
        {results.length > 5 && (
          <div className="text-center">
            <span className="text-xs text-muted-foreground">
              +{results.length - 5} more results
            </span>
          </div>
        )}
      </div>
    );
  }
  
  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-muted-foreground">
          {results.length} result{results.length !== 1 ? 's' : ''} found
          {query && ` for "${query}"`}
        </span>
        <Badge variant="outline">Sorted by {searchSettings.sortBy}</Badge>
      </div>
      
      <ScrollArea className="h-[500px]">
        <div className="space-y-4">
          {sortedAndGroupedResults.map((group) => (
            <div key={group.type}>
              {searchSettings.groupByType && group.type !== 'all' && (
                <div className="flex items-center gap-2 mb-3">
                  {getTypeIcon(group.type as SearchResult['type'])}
                  <h3 className="font-medium capitalize">{group.type}s</h3>
                  <Badge variant="outline">{group.items.length}</Badge>
                </div>
              )}
              
              <div className="space-y-3">
                {group.items.map((result) => (
                  <Card key={result.id} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardContent 
                      className="p-4" 
                      onClick={() => handleResultClick(result)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {getTypeIcon(result.type)}
                          <h3 className="font-medium">
                            {highlightText(result.title, query)}
                          </h3>
                          <Badge className={getTypeColor(result.type)}>
                            {result.type}
                          </Badge>
                          {searchSettings.showScore && (
                            <Badge variant="outline" className="text-xs">
                              {Math.round(result.score * 100)}% match
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">
                            {result.lastModified.toLocaleDateString()}
                          </span>
                          <ExternalLink size={14} className="text-muted-foreground" />
                        </div>
                      </div>
                      
                      {searchSettings.showSnippets && (
                        <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                          {highlightText(result.content, query)}
                        </p>
                      )}
                      
                      {searchSettings.showPath && result.path && (
                        <div className="text-xs text-muted-foreground mb-2">
                          <span className="font-mono">{result.path}</span>
                        </div>
                      )}
                      
                      {searchSettings.highlightMatches && result.highlights.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {result.highlights.map((highlight, index) => (
                            <Badge key={index} variant="secondary" className="text-xs">
                              {highlight}
                            </Badge>
                          ))}
                        </div>
                      )}
                      
                      <div className="flex items-center justify-between mt-3 pt-2 border-t">
                        <div className="text-xs text-muted-foreground">
                          Last modified: {result.lastModified.toLocaleString()}
                        </div>
                        <Button size="sm" variant="ghost">
                          Open
                          <ExternalLink size={12} className="ml-1" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

export default SearchResults;