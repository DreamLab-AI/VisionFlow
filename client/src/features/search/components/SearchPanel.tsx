import React, { useState, useMemo, useCallback } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Input } from '@/features/design-system/components/Input';
import { Badge } from '@/features/design-system/components/Badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Search, Filter, Clock, BookMark, X, Settings } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('SearchPanel');

interface SearchPanelProps {
  className?: string;
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

interface SearchHistory {
  id: string;
  query: string;
  timestamp: Date;
  resultCount: number;
}

export const SearchPanel: React.FC<SearchPanelProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [activeTab, setActiveTab] = useState('search');
  const [selectedScope, setSelectedScope] = useState('all');
  
  // Subscribe only to search-related settings
  const searchSettings = useSelectiveSettings({
    enabled: 'search.enabled',
    indexingEnabled: 'search.indexing.enabled',
    fuzzySearch: 'search.fuzzySearch.enabled',
    caseSensitive: 'search.caseSensitive',
    wholeWords: 'search.wholeWords',
    maxResults: 'search.limits.maxResults',
    saveHistory: 'search.history.enabled',
    historyLimit: 'search.history.maxEntries',
    highlightMatches: 'search.highlightMatches',
    searchTimeout: 'search.timeoutSeconds'
  });
  
  // Mock search results - in real app this would come from search API
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  
  const [searchHistory, setSearchHistory] = useState<SearchHistory[]>([
    {
      id: '1',
      query: 'user data analysis',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
      resultCount: 45
    },
    {
      id: '2',
      query: 'graph visualization settings',
      timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000),
      resultCount: 12
    },
    {
      id: '3',
      query: 'performance metrics',
      timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
      resultCount: 28
    }
  ]);
  
  const mockSearchResults: SearchResult[] = useMemo(() => [
    {
      id: '1',
      title: 'User Analytics Dashboard',
      content: 'Comprehensive analytics dashboard showing user behavior patterns, engagement metrics, and conversion rates.',
      type: 'document',
      score: 0.95,
      lastModified: new Date(Date.now() - 2 * 60 * 60 * 1000),
      path: '/analytics/dashboard',
      highlights: ['analytics', 'user', 'dashboard']
    },
    {
      id: '2',
      title: 'Data Visualization Settings',
      content: 'Configuration options for graph rendering, color schemes, and interactive features in the visualization engine.',
      type: 'setting',
      score: 0.88,
      lastModified: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
      path: '/settings/visualization',
      highlights: ['data', 'visualization', 'settings']
    },
    {
      id: '3',
      title: 'Performance Monitoring Bot',
      content: 'Automated bot that monitors system performance, tracks resource usage, and alerts on anomalies.',
      type: 'data',
      score: 0.82,
      lastModified: new Date(Date.now() - 3 * 60 * 60 * 1000),
      path: '/bots/performance-monitor',
      highlights: ['performance', 'monitoring', 'bot']
    },
    {
      id: '4',
      title: 'John Smith Profile',
      content: 'User profile containing personal information, preferences, and activity history for system administrator.',
      type: 'user',
      score: 0.76,
      lastModified: new Date(Date.now() - 6 * 60 * 60 * 1000),
      path: '/users/john-smith',
      highlights: ['user', 'profile', 'administrator']
    }
  ], []);
  
  const searchScopes = useMemo(() => [
    { value: 'all', label: 'All Content' },
    { value: 'documents', label: 'Documents' },
    { value: 'data', label: 'Data' },
    { value: 'users', label: 'Users' },
    { value: 'settings', label: 'Settings' }
  ], []);
  
  const performSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    
    setIsSearching(true);
    logger.info('Performing search', { query, scope: selectedScope });
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Filter and sort mock results based on query
    const filtered = mockSearchResults
      .filter(result => {
        const matchesScope = selectedScope === 'all' || 
                           (selectedScope === 'documents' && result.type === 'document') ||
                           (selectedScope === 'data' && result.type === 'data') ||
                           (selectedScope === 'users' && result.type === 'user') ||
                           (selectedScope === 'settings' && result.type === 'setting');
        
        const searchTerms = query.toLowerCase().split(' ');
        const text = `${result.title} ${result.content}`.toLowerCase();
        const matchesQuery = searchSettings.fuzzySearch 
          ? searchTerms.some(term => text.includes(term))
          : searchTerms.every(term => text.includes(term));
          
        return matchesScope && matchesQuery;
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, searchSettings.maxResults);
    
    setSearchResults(filtered);
    
    // Add to search history
    if (searchSettings.saveHistory) {
      const newHistoryEntry: SearchHistory = {
        id: Date.now().toString(),
        query,
        timestamp: new Date(),
        resultCount: filtered.length
      };
      
      setSearchHistory(prev => [
        newHistoryEntry,
        ...prev.slice(0, searchSettings.historyLimit - 1)
      ]);
    }
    
    setIsSearching(false);
  }, [mockSearchResults, selectedScope, searchSettings]);
  
  const handleSearch = useCallback(() => {
    performSearch(searchQuery);
  }, [searchQuery, performSearch]);
  
  const handleHistoryClick = (historyQuery: string) => {
    setSearchQuery(historyQuery);
    performSearch(historyQuery);
    setActiveTab('search');
  };
  
  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
  };
  
  const getTypeIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'document': return '📄';
      case 'data': return '📊';
      case 'user': return '👤';
      case 'setting': return '⚙️';
      default: return '📄';
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
  
  if (!searchSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search size={20} />
            Search
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Search size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">Search is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable search in settings to find content across the system
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
            <Search size={20} />
            Search Panel
            {!searchSettings.indexingEnabled && (
              <Badge variant="outline" className="text-xs">
                Limited
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="search">Search</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>
          
          <TabsContent value="search" className="space-y-4">
            {/* Search Input */}
            <div className="space-y-3">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Search size={20} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search for documents, data, users, settings..."
                    className="pl-10"
                    onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  />
                  {searchQuery && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={clearSearch}
                      className="absolute right-1 top-1/2 transform -translate-y-1/2 h-8 w-8 p-0"
                    >
                      <X size={16} />
                    </Button>
                  )}
                </div>
                <Select value={selectedScope} onValueChange={setSelectedScope}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {searchScopes.map(scope => (
                      <SelectItem key={scope.value} value={scope.value}>
                        {scope.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button onClick={handleSearch} disabled={!searchQuery.trim() || isSearching}>
                  {isSearching ? 'Searching...' : 'Search'}
                </Button>
              </div>
              
              {/* Search Options */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="case-sensitive"
                    checked={searchSettings.caseSensitive}
                    onChange={(e) => set('search.caseSensitive', e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="case-sensitive" className="text-sm">Case sensitive</label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="whole-words"
                    checked={searchSettings.wholeWords}
                    onChange={(e) => set('search.wholeWords', e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="whole-words" className="text-sm">Whole words</label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="fuzzy-search"
                    checked={searchSettings.fuzzySearch}
                    onChange={(e) => set('search.fuzzySearch.enabled', e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="fuzzy-search" className="text-sm">Fuzzy search</label>
                </div>
              </div>
            </div>
            
            {/* Search Results */}
            <ScrollArea className="h-[400px]">
              {searchResults.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">
                      Found {searchResults.length} results
                    </span>
                    <Badge variant="outline">{selectedScope}</Badge>
                  </div>
                  
                  {searchResults.map((result) => (
                    <div key={result.id} className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{getTypeIcon(result.type)}</span>
                          <h3 className="font-medium">{result.title}</h3>
                          <Badge className={getTypeColor(result.type)}>
                            {result.type}
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            {Math.round(result.score * 100)}% match
                          </Badge>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {result.lastModified.toLocaleDateString()}
                        </span>
                      </div>
                      
                      <p className="text-sm text-muted-foreground mb-2">
                        {result.content}
                      </p>
                      
                      {result.path && (
                        <div className="text-xs text-muted-foreground mb-2">
                          Path: {result.path}
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
                    </div>
                  ))}
                </div>
              )}
              
              {searchQuery && searchResults.length === 0 && !isSearching && (
                <div className="text-center py-8">
                  <Search size={48} className="mx-auto mb-4 text-gray-400" />
                  <p className="text-muted-foreground">No results found</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Try adjusting your search terms or scope
                  </p>
                </div>
              )}
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="history" className="space-y-4">
            {searchSettings.saveHistory ? (
              <ScrollArea className="h-[400px]">
                <div className="space-y-3">
                  {searchHistory.map((entry) => (
                    <div 
                      key={entry.id} 
                      className="border rounded-lg p-3 hover:bg-gray-50 cursor-pointer"
                      onClick={() => handleHistoryClick(entry.query)}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium">{entry.query}</span>
                        <Badge variant="outline">{entry.resultCount} results</Badge>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock size={12} />
                        {entry.timestamp.toLocaleString()}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <div className="text-center py-8">
                <Clock size={48} className="mx-auto mb-4 text-gray-400" />
                <p className="text-muted-foreground">Search history is disabled</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Enable search history in settings to track your searches
                </p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Full-text Indexing</span>
                <Button
                  variant={searchSettings.indexingEnabled ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('search.indexing.enabled', !searchSettings.indexingEnabled)}
                >
                  {searchSettings.indexingEnabled ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Save History</span>
                <Button
                  variant={searchSettings.saveHistory ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('search.history.enabled', !searchSettings.saveHistory)}
                >
                  {searchSettings.saveHistory ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Highlight Matches</span>
                <Button
                  variant={searchSettings.highlightMatches ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('search.highlightMatches', !searchSettings.highlightMatches)}
                >
                  {searchSettings.highlightMatches ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
            </div>
            
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">Search Limits</h3>
              <div className="text-sm text-muted-foreground space-y-1">
                <p>Max results per search: {searchSettings.maxResults}</p>
                <p>Search timeout: {searchSettings.searchTimeout}s</p>
                <p>History entries: {searchSettings.historyLimit}</p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default SearchPanel;