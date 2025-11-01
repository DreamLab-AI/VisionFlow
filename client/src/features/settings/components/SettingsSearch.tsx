import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Search, X, Filter } from 'lucide-react';
import { Input } from '../../design-system/components/Input';
import { Button } from '../../design-system/components/Button';
import { cn } from '../../../utils/classNameUtils';

export interface SettingsSearchProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  resultCount?: number;
  totalCount?: number;
  className?: string;
  debounceMs?: number;
  showFilters?: boolean;
  onFilterToggle?: () => void;
}


export const SettingsSearch: React.FC<SettingsSearchProps> = ({
  onSearch,
  placeholder = "Search 1,061 settings...",
  resultCount,
  totalCount = 1061,
  className,
  debounceMs = 200,
  showFilters = false,
  onFilterToggle
}) => {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const debounceTimerRef = useRef<NodeJS.Timeout>();
  const inputRef = useRef<HTMLInputElement>(null);

  
  const debouncedSearch = useCallback((searchQuery: string) => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    setIsSearching(true);
    debounceTimerRef.current = setTimeout(() => {
      onSearch(searchQuery);
      setIsSearching(false);
    }, debounceMs);
  }, [onSearch, debounceMs]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newQuery = e.target.value;
    setQuery(newQuery);
    debouncedSearch(newQuery);
  }, [debouncedSearch]);

  const handleClear = useCallback(() => {
    setQuery('');
    onSearch('');
    setIsSearching(false);
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    inputRef.current?.focus();
  }, [onSearch]);

  
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      
      if (e.key === 'Escape' && query) {
        e.preventDefault();
        handleClear();
      }
      
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [query, handleClear]);

  const showResultCount = query && resultCount !== undefined;
  const hasResults = resultCount !== undefined && resultCount > 0;

  return (
    <div className={cn("relative flex items-center gap-2", className)}>
      {}
      <div className="relative flex-1">
        <Search
          className={cn(
            "absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 transition-colors",
            isSearching ? "text-primary animate-pulse" : "text-muted-foreground"
          )}
          aria-hidden="true"
        />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={handleChange}
          placeholder={placeholder}
          className={cn(
            "w-full h-9 pl-10 pr-10 rounded-md border border-input",
            "bg-background text-foreground text-sm",
            "placeholder:text-muted-foreground",
            "focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary",
            "transition-all duration-200",
            query && "pr-20" 
          )}
          aria-label="Search settings"
          aria-describedby={showResultCount ? "search-results-count" : undefined}
          autoComplete="off"
          spellCheck={false}
        />

        {}
        {query && (
          <button
            onClick={handleClear}
            className={cn(
              "absolute right-3 top-1/2 transform -translate-y-1/2",
              "p-1 rounded-sm hover:bg-accent transition-colors",
              "focus:outline-none focus:ring-2 focus:ring-primary/20"
            )}
            aria-label="Clear search"
            title="Clear search (Esc)"
          >
            <X className="w-3.5 h-3.5 text-muted-foreground hover:text-foreground" />
          </button>
        )}
      </div>

      {}
      {showResultCount && (
        <div
          id="search-results-count"
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium",
            "transition-colors",
            hasResults
              ? "bg-primary/10 text-primary"
              : "bg-destructive/10 text-destructive"
          )}
          role="status"
          aria-live="polite"
        >
          <span className="tabular-nums">
            {resultCount} / {totalCount}
          </span>
          <span className="text-muted-foreground">
            {resultCount === 1 ? 'result' : 'results'}
          </span>
        </div>
      )}

      {}
      {showFilters && onFilterToggle && (
        <Button
          variant="outline"
          size="icon"
          onClick={onFilterToggle}
          className="h-9 w-9"
          title="Toggle filters"
          aria-label="Toggle advanced filters"
        >
          <Filter className="w-4 h-4" />
        </Button>
      )}

      {}
      {!query && (
        <div
          className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none"
          aria-hidden="true"
        >
          <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-border bg-muted text-[10px] font-mono text-muted-foreground">
            <span className="text-xs">⌘</span>K
          </kbd>
        </div>
      )}
    </div>
  );
};

// Export for testing
export { SettingsSearch as default };
