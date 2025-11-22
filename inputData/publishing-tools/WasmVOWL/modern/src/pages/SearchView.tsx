import { useState, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { search, getDocumentCount, type SearchResult } from '../services/searchService';
import './SearchView.css';

export default function SearchView() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get('q') || '';
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchTime, setSearchTime] = useState(0);
  const [filter, setFilter] = useState<'all' | 'page' | 'ontology'>('all');

  useEffect(() => {
    if (!query) {
      setResults([]);
      return;
    }

    const startTime = performance.now();
    const searchResults = search(query, 100);
    const endTime = performance.now();

    setResults(searchResults);
    setSearchTime(endTime - startTime);
  }, [query]);

  const filteredResults = results.filter(result => {
    if (filter === 'all') return true;
    return result.type === filter;
  });

  const handleFilterChange = (newFilter: typeof filter) => {
    setFilter(newFilter);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Search Results</h1>
        {query && (
          <p className="text-lg text-muted-foreground">
            Results for "<strong>{query}</strong>"
          </p>
        )}
      </header>

      {query ? (
        <>
          <div className="flex justify-between items-center mb-8 p-4 bg-muted rounded-lg">
            <div className="text-sm text-muted-foreground">
              {filteredResults.length} results in {searchTime.toFixed(0)}ms
            </div>
            <div className="flex gap-2">
              <button
                className={`px-4 py-2 rounded-md border text-sm transition-all ${
                  filter === 'all'
                    ? 'bg-accent text-accent-foreground border-accent'
                    : 'bg-card border-border text-muted-foreground hover:bg-muted'
                }`}
                onClick={() => handleFilterChange('all')}
              >
                All ({results.length})
              </button>
              <button
                className={`px-4 py-2 rounded-md border text-sm transition-all ${
                  filter === 'page'
                    ? 'bg-accent text-accent-foreground border-accent'
                    : 'bg-card border-border text-muted-foreground hover:bg-muted'
                }`}
                onClick={() => handleFilterChange('page')}
              >
                Pages ({results.filter(r => r.type === 'page').length})
              </button>
              <button
                className={`px-4 py-2 rounded-md border text-sm transition-all ${
                  filter === 'ontology'
                    ? 'bg-accent text-accent-foreground border-accent'
                    : 'bg-card border-border text-muted-foreground hover:bg-muted'
                }`}
                onClick={() => handleFilterChange('ontology')}
              >
                Ontology ({results.filter(r => r.type === 'ontology').length})
              </button>
            </div>
          </div>

          <div className="grid gap-6">
            {filteredResults.length > 0 ? (
              filteredResults.map((result) => (
                <div key={result.id} className="bg-card border border-border rounded-lg transition-all hover:shadow-md hover:-translate-y-0.5">
                  <Link to={`/page/${encodeURIComponent(result.title)}`} className="block p-6 no-underline">
                    <div className="flex justify-between items-start mb-3">
                      <h2 className="text-xl font-semibold text-foreground">{result.title}</h2>
                      <span className={`result-type-badge px-3 py-1 rounded text-sm font-medium whitespace-nowrap ${result.type}`}>
                        {result.type === 'ontology' ? '🏷️ Ontology' : '📄 Page'}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground mb-4">{result.excerpt}</p>
                    <div className="flex justify-end">
                      <span className="text-xs text-muted-foreground">
                        Relevance: {((1 - result.score) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </Link>
                </div>
              ))
            ) : (
              <div className="text-center py-16">
                <p className="text-lg text-muted-foreground mb-2">No results found for "{query}"</p>
                <p className="text-sm text-muted-foreground">
                  Try different keywords or check your spelling
                </p>
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="text-center py-16">
          <div className="text-6xl mb-4">🔍</div>
          <h2 className="text-3xl font-bold text-foreground mb-4">Start Searching</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Search across {getDocumentCount()} pages and ontology terms
          </p>
          <ul className="search-tips list-none p-0 inline-block text-left">
            <li className="py-2 text-muted-foreground">Use specific keywords for better results</li>
            <li className="py-2 text-muted-foreground">Try searching for technical terms, concepts, or domains</li>
            <li className="py-2 text-muted-foreground">Filter results by type (pages or ontology)</li>
          </ul>
        </div>
      )}
    </div>
  );
}
