# Settings Search System

## Overview

The Settings Search system provides powerful fuzzy search capabilities for navigating and filtering 1,061+ application settings. Built with performance and user experience in mind, it enables instant, intelligent search across all settings categories.

## Features

### 🔍 Advanced Fuzzy Matching
- **Intelligent Scoring**: Multi-factor relevance scoring considers:
  - Exact substring matches (100 points)
  - Word boundary matches (+20 points)
  - Start of string matches (+15 points)
  - Consecutive character matches (10 points/char)
  - Position-aware scoring (early matches scored higher)
  - Case-sensitive matching support

- **Forgiving Search**: Handles typos and partial matches gracefully
  - `"nde colr"` → finds "Node Color"
  - `"phys enble"` → finds "Physics Enable"

### ⚡ Performance Optimized
- **Sub-100ms Search**: Searches 1,000+ settings in <100ms
- **Debounced Input**: 200ms debounce prevents excessive searches
- **Memoized Index**: Build once, search many times
- **O(1) Lookups**: Fast path matching with Set data structures

### 🎨 Rich UI Features
- **Real-time Results**: Instant feedback as you type
- **Result Count Badge**: Shows "42 / 1061 results"
- **Keyboard Shortcuts**:
  - `Cmd/Ctrl + K`: Focus search
  - `Escape`: Clear search
- **Clear Button**: One-click search reset
- **Loading States**: Visual feedback during search

### 🏷️ Advanced Filtering
- **Category Filtering**: Search within specific categories (Visualization, Physics, XR, etc.)
- **Widget Type Filtering**: Filter by control type (slider, toggle, colorPicker, etc.)
- **Access Level Filtering**: Include/exclude advanced or power-user settings
- **Score Threshold**: Customize minimum relevance score

## Architecture

### Components

#### SettingsSearch Component
```typescript
<SettingsSearch
  onSearch={handleSearch}
  resultCount={42}
  totalCount={1061}
  placeholder="Search 1,061 settings..."
  debounceMs={200}
/>
```

**Props:**
- `onSearch`: Callback function receiving search query
- `resultCount`: Number of results found (optional)
- `totalCount`: Total searchable items
- `placeholder`: Custom placeholder text
- `debounceMs`: Debounce delay in milliseconds (default: 200)
- `showFilters`: Enable filter toggle button
- `onFilterToggle`: Filter button callback

#### Search Utilities (`settingsSearch.ts`)

**Core Functions:**

1. **`buildSearchIndex(settingsUIDefinition)`**
   ```typescript
   const searchIndex = buildSearchIndex(settingsUIDefinition);
   // Returns: SearchableSettingField[]
   ```
   - Indexes all settings with metadata
   - Builds tags for categorization
   - Extracts nested settings into flat structure

2. **`searchSettings(index, query, options)`**
   ```typescript
   const results = searchSettings(index, "node color", {
     minScore: 15,
     maxResults: 100,
     categories: ['visualization'],
     types: ['colorPicker'],
     includeAdvanced: true,
     includePowerUser: true
   });
   ```

3. **`fuzzyMatch(text, query, caseSensitive)`**
   ```typescript
   const { score, indices } = fuzzyMatch("Node Color Picker", "color", false);
   // score: 95, indices: [[5, 10]]
   ```

4. **`highlightMatches(text, indices)`**
   ```typescript
   const html = highlightMatches("Node Color", [[5, 10]]);
   // Returns: "Node <mark>Color</mark>"
   ```

### Data Structures

#### SearchableSettingField
```typescript
interface SearchableSettingField {
  key: string;                    // "visualization.nodes.baseColor"
  label: string;                  // "Base Color"
  path: string;                   // "visualisation.nodes.baseColor"
  description?: string;           // "Default color of nodes"
  category: string;               // "visualization"
  categoryLabel: string;          // "Visualization"
  subcategory?: string;           // "nodes"
  subcategoryLabel?: string;      // "Node Settings"
  type: string;                   // "colorPicker"
  tags?: string[];                // ["visualization", "nodes", "colorPicker"]
  isPowerUserOnly?: boolean;
  isAdvanced?: boolean;
  localStorage?: boolean;
}
```

#### SearchResult
```typescript
interface SearchResult extends SearchableSettingField {
  score: number;                  // 0-100 relevance score
  matches: {
    field: 'label' | 'path' | 'description' | 'category' | 'subcategory';
    text: string;
    indices: [number, number][];  // Match positions
  }[];
}
```

## Integration Guide

### Basic Integration

```typescript
import { SettingsSearch } from './features/settings/components/SettingsSearch';
import { buildSearchIndex, searchSettings } from './utils/settingsSearch';
import { settingsUIDefinition } from './features/settings/config/settingsUIDefinition';

function SettingsPanel() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

  // Build index once
  const searchIndex = useMemo(() => {
    return buildSearchIndex(settingsUIDefinition);
  }, []);

  // Handle search
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);

    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    const results = searchSettings(searchIndex, query, {
      minScore: 15,
      maxResults: 100
    });

    setSearchResults(results);
  }, [searchIndex]);

  return (
    <div>
      <SettingsSearch
        onSearch={handleSearch}
        resultCount={searchResults.length}
        totalCount={searchIndex.length}
      />

      {/* Render filtered settings */}
      {searchResults.map(result => (
        <SettingItem key={result.path} setting={result} />
      ))}
    </div>
  );
}
```

### Advanced Integration with Filtering

```typescript
const [filters, setFilters] = useState({
  categories: ['visualization', 'physics'],
  types: ['slider', 'toggle'],
  includeAdvanced: false
});

const handleSearch = useCallback((query: string) => {
  const results = searchSettings(searchIndex, query, {
    ...filters,
    minScore: 20,
    maxResults: 50
  });
  setSearchResults(results);
}, [searchIndex, filters]);
```

## Performance Characteristics

### Benchmarks (1,061 Settings)

| Operation | Time (avg) | Memory |
|-----------|-----------|--------|
| Index Build | ~50ms | ~2MB |
| Simple Search | <50ms | ~1MB |
| Complex Search | <100ms | ~2MB |
| Concurrent Searches (5x) | <200ms | ~5MB |

### Optimization Strategies

1. **Memoized Index**
   ```typescript
   const searchIndex = useMemo(() => buildSearchIndex(settingsUIDefinition), []);
   ```

2. **Debounced Input** (200ms default)
   - Prevents search on every keystroke
   - Configurable via `debounceMs` prop

3. **Result Limiting**
   ```typescript
   searchSettings(index, query, { maxResults: 100 })
   ```

4. **Score Thresholding**
   ```typescript
   searchSettings(index, query, { minScore: 15 })
   ```

5. **Fast Path Matching**
   - Uses `Set` for O(1) path lookups
   - Reduces filtering overhead

## Usage Examples

### Example 1: Basic Search
```typescript
// User types: "node color"
// Results (sorted by score):
// 1. Node Color (score: 100) - visualization.nodes.baseColor
// 2. Node Color Picker (score: 95) - visualization.nodes.colorPicker
// 3. Edge Color (score: 75) - visualization.edges.color
```

### Example 2: Fuzzy Search
```typescript
// User types: "phys enble bnd" (typos)
// Results:
// 1. Enable Bounds (score: 85) - physics.enableBounds
// 2. Physics Enable (score: 82) - physics.enabled
// 3. Boundary Damping (score: 65) - physics.boundaryDamping
```

### Example 3: Path Search
```typescript
// User types: "visualisation.nodes"
// Results: All node-related settings
// - visualisation.nodes.baseColor
// - visualisation.nodes.metalness
// - visualisation.nodes.opacity
```

### Example 4: Description Search
```typescript
// User types: "transparency"
// Results: Settings mentioning transparency in descriptions
// - Node Opacity (description: "transparency (0.1-1.0)")
// - Edge Opacity (description: "transparency for edges")
```

## Testing

### Running Tests
```bash
# Run all search tests
npm test -- settingsSearch.test.ts

# Run with coverage
npm test -- --coverage settingsSearch.test.ts

# Watch mode
npm test -- --watch settingsSearch.test.ts
```

### Test Coverage
- ✅ Fuzzy matching algorithm
- ✅ Search index building
- ✅ Search functionality
- ✅ Result ranking
- ✅ Filtering options
- ✅ Highlighting
- ✅ Performance benchmarks
- ✅ Edge cases (special chars, unicode, long queries)

## Accessibility

### Keyboard Navigation
- `Tab`: Focus search input
- `Cmd/Ctrl + K`: Focus search from anywhere
- `Escape`: Clear search and keep focus
- `Enter`: (Reserved for future: jump to first result)

### ARIA Labels
```html
<input
  aria-label="Search settings"
  aria-describedby="search-results-count"
/>

<div id="search-results-count" role="status" aria-live="polite">
  42 / 1061 results
</div>
```

### Screen Reader Support
- Live region announces result count
- Clear button has descriptive label
- Search input has meaningful placeholder

## Future Enhancements

### Planned Features
- [ ] **Search History**: Recent searches dropdown
- [ ] **Saved Searches**: Bookmark common searches
- [ ] **Search Suggestions**: Auto-complete suggestions
- [ ] **Category Chips**: Visual filter chips
- [ ] **Keyboard Navigation**: Arrow keys to navigate results
- [ ] **Jump to Result**: Enter key to focus first result
- [ ] **Search Analytics**: Track popular searches
- [ ] **Smart Grouping**: Group results by category
- [ ] **Export Results**: Save search results to file
- [ ] **Search Macros**: Define custom search shortcuts

### Performance Improvements
- [ ] **Web Workers**: Offload search to background thread
- [ ] **Indexed DB**: Cache search index persistently
- [ ] **Virtual Scrolling**: Render only visible results
- [ ] **Incremental Search**: Update results as index builds
- [ ] **Search Index Versioning**: Invalidate cache on schema changes

## Troubleshooting

### Common Issues

**Issue: Search is slow**
- Check `maxResults` - lower values improve performance
- Increase `minScore` threshold to filter low-quality results
- Ensure search index is memoized (built once)

**Issue: No results found**
- Check `minScore` - may be too high (try 10-15)
- Verify `includeAdvanced` and `includePowerUser` flags
- Check category/type filters aren't too restrictive

**Issue: Results seem irrelevant**
- Adjust scoring weights in `searchSettings()`
- Increase `minScore` to filter low-quality matches
- Check for typos in search query

**Issue: Search component not responding**
- Verify `onSearch` callback is provided
- Check console for errors
- Ensure settingsUIDefinition is loaded

## API Reference

### SettingsSearch Props
| Prop | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| onSearch | (query: string) => void | ✅ | - | Search callback |
| resultCount | number | ❌ | undefined | Number of results |
| totalCount | number | ❌ | 1061 | Total searchable items |
| placeholder | string | ❌ | "Search..." | Input placeholder |
| debounceMs | number | ❌ | 200 | Debounce delay (ms) |
| className | string | ❌ | "" | Additional CSS classes |
| showFilters | boolean | ❌ | false | Show filter button |
| onFilterToggle | () => void | ❌ | undefined | Filter button callback |

### Search Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| minScore | number | 20 | Minimum relevance score (0-100) |
| maxResults | number | 100 | Maximum results to return |
| categories | string[] | [] | Filter by categories |
| types | string[] | [] | Filter by widget types |
| includeAdvanced | boolean | true | Include advanced settings |
| includePowerUser | boolean | true | Include power-user settings |
| caseSensitive | boolean | false | Case-sensitive matching |
| fuzzyThreshold | number | 0.6 | Fuzzy matching threshold |

## Contributing

### Adding New Search Features

1. Update `settingsSearch.ts` with new functionality
2. Add tests to `settingsSearch.test.ts`
3. Update this documentation
4. Submit PR with benchmark results

### Reporting Issues

When reporting search-related issues, include:
- Search query used
- Expected vs actual results
- Browser/environment details
- Screenshots (if UI-related)

## License

This search system is part of the larger application and follows the project's license.

## Credits

- **Fuzzy Matching Algorithm**: Custom implementation with position-aware scoring
- **Performance Optimization**: Memoization, debouncing, and efficient data structures
- **UI/UX Design**: Keyboard shortcuts and accessibility features

---

**Version**: 1.0.0
**Last Updated**: 2025-10-22
**Maintainer**: Development Team
