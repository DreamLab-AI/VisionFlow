# Settings Search - Quick Start Guide

## 🚀 Quick Start (2 minutes)

### Basic Usage

```typescript
import { SettingsSearch } from '@/features/settings/components/SettingsSearch';
import { buildSearchIndex, searchSettings } from '@/utils/settingsSearch';

function MySettingsPanel() {
  const [results, setResults] = useState([]);

  const searchIndex = useMemo(() =>
    buildSearchIndex(settingsUIDefinition),
    []
  );

  const handleSearch = useCallback((query: string) => {
    const results = searchSettings(searchIndex, query);
    setResults(results);
  }, [searchIndex]);

  return (
    <SettingsSearch
      onSearch={handleSearch}
      resultCount={results.length}
      totalCount={searchIndex.length}
    />
  );
}
```

## 📖 Common Use Cases

### 1. Simple Search
```typescript
// Just search, no filters
const results = searchSettings(index, "node color");
```

### 2. Category Filter
```typescript
// Only visualization settings
const results = searchSettings(index, "enable", {
  categories: ['visualization']
});
```

### 3. Widget Type Filter
```typescript
// Only toggles and sliders
const results = searchSettings(index, "enable", {
  types: ['toggle', 'slider']
});
```

### 4. Advanced Settings Only
```typescript
// Power user settings
const results = searchSettings(index, "advanced", {
  includePowerUser: true,
  includeAdvanced: true
});
```

### 5. High Relevance Only
```typescript
// Only high-quality matches
const results = searchSettings(index, "color", {
  minScore: 50,  // 0-100 scale
  maxResults: 20
});
```

## 🎨 Customization

### Custom Debounce
```typescript
<SettingsSearch
  debounceMs={500}  // 500ms delay
  onSearch={handleSearch}
/>
```

### Custom Placeholder
```typescript
<SettingsSearch
  placeholder="Find your settings..."
  onSearch={handleSearch}
/>
```

### With Filter Button
```typescript
<SettingsSearch
  showFilters={true}
  onFilterToggle={() => setShowFilters(!showFilters)}
  onSearch={handleSearch}
/>
```

## ⌨️ Keyboard Shortcuts

- `Cmd/Ctrl + K`: Focus search
- `Escape`: Clear search
- `Tab`: Navigate interface

## 🎯 Search Tips

### Exact Match
```
"node color"  → Finds "Node Color" settings
```

### Fuzzy Match
```
"nde colr"    → Finds "Node Color" (handles typos)
```

### Path Search
```
"visualisation.nodes"  → Finds all node settings
```

### Category Search
```
"physics"     → Finds all physics settings
```

### Multi-word
```
"background color picker"  → Matches all words
```

## 📊 Result Scoring

Results are ranked by relevance (0-100):

| Score Range | Quality |
|-------------|---------|
| 90-100 | Excellent match (exact or near-exact) |
| 70-89 | Good match (clear relevance) |
| 50-69 | Moderate match (partial relevance) |
| 30-49 | Weak match (marginal relevance) |
| 0-29 | Poor match (filtered by default) |

## 🐛 Troubleshooting

### No Results?
1. Lower `minScore` threshold (try 10-15)
2. Check category/type filters
3. Verify `includeAdvanced` flag
4. Try simpler query

### Slow Search?
1. Increase `minScore` to filter more
2. Decrease `maxResults` limit
3. Ensure index is memoized
4. Check browser performance

### Wrong Results?
1. Increase `minScore` for better quality
2. Use more specific search terms
3. Add category filters
4. Check for typos

## 📚 Full Documentation

See [SETTINGS_SEARCH.md](./SETTINGS_SEARCH.md) for complete documentation.

## 🔗 Related Files

- Component: `/client/src/features/settings/components/SettingsSearch.tsx`
- Utilities: `/client/src/utils/settingsSearch.ts`
- Tests: `/tests/settingsSearch.test.ts`
- Panel Integration: `/client/src/features/settings/components/panels/SettingsPanelRedesign.tsx`

---

**Need Help?** Check the full documentation or tests for more examples.
