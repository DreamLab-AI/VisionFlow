/**
 * Settings Search Tests
 *
 * Comprehensive test suite for fuzzy search functionality
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import {
  fuzzyMatch,
  searchSettings,
  buildSearchIndex,
  highlightMatches,
  getSearchStats,
  SearchableSettingField,
  SearchResult
} from '../client/src/utils/settingsSearch';
import { settingsUIDefinition } from '../client/src/features/settings/config/settingsUIDefinition';

describe('Settings Search - Fuzzy Matching', () => {
  describe('fuzzyMatch', () => {
    it('should return 100 score for exact match', () => {
      const result = fuzzyMatch('node color', 'node color', false);
      expect(result.score).toBeGreaterThanOrEqual(100);
      expect(result.indices.length).toBeGreaterThan(0);
    });

    it('should match case-insensitively by default', () => {
      const result1 = fuzzyMatch('Node Color', 'node', false);
      const result2 = fuzzyMatch('node color', 'NODE', false);
      expect(result1.score).toBeGreaterThan(0);
      expect(result2.score).toBeGreaterThan(0);
    });

    it('should handle partial matches', () => {
      const result = fuzzyMatch('background color picker', 'color', false);
      expect(result.score).toBeGreaterThan(0);
      expect(result.indices.length).toBeGreaterThan(0);
    });

    it('should score consecutive matches higher', () => {
      const consecutive = fuzzyMatch('nodecolor', 'nodecolor', false);
      const nonConsecutive = fuzzyMatch('node-color-picker', 'nodecolor', false);
      expect(consecutive.score).toBeGreaterThan(nonConsecutive.score);
    });

    it('should return 0 for no match', () => {
      const result = fuzzyMatch('physics settings', 'visualization', false);
      expect(result.score).toBe(0);
      expect(result.indices.length).toBe(0);
    });

    it('should handle empty strings', () => {
      expect(fuzzyMatch('', 'query', false).score).toBe(0);
      expect(fuzzyMatch('text', '', false).score).toBe(0);
    });

    it('should bonus word boundary matches', () => {
      const wordBoundary = fuzzyMatch('node size', 'node', false);
      const midWord = fuzzyMatch('nodentity', 'node', false);
      expect(wordBoundary.score).toBeGreaterThan(midWord.score);
    });

    it('should return match indices', () => {
      const result = fuzzyMatch('background color', 'color', false);
      expect(result.indices).toEqual([[11, 16]]);
    });
  });
});

describe('Settings Search - Index Building', () => {
  let searchIndex: SearchableSettingField[];

  beforeEach(() => {
    searchIndex = buildSearchIndex(settingsUIDefinition);
  });

  it('should build search index from settings definition', () => {
    expect(searchIndex).toBeDefined();
    expect(searchIndex.length).toBeGreaterThan(1000); // We have 1,061+ settings
  });

  it('should include all required fields', () => {
    const setting = searchIndex[0];
    expect(setting).toHaveProperty('key');
    expect(setting).toHaveProperty('label');
    expect(setting).toHaveProperty('path');
    expect(setting).toHaveProperty('category');
    expect(setting).toHaveProperty('categoryLabel');
    expect(setting).toHaveProperty('type');
  });

  it('should index settings from all categories', () => {
    const categories = new Set(searchIndex.map(s => s.category));
    expect(categories.size).toBeGreaterThan(5); // Multiple categories
    expect(categories.has('visualization')).toBe(true);
    expect(categories.has('physics')).toBe(true);
  });

  it('should include descriptions when available', () => {
    const withDescription = searchIndex.filter(s => s.description);
    expect(withDescription.length).toBeGreaterThan(0);
  });

  it('should tag settings correctly', () => {
    const advancedSettings = searchIndex.filter(s => s.isAdvanced);
    expect(advancedSettings.length).toBeGreaterThan(0);
    expect(advancedSettings[0].tags).toContain('advanced');
  });
});

describe('Settings Search - Search Functionality', () => {
  let searchIndex: SearchableSettingField[];

  beforeEach(() => {
    searchIndex = buildSearchIndex(settingsUIDefinition);
  });

  it('should return empty array for empty query', () => {
    const results = searchSettings(searchIndex, '', {});
    expect(results.length).toBe(0);
  });

  it('should find settings by label', () => {
    const results = searchSettings(searchIndex, 'node color', {});
    expect(results.length).toBeGreaterThan(0);
    expect(results.some(r => r.label.toLowerCase().includes('color'))).toBe(true);
  });

  it('should find settings by path', () => {
    const results = searchSettings(searchIndex, 'visualisation.nodes', {});
    expect(results.length).toBeGreaterThan(0);
    expect(results.some(r => r.path.includes('visualisation.nodes'))).toBe(true);
  });

  it('should find settings by description', () => {
    const results = searchSettings(searchIndex, 'transparency', {});
    expect(results.length).toBeGreaterThan(0);
  });

  it('should rank results by relevance', () => {
    const results = searchSettings(searchIndex, 'node', {});
    expect(results.length).toBeGreaterThan(1);
    // First result should have highest score
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });

  it('should respect minScore threshold', () => {
    const results = searchSettings(searchIndex, 'xyz', { minScore: 50 });
    results.forEach(r => {
      expect(r.score).toBeGreaterThanOrEqual(50);
    });
  });

  it('should respect maxResults limit', () => {
    const results = searchSettings(searchIndex, 'node', { maxResults: 10 });
    expect(results.length).toBeLessThanOrEqual(10);
  });

  it('should filter by category', () => {
    const results = searchSettings(searchIndex, 'enable', {
      categories: ['visualization']
    });
    results.forEach(r => {
      expect(r.category).toBe('visualization');
    });
  });

  it('should filter by widget type', () => {
    const results = searchSettings(searchIndex, 'enable', {
      types: ['toggle']
    });
    results.forEach(r => {
      expect(r.type).toBe('toggle');
    });
  });

  it('should handle fuzzy matching', () => {
    // Typo: "colr" instead of "color"
    const results = searchSettings(searchIndex, 'nde colr', {});
    expect(results.length).toBeGreaterThan(0);
  });

  it('should include match highlights', () => {
    const results = searchSettings(searchIndex, 'node color', {});
    expect(results[0].matches.length).toBeGreaterThan(0);
    expect(results[0].matches[0]).toHaveProperty('field');
    expect(results[0].matches[0]).toHaveProperty('text');
    expect(results[0].matches[0]).toHaveProperty('indices');
  });

  it('should handle multi-word queries', () => {
    const results = searchSettings(searchIndex, 'background color picker', {});
    expect(results.length).toBeGreaterThan(0);
  });

  it('should filter advanced settings when requested', () => {
    const allResults = searchSettings(searchIndex, 'enable', {
      includeAdvanced: true
    });
    const nonAdvancedResults = searchSettings(searchIndex, 'enable', {
      includeAdvanced: false
    });
    expect(allResults.length).toBeGreaterThan(nonAdvancedResults.length);
  });
});

describe('Settings Search - Highlighting', () => {
  it('should highlight matched text', () => {
    const html = highlightMatches('node color picker', [[0, 4], [5, 10]]);
    expect(html).toContain('<mark');
    expect(html).toContain('node');
    expect(html).toContain('color');
  });

  it('should merge overlapping indices', () => {
    const html = highlightMatches('testing', [[0, 4], [3, 7]]);
    expect((html.match(/<mark/g) || []).length).toBeLessThanOrEqual(1);
  });

  it('should return original text for empty indices', () => {
    const html = highlightMatches('test text', []);
    expect(html).toBe('test text');
  });
});

describe('Settings Search - Statistics', () => {
  it('should calculate search statistics', () => {
    const mockResults: SearchResult[] = [
      {
        key: 'key1',
        label: 'Node Color',
        path: 'viz.node.color',
        category: 'visualization',
        categoryLabel: 'Visualization',
        type: 'colorPicker',
        score: 95,
        matches: []
      },
      {
        key: 'key2',
        label: 'Edge Width',
        path: 'viz.edge.width',
        category: 'visualization',
        categoryLabel: 'Visualization',
        type: 'slider',
        score: 85,
        matches: []
      }
    ];

    const stats = getSearchStats(mockResults);
    expect(stats.total).toBe(2);
    expect(stats.averageScore).toBeCloseTo(90, 0);
    expect(stats.byCategory).toHaveProperty('visualization');
    expect(stats.byType).toHaveProperty('colorPicker');
    expect(stats.byType).toHaveProperty('slider');
  });
});

describe('Settings Search - Performance', () => {
  let searchIndex: SearchableSettingField[];

  beforeEach(() => {
    searchIndex = buildSearchIndex(settingsUIDefinition);
  });

  it('should handle large index efficiently', () => {
    const start = Date.now();
    searchSettings(searchIndex, 'node', { maxResults: 50 });
    const duration = Date.now() - start;

    // Search should complete in under 100ms for 1000+ settings
    expect(duration).toBeLessThan(100);
  });

  it('should scale with query complexity', () => {
    const simpleQuery = 'node';
    const complexQuery = 'background color picker settings advanced';

    const start1 = Date.now();
    searchSettings(searchIndex, simpleQuery, {});
    const simple Duration = Date.now() - start1;

    const start2 = Date.now();
    searchSettings(searchIndex, complexQuery, {});
    const complexDuration = Date.now() - start2;

    // Complex queries shouldn't be more than 2x slower
    expect(complexDuration).toBeLessThan(simpleDuration * 2 + 50);
  });

  it('should handle concurrent searches', async () => {
    const queries = ['node', 'color', 'physics', 'enable', 'settings'];

    const start = Date.now();
    const results = await Promise.all(
      queries.map(q => Promise.resolve(searchSettings(searchIndex, q, {})))
    );
    const duration = Date.now() - start;

    expect(results.length).toBe(5);
    expect(duration).toBeLessThan(200); // All searches under 200ms
  });
});

describe('Settings Search - Edge Cases', () => {
  let searchIndex: SearchableSettingField[];

  beforeEach(() => {
    searchIndex = buildSearchIndex(settingsUIDefinition);
  });

  it('should handle special characters', () => {
    const results = searchSettings(searchIndex, 'node-color', {});
    expect(results.length).toBeGreaterThanOrEqual(0); // Should not throw
  });

  it('should handle very long queries', () => {
    const longQuery = 'a'.repeat(1000);
    const results = searchSettings(searchIndex, longQuery, {});
    expect(results.length).toBeGreaterThanOrEqual(0); // Should not throw
  });

  it('should handle unicode characters', () => {
    const results = searchSettings(searchIndex, 'node 色彩', {});
    expect(results.length).toBeGreaterThanOrEqual(0); // Should not throw
  });

  it('should handle numbers in queries', () => {
    const results = searchSettings(searchIndex, '0.5 opacity', {});
    expect(results.length).toBeGreaterThan(0);
  });
});
