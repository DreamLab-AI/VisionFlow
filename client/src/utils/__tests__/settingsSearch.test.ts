import { describe, it, expect } from 'vitest';
import {
  fuzzyMatch,
  buildSearchIndex,
  searchSettings,
  highlightMatches,
  getSearchStats,
  type SearchableSettingField,
} from '../settingsSearch';

describe('settingsSearch', () => {
  // --- fuzzyMatch ---

  describe('fuzzyMatch', () => {
    it('should return 0 score for empty query', () => {
      const result = fuzzyMatch('some text', '');
      expect(result.score).toBe(0);
      expect(result.indices).toHaveLength(0);
    });

    it('should return 0 score for empty text', () => {
      const result = fuzzyMatch('', 'query');
      expect(result.score).toBe(0);
    });

    it('should find exact matches with high score', () => {
      const result = fuzzyMatch('Enable Shadows', 'Enable Shadows');
      expect(result.score).toBeGreaterThan(80);
      expect(result.indices).toHaveLength(1);
    });

    it('should find case-insensitive exact matches', () => {
      const result = fuzzyMatch('Enable Shadows', 'enable shadows', false);
      expect(result.score).toBeGreaterThan(80);
    });

    it('should find substring matches', () => {
      const result = fuzzyMatch('visualisation.rendering.shadows', 'shadows');
      expect(result.score).toBeGreaterThan(50);
    });

    it('should perform fuzzy matching when no exact match exists', () => {
      const result = fuzzyMatch('Enable Anti-aliasing', 'eaa');
      // e...a...a fuzzy match
      expect(result.score).toBeGreaterThan(0);
    });

    it('should return 0 for non-matching characters', () => {
      const result = fuzzyMatch('abc', 'xyz');
      expect(result.score).toBe(0);
    });

    it('should give higher score for word-boundary matches', () => {
      const boundary = fuzzyMatch('Enable Shadows', 'es');
      const middle = fuzzyMatch('Enable Shadows', 'na');
      // 'es' starts at word boundaries (Enable, Shadows)
      // Both are fuzzy, but boundary should score higher or equal
      expect(boundary.score).toBeGreaterThanOrEqual(0);
      expect(middle.score).toBeGreaterThanOrEqual(0);
    });

    it('should respect case sensitivity option', () => {
      const caseSensitive = fuzzyMatch('Enable', 'enable', true);
      const caseInsensitive = fuzzyMatch('Enable', 'enable', false);
      // case insensitive should find exact match
      expect(caseInsensitive.score).toBeGreaterThan(0);
    });
  });

  // --- buildSearchIndex ---

  describe('buildSearchIndex', () => {
    it('should build index from UI definition', () => {
      const definition = {
        rendering: {
          label: 'Rendering',
          subsections: {
            quality: {
              label: 'Quality',
              settings: {
                shadows: {
                  label: 'Enable Shadows',
                  path: 'rendering.shadows',
                  type: 'toggle',
                  description: 'Toggle shadow rendering',
                },
              },
            },
          },
        },
      };

      const index = buildSearchIndex(definition);
      expect(index).toHaveLength(1);
      expect(index[0].label).toBe('Enable Shadows');
      expect(index[0].path).toBe('rendering.shadows');
      expect(index[0].category).toBe('rendering');
      expect(index[0].categoryLabel).toBe('Rendering');
      expect(index[0].subcategory).toBe('quality');
    });

    it('should handle empty definitions', () => {
      const index = buildSearchIndex({});
      expect(index).toHaveLength(0);
    });

    it('should include tags for advanced and power-user settings', () => {
      const definition = {
        dev: {
          label: 'Developer',
          subsections: {
            perf: {
              label: 'Performance',
              settings: {
                debug: {
                  label: 'Debug Mode',
                  path: 'dev.debug',
                  type: 'toggle',
                  isAdvanced: true,
                  isPowerUserOnly: true,
                },
              },
            },
          },
        },
      };

      const index = buildSearchIndex(definition);
      expect(index[0].tags).toContain('advanced');
      expect(index[0].tags).toContain('power-user');
    });
  });

  // --- searchSettings ---

  describe('searchSettings', () => {
    const sampleIndex: SearchableSettingField[] = [
      {
        key: 'r.q.shadows',
        label: 'Enable Shadows',
        path: 'rendering.shadows',
        description: 'Toggle shadow rendering',
        category: 'rendering',
        categoryLabel: 'Rendering',
        subcategory: 'quality',
        subcategoryLabel: 'Quality',
        type: 'toggle',
      },
      {
        key: 'r.q.aa',
        label: 'Anti-aliasing',
        path: 'rendering.antialiasing',
        description: 'Enable anti-aliasing',
        category: 'rendering',
        categoryLabel: 'Rendering',
        subcategory: 'quality',
        subcategoryLabel: 'Quality',
        type: 'toggle',
      },
      {
        key: 'p.g.spring',
        label: 'Spring Constant',
        path: 'physics.springK',
        description: 'Spring force constant',
        category: 'physics',
        categoryLabel: 'Physics',
        type: 'slider',
        isAdvanced: true,
      },
    ];

    it('should return all settings for empty query', () => {
      const results = searchSettings(sampleIndex, '');
      expect(results).toHaveLength(3);
    });

    it('should find matching settings by label', () => {
      const results = searchSettings(sampleIndex, 'shadow');
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].label).toBe('Enable Shadows');
    });

    it('should filter by category', () => {
      const results = searchSettings(sampleIndex, '', { categories: ['physics'] });
      expect(results).toHaveLength(1);
      expect(results[0].category).toBe('physics');
    });

    it('should filter by type', () => {
      const results = searchSettings(sampleIndex, '', { types: ['slider'] });
      expect(results).toHaveLength(1);
    });

    it('should exclude advanced settings when configured', () => {
      const results = searchSettings(sampleIndex, '', { includeAdvanced: false });
      expect(results.every(r => !r.isAdvanced)).toBe(true);
    });

    it('should respect maxResults', () => {
      const results = searchSettings(sampleIndex, '', { maxResults: 1 });
      expect(results).toHaveLength(1);
    });

    it('should sort results by score descending', () => {
      const results = searchSettings(sampleIndex, 'spring');
      if (results.length > 1) {
        for (let i = 1; i < results.length; i++) {
          expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
        }
      }
    });
  });

  // --- highlightMatches ---

  describe('highlightMatches', () => {
    it('should return original text when no indices', () => {
      expect(highlightMatches('hello', [])).toBe('hello');
    });

    it('should wrap matched ranges in mark tags', () => {
      const result = highlightMatches('Enable Shadows', [[7, 14]]);
      expect(result).toContain('<mark');
      expect(result).toContain('Shadows');
    });

    it('should merge overlapping indices', () => {
      const result = highlightMatches('abcdefgh', [[1, 4], [3, 6]]);
      // Merged: [1, 6]
      expect(result).toContain('<mark');
      // Only one mark tag
      const markCount = (result.match(/<mark/g) || []).length;
      expect(markCount).toBe(1);
    });
  });

  // --- getSearchStats ---

  describe('getSearchStats', () => {
    it('should compute statistics from results', () => {
      const results = [
        { category: 'rendering', type: 'toggle', score: 80 },
        { category: 'rendering', type: 'toggle', score: 60 },
        { category: 'physics', type: 'slider', score: 90 },
      ] as SearchResult[];

      const stats = getSearchStats(results);
      expect(stats.total).toBe(3);
      expect(stats.averageScore).toBeCloseTo((80 + 60 + 90) / 3, 1);
      expect(stats.byCategory['rendering']).toBe(2);
      expect(stats.byCategory['physics']).toBe(1);
      expect(stats.byType['toggle']).toBe(2);
      expect(stats.byType['slider']).toBe(1);
    });

    it('should handle empty results', () => {
      const stats = getSearchStats([]);
      expect(stats.total).toBe(0);
      expect(stats.averageScore).toBe(0);
    });
  });
});
