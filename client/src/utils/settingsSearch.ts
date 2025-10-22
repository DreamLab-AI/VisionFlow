/**
 * Settings Search Utilities
 *
 * High-performance fuzzy search and filtering for 1,061+ settings.
 * Optimized for real-time search with scoring, ranking, and highlighting.
 */

import { UICategoryDefinition, UISettingDefinition } from '../features/settings/config/widgetTypes';

/**
 * Searchable representation of a setting field with all indexed data
 */
export interface SearchableSettingField {
  /** Unique setting key */
  key: string;
  /** Display label */
  label: string;
  /** Dot-notation path (e.g., 'visualisation.nodes.baseColor') */
  path: string;
  /** Setting description (optional) */
  description?: string;
  /** Parent category (e.g., 'visualization', 'physics') */
  category: string;
  /** Parent category label (e.g., 'Visualization', 'Physics') */
  categoryLabel: string;
  /** Subsection name (e.g., 'nodes', 'edges') */
  subcategory?: string;
  /** Subsection label (e.g., 'Node Settings', 'Edge Settings') */
  subcategoryLabel?: string;
  /** Widget type for filtering (e.g., 'slider', 'toggle', 'colorPicker') */
  type: string;
  /** Tags for categorization */
  tags?: string[];
  /** Is this a power-user setting? */
  isPowerUserOnly?: boolean;
  /** Is this an advanced setting? */
  isAdvanced?: boolean;
  /** Is this stored in localStorage? */
  localStorage?: boolean;
}

/**
 * Search result with relevance score and match highlights
 */
export interface SearchResult extends SearchableSettingField {
  /** Relevance score (0-100, higher is better) */
  score: number;
  /** Matched search terms for highlighting */
  matches: {
    field: 'label' | 'path' | 'description' | 'category' | 'subcategory';
    text: string;
    indices: [number, number][];
  }[];
}

/**
 * Search configuration options
 */
export interface SearchOptions {
  /** Minimum score threshold (0-100) */
  minScore?: number;
  /** Maximum results to return */
  maxResults?: number;
  /** Search only in specific categories */
  categories?: string[];
  /** Search only specific widget types */
  types?: string[];
  /** Include advanced settings */
  includeAdvanced?: boolean;
  /** Include power-user settings */
  includePowerUser?: boolean;
  /** Case-sensitive search */
  caseSensitive?: boolean;
  /** Fuzzy matching threshold (0-1, lower = more fuzzy) */
  fuzzyThreshold?: number;
}

/**
 * Default search options
 */
const DEFAULT_OPTIONS: Required<SearchOptions> = {
  minScore: 20,
  maxResults: 100,
  categories: [],
  types: [],
  includeAdvanced: true,
  includePowerUser: true,
  caseSensitive: false,
  fuzzyThreshold: 0.6
};

/**
 * Advanced fuzzy matching algorithm with position-aware scoring
 *
 * Scoring factors:
 * - Exact substring match: 100 points
 * - Word boundary match: +20 points
 * - Start of string match: +15 points
 * - Consecutive character match: 10 points per char
 * - Non-consecutive match: 5 points per char
 * - Early position bonus: up to +10 points
 * - Case match bonus: +5 points
 *
 * @param text - The text to search in
 * @param query - The search query
 * @param caseSensitive - Whether to perform case-sensitive matching
 * @returns Score (0-100) and match indices
 */
export const fuzzyMatch = (
  text: string,
  query: string,
  caseSensitive: boolean = false
): { score: number; indices: [number, number][] } => {
  if (!query) return { score: 0, indices: [] };
  if (!text) return { score: 0, indices: [] };

  const originalText = text;
  const originalQuery = query;

  if (!caseSensitive) {
    text = text.toLowerCase();
    query = query.toLowerCase();
  }

  // Exact match - highest score
  const exactIndex = text.indexOf(query);
  if (exactIndex !== -1) {
    let score = 100;

    // Bonus for word boundary
    if (exactIndex === 0 || /\s/.test(text[exactIndex - 1])) {
      score += 20;
    }

    // Bonus for start of string
    if (exactIndex === 0) {
      score += 15;
    }

    // Case match bonus
    if (caseSensitive && originalText.substring(exactIndex, exactIndex + query.length) === originalQuery) {
      score += 5;
    }

    return {
      score: Math.min(score, 100),
      indices: [[exactIndex, exactIndex + query.length]]
    };
  }

  // Fuzzy match - character by character with position tracking
  let score = 0;
  let queryIndex = 0;
  let textIndex = 0;
  const indices: [number, number][] = [];
  let consecutiveMatches = 0;
  let matchStart = -1;

  while (textIndex < text.length && queryIndex < query.length) {
    if (text[textIndex] === query[queryIndex]) {
      if (matchStart === -1) {
        matchStart = textIndex;
      }

      consecutiveMatches++;

      // Consecutive character bonus
      if (consecutiveMatches > 1) {
        score += 10;
      } else {
        score += 5;
      }

      // Word boundary bonus
      if (textIndex === 0 || /\s/.test(text[textIndex - 1])) {
        score += 10;
      }

      // Early position bonus (first 20% of string)
      if (textIndex < text.length * 0.2) {
        score += 5;
      }

      // Case match bonus
      if (caseSensitive && originalText[textIndex] === originalQuery[queryIndex]) {
        score += 2;
      }

      queryIndex++;
    } else {
      // End consecutive sequence
      if (matchStart !== -1) {
        indices.push([matchStart, textIndex]);
        matchStart = -1;
      }
      consecutiveMatches = 0;
    }
    textIndex++;
  }

  // Close final sequence
  if (matchStart !== -1) {
    indices.push([matchStart, textIndex]);
  }

  // Did we match all query characters?
  if (queryIndex < query.length) {
    return { score: 0, indices: [] };
  }

  // Normalize score to 0-100 range
  const maxPossibleScore = query.length * 15; // ~max score per char
  const normalizedScore = Math.min(100, (score / maxPossibleScore) * 100);

  return {
    score: normalizedScore,
    indices: indices.length > 0 ? indices : []
  };
};

/**
 * Build searchable index from settings UI definition
 *
 * @param settingsUIDefinition - The complete settings configuration
 * @returns Array of searchable setting fields
 */
export const buildSearchIndex = (
  settingsUIDefinition: Record<string, UICategoryDefinition>
): SearchableSettingField[] => {
  const index: SearchableSettingField[] = [];

  Object.entries(settingsUIDefinition).forEach(([categoryKey, categoryDef]) => {
    Object.entries(categoryDef.subsections || {}).forEach(([subsectionKey, subsection]) => {
      Object.entries(subsection.settings || {}).forEach(([settingKey, setting]) => {
        // Type assertion since we know the structure
        const settingDef = setting as UISettingDefinition;

        index.push({
          key: `${categoryKey}.${subsectionKey}.${settingKey}`,
          label: settingDef.label,
          path: settingDef.path,
          description: settingDef.description,
          category: categoryKey,
          categoryLabel: categoryDef.label,
          subcategory: subsectionKey,
          subcategoryLabel: subsection.label,
          type: settingDef.type,
          isPowerUserOnly: settingDef.isPowerUserOnly,
          isAdvanced: settingDef.isAdvanced,
          localStorage: settingDef.localStorage,
          tags: [
            categoryKey,
            subsectionKey,
            settingDef.type,
            ...(settingDef.isAdvanced ? ['advanced'] : []),
            ...(settingDef.isPowerUserOnly ? ['power-user'] : []),
            ...(settingDef.localStorage ? ['local'] : [])
          ]
        });
      });
    });
  });

  return index;
};

/**
 * Search settings with advanced fuzzy matching and scoring
 *
 * @param index - Searchable settings index
 * @param query - Search query string
 * @param options - Search configuration options
 * @returns Ranked array of search results
 */
export const searchSettings = (
  index: SearchableSettingField[],
  query: string,
  options: SearchOptions = {}
): SearchResult[] => {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Empty query returns all settings (filtered by options)
  if (!query.trim()) {
    return index
      .filter(setting => filterSetting(setting, opts))
      .slice(0, opts.maxResults)
      .map(setting => ({ ...setting, score: 0, matches: [] }));
  }

  const results: SearchResult[] = [];

  // Search each setting across multiple fields
  for (const setting of index) {
    // Apply filters
    if (!filterSetting(setting, opts)) {
      continue;
    }

    // Calculate scores for each searchable field
    const labelMatch = fuzzyMatch(setting.label, query, opts.caseSensitive);
    const pathMatch = fuzzyMatch(setting.path, query, opts.caseSensitive);
    const descMatch = setting.description
      ? fuzzyMatch(setting.description, query, opts.caseSensitive)
      : { score: 0, indices: [] };
    const categoryMatch = fuzzyMatch(setting.categoryLabel, query, opts.caseSensitive);
    const subcategoryMatch = setting.subcategoryLabel
      ? fuzzyMatch(setting.subcategoryLabel, query, opts.caseSensitive)
      : { score: 0, indices: [] };

    // Weighted score calculation (prioritize label > path > description)
    const totalScore =
      labelMatch.score * 3.0 +          // Label is most important
      pathMatch.score * 2.0 +            // Path is secondary
      descMatch.score * 1.0 +            // Description is helpful
      categoryMatch.score * 0.5 +        // Category provides context
      subcategoryMatch.score * 0.5;      // Subcategory provides context

    // Normalize to 0-100 range
    const normalizedScore = Math.min(100, totalScore / 7.0);

    // Filter by minimum score threshold
    if (normalizedScore < opts.minScore) {
      continue;
    }

    // Build match highlights
    const matches: SearchResult['matches'] = [];
    if (labelMatch.score > 0) {
      matches.push({ field: 'label', text: setting.label, indices: labelMatch.indices });
    }
    if (pathMatch.score > 0) {
      matches.push({ field: 'path', text: setting.path, indices: pathMatch.indices });
    }
    if (descMatch.score > 0 && setting.description) {
      matches.push({ field: 'description', text: setting.description, indices: descMatch.indices });
    }
    if (categoryMatch.score > 0) {
      matches.push({ field: 'category', text: setting.categoryLabel, indices: categoryMatch.indices });
    }
    if (subcategoryMatch.score > 0 && setting.subcategoryLabel) {
      matches.push({ field: 'subcategory', text: setting.subcategoryLabel, indices: subcategoryMatch.indices });
    }

    results.push({
      ...setting,
      score: normalizedScore,
      matches
    });
  }

  // Sort by score (descending) and limit results
  return results
    .sort((a, b) => b.score - a.score)
    .slice(0, opts.maxResults);
};

/**
 * Filter settings based on search options
 */
const filterSetting = (setting: SearchableSettingField, options: Required<SearchOptions>): boolean => {
  // Filter by category
  if (options.categories.length > 0 && !options.categories.includes(setting.category)) {
    return false;
  }

  // Filter by type
  if (options.types.length > 0 && !options.types.includes(setting.type)) {
    return false;
  }

  // Filter advanced settings
  if (!options.includeAdvanced && setting.isAdvanced) {
    return false;
  }

  // Filter power-user settings
  if (!options.includePowerUser && setting.isPowerUserOnly) {
    return false;
  }

  return true;
};

/**
 * Highlight matched text with HTML mark tags
 *
 * @param text - Original text
 * @param indices - Array of match indices [start, end]
 * @returns HTML string with <mark> tags
 */
export const highlightMatches = (text: string, indices: [number, number][]): string => {
  if (!indices || indices.length === 0) return text;

  let result = '';
  let lastIndex = 0;

  // Sort and merge overlapping indices
  const sortedIndices = [...indices].sort((a, b) => a[0] - b[0]);
  const mergedIndices: [number, number][] = [];

  for (const [start, end] of sortedIndices) {
    if (mergedIndices.length === 0) {
      mergedIndices.push([start, end]);
    } else {
      const last = mergedIndices[mergedIndices.length - 1];
      if (start <= last[1]) {
        // Merge overlapping ranges
        last[1] = Math.max(last[1], end);
      } else {
        mergedIndices.push([start, end]);
      }
    }
  }

  // Build highlighted string
  for (const [start, end] of mergedIndices) {
    result += text.substring(lastIndex, start);
    result += `<mark class="bg-primary/20 text-primary font-medium">${text.substring(start, end)}</mark>`;
    lastIndex = end;
  }
  result += text.substring(lastIndex);

  return result;
};

/**
 * Get search statistics
 */
export const getSearchStats = (results: SearchResult[]) => {
  const categoryCount = new Map<string, number>();
  const typeCount = new Map<string, number>();

  for (const result of results) {
    categoryCount.set(result.category, (categoryCount.get(result.category) || 0) + 1);
    typeCount.set(result.type, (typeCount.get(result.type) || 0) + 1);
  }

  return {
    total: results.length,
    averageScore: results.reduce((sum, r) => sum + r.score, 0) / results.length || 0,
    byCategory: Object.fromEntries(categoryCount),
    byType: Object.fromEntries(typeCount)
  };
};
