

import { UICategoryDefinition, UISettingDefinition } from '../features/settings/config/widgetTypes';


export interface SearchableSettingField {
  
  key: string;
  
  label: string;
  
  path: string;
  
  description?: string;
  
  category: string;
  
  categoryLabel: string;
  
  subcategory?: string;
  
  subcategoryLabel?: string;
  
  type: string;
  
  tags?: string[];
  
  isPowerUserOnly?: boolean;
  
  isAdvanced?: boolean;
  
  localStorage?: boolean;
}


export interface SearchResult extends SearchableSettingField {
  
  score: number;
  
  matches: {
    field: 'label' | 'path' | 'description' | 'category' | 'subcategory';
    text: string;
    indices: [number, number][];
  }[];
}


export interface SearchOptions {
  
  minScore?: number;
  
  maxResults?: number;
  
  categories?: string[];
  
  types?: string[];
  
  includeAdvanced?: boolean;
  
  includePowerUser?: boolean;
  
  caseSensitive?: boolean;
  
  fuzzyThreshold?: number;
}


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

  
  const exactIndex = text.indexOf(query);
  if (exactIndex !== -1) {
    let score = 100;

    
    if (exactIndex === 0 || /\s/.test(text[exactIndex - 1])) {
      score += 20;
    }

    
    if (exactIndex === 0) {
      score += 15;
    }

    
    if (caseSensitive && originalText.substring(exactIndex, exactIndex + query.length) === originalQuery) {
      score += 5;
    }

    return {
      score: Math.min(score, 100),
      indices: [[exactIndex, exactIndex + query.length]]
    };
  }

  
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

      
      if (consecutiveMatches > 1) {
        score += 10;
      } else {
        score += 5;
      }

      
      if (textIndex === 0 || /\s/.test(text[textIndex - 1])) {
        score += 10;
      }

      
      if (textIndex < text.length * 0.2) {
        score += 5;
      }

      
      if (caseSensitive && originalText[textIndex] === originalQuery[queryIndex]) {
        score += 2;
      }

      queryIndex++;
    } else {
      
      if (matchStart !== -1) {
        indices.push([matchStart, textIndex]);
        matchStart = -1;
      }
      consecutiveMatches = 0;
    }
    textIndex++;
  }

  
  if (matchStart !== -1) {
    indices.push([matchStart, textIndex]);
  }

  
  if (queryIndex < query.length) {
    return { score: 0, indices: [] };
  }

  
  const maxPossibleScore = query.length * 15; 
  const normalizedScore = Math.min(100, (score / maxPossibleScore) * 100);

  return {
    score: normalizedScore,
    indices: indices.length > 0 ? indices : []
  };
};


export const buildSearchIndex = (
  settingsUIDefinition: Record<string, UICategoryDefinition>
): SearchableSettingField[] => {
  const index: SearchableSettingField[] = [];

  Object.entries(settingsUIDefinition).forEach(([categoryKey, categoryDef]) => {
    Object.entries(categoryDef.subsections || {}).forEach(([subsectionKey, subsection]) => {
      Object.entries(subsection.settings || {}).forEach(([settingKey, setting]) => {
        
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


export const searchSettings = (
  index: SearchableSettingField[],
  query: string,
  options: SearchOptions = {}
): SearchResult[] => {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  
  if (!query.trim()) {
    return index
      .filter(setting => filterSetting(setting, opts))
      .slice(0, opts.maxResults)
      .map(setting => ({ ...setting, score: 0, matches: [] }));
  }

  const results: SearchResult[] = [];

  
  for (const setting of index) {
    
    if (!filterSetting(setting, opts)) {
      continue;
    }

    
    const labelMatch = fuzzyMatch(setting.label, query, opts.caseSensitive);
    const pathMatch = fuzzyMatch(setting.path, query, opts.caseSensitive);
    const descMatch = setting.description
      ? fuzzyMatch(setting.description, query, opts.caseSensitive)
      : { score: 0, indices: [] };
    const categoryMatch = fuzzyMatch(setting.categoryLabel, query, opts.caseSensitive);
    const subcategoryMatch = setting.subcategoryLabel
      ? fuzzyMatch(setting.subcategoryLabel, query, opts.caseSensitive)
      : { score: 0, indices: [] };

    
    const totalScore =
      labelMatch.score * 3.0 +          
      pathMatch.score * 2.0 +            
      descMatch.score * 1.0 +            
      categoryMatch.score * 0.5 +        
      subcategoryMatch.score * 0.5;      

    
    const normalizedScore = Math.min(100, totalScore / 7.0);

    
    if (normalizedScore < opts.minScore) {
      continue;
    }

    
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

  
  return results
    .sort((a, b) => b.score - a.score)
    .slice(0, opts.maxResults);
};


const filterSetting = (setting: SearchableSettingField, options: Required<SearchOptions>): boolean => {
  
  if (options.categories.length > 0 && !options.categories.includes(setting.category)) {
    return false;
  }

  
  if (options.types.length > 0 && !options.types.includes(setting.type)) {
    return false;
  }

  
  if (!options.includeAdvanced && setting.isAdvanced) {
    return false;
  }

  
  if (!options.includePowerUser && setting.isPowerUserOnly) {
    return false;
  }

  return true;
};


export const highlightMatches = (text: string, indices: [number, number][]): string => {
  if (!indices || indices.length === 0) return text;

  let result = '';
  let lastIndex = 0;

  
  const sortedIndices = [...indices].sort((a, b) => a[0] - b[0]);
  const mergedIndices: [number, number][] = [];

  for (const [start, end] of sortedIndices) {
    if (mergedIndices.length === 0) {
      mergedIndices.push([start, end]);
    } else {
      const last = mergedIndices[mergedIndices.length - 1];
      if (start <= last[1]) {
        
        last[1] = Math.max(last[1], end);
      } else {
        mergedIndices.push([start, end]);
      }
    }
  }

  
  for (const [start, end] of mergedIndices) {
    result += text.substring(lastIndex, start);
    result += `<mark class="bg-primary/20 text-primary font-medium">${text.substring(start, end)}</mark>`;
    lastIndex = end;
  }
  result += text.substring(lastIndex);

  return result;
};


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
