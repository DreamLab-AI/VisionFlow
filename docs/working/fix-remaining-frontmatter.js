#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const DOCS_DIR = '/home/devuser/workspace/project/docs';

const STANDARD_TAGS = [
  'architecture', 'api', 'authentication', 'agents', 'automation', 'backend',
  'ci-cd', 'configuration', 'database', 'debugging', 'deployment', 'development',
  'docker', 'documentation', 'features', 'frontend', 'getting-started', 'gpu',
  'infrastructure', 'installation', 'integration', 'kubernetes', 'learning',
  'mcp', 'memory', 'migration', 'monitoring', 'neural', 'optimization', 'performance',
  'plugins', 'reference', 'security', 'setup', 'swarm', 'testing', 'tools', 'tutorial',
  'ui', 'validation', 'workflow'
];

const INVALID_TAGS = [
  'rust', 'rest', 'websocket', 'visionflow', 'neo4j', 'guide', 'react',
  'design', 'patterns', 'structure', 'contribution', 'standards', 'http',
  'ai', 'maintenance', 'procedures', 'quality', 'navigation', 'search',
  'checklist', 'guides', 'types', 'typescript', 'implementation', 'components',
  'protocol', 'migration-plan', 'ontology', 'visualization', 'gpu-compute',
  'physics', 'semantic', 'graph', 'client', 'server', 'data-flow', 'cqrs',
  'pipeline', 'services', 'ports', 'handlers', 'pathfinding', 'parser',
  'analytics', 'pagerank', 'clustering', 'lod', 'rendering', 'webgl',
  'sync', 'github', 'history', 'changelog', 'roadmap', 'archive', 'fixes',
  'troubleshooting', 'errors', 'refactor', 'migration-guide', 'audit',
  'report', 'summary', 'logs', 'sprint', 'tasks', 'completion', 'http-api',
  'turboflow', 'multi-agent', 'model', 'llm', 'embedding', 'vector',
  'search-algorithms', 'real-time', 'hnsw', 'versioning', 'update', 'system'
];

const TAG_MAPPING = {
  'rust': 'backend',
  'rest': 'api',
  'websocket': 'api',
  'visionflow': 'documentation',
  'neo4j': 'database',
  'guide': 'tutorial',
  'react': 'frontend',
  'design': 'architecture',
  'patterns': 'architecture',
  'structure': 'architecture',
  'contribution': 'documentation',
  'standards': 'documentation',
  'http': 'api',
  'ai': 'agents',
  'maintenance': 'documentation',
  'procedures': 'workflow',
  'quality': 'validation',
  'navigation': 'documentation',
  'search': 'documentation',
  'checklist': 'validation',
  'guides': 'tutorial',
  'types': 'reference',
  'typescript': 'reference',
  'implementation': 'development',
  'components': 'architecture',
  'protocol': 'api',
  'migration-plan': 'migration',
  'ontology': 'architecture',
  'visualization': 'ui',
  'gpu-compute': 'gpu',
  'physics': 'features',
  'semantic': 'features',
  'graph': 'architecture',
  'client': 'frontend',
  'server': 'backend',
  'data-flow': 'architecture',
  'cqrs': 'architecture',
  'pipeline': 'workflow',
  'services': 'architecture',
  'ports': 'architecture',
  'handlers': 'api',
  'pathfinding': 'features',
  'parser': 'features',
  'analytics': 'monitoring',
  'pagerank': 'features',
  'clustering': 'features',
  'lod': 'features',
  'rendering': 'ui',
  'webgl': 'ui',
  'sync': 'integration',
  'github': 'integration',
  'history': 'documentation',
  'changelog': 'documentation',
  'roadmap': 'documentation',
  'archive': 'documentation',
  'fixes': 'debugging',
  'troubleshooting': 'debugging',
  'errors': 'debugging',
  'refactor': 'development',
  'migration-guide': 'migration',
  'audit': 'validation',
  'report': 'documentation',
  'summary': 'documentation',
  'logs': 'monitoring',
  'sprint': 'workflow',
  'tasks': 'workflow',
  'completion': 'documentation',
  'http-api': 'api',
  'turboflow': 'documentation',
  'multi-agent': 'swarm',
  'model': 'agents',
  'llm': 'agents',
  'embedding': 'neural',
  'vector': 'neural',
  'search-algorithms': 'features',
  'real-time': 'performance',
  'hnsw': 'optimization',
  'versioning': 'documentation',
  'update': 'documentation',
  'system': 'architecture'
};

const INVALID_CATEGORIES = ['guides', 'howto', 'report'];
const CATEGORY_MAPPING = {
  'guides': 'guide',
  'howto': 'guide',
  'report': 'explanation'
};

function sanitizeTags(tags) {
  const sanitized = new Set();
  for (const tag of tags) {
    const cleaned = tag.toLowerCase().trim();
    if (STANDARD_TAGS.includes(cleaned)) {
      sanitized.add(cleaned);
    } else if (TAG_MAPPING[cleaned]) {
      sanitized.add(TAG_MAPPING[cleaned]);
    }
  }

  if (sanitized.size === 0) {
    sanitized.add('documentation');
  }

  return Array.from(sanitized).slice(0, 5);
}

function sanitizeCategory(category) {
  if (CATEGORY_MAPPING[category]) {
    return CATEGORY_MAPPING[category];
  }
  if (['tutorial', 'guide', 'explanation', 'reference'].includes(category)) {
    return category;
  }
  return 'explanation';
}

function sanitizeDifficulty(difficulty) {
  if (difficulty && difficulty.startsWith('adva')) {
    return 'advanced';
  }
  if (['beginner', 'intermediate', 'advanced'].includes(difficulty)) {
    return difficulty;
  }
  return 'intermediate';
}

function generateTitle(filePath, content) {
  const headingMatch = content.match(/^#\s+(.+)$/m);
  if (headingMatch) {
    return headingMatch[1].trim();
  }

  const filename = path.basename(filePath, '.md');
  return filename
    .replace(/-/g, ' ')
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function generateDescription(content) {
  const withoutFrontmatter = content.replace(/^---\n[\s\S]*?\n---\n/, '');
  const withoutHeadings = withoutFrontmatter.replace(/^#+\s+.+$/gm, '');
  const paragraphs = withoutHeadings.split('\n\n').filter(p => p.trim().length > 20);

  if (paragraphs.length > 0) {
    let desc = paragraphs[0].trim().replace(/\n/g, ' ').substring(0, 150);
    if (desc.length === 150) desc += '...';
    return desc;
  }

  return 'Documentation file';
}

function extractExistingFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n/);
  if (!match) return null;

  const yaml = match[1];
  const frontmatter = {};

  const lines = yaml.split('\n');
  let currentKey = null;
  let currentArray = null;

  for (const line of lines) {
    if (line.match(/^[a-z-]+:/)) {
      const colonIndex = line.indexOf(':');
      const key = line.substring(0, colonIndex).trim();
      const value = line.substring(colonIndex + 1).trim();
      currentKey = key;

      if (value) {
        frontmatter[key] = value.replace(/^["']|["']$/g, '');
      } else {
        currentArray = [];
        frontmatter[key] = currentArray;
      }
    } else if (currentArray && line.trim().startsWith('-')) {
      currentArray.push(line.trim().substring(1).trim());
    }
  }

  return frontmatter;
}

function processFile(filePath) {
  try {
    const fullPath = path.join(DOCS_DIR, filePath);
    const content = fs.readFileSync(fullPath, 'utf8');

    const existing = extractExistingFrontmatter(content);
    if (!existing) {
      // No frontmatter - add minimal
      const contentWithoutFrontmatter = content;
      const title = generateTitle(filePath, content);
      const description = generateDescription(content);

      let yaml = '---\n';
      yaml += `title: "${title}"\n`;
      yaml += `description: "${description}"\n`;
      yaml += 'category: explanation\n';
      yaml += 'tags:\n  - documentation\n';
      yaml += 'updated-date: 2025-12-19\n';
      yaml += 'difficulty-level: intermediate\n';
      yaml += '---\n\n';

      fs.writeFileSync(fullPath, yaml + contentWithoutFrontmatter, 'utf8');
      return { success: true, file: filePath, action: 'added' };
    }

    // Has frontmatter - sanitize it
    const contentWithoutFrontmatter = content.replace(/^---\n[\s\S]*?\n---\n/, '');

    const sanitized = {
      title: existing.title || generateTitle(filePath, content),
      description: existing.description || generateDescription(contentWithoutFrontmatter),
      category: sanitizeCategory(existing.category || 'explanation'),
      tags: sanitizeTags(existing.tags || []),
      'updated-date': '2025-12-19',
      'difficulty-level': sanitizeDifficulty(existing['difficulty-level'] || 'intermediate')
    };

    let yaml = '---\n';
    yaml += `title: "${sanitized.title}"\n`;
    yaml += `description: "${sanitized.description}"\n`;
    yaml += `category: ${sanitized.category}\n`;
    yaml += 'tags:\n';
    for (const tag of sanitized.tags) {
      yaml += `  - ${tag}\n`;
    }
    yaml += `updated-date: ${sanitized['updated-date']}\n`;
    yaml += `difficulty-level: ${sanitized['difficulty-level']}\n`;
    yaml += '---\n\n';

    fs.writeFileSync(fullPath, yaml + contentWithoutFrontmatter, 'utf8');
    return { success: true, file: filePath, action: 'sanitized' };
  } catch (error) {
    return { success: false, file: filePath, error: error.message };
  }
}

function main() {
  const filesToProcess = [
    'working/DIAGRAM_INSPECTOR_REPORT.md',
    'working/ASCII_CONVERSION_BATCH_3_REPORT.md',
    'working/hive-content-audit.md',
    'working/UNIFIED_HIVE_REPORT.md',
    'working/DOCUMENTATION_ALIGNMENT_FINAL_REPORT.md',
    'working/hive-diagram-validation.md',
    'working/hive-spelling-audit.md',
    'working/hive-link-validation.md',
    'working/FINAL_QUALITY_SCORECARD.md',
    'working/hive-frontmatter-validation.md',
    'working/hive-coordination/HIVE_COORDINATION_PLAN.md',
    'working/hive-coordination/FINAL_ROYAL_DECREE.md',
    'working/hive-diataxis-validation.md',
    'working/validation-reports/WAVE_1_INTELLIGENCE_SUMMARY.md',
    'working/HIVE_QUALITY_REPORT.md',
    'working/hive-corpus-analysis.md',
    'LINK_REPAIR_REPORT.md',
    'scripts/AUTOMATION_COMPLETE.md'
  ];

  console.log(`Processing ${filesToProcess.length} remaining files...`);

  const results = [];
  let processed = 0;

  for (const file of filesToProcess) {
    const result = processFile(file);
    results.push(result);
    if (result.success) {
      processed++;
      console.log(`${result.action}: ${file}`);
    } else {
      console.error(`Error: ${file} - ${result.error}`);
    }
  }

  console.log(`\nComplete: ${processed}/${filesToProcess.length} files updated`);

  fs.writeFileSync(
    '/home/devuser/workspace/project/docs/working/remaining-frontmatter-results.json',
    JSON.stringify({ processed, total: filesToProcess.length, results }, null, 2)
  );
}

main();
