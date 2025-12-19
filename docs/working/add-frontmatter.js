#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const DOCS_DIR = '/home/devuser/workspace/project/docs';
const REPORT_PATH = '/home/devuser/workspace/project/docs/working/hive-frontmatter-validation.json';

const STANDARD_TAGS = [
  'architecture', 'api', 'authentication', 'agents', 'automation', 'backend',
  'ci-cd', 'configuration', 'database', 'debugging', 'deployment', 'development',
  'docker', 'documentation', 'features', 'frontend', 'getting-started', 'gpu',
  'infrastructure', 'installation', 'integration', 'kubernetes', 'learning',
  'mcp', 'memory', 'migration', 'monitoring', 'neural', 'optimization', 'performance',
  'plugins', 'reference', 'security', 'setup', 'swarm', 'testing', 'tools', 'tutorial',
  'ui', 'validation', 'workflow'
];

function sanitizeTag(tag) {
  const mapping = {
    'patterns': 'architecture',
    'structure': 'architecture',
    'rest': 'api',
    'contribution': 'documentation',
    'standards': 'documentation',
    'design': 'architecture',
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
    'websocket': 'api',
    'protocol': 'api',
    'migration-plan': 'migration',
    'ontology': 'architecture',
    'visualization': 'ui',
    'gpu-compute': 'gpu',
    'physics': 'features',
    'semantic': 'features',
    'graph': 'architecture',
    'neo4j': 'database',
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
    'completion': 'documentation'
  };

  return mapping[tag.toLowerCase()] || null;
}

function sanitizeTags(tags) {
  const sanitized = new Set();
  for (const tag of tags) {
    const cleaned = sanitizeTag(tag);
    if (cleaned && STANDARD_TAGS.includes(cleaned)) {
      sanitized.add(cleaned);
    }
  }
  return Array.from(sanitized);
}

function detectCategory(filePath, content) {
  const lower = content.toLowerCase();
  const pathLower = filePath.toLowerCase();

  if (pathLower.includes('/tutorial') || lower.includes('step-by-step') || lower.includes('walkthrough')) {
    return 'tutorial';
  }
  if (pathLower.includes('/guides/') || lower.includes('how to') || lower.includes('guide')) {
    return 'guide';
  }
  if (pathLower.includes('/reference/') || pathLower.includes('/api/') || lower.includes('reference')) {
    return 'reference';
  }
  return 'explanation';
}

function detectTags(filePath, content) {
  const tags = new Set();
  const lower = content.toLowerCase();
  const pathLower = filePath.toLowerCase();

  // Path-based tags
  if (pathLower.includes('architecture')) tags.add('architecture');
  if (pathLower.includes('api')) tags.add('api');
  if (pathLower.includes('gpu')) tags.add('gpu');
  if (pathLower.includes('tutorial')) tags.add('tutorial');
  if (pathLower.includes('guide')) tags.add('tutorial');
  if (pathLower.includes('ontology')) tags.add('architecture');
  if (pathLower.includes('database')) tags.add('database');
  if (pathLower.includes('migration')) tags.add('migration');
  if (pathLower.includes('docker')) tags.add('docker');
  if (pathLower.includes('deployment')) tags.add('deployment');
  if (pathLower.includes('testing')) tags.add('testing');
  if (pathLower.includes('setup')) tags.add('setup');
  if (pathLower.includes('installation')) tags.add('installation');
  if (pathLower.includes('getting-started')) tags.add('getting-started');
  if (pathLower.includes('reference')) tags.add('reference');
  if (pathLower.includes('multi-agent')) tags.add('agents');
  if (pathLower.includes('swarm')) tags.add('swarm');
  if (pathLower.includes('neural')) tags.add('neural');
  if (pathLower.includes('memory')) tags.add('memory');
  if (pathLower.includes('performance')) tags.add('performance');
  if (pathLower.includes('optimization')) tags.add('optimization');
  if (pathLower.includes('security')) tags.add('security');
  if (pathLower.includes('authentication')) tags.add('authentication');
  if (pathLower.includes('configuration')) tags.add('configuration');
  if (pathLower.includes('workflow')) tags.add('workflow');
  if (pathLower.includes('ci-cd')) tags.add('ci-cd');
  if (pathLower.includes('monitoring')) tags.add('monitoring');
  if (pathLower.includes('debugging')) tags.add('debugging');
  if (pathLower.includes('archive')) tags.add('documentation');
  if (pathLower.includes('mcp')) tags.add('mcp');

  // Content-based tags
  if (lower.includes('neural') || lower.includes('machine learning')) tags.add('neural');
  if (lower.includes('gpu') || lower.includes('cuda')) tags.add('gpu');
  if (lower.includes('docker') || lower.includes('container')) tags.add('docker');
  if (lower.includes('kubernetes')) tags.add('kubernetes');
  if (lower.includes('api') || lower.includes('endpoint')) tags.add('api');
  if (lower.includes('database') || lower.includes('neo4j')) tags.add('database');
  if (lower.includes('authentication') || lower.includes('auth')) tags.add('authentication');
  if (lower.includes('swarm') || lower.includes('multi-agent')) tags.add('swarm');
  if (lower.includes('performance') || lower.includes('optimization')) tags.add('performance');
  if (lower.includes('testing') || lower.includes('test')) tags.add('testing');
  if (lower.includes('deployment') || lower.includes('deploy')) tags.add('deployment');
  if (lower.includes('migration')) tags.add('migration');
  if (lower.includes('installation') || lower.includes('install')) tags.add('installation');
  if (lower.includes('configuration') || lower.includes('config')) tags.add('configuration');
  if (lower.includes('visualization')) tags.add('ui');
  if (lower.includes('frontend') || lower.includes('client')) tags.add('frontend');
  if (lower.includes('backend') || lower.includes('server')) tags.add('backend');
  if (lower.includes('workflow')) tags.add('workflow');
  if (lower.includes('ci/cd') || lower.includes('pipeline')) tags.add('ci-cd');
  if (lower.includes('monitoring') || lower.includes('metrics')) tags.add('monitoring');
  if (lower.includes('debugging') || lower.includes('troubleshoot')) tags.add('debugging');
  if (lower.includes('security')) tags.add('security');

  // Ensure at least one tag
  if (tags.size === 0) {
    tags.add('documentation');
  }

  return Array.from(tags).slice(0, 5);
}

function detectDifficulty(content) {
  const lower = content.toLowerCase();

  if (lower.includes('beginner') || lower.includes('getting started') || lower.includes('quick start')) {
    return 'beginner';
  }
  if (lower.includes('advanced') || lower.includes('expert') || lower.includes('optimization')) {
    return 'advanced';
  }
  return 'intermediate';
}

function generateTitle(filePath, content) {
  // Try to find first heading
  const headingMatch = content.match(/^#\s+(.+)$/m);
  if (headingMatch) {
    return headingMatch[1].trim();
  }

  // Generate from filename
  const filename = path.basename(filePath, '.md');
  return filename
    .replace(/-/g, ' ')
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function generateDescription(content) {
  // Remove frontmatter if exists
  const withoutFrontmatter = content.replace(/^---\n[\s\S]*?\n---\n/, '');

  // Remove headings
  const withoutHeadings = withoutFrontmatter.replace(/^#+\s+.+$/gm, '');

  // Find first substantial paragraph
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
      const [key, value] = line.split(':').map(s => s.trim());
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

function createFrontmatter(filePath, content) {
  const existing = extractExistingFrontmatter(content);
  const contentWithoutFrontmatter = content.replace(/^---\n[\s\S]*?\n---\n/, '');

  const frontmatter = {
    title: existing?.title || generateTitle(filePath, content),
    description: existing?.description || generateDescription(contentWithoutFrontmatter),
    category: existing?.category || detectCategory(filePath, contentWithoutFrontmatter),
    tags: existing?.tags ? sanitizeTags(existing.tags) : detectTags(filePath, contentWithoutFrontmatter),
    'updated-date': '2025-12-19',
    'difficulty-level': existing?.['difficulty-level'] || detectDifficulty(contentWithoutFrontmatter)
  };

  // Ensure tags are valid
  if (frontmatter.tags.length === 0) {
    frontmatter.tags = ['documentation'];
  }

  let yaml = '---\n';
  yaml += `title: "${frontmatter.title}"\n`;
  yaml += `description: "${frontmatter.description}"\n`;
  yaml += `category: ${frontmatter.category}\n`;
  yaml += 'tags:\n';
  for (const tag of frontmatter.tags) {
    yaml += `  - ${tag}\n`;
  }
  yaml += `updated-date: ${frontmatter['updated-date']}\n`;
  yaml += `difficulty-level: ${frontmatter['difficulty-level']}\n`;
  yaml += '---\n\n';

  return yaml + contentWithoutFrontmatter;
}

function processFile(filePath) {
  try {
    const fullPath = path.join(DOCS_DIR, filePath);
    const content = fs.readFileSync(fullPath, 'utf8');
    const updated = createFrontmatter(filePath, content);
    fs.writeFileSync(fullPath, updated, 'utf8');
    return { success: true, file: filePath };
  } catch (error) {
    return { success: false, file: filePath, error: error.message };
  }
}

function main() {
  const report = JSON.parse(fs.readFileSync(REPORT_PATH, 'utf8'));

  const filesToProcess = new Set();

  // Add files missing frontmatter
  for (const file of report.files_missing_frontmatter) {
    filesToProcess.add(file);
  }

  // Add files with missing required fields
  for (const item of report.missing_required_fields) {
    filesToProcess.add(item.file);
  }

  // Add files with invalid tags
  for (const item of report.invalid_tags) {
    filesToProcess.add(item.file);
  }

  console.log(`Processing ${filesToProcess.size} files...`);

  const results = [];
  let processed = 0;

  for (const file of filesToProcess) {
    const result = processFile(file);
    results.push(result);
    if (result.success) {
      processed++;
      if (processed % 10 === 0) {
        console.log(`Processed ${processed}/${filesToProcess.size} files...`);
      }
    } else {
      console.error(`Error processing ${file}: ${result.error}`);
    }
  }

  console.log(`\nComplete: ${processed}/${filesToProcess.size} files updated successfully`);

  // Write results
  fs.writeFileSync(
    '/home/devuser/workspace/project/docs/working/frontmatter-update-results.json',
    JSON.stringify({ processed, total: filesToProcess.size, results }, null, 2)
  );
}

main();
