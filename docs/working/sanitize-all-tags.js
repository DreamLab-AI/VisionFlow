#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

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

const TAG_MAPPING = {
  'rust': 'backend', 'rest': 'api', 'websocket': 'api', 'visionflow': 'documentation',
  'neo4j': 'database', 'guide': 'tutorial', 'react': 'frontend', 'design': 'architecture',
  'patterns': 'architecture', 'structure': 'architecture', 'http': 'api', 'ai': 'agents',
  'ontology': 'architecture', 'visualization': 'ui', 'gpu-compute': 'gpu', 'semantic': 'features',
  'graph': 'architecture', 'client': 'frontend', 'server': 'backend', 'cqrs': 'architecture',
  'pipeline': 'workflow', 'services': 'architecture', 'ports': 'architecture', 'handlers': 'api',
  'analytics': 'monitoring', 'rendering': 'ui', 'webgl': 'ui', 'github': 'integration',
  'multi-agent': 'swarm', 'llm': 'agents', 'embedding': 'neural', 'vector': 'neural',
  'model': 'agents', 'system': 'architecture'
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

function getAllMarkdownFiles() {
  const output = execSync(`find "${DOCS_DIR}" -name "*.md" -type f`, { encoding: 'utf8' });
  return output.trim().split('\n').filter(f => f).map(f => f.replace(DOCS_DIR + '/', ''));
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
    if (!existing || !existing.tags) {
      return { success: false, file: filePath, reason: 'no-frontmatter' };
    }

    const originalTags = existing.tags;
    const sanitized = sanitizeTags(originalTags);

    // Check if any tag was invalid
    const hadInvalidTags = originalTags.some(tag => !STANDARD_TAGS.includes(tag.toLowerCase()));

    if (!hadInvalidTags) {
      return { success: false, file: filePath, reason: 'already-valid' };
    }

    const contentWithoutFrontmatter = content.replace(/^---\n[\s\S]*?\n---\n/, '');

    let yaml = '---\n';
    yaml += `title: "${existing.title}"\n`;
    yaml += `description: "${existing.description}"\n`;
    yaml += `category: ${existing.category}\n`;
    yaml += 'tags:\n';
    for (const tag of sanitized) {
      yaml += `  - ${tag}\n`;
    }
    yaml += `updated-date: ${existing['updated-date']}\n`;
    yaml += `difficulty-level: ${existing['difficulty-level']}\n`;
    yaml += '---\n\n';

    fs.writeFileSync(fullPath, yaml + contentWithoutFrontmatter, 'utf8');
    return { success: true, file: filePath, before: originalTags, after: sanitized };
  } catch (error) {
    return { success: false, file: filePath, error: error.message };
  }
}

function main() {
  console.log('Scanning all markdown files for invalid tags...');

  const allFiles = getAllMarkdownFiles();
  console.log(`Found ${allFiles.length} markdown files`);

  const results = [];
  let sanitized = 0;

  for (const file of allFiles) {
    const result = processFile(file);
    if (result.success) {
      sanitized++;
      if (sanitized % 10 === 0) {
        console.log(`Sanitized ${sanitized} files...`);
      }
    }
    if (result.success || result.error) {
      results.push(result);
    }
  }

  console.log(`\nComplete: ${sanitized} files had tags sanitized`);

  fs.writeFileSync(
    '/home/devuser/workspace/project/docs/working/tag-sanitization-results.json',
    JSON.stringify({ sanitized, total: allFiles.length, results }, null, 2)
  );
}

main();
