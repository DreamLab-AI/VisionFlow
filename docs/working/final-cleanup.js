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

const TAG_MAP = {
  'websocket': 'api', 'neo4j': 'database', 'rust': 'backend', 'rest': 'api',
  'react': 'frontend', 'structure': 'architecture'
};

function sanitizeTags(tags) {
  const result = new Set();
  for (const tag of tags) {
    if (STANDARD_TAGS.includes(tag)) {
      result.add(tag);
    } else if (TAG_MAP[tag]) {
      result.add(TAG_MAP[tag]);
    }
  }
  if (result.size === 0) result.add('documentation');
  return Array.from(result).slice(0, 5);
}

function extractFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n/);
  if (!match) return null;

  const fm = {};
  const yaml = match[1];
  const lines = yaml.split('\n');
  let currentKey = null;
  let currentArray = [];

  for (const line of lines) {
    if (line.match(/^[a-z-]+:/)) {
      const idx = line.indexOf(':');
      const key = line.substring(0, idx).trim();
      const value = line.substring(idx + 1).trim();
      currentKey = key;
      if (value) {
        fm[key] = value.replace(/^["']|["']$/g, '');
      } else {
        currentArray = [];
        fm[key] = currentArray;
      }
    } else if (line.trim().startsWith('-') && Array.isArray(fm[currentKey])) {
      fm[currentKey].push(line.trim().substring(1).trim());
    }
  }

  return fm;
}

function writeFrontmatter(fm, content) {
  const sanitized = {
    title: fm.title || 'Documentation',
    description: fm.description || 'Documentation file',
    category: fm.category || 'explanation',
    tags: fm.tags ? sanitizeTags(fm.tags) : ['documentation'],
    'updated-date': '2025-12-19',
    'difficulty-level': fm['difficulty-level'] || 'intermediate'
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

  const body = content.replace(/^---\n[\s\S]*?\n---\n/, '');
  return yaml + body;
}

const filesToFix = [
  'CONTRIBUTION.md',
  'MAINTENANCE.md',
  'VALIDATION_CHECKLIST.md',
  'NAVIGATION.md',
  'INDEX.md',
  '01-GETTING_STARTED.md',
  'reference/README.md',
  'reference/implementation-status.md',
  'reference/API_REFERENCE.md',
  'reference/PROTOCOL_REFERENCE.md',
  'reference/code-quality-status.md',
  'reference/performance-benchmarks.md',
  'reference/INDEX.md',
  'archive/README.md',
  'archive/deprecated-patterns/README.md',
  'working/spelling-remediation-wave-1-complete.md'
];

let fixed = 0;
for (const file of filesToFix) {
  try {
    const fullPath = path.join(DOCS_DIR, file);
    const content = fs.readFileSync(fullPath, 'utf8');

    // Check if has frontmatter
    if (!content.startsWith('---\n')) {
      // Add minimal frontmatter
      const title = file.split('/').pop().replace('.md', '').replace(/-/g, ' ').replace(/_/g, ' ');
      const yaml = `---
title: "${title.charAt(0).toUpperCase() + title.slice(1)}"
description: "Documentation file"
category: explanation
tags:
  - documentation
updated-date: 2025-12-19
difficulty-level: intermediate
---

`;
      fs.writeFileSync(fullPath, yaml + content, 'utf8');
      console.log(`Added frontmatter: ${file}`);
      fixed++;
      continue;
    }

    // Has frontmatter - fix it
    const fm = extractFrontmatter(content);
    if (fm) {
      const updated = writeFrontmatter(fm, content);
      fs.writeFileSync(fullPath, updated, 'utf8');
      console.log(`Fixed frontmatter: ${file}`);
      fixed++;
    }
  } catch (error) {
    console.error(`Error: ${file} - ${error.message}`);
  }
}

console.log(`\nFixed ${fixed}/${filesToFix.length} files`);
