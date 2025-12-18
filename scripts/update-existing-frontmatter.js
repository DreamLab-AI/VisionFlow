#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Migration mappings for old fields
const FIELD_MIGRATIONS = {
  'type': 'category',
  'status': null // Remove status field
};

// Valid category values
const VALID_CATEGORIES = ['tutorial', 'howto', 'reference', 'explanation'];

class FrontMatterUpdater {
  constructor(docsRoot) {
    this.docsRoot = docsRoot;
    this.errors = [];
    this.stats = {
      total: 0,
      updated: 0,
      skipped: 0,
      errors: 0
    };
  }

  // Parse YAML front matter
  parseFrontMatter(content) {
    if (!content.startsWith('---\n')) {
      return { frontMatter: null, body: content };
    }

    const endMatch = content.indexOf('\n---\n', 4);
    if (endMatch === -1) {
      return { frontMatter: null, body: content };
    }

    const yamlContent = content.substring(4, endMatch);
    const body = content.substring(endMatch + 5);
    const fm = {};

    // Simple YAML parser
    let currentKey = null;
    let currentArray = null;

    const lines = yamlContent.split('\n');
    for (const line of lines) {
      if (line.trim() === '') continue;

      // Array item
      if (line.match(/^\s+-\s+(.+)$/)) {
        const value = line.match(/^\s+-\s+(.+)$/)[1];
        if (currentArray) {
          currentArray.push(value);
        }
        continue;
      }

      // Key-value pair
      const kvMatch = line.match(/^([a-z-]+):\s*(.*)$/);
      if (kvMatch) {
        currentKey = kvMatch[1];
        const value = kvMatch[2];

        if (value === '') {
          // Start of array
          currentArray = [];
          fm[currentKey] = currentArray;
        } else {
          // Direct value
          fm[currentKey] = value;
          currentArray = null;
        }
      }
    }

    return { frontMatter: fm, body };
  }

  // Normalize category value
  normalizeCategory(value) {
    if (!value) return null;

    const lower = value.toLowerCase();

    // Map common aliases
    const mappings = {
      'guide': 'howto',
      'guides': 'howto',
      'how-to': 'howto',
      'tutorial': 'tutorial',
      'tutorials': 'tutorial',
      'reference': 'reference',
      'documentation': 'reference',
      'explanation': 'explanation',
      'concept': 'explanation',
      'overview': 'explanation'
    };

    if (mappings[lower]) {
      return mappings[lower];
    }

    // Default to explanation
    return 'explanation';
  }

  // Generate missing fields based on existing data and file path
  generateMissingFields(existingFm, filePath, body) {
    const relativePath = path.relative(this.docsRoot, filePath);
    const newFields = {};

    // Category
    if (!existingFm.category) {
      if (existingFm.type) {
        newFields.category = this.normalizeCategory(existingFm.type);
      } else {
        // Infer from path
        if (relativePath.startsWith('tutorials/')) newFields.category = 'tutorial';
        else if (relativePath.startsWith('guides/')) newFields.category = 'howto';
        else if (relativePath.startsWith('reference/')) newFields.category = 'reference';
        else newFields.category = 'explanation';
      }
    }

    // Tags
    if (!existingFm.tags || !Array.isArray(existingFm.tags) || existingFm.tags.length === 0) {
      newFields.tags = this.generateTags(relativePath, body);
    }

    // Updated date
    if (!existingFm['updated-date']) {
      newFields['updated-date'] = new Date().toISOString().split('T')[0];
    }

    // Difficulty level
    if (!existingFm['difficulty-level']) {
      newFields['difficulty-level'] = this.inferDifficulty(relativePath, body);
    }

    // Title (keep existing if present)
    if (!existingFm.title) {
      newFields.title = this.extractTitle(body, filePath);
    }

    // Description (keep existing if present)
    if (!existingFm.description) {
      newFields.description = this.extractDescription(body, filePath);
    }

    return newFields;
  }

  extractTitle(content, filePath) {
    const h1Match = content.match(/^#\s+(.+)$/m);
    if (h1Match) {
      return h1Match[1].trim();
    }
    const fileName = path.basename(filePath, '.md');
    return fileName
      .split(/[-_]/)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  extractDescription(content, filePath) {
    content = content.replace(/^#\s+.+$/m, '');
    const paragraphs = content
      .split('\n\n')
      .map(p => p.trim())
      .filter(p => p && !p.startsWith('#') && !p.startsWith('```') && !p.startsWith('-') && !p.startsWith('*'));

    if (paragraphs.length > 0) {
      let desc = paragraphs[0].replace(/\n/g, ' ').trim();
      const sentences = desc.match(/[^.!?]+[.!?]+/g) || [desc];
      desc = sentences.slice(0, 2).join(' ').trim();
      if (desc.length > 200) {
        desc = desc.substring(0, 197) + '...';
      }
      return desc;
    }

    return 'Documentation for ' + path.basename(filePath, '.md');
  }

  generateTags(relativePath, content) {
    const tags = new Set();
    const contentLower = content.toLowerCase();

    // Add based on path
    if (relativePath.includes('api')) tags.add('api');
    if (relativePath.includes('architecture')) tags.add('architecture');
    if (relativePath.includes('guide')) tags.add('guide');
    if (relativePath.includes('server')) tags.add('server');
    if (relativePath.includes('client')) tags.add('client');
    if (relativePath.includes('database')) tags.add('database');
    if (relativePath.includes('deployment')) tags.add('deployment');
    if (relativePath.includes('testing')) tags.add('testing');

    // Add based on content
    if (contentLower.includes('rest')) tags.add('rest');
    if (contentLower.includes('websocket')) tags.add('websocket');
    if (contentLower.includes('docker')) tags.add('docker');
    if (contentLower.includes('neo4j')) tags.add('neo4j');
    if (contentLower.includes('rust')) tags.add('rust');
    if (contentLower.includes('react')) tags.add('react');

    // Ensure at least 3 tags
    if (tags.size < 3) {
      tags.add('documentation');
      tags.add('reference');
      tags.add('visionflow');
    }

    return Array.from(tags).slice(0, 5);
  }

  inferDifficulty(relativePath, content) {
    const combined = relativePath + ' ' + content.substring(0, 500);

    if (/getting-started|quickstart|introduction|basics|beginner/i.test(combined)) {
      return 'beginner';
    }

    if (/advanced|optimization|performance|internals|architecture/i.test(combined)) {
      return 'advanced';
    }

    return 'intermediate';
  }

  // Format front matter as YAML
  formatFrontMatter(fm) {
    let yaml = '---\n';

    // Order: title, description, category, tags, related-docs, updated-date, difficulty-level, dependencies
    const order = ['title', 'description', 'category', 'tags', 'related-docs', 'updated-date', 'difficulty-level', 'dependencies'];

    for (const key of order) {
      if (fm[key] !== undefined) {
        if (Array.isArray(fm[key])) {
          if (fm[key].length === 0) continue;
          yaml += `${key}:\n`;
          fm[key].forEach(item => {
            yaml += `  - ${item}\n`;
          });
        } else {
          yaml += `${key}: ${fm[key]}\n`;
        }
      }
    }

    // Add any other fields not in the order
    for (const [key, value] of Object.entries(fm)) {
      if (order.includes(key)) continue;

      if (Array.isArray(value)) {
        if (value.length === 0) continue;
        yaml += `${key}:\n`;
        value.forEach(item => {
          yaml += `  - ${item}\n`;
        });
      } else {
        yaml += `${key}: ${value}\n`;
      }
    }

    yaml += '---\n\n';
    return yaml;
  }

  // Update a single file
  updateFile(filePath, dryRun = false) {
    try {
      this.stats.total++;

      const content = fs.readFileSync(filePath, 'utf8');
      const { frontMatter, body } = this.parseFrontMatter(content);

      if (!frontMatter) {
        this.stats.skipped++;
        return false;
      }

      // Generate missing fields
      const newFields = this.generateMissingFields(frontMatter, filePath, body);

      // Check if any updates needed
      if (Object.keys(newFields).length === 0) {
        this.stats.skipped++;
        return false;
      }

      // Merge fields
      const updatedFm = { ...frontMatter, ...newFields };

      // Remove deprecated fields
      delete updatedFm.type;
      delete updatedFm.status;

      // Format and write
      const yaml = this.formatFrontMatter(updatedFm);
      const newContent = yaml + body;

      if (!dryRun) {
        fs.writeFileSync(filePath, newContent, 'utf8');
        console.log(`✓ Updated: ${path.relative(this.docsRoot, filePath)}`);
      } else {
        console.log(`[DRY RUN] Would update: ${path.relative(this.docsRoot, filePath)}`);
      }

      this.stats.updated++;
      return true;
    } catch (error) {
      this.errors.push({
        file: path.relative(this.docsRoot, filePath),
        error: error.message
      });
      this.stats.errors++;
      return false;
    }
  }

  // Process all files
  processAll(dryRun = false) {
    const findMarkdown = (dir) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          findMarkdown(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.md')) {
          this.updateFile(fullPath, dryRun);
        }
      }
    };

    findMarkdown(this.docsRoot);

    console.log('\n' + '='.repeat(60));
    console.log('Summary:');
    console.log(`  Total files: ${this.stats.total}`);
    console.log(`  Updated: ${this.stats.updated}`);
    console.log(`  Skipped: ${this.stats.skipped}`);
    console.log(`  Errors: ${this.stats.errors}`);
    console.log('='.repeat(60) + '\n');
  }
}

// Main execution
const docsRoot = path.join(__dirname, '..', 'docs');
const updater = new FrontMatterUpdater(docsRoot);

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');

updater.processAll(dryRun);

if (updater.errors.length > 0) {
  console.log('\nErrors:');
  updater.errors.forEach(err => {
    console.log(`  ${err.file}: ${err.error}`);
  });
}
