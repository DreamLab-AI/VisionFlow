#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Standard tag vocabulary
const STANDARD_TAGS = {
  architecture: ['architecture', 'design', 'patterns', 'structure', 'system-design'],
  api: ['api', 'rest', 'websocket', 'endpoints', 'http'],
  database: ['database', 'neo4j', 'schema', 'queries', 'cypher'],
  deployment: ['deployment', 'docker', 'kubernetes', 'devops', 'infrastructure'],
  testing: ['testing', 'jest', 'playwright', 'e2e', 'unit-tests'],
  client: ['client', 'react', 'three.js', 'xr', 'frontend'],
  server: ['server', 'actix', 'rust', 'backend', 'actors'],
  physics: ['physics', 'rapier', 'simulation', 'collision', 'forces'],
  ai: ['ai', 'agents', 'llm', 'claude', 'semantic'],
  gpu: ['gpu', 'wgpu', 'compute', 'shaders', 'performance'],
  guides: ['guide', 'tutorial', 'howto', 'setup', 'quickstart'],
  reference: ['reference', 'documentation', 'api-docs', 'specification'],
  migration: ['migration', 'upgrade', 'changelog', 'breaking-changes'],
  security: ['security', 'authentication', 'authorization', 'jwt', 'permissions'],
  ontology: ['ontology', 'knowledge-graph', 'semantic', 'rdf', 'owl']
};

// Category inference rules
const CATEGORY_RULES = {
  tutorial: [
    /getting-started/i,
    /quickstart/i,
    /tutorial/i,
    /step-by-step/i,
    /^guides\/.*tutorial/i
  ],
  howto: [
    /how-to/i,
    /guide/i,
    /setup/i,
    /configure/i,
    /^guides\//i
  ],
  reference: [
    /reference/i,
    /api/i,
    /specification/i,
    /schema/i,
    /protocol/i
  ],
  explanation: [
    /explanation/i,
    /concept/i,
    /architecture/i,
    /overview/i,
    /design/i
  ]
};

// Difficulty inference
const DIFFICULTY_RULES = {
  beginner: [/getting-started/i, /quickstart/i, /introduction/i, /basics/i],
  advanced: [/advanced/i, /optimization/i, /performance/i, /internals/i, /architecture/i],
  intermediate: [] // default
};

class FrontMatterGenerator {
  constructor(docsRoot) {
    this.docsRoot = docsRoot;
    this.allFiles = [];
    this.linkGraph = new Map(); // file -> [linked files]
    this.errors = [];
    this.warnings = [];
  }

  // Find all markdown files
  findAllMarkdownFiles() {
    const find = (dir) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          find(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.md')) {
          this.allFiles.push(fullPath);
        }
      }
    };
    find(this.docsRoot);
    console.log(`Found ${this.allFiles.length} markdown files`);
  }

  // Extract title from first H1 heading
  extractTitle(content, filePath) {
    const h1Match = content.match(/^#\s+(.+)$/m);
    if (h1Match) {
      return h1Match[1].trim();
    }
    // Fallback to filename
    const fileName = path.basename(filePath, '.md');
    return fileName
      .split(/[-_]/)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  // Extract description from content
  extractDescription(content, filePath) {
    // Remove front matter if exists
    content = content.replace(/^---\n[\s\S]*?\n---\n/, '');

    // Remove H1 heading
    content = content.replace(/^#\s+.+$/m, '');

    // Find first paragraph
    const paragraphs = content
      .split('\n\n')
      .map(p => p.trim())
      .filter(p => p && !p.startsWith('#') && !p.startsWith('```') && !p.startsWith('-') && !p.startsWith('*'));

    if (paragraphs.length > 0) {
      let desc = paragraphs[0].replace(/\n/g, ' ').trim();
      // Limit to 2 sentences
      const sentences = desc.match(/[^.!?]+[.!?]+/g) || [desc];
      desc = sentences.slice(0, 2).join(' ').trim();
      if (desc.length > 200) {
        desc = desc.substring(0, 197) + '...';
      }
      return desc;
    }

    return 'Documentation for ' + path.basename(filePath, '.md');
  }

  // Infer category based on path and content
  inferCategory(filePath, content) {
    const relativePath = path.relative(this.docsRoot, filePath);

    for (const [category, patterns] of Object.entries(CATEGORY_RULES)) {
      for (const pattern of patterns) {
        if (pattern.test(relativePath) || pattern.test(content.substring(0, 500))) {
          return category;
        }
      }
    }

    // Default based on directory structure
    if (relativePath.startsWith('tutorials/')) return 'tutorial';
    if (relativePath.startsWith('guides/')) return 'howto';
    if (relativePath.startsWith('reference/')) return 'reference';
    if (relativePath.startsWith('explanations/') || relativePath.startsWith('concepts/')) return 'explanation';

    return 'explanation'; // default
  }

  // Generate tags based on path and content
  generateTags(filePath, content) {
    const tags = new Set();
    const relativePath = path.relative(this.docsRoot, filePath).toLowerCase();
    const contentLower = content.toLowerCase();

    // Add tags based on path
    const pathParts = relativePath.split('/');
    for (const part of pathParts) {
      for (const [category, tagList] of Object.entries(STANDARD_TAGS)) {
        for (const tag of tagList) {
          if (part.includes(tag) || contentLower.includes(tag)) {
            tags.add(tag);
          }
        }
      }
    }

    // Ensure we have at least 3 tags
    if (tags.size < 3) {
      // Add category-based defaults
      if (relativePath.includes('api')) tags.add('api');
      if (relativePath.includes('guide')) tags.add('guide');
      if (relativePath.includes('architecture')) tags.add('architecture');
      if (relativePath.includes('server')) tags.add('server');
      if (relativePath.includes('client')) tags.add('client');
    }

    return Array.from(tags).slice(0, 5);
  }

  // Infer difficulty level
  inferDifficulty(filePath, content) {
    const combined = filePath + ' ' + content.substring(0, 500);

    for (const [level, patterns] of Object.entries(DIFFICULTY_RULES)) {
      for (const pattern of patterns) {
        if (pattern.test(combined)) {
          return level;
        }
      }
    }

    return 'intermediate';
  }

  // Extract dependencies from content
  extractDependencies(content) {
    const deps = new Set();

    // Look for "Prerequisites", "Requirements", etc.
    const prereqMatch = content.match(/(?:Prerequisites?|Requirements?|Dependencies):?\s*\n((?:[-*]\s+.+\n?)+)/i);
    if (prereqMatch) {
      const items = prereqMatch[1].match(/[-*]\s+(.+)/g) || [];
      items.forEach(item => {
        const cleaned = item.replace(/^[-*]\s+/, '').trim();
        if (cleaned) deps.add(cleaned);
      });
    }

    // Look for common dependencies
    if (content.includes('Docker')) deps.add('Docker installation');
    if (content.includes('Rust') && content.includes('cargo')) deps.add('Rust toolchain');
    if (content.includes('Node.js') || content.includes('npm')) deps.add('Node.js runtime');
    if (content.includes('Neo4j')) deps.add('Neo4j database');

    return Array.from(deps);
  }

  // Build link graph
  buildLinkGraph() {
    console.log('Building link graph...');

    for (const filePath of this.allFiles) {
      const content = fs.readFileSync(filePath, 'utf8');
      const links = new Set();

      // Find markdown links
      const linkMatches = content.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g);
      for (const match of linkMatches) {
        let linkPath = match[2];

        // Skip external links
        if (linkPath.startsWith('http://') || linkPath.startsWith('https://')) continue;

        // Remove anchors
        linkPath = linkPath.split('#')[0];

        // Skip empty links
        if (!linkPath || linkPath.startsWith('#')) continue;

        // Resolve relative path
        const dir = path.dirname(filePath);
        const resolvedPath = path.resolve(dir, linkPath);

        // Normalize to relative from docs root
        const relativePath = path.relative(this.docsRoot, resolvedPath);

        if (fs.existsSync(resolvedPath) && resolvedPath.endsWith('.md')) {
          links.add(relativePath);
        } else {
          this.warnings.push(`Broken link in ${path.relative(this.docsRoot, filePath)}: ${linkPath}`);
        }
      }

      this.linkGraph.set(filePath, Array.from(links));
    }
  }

  // Find related documents
  findRelatedDocs(filePath) {
    const related = new Set();
    const relativePath = path.relative(this.docsRoot, filePath);

    // Add documents linked from this file
    const outgoingLinks = this.linkGraph.get(filePath) || [];
    outgoingLinks.forEach(link => related.add(link));

    // Add documents linking to this file
    for (const [otherFile, links] of this.linkGraph.entries()) {
      const otherRelative = path.relative(this.docsRoot, otherFile);
      if (links.includes(relativePath)) {
        related.add(otherRelative);
      }
    }

    // Add documents in same directory
    const dir = path.dirname(filePath);
    const siblings = this.allFiles
      .filter(f => path.dirname(f) === dir && f !== filePath)
      .map(f => path.relative(this.docsRoot, f))
      .slice(0, 3);
    siblings.forEach(s => related.add(s));

    return Array.from(related).slice(0, 5);
  }

  // Generate front matter for a file
  generateFrontMatter(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');

    // Check if already has front matter
    if (content.startsWith('---\n')) {
      this.warnings.push(`File already has front matter: ${path.relative(this.docsRoot, filePath)}`);
      return null;
    }

    const title = this.extractTitle(content, filePath);
    const description = this.extractDescription(content, filePath);
    const category = this.inferCategory(filePath, content);
    const tags = this.generateTags(filePath, content);
    const difficulty = this.inferDifficulty(filePath, content);
    const dependencies = this.extractDependencies(content);
    const relatedDocs = this.findRelatedDocs(filePath);

    const frontMatter = {
      title,
      description,
      category,
      tags,
      'related-docs': relatedDocs,
      'updated-date': new Date().toISOString().split('T')[0],
      'difficulty-level': difficulty,
      dependencies: dependencies.length > 0 ? dependencies : undefined
    };

    // Remove undefined fields
    Object.keys(frontMatter).forEach(key => {
      if (frontMatter[key] === undefined) delete frontMatter[key];
    });

    return frontMatter;
  }

  // Format front matter as YAML
  formatFrontMatter(fm) {
    let yaml = '---\n';

    for (const [key, value] of Object.entries(fm)) {
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

  // Update file with front matter
  updateFile(filePath, dryRun = false) {
    try {
      const frontMatter = this.generateFrontMatter(filePath);
      if (!frontMatter) return false;

      const content = fs.readFileSync(filePath, 'utf8');
      const yaml = this.formatFrontMatter(frontMatter);
      const newContent = yaml + content;

      if (!dryRun) {
        fs.writeFileSync(filePath, newContent, 'utf8');
        console.log(`✓ Updated: ${path.relative(this.docsRoot, filePath)}`);
      } else {
        console.log(`[DRY RUN] Would update: ${path.relative(this.docsRoot, filePath)}`);
      }

      return true;
    } catch (error) {
      this.errors.push(`Error processing ${path.relative(this.docsRoot, filePath)}: ${error.message}`);
      return false;
    }
  }

  // Process all files
  processAll(dryRun = false) {
    console.log(`\n${dryRun ? '[DRY RUN] ' : ''}Processing ${this.allFiles.length} files...\n`);

    let updated = 0;
    let skipped = 0;

    for (const filePath of this.allFiles) {
      const result = this.updateFile(filePath, dryRun);
      if (result) {
        updated++;
      } else {
        skipped++;
      }
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Summary:`);
    console.log(`  Total files: ${this.allFiles.length}`);
    console.log(`  Updated: ${updated}`);
    console.log(`  Skipped: ${skipped}`);
    console.log(`  Errors: ${this.errors.length}`);
    console.log(`  Warnings: ${this.warnings.length}`);
    console.log(`${'='.repeat(60)}\n`);
  }

  // Generate validation report
  generateValidationReport() {
    const report = {
      timestamp: new Date().toISOString(),
      totalFiles: this.allFiles.length,
      errors: this.errors,
      warnings: this.warnings,
      linkGraph: {
        totalLinks: Array.from(this.linkGraph.values()).reduce((sum, links) => sum + links.length, 0),
        brokenLinks: this.warnings.filter(w => w.includes('Broken link')).length
      },
      categories: {},
      tags: {},
      difficultyLevels: {}
    };

    // Analyze all files
    for (const filePath of this.allFiles) {
      const fm = this.generateFrontMatter(filePath);
      if (!fm) continue;

      // Count categories
      report.categories[fm.category] = (report.categories[fm.category] || 0) + 1;

      // Count tags
      fm.tags.forEach(tag => {
        report.tags[tag] = (report.tags[tag] || 0) + 1;
      });

      // Count difficulty levels
      report.difficultyLevels[fm['difficulty-level']] =
        (report.difficultyLevels[fm['difficulty-level']] || 0) + 1;
    }

    return report;
  }

  // Save validation report
  saveValidationReport(outputPath) {
    const report = this.generateValidationReport();

    let md = `# Front Matter Validation Report\n\n`;
    md += `**Generated:** ${report.timestamp}\n\n`;
    md += `## Summary\n\n`;
    md += `- **Total Files:** ${report.totalFiles}\n`;
    md += `- **Total Links:** ${report.linkGraph.totalLinks}\n`;
    md += `- **Broken Links:** ${report.linkGraph.brokenLinks}\n`;
    md += `- **Errors:** ${report.errors.length}\n`;
    md += `- **Warnings:** ${report.warnings.length}\n\n`;

    md += `## Category Distribution\n\n`;
    md += `| Category | Count |\n`;
    md += `|----------|-------|\n`;
    Object.entries(report.categories)
      .sort((a, b) => b[1] - a[1])
      .forEach(([cat, count]) => {
        md += `| ${cat} | ${count} |\n`;
      });

    md += `\n## Tag Distribution\n\n`;
    md += `| Tag | Count |\n`;
    md += `|-----|-------|\n`;
    Object.entries(report.tags)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .forEach(([tag, count]) => {
        md += `| ${tag} | ${count} |\n`;
      });

    md += `\n## Difficulty Distribution\n\n`;
    md += `| Level | Count |\n`;
    md += `|-------|-------|\n`;
    Object.entries(report.difficultyLevels)
      .sort((a, b) => b[1] - a[1])
      .forEach(([level, count]) => {
        md += `| ${level} | ${count} |\n`;
      });

    if (report.errors.length > 0) {
      md += `\n## Errors\n\n`;
      report.errors.forEach(err => {
        md += `- ${err}\n`;
      });
    }

    if (report.warnings.length > 0) {
      md += `\n## Warnings\n\n`;
      report.warnings.slice(0, 50).forEach(warn => {
        md += `- ${warn}\n`;
      });
      if (report.warnings.length > 50) {
        md += `\n*... and ${report.warnings.length - 50} more warnings*\n`;
      }
    }

    md += `\n## Validation Rules\n\n`;
    md += `### Required Fields\n`;
    md += `- ✓ title (extracted from H1 or filename)\n`;
    md += `- ✓ description (1-2 sentences)\n`;
    md += `- ✓ category (tutorial|howto|reference|explanation)\n`;
    md += `- ✓ tags (3-5 standardized tags)\n`;
    md += `- ✓ related-docs (up to 5 related files)\n`;
    md += `- ✓ updated-date (YYYY-MM-DD format)\n`;
    md += `- ✓ difficulty-level (beginner|intermediate|advanced)\n\n`;
    md += `### Optional Fields\n`;
    md += `- dependencies (array of prerequisites)\n\n`;
    md += `### Tag Vocabulary\n`;
    md += `Standardized tags organized by category:\n\n`;
    Object.entries(STANDARD_TAGS).forEach(([category, tags]) => {
      md += `**${category}:** ${tags.join(', ')}\n\n`;
    });

    fs.writeFileSync(outputPath, md, 'utf8');
    console.log(`\nValidation report saved to: ${outputPath}`);
  }
}

// Main execution
const docsRoot = path.join(__dirname, '..', 'docs');
const generator = new FrontMatterGenerator(docsRoot);

// Parse command line args
const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const reportOnly = args.includes('--report-only');

generator.findAllMarkdownFiles();
generator.buildLinkGraph();

if (!reportOnly) {
  generator.processAll(dryRun);
}

// Always generate validation report
const reportPath = path.join(docsRoot, 'working', 'frontmatter-validation.md');
generator.saveValidationReport(reportPath);
