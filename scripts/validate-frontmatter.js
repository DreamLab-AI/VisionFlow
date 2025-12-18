#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const REQUIRED_FIELDS = ['title', 'description', 'category', 'tags', 'updated-date', 'difficulty-level'];
const VALID_CATEGORIES = ['tutorial', 'howto', 'reference', 'explanation'];
const VALID_DIFFICULTIES = ['beginner', 'intermediate', 'advanced'];

class FrontMatterValidator {
  constructor(docsRoot) {
    this.docsRoot = docsRoot;
    this.errors = [];
    this.warnings = [];
    this.stats = {
      totalFiles: 0,
      withFrontMatter: 0,
      withoutFrontMatter: 0,
      validFrontMatter: 0,
      invalidFrontMatter: 0
    };
  }

  // Parse YAML front matter
  parseFrontMatter(content) {
    if (!content.startsWith('---\n')) {
      return null;
    }

    const endMatch = content.indexOf('\n---\n', 4);
    if (endMatch === -1) {
      return null;
    }

    const yamlContent = content.substring(4, endMatch);
    const fm = {};

    // Simple YAML parser (handles our use case)
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

    return fm;
  }

  // Validate a single file
  validateFile(filePath) {
    this.stats.totalFiles++;

    const content = fs.readFileSync(filePath, 'utf8');
    const fm = this.parseFrontMatter(content);

    if (!fm) {
      this.stats.withoutFrontMatter++;
      this.errors.push({
        file: path.relative(this.docsRoot, filePath),
        error: 'Missing front matter'
      });
      return false;
    }

    this.stats.withFrontMatter++;

    let isValid = true;

    // Check required fields
    for (const field of REQUIRED_FIELDS) {
      if (!fm[field]) {
        this.errors.push({
          file: path.relative(this.docsRoot, filePath),
          error: `Missing required field: ${field}`
        });
        isValid = false;
      }
    }

    // Validate category
    if (fm.category && !VALID_CATEGORIES.includes(fm.category)) {
      this.warnings.push({
        file: path.relative(this.docsRoot, filePath),
        warning: `Invalid category: ${fm.category}. Must be one of: ${VALID_CATEGORIES.join(', ')}`
      });
      isValid = false;
    }

    // Validate difficulty
    if (fm['difficulty-level'] && !VALID_DIFFICULTIES.includes(fm['difficulty-level'])) {
      this.warnings.push({
        file: path.relative(this.docsRoot, filePath),
        warning: `Invalid difficulty: ${fm['difficulty-level']}. Must be one of: ${VALID_DIFFICULTIES.join(', ')}`
      });
      isValid = false;
    }

    // Validate tags
    if (fm.tags) {
      if (!Array.isArray(fm.tags)) {
        this.errors.push({
          file: path.relative(this.docsRoot, filePath),
          error: 'Tags must be an array'
        });
        isValid = false;
      } else if (fm.tags.length < 3) {
        this.warnings.push({
          file: path.relative(this.docsRoot, filePath),
          warning: `Only ${fm.tags.length} tags (recommended: 3-5)`
        });
      } else if (fm.tags.length > 5) {
        this.warnings.push({
          file: path.relative(this.docsRoot, filePath),
          warning: `Too many tags: ${fm.tags.length} (recommended: 3-5)`
        });
      }
    }

    // Validate related-docs
    if (fm['related-docs']) {
      if (!Array.isArray(fm['related-docs'])) {
        this.errors.push({
          file: path.relative(this.docsRoot, filePath),
          error: 'related-docs must be an array'
        });
        isValid = false;
      } else {
        // Check if referenced files exist
        for (const relatedPath of fm['related-docs']) {
          const fullPath = path.join(this.docsRoot, relatedPath);
          if (!fs.existsSync(fullPath)) {
            this.warnings.push({
              file: path.relative(this.docsRoot, filePath),
              warning: `Broken reference in related-docs: ${relatedPath}`
            });
          }
        }
      }
    }

    // Validate date format
    if (fm['updated-date'] && !fm['updated-date'].match(/^\d{4}-\d{2}-\d{2}$/)) {
      this.warnings.push({
        file: path.relative(this.docsRoot, filePath),
        warning: `Invalid date format: ${fm['updated-date']} (expected YYYY-MM-DD)`
      });
    }

    if (isValid) {
      this.stats.validFrontMatter++;
    } else {
      this.stats.invalidFrontMatter++;
    }

    return isValid;
  }

  // Validate all files
  validateAll() {
    const findMarkdown = (dir) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          findMarkdown(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.md')) {
          this.validateFile(fullPath);
        }
      }
    };

    findMarkdown(this.docsRoot);
  }

  // Generate report
  generateReport() {
    let report = `# Front Matter Validation Report\n\n`;
    report += `**Generated:** ${new Date().toISOString()}\n\n`;

    report += `## Summary\n\n`;
    report += `| Metric | Count |\n`;
    report += `|--------|-------|\n`;
    report += `| Total Files | ${this.stats.totalFiles} |\n`;
    report += `| With Front Matter | ${this.stats.withFrontMatter} |\n`;
    report += `| Without Front Matter | ${this.stats.withoutFrontMatter} |\n`;
    report += `| Valid Front Matter | ${this.stats.validFrontMatter} |\n`;
    report += `| Invalid Front Matter | ${this.stats.invalidFrontMatter} |\n`;
    report += `| Errors | ${this.errors.length} |\n`;
    report += `| Warnings | ${this.warnings.length} |\n\n`;

    const validPercent = ((this.stats.validFrontMatter / this.stats.totalFiles) * 100).toFixed(1);
    report += `**Coverage:** ${validPercent}% of files have valid front matter\n\n`;

    if (this.errors.length > 0) {
      report += `## Errors\n\n`;
      const errorsByFile = this.errors.reduce((acc, err) => {
        if (!acc[err.file]) acc[err.file] = [];
        acc[err.file].push(err.error);
        return acc;
      }, {});

      for (const [file, errors] of Object.entries(errorsByFile)) {
        report += `### ${file}\n\n`;
        errors.forEach(err => {
          report += `- ❌ ${err}\n`;
        });
        report += `\n`;
      }
    }

    if (this.warnings.length > 0) {
      report += `## Warnings\n\n`;
      const warningsByFile = this.warnings.reduce((acc, warn) => {
        if (!acc[warn.file]) acc[warn.file] = [];
        acc[warn.file].push(warn.warning);
        return acc;
      }, {});

      const files = Object.keys(warningsByFile).slice(0, 30);
      for (const file of files) {
        report += `### ${file}\n\n`;
        warningsByFile[file].forEach(warn => {
          report += `- ⚠️ ${warn}\n`;
        });
        report += `\n`;
      }

      if (Object.keys(warningsByFile).length > 30) {
        report += `*... and ${Object.keys(warningsByFile).length - 30} more files with warnings*\n\n`;
      }
    }

    report += `## Validation Rules\n\n`;
    report += `### Required Fields\n\n`;
    REQUIRED_FIELDS.forEach(field => {
      report += `- ✓ \`${field}\`\n`;
    });

    report += `\n### Valid Categories\n\n`;
    VALID_CATEGORIES.forEach(cat => {
      report += `- \`${cat}\`\n`;
    });

    report += `\n### Valid Difficulty Levels\n\n`;
    VALID_DIFFICULTIES.forEach(diff => {
      report += `- \`${diff}\`\n`;
    });

    report += `\n### Tag Guidelines\n\n`;
    report += `- Minimum: 3 tags\n`;
    report += `- Maximum: 5 tags\n`;
    report += `- Use standardized tag vocabulary\n\n`;

    return report;
  }

  // Print summary
  printSummary() {
    console.log('\n' + '='.repeat(60));
    console.log('Front Matter Validation Summary');
    console.log('='.repeat(60));
    console.log(`Total Files:         ${this.stats.totalFiles}`);
    console.log(`With Front Matter:   ${this.stats.withFrontMatter}`);
    console.log(`Without Front Matter: ${this.stats.withoutFrontMatter}`);
    console.log(`Valid:               ${this.stats.validFrontMatter}`);
    console.log(`Invalid:             ${this.stats.invalidFrontMatter}`);
    console.log(`Errors:              ${this.errors.length}`);
    console.log(`Warnings:            ${this.warnings.length}`);
    console.log('='.repeat(60));

    const validPercent = ((this.stats.validFrontMatter / this.stats.totalFiles) * 100).toFixed(1);
    console.log(`\nCoverage: ${validPercent}%`);

    if (this.stats.withoutFrontMatter > 0) {
      console.log(`\n⚠️  ${this.stats.withoutFrontMatter} files missing front matter`);
    }

    if (this.errors.length > 0) {
      console.log(`\n❌ ${this.errors.length} errors found`);
    }

    if (this.warnings.length > 0) {
      console.log(`\n⚠️  ${this.warnings.length} warnings found`);
    }

    console.log('');
  }
}

// Main execution
const docsRoot = path.join(__dirname, '..', 'docs');
const validator = new FrontMatterValidator(docsRoot);

validator.validateAll();
validator.printSummary();

const report = validator.generateReport();
const reportPath = path.join(docsRoot, 'working', 'frontmatter-validation.md');
fs.writeFileSync(reportPath, report, 'utf8');
console.log(`\nValidation report saved to: ${reportPath}\n`);

process.exit(validator.errors.length > 0 ? 1 : 0);
