const fs = require('fs');

const data = JSON.parse(fs.readFileSync('/home/devuser/workspace/project/docs/working/hive-spelling-audit.json', 'utf8'));

let markdown = `# UK English Spelling Audit Report
*Generated: ${new Date().toISOString()}*

## Executive Summary

- **Total Files Scanned**: ${data.total_files}
- **Files with US Spellings**: ${data.files_with_violations.length}
- **Clean Files**: ${data.clean_files_count}
- **Total Violations**: ${data.total_violations}
- **UK Compliance**: ${data.uk_compliance_percentage}%

**Status**: ${parseFloat(data.uk_compliance_percentage) === 100 ? '✅ COMPLIANT' : '⚠️ NON-COMPLIANT'}

---

## Violations by Word

`;

// Sort by frequency
const wordStats = Object.entries(data.violations_by_word)
  .map(([word, violations]) => ({
    word,
    count: violations.length,
    uk_spelling: violations[0].uk_spelling
  }))
  .sort((a, b) => b.count - a.count);

markdown += '| US Spelling | UK Spelling | Occurrences |\n';
markdown += '|-------------|-------------|-------------|\n';
wordStats.forEach(({ word, uk_spelling, count }) => {
  markdown += `| ${word} | ${uk_spelling} | ${count} |\n`;
});

markdown += `\n---\n\n## Files Requiring Correction\n\n`;

// Group violations by file
const fileViolations = {};
data.violations.forEach(v => {
  if (!fileViolations[v.file]) {
    fileViolations[v.file] = [];
  }
  fileViolations[v.file].push(v);
});

const sortedFiles = Object.keys(fileViolations).sort();

sortedFiles.forEach(file => {
  const violations = fileViolations[file];
  markdown += `### ${file}\n\n`;
  markdown += `**${violations.length} violation(s)**\n\n`;

  violations.forEach(v => {
    markdown += `- Line ${v.line}: \`${v.us_spelling}\` → \`${v.uk_spelling}\`\n`;
    markdown += `  - Context: "${v.context}"\n`;
  });

  markdown += '\n';
});

markdown += `---\n\n## Automated Fix Commands\n\n`;
markdown += `Run these commands to fix all US spellings:\n\n`;
markdown += '```bash\n';

// Generate sed commands for each file
sortedFiles.forEach(file => {
  const violations = fileViolations[file];
  const uniqueReplacements = {};

  violations.forEach(v => {
    const usWord = v.us_spelling;
    const ukWord = v.uk_spelling;
    uniqueReplacements[usWord] = ukWord;
  });

  Object.entries(uniqueReplacements).forEach(([us, uk]) => {
    markdown += `sed -i 's/\\\\b${us}\\\\b/${uk}/g' "docs/${file}"\n`;
  });
});

markdown += '```\n\n';

markdown += `---\n\n## Verification\n\n`;
markdown += `After running the fix commands, verify compliance:\n\n`;
markdown += '```bash\n';
markdown += 'node docs/working/spelling-scanner.js\n';
markdown += '```\n\n';
markdown += `Target: **100% UK English compliance**\n`;

fs.writeFileSync('/home/devuser/workspace/project/docs/working/hive-spelling-audit.md', markdown);

console.log('Markdown report generated');
