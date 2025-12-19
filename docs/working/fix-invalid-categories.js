#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const DOCS_DIR = '/home/devuser/workspace/project/docs';

const filesToFix = [
  'archive/docs/guides/working-with-gui-sandbox.md',
  'archive/docs/guides/xr-setup.md',
  'archive/docs/guides/developer/05-testing-guide.md',
  'archive/docs/guides/user/working-with-agents.md',
  'archive/reports/consolidation/link-fix-suggestions-2025-12.md',
  'tutorials/advanced/06-extending-system.md'
];

function processFile(filePath) {
  try {
    const fullPath = path.join(DOCS_DIR, filePath);
    const content = fs.readFileSync(fullPath, 'utf8');

    // Replace howto with guide
    const updated = content.replace(/^category:\s*howto$/m, 'category: guide');

    if (updated !== content) {
      fs.writeFileSync(fullPath, updated, 'utf8');
      return { success: true, file: filePath };
    }

    return { success: false, file: filePath, reason: 'no-change' };
  } catch (error) {
    return { success: false, file: filePath, error: error.message };
  }
}

function main() {
  console.log(`Fixing ${filesToFix.length} files with invalid categories...`);

  let fixed = 0;
  for (const file of filesToFix) {
    const result = processFile(file);
    if (result.success) {
      fixed++;
      console.log(`Fixed: ${file}`);
    } else if (result.error) {
      console.error(`Error: ${file} - ${result.error}`);
    }
  }

  console.log(`\nComplete: ${fixed}/${filesToFix.length} files fixed`);
}

main();
