#!/usr/bin/env node

/**
 * Security Script: Block installation of compromised testing packages
 * Run this as a pre-install script to prevent malware infection
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const BLOCKED_PACKAGES = [
  '@testing-library/react',
  '@testing-library/jest-dom',
  '@testing-library/dom',
  '@testing-library/user-event',
  'vitest',
  '@vitest/ui',
  '@vitest/coverage-v8',
  '@vitest/coverage-c8',
  '@vitest/coverage-istanbul',
  'jest',
  '@jest/core',
  '@jest/test-result',
  'pretty-format', // Direct malware vector
];

function checkPackageJson() {
  const packagePath = path.join(process.cwd(), 'package.json');
  
  if (!fs.existsSync(packagePath)) {
    console.log('✅ No package.json found, skipping security check');
    return;
  }
  
  const packageContent = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
  const allDeps = {
    ...packageContent.dependencies || {},
    ...packageContent.devDependencies || {},
    ...packageContent.peerDependencies || {}
  };
  
  const foundBlocked = [];
  
  for (const [dep, version] of Object.entries(allDeps)) {
    if (BLOCKED_PACKAGES.includes(dep)) {
      foundBlocked.push(`${dep}@${version}`);
    }
  }
  
  if (foundBlocked.length > 0) {
    console.error('\n🚨 SECURITY ALERT: BLOCKED PACKAGES DETECTED 🚨');
    console.error('\nThe following packages contain MALWARE in their dependency chain:');
    foundBlocked.forEach(pkg => console.error(`  ❌ ${pkg}`));
    console.error('\nThese packages MUST NOT be installed due to an active supply chain attack.');
    console.error('See SECURITY_ALERT.md for more information.\n');
    process.exit(1);
  }
  
  console.log('✅ Security check passed - no compromised packages detected');
}

// Check on script run
checkPackageJson();

// Also export for use in other scripts
export { checkPackageJson, BLOCKED_PACKAGES };