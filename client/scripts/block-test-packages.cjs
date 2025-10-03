#!/usr/bin/env node

/**
 * Security script to block installation of testing packages
 * See SECURITY_ALERT.md for details
 */

const fs = require('fs');
const path = require('path');

const BLOCKED_PACKAGES = [
  '@vitest/ui',
  'vitest',
  '@testing-library/react',
  '@testing-library/jest-dom',
  'jest',
  'jest-environment-jsdom'
];

try {
  const packageJsonPath = path.join(process.cwd(), 'package.json');

  if (!fs.existsSync(packageJsonPath)) {
    console.log('No package.json found, skipping test package check');
    process.exit(0);
  }

  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const dependencies = {
    ...packageJson.dependencies,
    ...packageJson.devDependencies
  };

  const foundBlockedPackages = BLOCKED_PACKAGES.filter(pkg => dependencies[pkg]);

  if (foundBlockedPackages.length > 0) {
    console.error('\n❌ SECURITY ALERT: Blocked test packages detected!');
    console.error('The following packages are blocked due to supply chain security concerns:');
    foundBlockedPackages.forEach(pkg => console.error(`  - ${pkg}`));
    console.error('\nSee SECURITY_ALERT.md for details.');
    console.error('Testing is currently disabled for security reasons.\n');
    process.exit(1);
  }

  console.log('✓ No blocked test packages detected');
  process.exit(0);
} catch (error) {
  console.error('Error checking for blocked packages:', error.message);
  // Don't block installation on script errors
  process.exit(0);
}
