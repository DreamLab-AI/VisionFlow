#!/usr/bin/env node

/**
 * Environment Variable Migration Script
 * Helps migrate existing .env files to remove exposed secrets
 * and generate secure replacements
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Sensitive keys that should be replaced
const SENSITIVE_KEYS = [
  'GITHUB_TOKEN',
  'RAGFLOW_API_KEY',
  'PERPLEXITY_API_KEY',
  'OPENAI_API_KEY',
  'CLOUDFLARE_TUNNEL_TOKEN',
  'JWT_SECRET',
  'SESSION_SECRET',
  'WS_AUTH_TOKEN'
];

// Keys that need to be added for security
const REQUIRED_SECURITY_KEYS = {
  'JWT_SECRET': () => generateSecret(64),
  'SESSION_SECRET': () => generateSecret(64),
  'WS_AUTH_TOKEN': () => generateSecret(32),
  'WS_AUTH_ENABLED': 'true',
  'WS_MAX_CONNECTIONS': '100',
  'WS_CONNECTION_TIMEOUT': '300000',
  'RATE_LIMIT_WINDOW_MS': '60000',
  'RATE_LIMIT_MAX_REQUESTS': '100',
  'MAX_REQUEST_SIZE': '10485760',
  'SANITIZE_HTML': 'true',
  'TCP_MAX_CONNECTIONS': '50',
  'TCP_CONNECTION_TIMEOUT': '300000'
};

function generateSecret(length = 32) {
  return crypto.randomBytes(length).toString('hex');
}

function parseEnvFile(content) {
  const lines = content.split('\n');
  const env = {};
  const comments = [];
  
  lines.forEach((line, index) => {
    const trimmed = line.trim();
    
    // Skip empty lines
    if (!trimmed) {
      comments.push({ index, content: '' });
      return;
    }
    
    // Handle comments
    if (trimmed.startsWith('#')) {
      comments.push({ index, content: line });
      return;
    }
    
    // Parse key=value pairs
    const match = line.match(/^([^=]+)=(.*)$/);
    if (match) {
      const key = match[1].trim();
      const value = match[2].trim();
      env[key] = { value, index, originalLine: line };
    }
  });
  
  return { env, comments, totalLines: lines.length };
}

function maskSecret(value) {
  if (!value || value.length < 8) return value;
  const visibleChars = 4;
  return value.substring(0, visibleChars) + '*'.repeat(value.length - visibleChars);
}

async function promptUser(question) {
  return new Promise(resolve => {
    rl.question(question, resolve);
  });
}

async function migrateEnvFile(envPath) {
  console.log('🔒 Environment Variable Security Migration Tool\n');
  
  // Check if .env exists
  if (!fs.existsSync(envPath)) {
    console.error(`❌ Error: ${envPath} not found`);
    process.exit(1);
  }
  
  // Read and parse .env file
  const content = fs.readFileSync(envPath, 'utf8');
  const { env, comments } = parseEnvFile(content);
  
  // Create backup
  const backupPath = `${envPath}.backup.${Date.now()}`;
  fs.copyFileSync(envPath, backupPath);
  console.log(`✅ Created backup: ${backupPath}\n`);
  
  // Check for exposed secrets
  const exposedSecrets = [];
  const secureEnv = {};
  
  for (const [key, data] of Object.entries(env)) {
    if (SENSITIVE_KEYS.includes(key) && data.value && !data.value.includes('xxx')) {
      exposedSecrets.push({ key, value: maskSecret(data.value) });
      
      // Ask user if they want to keep the value
      const keep = await promptUser(
        `\n⚠️  Found exposed ${key}: ${maskSecret(data.value)}\n` +
        `Do you want to:\n` +
        `1. Replace with placeholder (recommended for repository)\n` +
        `2. Keep current value (only for local development)\n` +
        `Enter choice (1 or 2): `
      );
      
      if (keep === '1') {
        secureEnv[key] = generatePlaceholder(key);
      } else {
        secureEnv[key] = data.value;
      }
    } else {
      secureEnv[key] = data.value;
    }
  }
  
  // Add missing security keys
  console.log('\n📋 Adding security configuration...');
  for (const [key, defaultValue] of Object.entries(REQUIRED_SECURITY_KEYS)) {
    if (!secureEnv[key]) {
      const value = typeof defaultValue === 'function' ? defaultValue() : defaultValue;
      secureEnv[key] = value;
      console.log(`  ✅ Added ${key}`);
    }
  }
  
  // Generate new .env content
  let newContent = generateEnvContent(secureEnv, comments, content);
  
  // Write updated .env
  const outputPath = await promptUser(
    `\n💾 Where to save the migrated file?\n` +
    `1. Overwrite ${envPath}\n` +
    `2. Save as ${envPath}.secure\n` +
    `Enter choice (1 or 2): `
  );
  
  const finalPath = outputPath === '1' ? envPath : `${envPath}.secure`;
  fs.writeFileSync(finalPath, newContent);
  
  console.log(`\n✅ Migration complete! Saved to: ${finalPath}`);
  
  // Create .env.local with actual values if user chose placeholders
  if (exposedSecrets.length > 0 && outputPath === '1') {
    const createLocal = await promptUser(
      `\n📝 Create .env.local with your actual values? (y/n): `
    );
    
    if (createLocal.toLowerCase() === 'y') {
      const localEnv = { ...env };
      const localContent = generateEnvContent(
        Object.fromEntries(
          Object.entries(localEnv).map(([k, v]) => [k, v.value])
        ),
        comments,
        content
      );
      fs.writeFileSync(`${envPath}.local`, localContent);
      console.log(`✅ Created ${envPath}.local with actual values`);
      console.log(`⚠️  Remember to add .env.local to .gitignore!`);
    }
  }
  
  // Summary
  console.log('\n📊 Migration Summary:');
  console.log(`  - Exposed secrets found: ${exposedSecrets.length}`);
  console.log(`  - Security keys added: ${Object.keys(REQUIRED_SECURITY_KEYS).filter(k => !env[k]).length}`);
  console.log(`  - Backup saved to: ${backupPath}`);
  
  if (exposedSecrets.length > 0) {
    console.log('\n⚠️  Security Recommendations:');
    console.log('  1. Rotate all exposed API keys and tokens');
    console.log('  2. Use environment-specific secret management');
    console.log('  3. Never commit .env files with real secrets');
    console.log('  4. Review the SECURITY.md documentation');
  }
  
  rl.close();
}

function generatePlaceholder(key) {
  const placeholders = {
    'GITHUB_TOKEN': 'ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'RAGFLOW_API_KEY': 'ragflow-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'PERPLEXITY_API_KEY': 'pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'OPENAI_API_KEY': 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'CLOUDFLARE_TUNNEL_TOKEN': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
  };
  
  return placeholders[key] || 'your-' + key.toLowerCase().replace(/_/g, '-') + '-here';
}

function generateEnvContent(env, comments, originalContent) {
  const lines = originalContent.split('\n');
  const newLines = [];
  const processedKeys = new Set();
  
  // First pass: update existing lines
  lines.forEach((line, index) => {
    const trimmed = line.trim();
    
    // Keep empty lines and comments
    if (!trimmed || trimmed.startsWith('#')) {
      newLines.push(line);
      return;
    }
    
    // Check if this line contains a key=value pair
    const match = line.match(/^([^=]+)=(.*)$/);
    if (match) {
      const key = match[1].trim();
      if (env[key] !== undefined) {
        newLines.push(`${key}=${env[key]}`);
        processedKeys.add(key);
      } else {
        newLines.push(line);
      }
    } else {
      newLines.push(line);
    }
  });
  
  // Second pass: add new keys that weren't in the original file
  const newKeys = Object.keys(env).filter(key => !processedKeys.has(key));
  if (newKeys.length > 0) {
    newLines.push('');
    newLines.push('# ===========================================');
    newLines.push('# Security Configuration (Auto-generated)');
    newLines.push('# ===========================================');
    
    newKeys.forEach(key => {
      newLines.push(`${key}=${env[key]}`);
    });
  }
  
  return newLines.join('\n');
}

// Main execution
const envPath = process.argv[2] || path.join(process.cwd(), '.env');

migrateEnvFile(envPath).catch(error => {
  console.error('❌ Migration failed:', error.message);
  process.exit(1);
});