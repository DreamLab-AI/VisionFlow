#!/usr/bin/env node

/**
 * AgentDB to JSS (JavaScript Solid Server) Migration Script
 *
 * Migrates agent memory from local SQLite database (.agentdb or data/agentdb.db)
 * to federated Solid pods with JSON-LD format matching agent-memory.jsonld schema.
 *
 * Features:
 * - Dry-run mode for validation before actual migration
 * - Rollback capability with backup manifests
 * - Progress tracking and resumable migrations
 * - Validation of migrated data
 *
 * Usage:
 *   node migrate-agentdb-to-jss.js [options]
 *
 * Options:
 *   --dry-run           Preview migration without making changes
 *   --rollback <file>   Rollback using specified manifest file
 *   --pod-url <url>     Target Solid pod URL (default: http://localhost:3000)
 *   --db-path <path>    Source database path (default: ./data/agentdb.db)
 *   --batch-size <n>    Records per batch (default: 100)
 *   --verbose           Show detailed progress
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Dynamic import for better-sqlite3
let Database = null;
try {
  const sqlite = await import('better-sqlite3');
  Database = sqlite.default;
} catch {
  // Will be handled at runtime
}

// Configuration
const DEFAULT_CONFIG = {
  podUrl: process.env.JSS_POD_URL || 'http://localhost:3000',
  dbPath: process.env.AGENTDB_PATH || path.join(__dirname, '..', 'data', 'agentdb.db'),
  batchSize: 100,
  dryRun: false,
  verbose: false,
  backupDir: path.join(__dirname, '..', 'data', 'migration-backups'),
  agentId: process.env.AGENT_ID || 'default-agent'
};

// Agent Memory JSON-LD Context (from agent-memory.jsonld schema)
const AGENT_MEMORY_CONTEXT = {
  "@version": 1.1,
  "@vocab": "https://visionflow.local/ontology/agent#",
  "@base": "https://visionflow.local/ontology/agent-memory/",
  "agent": "https://visionflow.local/ontology/agent#",
  "memory": "https://visionflow.local/ontology/memory#",
  "xsd": "http://www.w3.org/2001/XMLSchema#",
  "dcterms": "http://purl.org/dc/terms/",
  "schema": "http://schema.org/",
  "prov": "http://www.w3.org/ns/prov#"
};

// Memory type mappings from SQLite tables to JSON-LD types
const MEMORY_TYPE_MAP = {
  episodes: 'EpisodicMemory',
  skills: 'ProceduralMemory',
  facts: 'SemanticMemory',
  notes: 'WorkingMemory',
  consolidated_memories: 'ConsolidatedMemory'
};

// Pod path structure for different memory types
const POD_PATHS = {
  EpisodicMemory: '/agent-memory/episodic/',
  ProceduralMemory: '/agent-memory/procedural/',
  SemanticMemory: '/agent-memory/semantic/',
  WorkingMemory: '/agent-memory/working/',
  ConsolidatedMemory: '/agent-memory/consolidated/',
  SessionSummary: '/agent-memory/sessions/'
};

class MigrationError extends Error {
  constructor(message, code, details = {}) {
    super(message);
    this.name = 'MigrationError';
    this.code = code;
    this.details = details;
  }
}

class AgentDBMigrator {
  constructor(config = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.db = null;
    this.stats = {
      total: 0,
      migrated: 0,
      failed: 0,
      skipped: 0,
      startTime: null,
      endTime: null
    };
    this.manifest = {
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      source: this.config.dbPath,
      target: this.config.podUrl,
      dryRun: this.config.dryRun,
      records: []
    };
  }

  log(message, level = 'info') {
    const prefix = {
      info: '[INFO]',
      warn: '[WARN]',
      error: '[ERROR]',
      debug: '[DEBUG]'
    };
    if (level === 'debug' && !this.config.verbose) return;
    console.log(`${prefix[level] || '[LOG]'} ${message}`);
  }

  pathExists(p) {
    try {
      fs.accessSync(p);
      return true;
    } catch {
      return false;
    }
  }

  ensureDir(dir) {
    if (!this.pathExists(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  }

  writeJson(filePath, data) {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
  }

  async initialize() {
    // Check if better-sqlite3 is available
    if (!Database) {
      throw new MigrationError(
        'better-sqlite3 is not installed. Run: npm install better-sqlite3',
        'MISSING_DEPENDENCY'
      );
    }

    // Check database exists
    if (!this.pathExists(this.config.dbPath)) {
      // Try alternative paths
      const altPaths = [
        path.join(__dirname, '..', '.agentdb', 'memory.db'),
        path.join(process.cwd(), '.agentdb', 'memory.db'),
        path.join(process.cwd(), 'data', 'agentdb.db')
      ];

      for (const altPath of altPaths) {
        if (this.pathExists(altPath)) {
          this.config.dbPath = altPath;
          this.log(`Found database at: ${altPath}`);
          break;
        }
      }

      if (!this.pathExists(this.config.dbPath)) {
        throw new MigrationError(
          `Database not found at ${this.config.dbPath}`,
          'DB_NOT_FOUND',
          { searchedPaths: [this.config.dbPath, ...altPaths] }
        );
      }
    }

    // Open database
    this.db = new Database(this.config.dbPath, { readonly: true });
    this.log(`Opened database: ${this.config.dbPath}`);

    // Ensure backup directory exists
    this.ensureDir(this.config.backupDir);
  }

  async getTableSchema() {
    const tables = this.db.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).all();

    const schema = {};
    for (const { name } of tables) {
      const columns = this.db.prepare(`PRAGMA table_info(${name})`).all();
      const count = this.db.prepare(`SELECT COUNT(*) as count FROM ${name}`).get();
      schema[name] = {
        columns: columns.map(c => c.name),
        rowCount: count.count
      };
    }
    return schema;
  }

  convertEpisodeToJsonLd(episode) {
    const id = `urn:uuid:${crypto.randomUUID()}`;

    let input = null;
    let output = null;

    try {
      input = episode.input ? JSON.parse(episode.input) : null;
    } catch {
      input = episode.input;
    }

    try {
      output = episode.output ? JSON.parse(episode.output) : null;
    } catch {
      output = episode.output;
    }

    let tags = [];
    try {
      tags = episode.tags ? JSON.parse(episode.tags) : [];
    } catch {
      tags = episode.tags ? [episode.tags] : [];
    }

    return {
      "@context": AGENT_MEMORY_CONTEXT,
      "@type": "EpisodicMemory",
      "@id": id,
      "agentId": this.config.agentId,
      "sessionId": episode.session_id,
      "taskId": episode.task,
      "content": {
        input,
        output,
        critique: episode.critique
      },
      "confidence": episode.reward || 0.0,
      "timestamp": new Date((episode.ts || 0) * 1000).toISOString(),
      "modifiedAt": new Date((episode.created_at || 0) * 1000).toISOString(),
      "metadata": {
        success: Boolean(episode.success),
        latencyMs: episode.latency_ms,
        tokensUsed: episode.tokens_used,
        tags,
        originalId: episode.id,
        migratedFrom: 'agentdb-sqlite'
      },
      "version": "1.0.0",
      "createdBy": "migrate-agentdb-to-jss"
    };
  }

  convertSkillToJsonLd(skill) {
    const id = `urn:uuid:${crypto.randomUUID()}`;

    let signature = {};
    try {
      signature = skill.signature ? JSON.parse(skill.signature) : {};
    } catch {
      signature = { raw: skill.signature };
    }

    return {
      "@context": AGENT_MEMORY_CONTEXT,
      "@type": "ProceduralMemory",
      "@id": id,
      "agentId": this.config.agentId,
      "pattern": skill.name,
      "content": {
        description: skill.description,
        signature,
        code: skill.code
      },
      "successRate": skill.success_rate || 0.0,
      "confidence": skill.avg_reward || 0.0,
      "accessCount": skill.uses || 0,
      "timestamp": new Date((skill.created_at || 0) * 1000).toISOString(),
      "modifiedAt": new Date((skill.updated_at || 0) * 1000).toISOString(),
      "lastAccessed": skill.last_used_at ? new Date(skill.last_used_at * 1000).toISOString() : null,
      "metadata": {
        avgLatencyMs: skill.avg_latency_ms,
        createdFromEpisode: skill.created_from_episode,
        originalId: skill.id,
        migratedFrom: 'agentdb-sqlite'
      },
      "version": "1.0.0",
      "createdBy": "migrate-agentdb-to-jss"
    };
  }

  convertFactToJsonLd(fact) {
    const id = `urn:uuid:${crypto.randomUUID()}`;

    return {
      "@context": AGENT_MEMORY_CONTEXT,
      "@type": "SemanticMemory",
      "@id": id,
      "agentId": this.config.agentId,
      "domain": "facts",
      "content": {
        subject: fact.subject,
        predicate: fact.predicate,
        object: fact.object,
        sourceType: fact.source_type,
        sourceId: fact.source_id
      },
      "confidence": fact.confidence || 1.0,
      "timestamp": new Date((fact.created_at || 0) * 1000).toISOString(),
      "expiresAt": fact.expires_at ? new Date(fact.expires_at * 1000).toISOString() : null,
      "metadata": {
        originalId: fact.id,
        migratedFrom: 'agentdb-sqlite'
      },
      "version": "1.0.0",
      "createdBy": "migrate-agentdb-to-jss"
    };
  }

  convertNoteToJsonLd(note) {
    const id = `urn:uuid:${crypto.randomUUID()}`;

    return {
      "@context": AGENT_MEMORY_CONTEXT,
      "@type": "WorkingMemory",
      "@id": id,
      "agentId": this.config.agentId,
      "content": {
        title: note.title,
        text: note.text,
        summary: note.summary,
        noteType: note.note_type
      },
      "importance": note.importance || 0.5,
      "accessCount": note.access_count || 0,
      "lastAccessed": note.last_accessed_at ? new Date(note.last_accessed_at * 1000).toISOString() : null,
      "timestamp": new Date((note.created_at || 0) * 1000).toISOString(),
      "modifiedAt": new Date((note.updated_at || 0) * 1000).toISOString(),
      "metadata": {
        originalId: note.id,
        migratedFrom: 'agentdb-sqlite'
      },
      "consolidationStatus": "working",
      "version": "1.0.0",
      "createdBy": "migrate-agentdb-to-jss"
    };
  }

  async uploadToSolidPod(jsonLd, memoryType) {
    const podPath = POD_PATHS[memoryType] || '/agent-memory/other/';
    const resourceId = jsonLd['@id'].replace('urn:uuid:', '');
    const url = `${this.config.podUrl}${podPath}${resourceId}.jsonld`;

    if (this.config.dryRun) {
      this.log(`[DRY-RUN] Would upload to: ${url}`, 'debug');
      return { success: true, url, dryRun: true };
    }

    try {
      const response = await fetch(url, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/ld+json',
          'Accept': 'application/ld+json'
        },
        body: JSON.stringify(jsonLd, null, 2)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return { success: true, url };
    } catch (error) {
      return { success: false, url, error: error.message };
    }
  }

  async migrateTable(tableName, converter, memoryType) {
    this.log(`\nMigrating table: ${tableName} -> ${memoryType}`);

    const count = this.db.prepare(`SELECT COUNT(*) as count FROM ${tableName}`).get();
    this.log(`Found ${count.count} records`);

    if (count.count === 0) {
      this.log(`No records to migrate in ${tableName}`, 'debug');
      return { migrated: 0, failed: 0 };
    }

    let offset = 0;
    let migrated = 0;
    let failed = 0;

    while (offset < count.count) {
      const rows = this.db.prepare(
        `SELECT * FROM ${tableName} LIMIT ${this.config.batchSize} OFFSET ${offset}`
      ).all();

      for (const row of rows) {
        try {
          const jsonLd = converter.call(this, row);
          const result = await this.uploadToSolidPod(jsonLd, memoryType);

          if (result.success) {
            migrated++;
            this.manifest.records.push({
              sourceTable: tableName,
              sourceId: row.id,
              targetUrl: result.url,
              type: memoryType,
              status: 'success'
            });
          } else {
            failed++;
            this.manifest.records.push({
              sourceTable: tableName,
              sourceId: row.id,
              type: memoryType,
              status: 'failed',
              error: result.error
            });
            this.log(`Failed to migrate ${tableName}:${row.id}: ${result.error}`, 'error');
          }
        } catch (error) {
          failed++;
          this.log(`Error converting ${tableName}:${row.id}: ${error.message}`, 'error');
        }

        // Progress indicator
        if ((migrated + failed) % 10 === 0) {
          process.stdout.write(`\r  Progress: ${migrated + failed}/${count.count}`);
        }
      }

      offset += this.config.batchSize;
    }

    console.log(); // New line after progress
    this.log(`Completed ${tableName}: ${migrated} migrated, ${failed} failed`);

    return { migrated, failed };
  }

  async runMigration() {
    this.stats.startTime = Date.now();

    try {
      await this.initialize();

      // Get schema info
      const schema = await this.getTableSchema();
      this.log('\nDatabase Schema:');
      for (const [table, info] of Object.entries(schema)) {
        this.log(`  ${table}: ${info.rowCount} rows`);
        this.stats.total += info.rowCount;
      }

      if (this.config.dryRun) {
        this.log('\n=== DRY RUN MODE - No changes will be made ===\n', 'warn');
      }

      // Create container directories on pod
      if (!this.config.dryRun) {
        for (const podPath of Object.values(POD_PATHS)) {
          try {
            await fetch(`${this.config.podUrl}${podPath}`, {
              method: 'PUT',
              headers: { 'Content-Type': 'text/turtle' }
            });
          } catch (e) {
            this.log(`Could not create container ${podPath}: ${e.message}`, 'debug');
          }
        }
      }

      // Migrate each table
      const results = {};

      if (schema.episodes?.rowCount > 0) {
        results.episodes = await this.migrateTable(
          'episodes',
          this.convertEpisodeToJsonLd,
          'EpisodicMemory'
        );
      }

      if (schema.skills?.rowCount > 0) {
        results.skills = await this.migrateTable(
          'skills',
          this.convertSkillToJsonLd,
          'ProceduralMemory'
        );
      }

      if (schema.facts?.rowCount > 0) {
        results.facts = await this.migrateTable(
          'facts',
          this.convertFactToJsonLd,
          'SemanticMemory'
        );
      }

      if (schema.notes?.rowCount > 0) {
        results.notes = await this.migrateTable(
          'notes',
          this.convertNoteToJsonLd,
          'WorkingMemory'
        );
      }

      // Calculate totals
      for (const result of Object.values(results)) {
        this.stats.migrated += result.migrated || 0;
        this.stats.failed += result.failed || 0;
      }

      this.stats.endTime = Date.now();

      // Save manifest
      await this.saveManifest();

      return this.stats;
    } finally {
      if (this.db) {
        this.db.close();
      }
    }
  }

  async saveManifest() {
    this.manifest.stats = this.stats;
    this.manifest.completedAt = new Date().toISOString();

    const manifestPath = path.join(
      this.config.backupDir,
      `migration-${this.manifest.id}.json`
    );

    this.writeJson(manifestPath, this.manifest);
    this.log(`\nManifest saved: ${manifestPath}`);

    return manifestPath;
  }

  async validateMigration() {
    this.log('\n=== Validating Migration ===\n');

    let validated = 0;
    let errors = 0;

    for (const record of this.manifest.records) {
      if (record.status !== 'success') continue;

      try {
        const response = await fetch(record.targetUrl, {
          headers: { 'Accept': 'application/ld+json' }
        });

        if (response.ok) {
          const data = await response.json();
          if (data['@type'] === record.type) {
            validated++;
          } else {
            errors++;
            this.log(`Type mismatch for ${record.targetUrl}`, 'warn');
          }
        } else {
          errors++;
          this.log(`Cannot fetch ${record.targetUrl}: ${response.status}`, 'warn');
        }
      } catch (e) {
        errors++;
        this.log(`Validation error for ${record.targetUrl}: ${e.message}`, 'error');
      }

      if ((validated + errors) % 10 === 0) {
        process.stdout.write(`\r  Validated: ${validated}/${this.manifest.records.filter(r => r.status === 'success').length}`);
      }
    }

    console.log();
    this.log(`Validation complete: ${validated} valid, ${errors} errors`);

    return { validated, errors };
  }

  async rollback(manifestPath) {
    this.log(`\n=== Rolling Back Migration ===`);
    this.log(`Using manifest: ${manifestPath}`);

    if (!this.pathExists(manifestPath)) {
      throw new MigrationError('Manifest file not found', 'MANIFEST_NOT_FOUND');
    }

    const manifest = this.readJson(manifestPath);
    let deleted = 0;
    let errors = 0;

    for (const record of manifest.records) {
      if (record.status !== 'success') continue;

      try {
        const response = await fetch(record.targetUrl, {
          method: 'DELETE'
        });

        if (response.ok || response.status === 404) {
          deleted++;
        } else {
          errors++;
          this.log(`Failed to delete ${record.targetUrl}: ${response.status}`, 'warn');
        }
      } catch (e) {
        errors++;
        this.log(`Error deleting ${record.targetUrl}: ${e.message}`, 'error');
      }

      if ((deleted + errors) % 10 === 0) {
        process.stdout.write(`\r  Deleted: ${deleted}/${manifest.records.filter(r => r.status === 'success').length}`);
      }
    }

    console.log();
    this.log(`Rollback complete: ${deleted} deleted, ${errors} errors`);

    // Mark manifest as rolled back
    manifest.rolledBack = true;
    manifest.rollbackAt = new Date().toISOString();
    this.writeJson(manifestPath, manifest);

    return { deleted, errors };
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  const config = { ...DEFAULT_CONFIG };

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--dry-run':
        config.dryRun = true;
        break;
      case '--verbose':
      case '-v':
        config.verbose = true;
        break;
      case '--pod-url':
        config.podUrl = args[++i];
        break;
      case '--db-path':
        config.dbPath = args[++i];
        break;
      case '--batch-size':
        config.batchSize = parseInt(args[++i], 10);
        break;
      case '--rollback':
        config.rollbackManifest = args[++i];
        break;
      case '--validate':
        config.validateOnly = args[++i];
        break;
      case '--help':
      case '-h':
        console.log(`
AgentDB to JSS Migration Script

Migrates agent memory from SQLite to Solid pods with JSON-LD format.

Usage:
  node migrate-agentdb-to-jss.js [options]

Options:
  --dry-run             Preview migration without making changes
  --rollback <file>     Rollback using specified manifest file
  --validate <file>     Validate migration using manifest file
  --pod-url <url>       Target Solid pod URL (default: http://localhost:3000)
  --db-path <path>      Source database path (default: ./data/agentdb.db)
  --batch-size <n>      Records per batch (default: 100)
  --verbose, -v         Show detailed progress
  --help, -h            Show this help message

Environment Variables:
  JSS_POD_URL           Default pod URL
  AGENTDB_PATH          Default database path
  AGENT_ID              Agent identifier for migrated records

Examples:
  # Preview migration
  node migrate-agentdb-to-jss.js --dry-run

  # Migrate to local Solid server
  node migrate-agentdb-to-jss.js --pod-url http://localhost:3000

  # Rollback a migration
  node migrate-agentdb-to-jss.js --rollback ./data/migration-backups/migration-xxx.json
`);
        process.exit(0);
    }
  }

  const migrator = new AgentDBMigrator(config);

  try {
    if (config.rollbackManifest) {
      const result = await migrator.rollback(config.rollbackManifest);
      console.log('\nRollback Summary:');
      console.log(`  Deleted: ${result.deleted}`);
      console.log(`  Errors: ${result.errors}`);
    } else if (config.validateOnly) {
      await migrator.initialize();
      migrator.manifest = migrator.readJson(config.validateOnly);
      const result = await migrator.validateMigration();
      console.log('\nValidation Summary:');
      console.log(`  Valid: ${result.validated}`);
      console.log(`  Errors: ${result.errors}`);
    } else {
      const stats = await migrator.runMigration();

      console.log('\n=== Migration Summary ===');
      console.log(`  Mode: ${config.dryRun ? 'DRY RUN' : 'LIVE'}`);
      console.log(`  Source: ${config.dbPath}`);
      console.log(`  Target: ${config.podUrl}`);
      console.log(`  Total Records: ${stats.total}`);
      console.log(`  Migrated: ${stats.migrated}`);
      console.log(`  Failed: ${stats.failed}`);
      console.log(`  Duration: ${((stats.endTime - stats.startTime) / 1000).toFixed(2)}s`);

      if (!config.dryRun && stats.migrated > 0) {
        console.log('\nTo validate the migration:');
        console.log(`  node migrate-agentdb-to-jss.js --validate ${path.join(config.backupDir, `migration-${migrator.manifest.id}.json`)}`);
        console.log('\nTo rollback:');
        console.log(`  node migrate-agentdb-to-jss.js --rollback ${path.join(config.backupDir, `migration-${migrator.manifest.id}.json`)}`);
      }
    }
  } catch (error) {
    console.error(`\nMigration failed: ${error.message}`);
    if (config.verbose && error.details) {
      console.error('Details:', JSON.stringify(error.details, null, 2));
    }
    process.exit(1);
  }
}

// Run main
main();

export { AgentDBMigrator, AGENT_MEMORY_CONTEXT, POD_PATHS, MEMORY_TYPE_MAP };
