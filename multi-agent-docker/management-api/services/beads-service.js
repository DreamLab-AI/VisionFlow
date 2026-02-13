/**
 * Beads Service - Wrapper around the `bd` CLI for structured task tracking.
 *
 * Provides user-scoped bead creation, claiming, closing, and sync operations.
 * Each bead is attributed to the requesting VisionFlow user via EntityRef.
 */

const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

class BeadsService {
  constructor(logger) {
    this.logger = logger;
    this.bdPath = this._findBd();
  }

  /**
   * Locate the bd binary.
   */
  _findBd() {
    const candidates = [
      '/usr/local/bin/bd',
      path.join(process.env.HOME || '/home/devuser', '.local/bin/bd'),
      'bd', // fallback to PATH
    ];
    for (const candidate of candidates) {
      try {
        if (candidate === 'bd' || fs.existsSync(candidate)) {
          return candidate;
        }
      } catch {
        // continue
      }
    }
    return 'bd';
  }

  /**
   * Execute a bd command in a given workspace directory.
   */
  _exec(args, cwd, env = {}) {
    return new Promise((resolve, reject) => {
      const opts = {
        cwd,
        env: { ...process.env, ...env },
        timeout: 30000,
      };

      execFile(this.bdPath, args, opts, (error, stdout, stderr) => {
        if (error) {
          this.logger.warn({ args, stderr: stderr?.trim(), code: error.code }, 'bd command failed');
          // Don't reject for non-zero exits — bd returns useful output anyway
          if (error.killed || error.signal) {
            return reject(new Error(`bd killed by ${error.signal || 'timeout'}`));
          }
        }
        resolve({ stdout: stdout?.trim() || '', stderr: stderr?.trim() || '' });
      });
    });
  }

  /**
   * Initialize beads in a workspace directory (if not already initialized).
   */
  async init(workspaceDir, prefix = 'vf') {
    const beadsDir = path.join(workspaceDir, '.beads');
    if (fs.existsSync(beadsDir)) {
      this.logger.debug({ workspaceDir }, 'Beads already initialized');
      return;
    }

    this.logger.info({ workspaceDir, prefix }, 'Initializing beads');
    await this._exec(['init', '--prefix', prefix, '--quiet'], workspaceDir);
  }

  /**
   * Create an epic bead for a top-level task or brief.
   *
   * @param {string} workspaceDir - The workspace directory
   * @param {string} title - Epic title
   * @param {object} userContext - The requesting user's context
   * @param {object} opts - Additional options (priority, labels, metadata)
   * @returns {object} The created bead (parsed JSON)
   */
  async createEpic(workspaceDir, title, userContext, opts = {}) {
    await this.init(workspaceDir);

    const args = [
      'create', title,
      '-p', String(opts.priority || 0),
      '--type', 'epic',
      '--json',
    ];

    if (userContext?.display_name) {
      args.push('--tag', `user:${userContext.display_name}`);
    }
    if (userContext?.pubkey) {
      args.push('--tag', `pubkey:${userContext.pubkey.substring(0, 8)}`);
    }
    if (opts.labels) {
      for (const label of opts.labels) {
        args.push('--label', label);
      }
    }

    const env = this._userEnv(userContext);
    const { stdout } = await this._exec(args, workspaceDir, env);
    return this._parseJson(stdout);
  }

  /**
   * Create a child bead under an epic (for swarm sub-tasks or role responses).
   */
  async createChild(workspaceDir, title, parentBeadId, userContext, opts = {}) {
    const args = [
      'create', title,
      '-p', String(opts.priority || 1),
      '--parent', parentBeadId,
      '--json',
    ];

    if (opts.role) {
      args.push('--tag', `role:${opts.role}`);
    }
    if (userContext?.display_name) {
      args.push('--tag', `user:${userContext.display_name}`);
    }

    const env = this._userEnv(userContext);
    const { stdout } = await this._exec(args, workspaceDir, env);
    return this._parseJson(stdout);
  }

  /**
   * Get ready (unblocked) work items.
   */
  async getReady(workspaceDir, opts = {}) {
    const args = ['ready', '--json'];
    if (opts.limit) args.push('--limit', String(opts.limit));
    if (opts.assignee) args.push('--assignee', opts.assignee);
    if (opts.unassigned) args.push('--unassigned');

    const { stdout } = await this._exec(args, workspaceDir);
    return this._parseJson(stdout);
  }

  /**
   * Claim a bead (atomic compare-and-swap assignment).
   */
  async claim(workspaceDir, beadId, actor) {
    const args = ['update', beadId, '--claim', '--json'];
    const { stdout } = await this._exec(args, workspaceDir, { BEADS_ACTOR: actor });
    return this._parseJson(stdout);
  }

  /**
   * Close a bead with a reason.
   */
  async close(workspaceDir, beadId, reason) {
    const args = ['close', beadId, '--reason', reason, '--json'];
    const { stdout } = await this._exec(args, workspaceDir);
    return this._parseJson(stdout);
  }

  /**
   * Add a dependency between beads.
   */
  async addDep(workspaceDir, childId, parentId, depType = 'blocks') {
    const args = ['dep', 'add', childId, parentId, '--type', depType];
    await this._exec(args, workspaceDir);
  }

  /**
   * Show details of a bead.
   */
  async show(workspaceDir, beadId) {
    const args = ['show', beadId, '--json'];
    const { stdout } = await this._exec(args, workspaceDir);
    return this._parseJson(stdout);
  }

  /**
   * Sync beads state (export JSONL, commit if in git repo).
   */
  async sync(workspaceDir) {
    const args = ['sync'];
    await this._exec(args, workspaceDir);
  }

  /**
   * Build environment variables for user attribution.
   */
  _userEnv(userContext) {
    if (!userContext) return {};
    return {
      BEADS_ACTOR: `visionflow/${userContext.display_name || userContext.user_id}`,
    };
  }

  /**
   * Safely parse JSON output from bd, falling back to raw string.
   */
  _parseJson(str) {
    if (!str) return null;
    try {
      return JSON.parse(str);
    } catch {
      // bd sometimes outputs non-JSON info lines before the JSON
      const jsonStart = str.indexOf('{');
      const jsonArrayStart = str.indexOf('[');
      const start = jsonStart >= 0 && (jsonArrayStart < 0 || jsonStart < jsonArrayStart)
        ? jsonStart : jsonArrayStart;
      if (start >= 0) {
        try {
          return JSON.parse(str.substring(start));
        } catch {
          // fall through
        }
      }
      return { raw: str };
    }
  }
}

module.exports = BeadsService;
