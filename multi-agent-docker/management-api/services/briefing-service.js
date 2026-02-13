/**
 * Briefing Service — Orchestrates the brief → execute → debrief workflow.
 *
 * Implements the symmetrical briefing pattern:
 * - Briefs flow down: human → team (stored in team/humans/{name}/briefs/)
 * - Debriefs flow up: team → human (stored in team/humans/{name}/debriefs/)
 * - Role responses: team/roles/{role}/reviews/{date}/
 */

const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

class BriefingService {
  constructor(logger, processManager, beadsService) {
    this.logger = logger;
    this.processManager = processManager;
    this.beadsService = beadsService;
    this.teamRoot = path.join(
      process.env.WORKSPACE || path.join(process.env.HOME || '/home/devuser', 'workspace'),
      'team'
    );
  }

  /**
   * Create a brief file and its tracking bead.
   *
   * @param {object} userContext - The requesting user
   * @param {string} content - Markdown brief content
   * @param {object} opts - { version, briefType, slug, roles }
   * @returns {object} { briefId, briefPath, beadId }
   */
  async createBrief(userContext, content, opts = {}) {
    const briefId = uuidv4().substring(0, 8);
    const now = new Date();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const displayName = userContext.display_name || 'anonymous';
    const version = opts.version || 'v0.1.0';
    const briefType = opts.briefType || 'brief';
    const seq = opts.seq || 1;
    const slug = opts.slug || `brief-${briefId}`;

    // Build file path: team/humans/{name}/briefs/{MM}/{DD}/
    const briefDir = path.join(
      this.teamRoot, 'humans', displayName, 'briefs', month, day
    );
    fs.mkdirSync(briefDir, { recursive: true });

    const fileName = `${version}__${briefType}__${seq}__${slug}.md`;
    const briefPath = path.join(briefDir, fileName);
    const relativePath = path.relative(this.teamRoot, briefPath);

    // Write the brief
    const header = [
      `# ${briefType.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}`,
      '',
      `**Author**: ${displayName}`,
      `**Date**: ${now.toISOString().split('T')[0]}`,
      `**Version**: ${version}`,
      `**Brief ID**: ${briefId}`,
      opts.roles ? `**Roles**: ${opts.roles.join(', ')}` : '',
      '',
      '---',
      '',
    ].filter(Boolean).join('\n');

    fs.writeFileSync(briefPath, header + content, 'utf-8');
    this.logger.info({ briefId, briefPath: relativePath, user: displayName }, 'Brief created');

    // Create epic bead for this brief
    let beadId = null;
    if (this.beadsService) {
      try {
        const epic = await this.beadsService.createEpic(
          this.teamRoot,
          `Brief: ${slug} (${displayName})`,
          userContext,
          { labels: ['brief', briefType] }
        );
        beadId = epic?.id || null;
      } catch (err) {
        this.logger.warn({ error: err.message }, 'Beads epic creation failed (non-fatal)');
      }
    }

    return { briefId, briefPath: relativePath, beadId, fullPath: briefPath };
  }

  /**
   * Execute a brief by spawning role-specific agents.
   *
   * @param {string} briefPath - Path to the brief file
   * @param {string[]} roles - Role names to activate
   * @param {object} userContext - Requesting user
   * @param {string} [epicBeadId] - Optional epic bead to parent child beads under
   * @returns {object[]} Array of { role, taskId, beadId, responsePath }
   */
  async executeBrief(briefPath, roles, userContext, epicBeadId = null) {
    const displayName = userContext.display_name || 'anonymous';
    const now = new Date();
    const dateStr = `${String(now.getFullYear()).slice(2)}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;

    const results = [];

    for (const role of roles) {
      // Create role response directory
      const roleDir = path.join(this.teamRoot, 'roles', role, 'reviews', dateStr);
      fs.mkdirSync(roleDir, { recursive: true });

      const responseFile = path.basename(briefPath).replace('__brief__', `__response-from-${role}__`);
      const responsePath = path.join(roleDir, responseFile);
      const relativeResponsePath = path.relative(this.teamRoot, responsePath);
      const relativeBriefPath = path.relative(roleDir, path.dirname(briefPath) === '.' ? briefPath : path.join(this.teamRoot, briefPath));

      // Build role-specific task prompt
      const taskPrompt = [
        `You are the ${role.toUpperCase()} role on this team.`,
        `Read the brief at: ${briefPath}`,
        `Respond from your ${role} perspective. Focus on:`,
        this._roleGuidance(role),
        '',
        `Write your response to: ${responsePath}`,
        `Include a header with your role, the date, and a reference to the brief.`,
        `Be thorough but concise. Flag risks, dependencies, and recommendations.`,
      ].join('\n');

      // Create child bead for this role
      let childBeadId = null;
      if (epicBeadId && this.beadsService) {
        try {
          const child = await this.beadsService.createChild(
            this.teamRoot,
            `${role} response to brief`,
            epicBeadId,
            userContext,
            { role, priority: 1 }
          );
          childBeadId = child?.id || null;
        } catch (err) {
          this.logger.warn({ role, error: err.message }, 'Child bead creation failed');
        }
      }

      // Spawn agent for this role
      const processInfo = this.processManager.spawnTask(
        role, taskPrompt, 'claude-flow', userContext,
        { withBeads: !!epicBeadId, parentBeadId: epicBeadId, beadId: childBeadId }
      );

      results.push({
        role,
        taskId: processInfo.taskId,
        beadId: childBeadId,
        responsePath: relativeResponsePath,
      });
    }

    this.logger.info({
      roles,
      user: displayName,
      taskCount: results.length,
      epicBeadId,
    }, 'Brief execution spawned');

    return results;
  }

  /**
   * Create a consolidated debrief from role responses.
   */
  async createDebrief(userContext, briefRef, roleResponses) {
    const displayName = userContext.display_name || 'anonymous';
    const now = new Date();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');

    const debriefDir = path.join(
      this.teamRoot, 'humans', displayName, 'debriefs', month, day
    );
    fs.mkdirSync(debriefDir, { recursive: true });

    const fileName = `debrief__${briefRef}.md`;
    const debriefPath = path.join(debriefDir, fileName);

    // Build debrief content
    const lines = [
      `# Debrief: ${briefRef}`,
      '',
      `**For**: ${displayName}`,
      `**Date**: ${now.toISOString().split('T')[0]}`,
      `**Roles**: ${roleResponses.map(r => r.role).join(', ')}`,
      '',
      '---',
      '',
      '## Summary',
      '',
      'Role responses are linked below. Review each for domain-specific analysis.',
      '',
      '## Role Responses',
      '',
    ];

    for (const { role, responsePath, taskId, status } of roleResponses) {
      const relPath = path.relative(debriefDir, path.join(this.teamRoot, responsePath));
      lines.push(`### ${role.charAt(0).toUpperCase() + role.slice(1)}`);
      lines.push(`- **Response**: [${responsePath}](${relPath})`);
      lines.push(`- **Task**: ${taskId}`);
      lines.push(`- **Status**: ${status || 'pending'}`);
      lines.push('');
    }

    lines.push('---');
    lines.push('');
    lines.push('*Generated by VisionFlow Briefing Workflow*');

    fs.writeFileSync(debriefPath, lines.join('\n'), 'utf-8');

    this.logger.info({ debriefPath, user: displayName, briefRef }, 'Debrief created');

    return {
      debriefPath: path.relative(this.teamRoot, debriefPath),
      fullPath: debriefPath,
    };
  }

  /**
   * Role-specific guidance for task prompts.
   */
  _roleGuidance(role) {
    const guidance = {
      architect: '- System design implications\n- Architecture fitness\n- Technical debt risks\n- Integration concerns',
      dev: '- Implementation approach\n- Effort estimates\n- Technical risks\n- Dependencies on other work',
      ciso: '- Security implications\n- Threat model changes\n- Compliance concerns\n- Access control impacts',
      designer: '- UX implications\n- User journey impacts\n- Accessibility concerns\n- Visual consistency',
      dpo: '- Data protection impacts\n- GDPR/privacy concerns\n- Data flow changes\n- Consent requirements',
      devops: '- Infrastructure requirements\n- Deployment changes\n- Monitoring needs\n- Scaling implications',
      appsec: '- Application security review\n- OWASP considerations\n- Input validation\n- Authentication impacts',
      advocate: '- User impact assessment\n- Community communication\n- Documentation needs\n- Onboarding implications',
    };
    return guidance[role] || '- Respond from your domain expertise\n- Flag risks and dependencies';
  }
}

module.exports = BriefingService;
