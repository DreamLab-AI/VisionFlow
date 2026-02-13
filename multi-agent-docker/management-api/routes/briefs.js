/**
 * Briefing workflow routes
 *
 * POST /v1/briefs            - Create a new brief
 * POST /v1/briefs/:id/execute - Execute a brief (spawn role agents)
 * GET  /v1/briefs/:userId     - List briefs for a user
 * POST /v1/briefs/:id/debrief - Create consolidated debrief
 * GET  /v1/debriefs/:userId   - List debriefs for a user
 */

const path = require('path');
const fs = require('fs');

async function briefsRoutes(fastify, options) {
  const { logger, processManager, briefingService, beadsService } = options;

  /**
   * Create a new brief from user content.
   */
  fastify.post('/v1/briefs', {
    schema: {
      body: {
        type: 'object',
        required: ['content', 'roles'],
        properties: {
          content: { type: 'string' },
          roles: { type: 'array', items: { type: 'string' } },
          version: { type: 'string' },
          brief_type: { type: 'string' },
          slug: { type: 'string' },
          user_context: {
            type: 'object',
            properties: {
              user_id: { type: 'string' },
              pubkey: { type: 'string' },
              display_name: { type: 'string' },
              session_id: { type: 'string' },
              is_power_user: { type: 'boolean' }
            }
          }
        }
      },
      response: {
        201: {
          type: 'object',
          properties: {
            briefId: { type: 'string' },
            briefPath: { type: 'string' },
            beadId: { type: ['string', 'null'] }
          }
        }
      }
    }
  }, async (request, reply) => {
    const {
      content,
      roles,
      version,
      brief_type: briefType,
      slug,
      user_context: userContext
    } = request.body;

    logger.info({
      user: userContext?.display_name || 'anonymous',
      roles,
      briefType
    }, 'Creating brief');

    try {
      const result = await briefingService.createBrief(
        userContext || { display_name: 'anonymous' },
        content,
        { version, briefType, slug, roles }
      );

      reply.code(201).send({
        briefId: result.briefId,
        briefPath: result.briefPath,
        beadId: result.beadId
      });
    } catch (error) {
      logger.error({ error: error.message }, 'Failed to create brief');
      reply.code(500).send({
        error: 'Internal Server Error',
        message: 'Failed to create brief',
        details: error.message
      });
    }
  });

  /**
   * Execute a brief — spawn role-specific agents.
   */
  fastify.post('/v1/briefs/:briefId/execute', {
    schema: {
      params: {
        type: 'object',
        properties: {
          briefId: { type: 'string' }
        }
      },
      body: {
        type: 'object',
        required: ['brief_path', 'roles'],
        properties: {
          brief_path: { type: 'string' },
          roles: { type: 'array', items: { type: 'string' } },
          epic_bead_id: { type: 'string' },
          user_context: {
            type: 'object',
            properties: {
              user_id: { type: 'string' },
              pubkey: { type: 'string' },
              display_name: { type: 'string' },
              session_id: { type: 'string' },
              is_power_user: { type: 'boolean' }
            }
          }
        }
      },
      response: {
        202: {
          type: 'object',
          properties: {
            briefId: { type: 'string' },
            roleTasks: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  role: { type: 'string' },
                  taskId: { type: 'string' },
                  beadId: { type: ['string', 'null'] },
                  responsePath: { type: 'string' }
                }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { briefId } = request.params;
    const {
      brief_path: briefPath,
      roles,
      epic_bead_id: epicBeadId,
      user_context: userContext
    } = request.body;

    logger.info({
      briefId,
      roles,
      user: userContext?.display_name || 'anonymous'
    }, 'Executing brief');

    try {
      const results = await briefingService.executeBrief(
        briefPath,
        roles,
        userContext || { display_name: 'anonymous' },
        epicBeadId
      );

      reply.code(202).send({
        briefId,
        roleTasks: results
      });
    } catch (error) {
      logger.error({ briefId, error: error.message }, 'Failed to execute brief');
      reply.code(500).send({
        error: 'Internal Server Error',
        message: 'Failed to execute brief',
        details: error.message
      });
    }
  });

  /**
   * List briefs for a specific user.
   */
  fastify.get('/v1/briefs/:userId', {
    schema: {
      params: {
        type: 'object',
        properties: {
          userId: { type: 'string' }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            briefs: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  path: { type: 'string' },
                  name: { type: 'string' },
                  date: { type: 'string' }
                }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { userId } = request.params;
    const teamRoot = briefingService.teamRoot;
    const briefsBase = path.join(teamRoot, 'humans', userId, 'briefs');

    const briefs = [];
    try {
      if (fs.existsSync(briefsBase)) {
        // Walk month/day directories
        const months = fs.readdirSync(briefsBase).filter(f =>
          fs.statSync(path.join(briefsBase, f)).isDirectory()
        );
        for (const month of months) {
          const monthDir = path.join(briefsBase, month);
          const days = fs.readdirSync(monthDir).filter(f =>
            fs.statSync(path.join(monthDir, f)).isDirectory()
          );
          for (const day of days) {
            const dayDir = path.join(monthDir, day);
            const files = fs.readdirSync(dayDir).filter(f => f.endsWith('.md'));
            for (const file of files) {
              briefs.push({
                path: path.relative(teamRoot, path.join(dayDir, file)),
                name: file,
                date: `${month}-${day}`
              });
            }
          }
        }
      }
    } catch (error) {
      logger.warn({ userId, error: error.message }, 'Failed to list briefs');
    }

    reply.send({ briefs });
  });

  /**
   * Create a consolidated debrief from role responses.
   */
  fastify.post('/v1/briefs/:briefId/debrief', {
    schema: {
      params: {
        type: 'object',
        properties: {
          briefId: { type: 'string' }
        }
      },
      body: {
        type: 'object',
        required: ['role_responses'],
        properties: {
          role_responses: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                role: { type: 'string' },
                responsePath: { type: 'string' },
                taskId: { type: 'string' },
                status: { type: 'string' }
              }
            }
          },
          user_context: {
            type: 'object',
            properties: {
              user_id: { type: 'string' },
              pubkey: { type: 'string' },
              display_name: { type: 'string' },
              session_id: { type: 'string' },
              is_power_user: { type: 'boolean' }
            }
          }
        }
      },
      response: {
        201: {
          type: 'object',
          properties: {
            debriefPath: { type: 'string' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { briefId } = request.params;
    const { role_responses: roleResponses, user_context: userContext } = request.body;

    logger.info({
      briefId,
      roles: roleResponses.map(r => r.role),
      user: userContext?.display_name || 'anonymous'
    }, 'Creating debrief');

    try {
      const result = await briefingService.createDebrief(
        userContext || { display_name: 'anonymous' },
        briefId,
        roleResponses
      );

      reply.code(201).send({
        debriefPath: result.debriefPath
      });
    } catch (error) {
      logger.error({ briefId, error: error.message }, 'Failed to create debrief');
      reply.code(500).send({
        error: 'Internal Server Error',
        message: 'Failed to create debrief',
        details: error.message
      });
    }
  });

  /**
   * List debriefs for a specific user.
   */
  fastify.get('/v1/debriefs/:userId', {
    schema: {
      params: {
        type: 'object',
        properties: {
          userId: { type: 'string' }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            debriefs: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  path: { type: 'string' },
                  name: { type: 'string' },
                  date: { type: 'string' }
                }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { userId } = request.params;
    const teamRoot = briefingService.teamRoot;
    const debriefsBase = path.join(teamRoot, 'humans', userId, 'debriefs');

    const debriefs = [];
    try {
      if (fs.existsSync(debriefsBase)) {
        const months = fs.readdirSync(debriefsBase).filter(f =>
          fs.statSync(path.join(debriefsBase, f)).isDirectory()
        );
        for (const month of months) {
          const monthDir = path.join(debriefsBase, month);
          const days = fs.readdirSync(monthDir).filter(f =>
            fs.statSync(path.join(monthDir, f)).isDirectory()
          );
          for (const day of days) {
            const dayDir = path.join(monthDir, day);
            const files = fs.readdirSync(dayDir).filter(f => f.endsWith('.md'));
            for (const file of files) {
              debriefs.push({
                path: path.relative(teamRoot, path.join(dayDir, file)),
                name: file,
                date: `${month}-${day}`
              });
            }
          }
        }
      }
    } catch (error) {
      logger.warn({ userId, error: error.message }, 'Failed to list debriefs');
    }

    reply.send({ debriefs });
  });
}

module.exports = briefsRoutes;
