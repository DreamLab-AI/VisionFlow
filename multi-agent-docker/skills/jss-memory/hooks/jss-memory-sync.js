/**
 * JSS Memory Sync - Claude-Flow Hooks for Solid Pod Integration
 *
 * Provides persistent memory for AI agents using Solid Pods as storage backend.
 * Integrates with claude-flow hooks system for pre/post task coordination,
 * session management, and cross-agent memory sharing.
 *
 * Memory Types:
 *   - Episodic: Task-specific memories with temporal context
 *   - Semantic: Factual knowledge and relationships
 *   - Procedural: Learned patterns and workflows
 *
 * Usage:
 *   const hooks = require('./jss-memory-sync');
 *   await hooks['pre-task']({ taskId: 'task-123', agentType: 'coder' });
 *   await hooks['post-task']({ taskId: 'task-123', result: {...} });
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

/**
 * Configuration from environment
 */
const CONFIG = {
  solidPodUrl: process.env.SOLID_POD_URL || 'http://localhost:3000',
  nostrToken: process.env.NOSTR_SESSION_TOKEN || '',
  memoryNamespace: 'agent-memory',
  claudeFlowBin: 'npx claude-flow@alpha',
  maxRetries: 3,
  retryDelay: 1000,
  cacheDir: process.env.JSS_CACHE_DIR || '/tmp/jss-memory-cache',
  debugMode: process.env.JSS_DEBUG === 'true'
};

/**
 * Memory type definitions aligned with JSON-LD schema
 */
const MemoryType = {
  EPISODIC: 'EpisodicMemory',
  SEMANTIC: 'SemanticMemory',
  PROCEDURAL: 'ProceduralMemory'
};

/**
 * Generate unique ID for memory entries
 */
function generateMemoryId() {
  return `mem-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
}

/**
 * Get pod path for agent memory storage
 * @param {string} agentType - Type of agent (coder, tester, researcher, etc.)
 * @param {string} memoryType - Type of memory (episodic, semantic, procedural)
 * @returns {string} Full pod path
 */
function getPodPath(agentType, memoryType = 'episodic') {
  return `/agent-${agentType}/memories/${memoryType}/`;
}

/**
 * Get session path for agent
 * @param {string} agentType - Type of agent
 * @returns {string} Session storage path
 */
function getSessionPath(agentType) {
  return `/agent-${agentType}/sessions/`;
}

/**
 * Create JSON-LD formatted memory object
 * @param {Object} params - Memory parameters
 * @returns {Object} JSON-LD formatted memory
 */
function createMemoryObject(params) {
  const {
    id = generateMemoryId(),
    type = MemoryType.EPISODIC,
    agentId,
    sessionId,
    content,
    confidence = 1.0,
    embedding = [],
    relatedMemories = [],
    metadata = {}
  } = params;

  return {
    '@context': 'https://visionflow.local/ontology/agent-memory.jsonld',
    '@type': type,
    '@id': id,
    agentId: `agent:${agentId}`,
    sessionId,
    content,
    confidence,
    embedding,
    timestamp: new Date().toISOString(),
    relatedMemories: relatedMemories.map(m => `memory:${m}`),
    metadata: {
      ...metadata,
      version: '1.0.0',
      createdBy: 'jss-memory-sync'
    }
  };
}

/**
 * Execute claude-flow hook command
 * @param {string} hookType - Hook type (pre-task, post-task, etc.)
 * @param {Object} params - Hook parameters
 * @returns {Object|null} Hook result or null on failure
 */
async function executeClaudeFlowHook(hookType, params) {
  try {
    const paramStr = Object.entries(params)
      .map(([k, v]) => `--${k} "${typeof v === 'object' ? JSON.stringify(v) : v}"`)
      .join(' ');

    const cmd = `${CONFIG.claudeFlowBin} hooks ${hookType} ${paramStr}`;

    if (CONFIG.debugMode) {
      console.log(`[JSS-Memory] Executing: ${cmd}`);
    }

    const result = execSync(cmd, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 30000
    });

    return JSON.parse(result);
  } catch (error) {
    if (CONFIG.debugMode) {
      console.error(`[JSS-Memory] Hook ${hookType} failed:`, error.message);
    }
    return null;
  }
}

/**
 * Store memory to Solid Pod with retry logic
 * @param {string} podPath - Path in pod
 * @param {Object} memory - Memory object to store
 * @returns {boolean} Success status
 */
async function storeToPod(podPath, memory) {
  const fullUrl = `${CONFIG.solidPodUrl}${podPath}${memory['@id']}.jsonld`;

  for (let attempt = 1; attempt <= CONFIG.maxRetries; attempt++) {
    try {
      // Use fetch or curl depending on environment
      const headers = {
        'Content-Type': 'application/ld+json',
        'Authorization': CONFIG.nostrToken ? `Bearer ${CONFIG.nostrToken}` : ''
      };

      // For now, cache locally and attempt pod storage
      const cacheFile = path.join(CONFIG.cacheDir, `${memory['@id']}.jsonld`);
      await fs.mkdir(path.dirname(cacheFile), { recursive: true });
      await fs.writeFile(cacheFile, JSON.stringify(memory, null, 2));

      if (CONFIG.debugMode) {
        console.log(`[JSS-Memory] Cached memory: ${cacheFile}`);
      }

      // Attempt actual pod storage if URL is configured
      if (CONFIG.solidPodUrl !== 'http://localhost:3000') {
        const curlCmd = `curl -s -X PUT "${fullUrl}" \
          -H "Content-Type: application/ld+json" \
          ${CONFIG.nostrToken ? `-H "Authorization: Bearer ${CONFIG.nostrToken}"` : ''} \
          -d '${JSON.stringify(memory)}'`;

        execSync(curlCmd, { stdio: 'pipe', timeout: 10000 });
      }

      return true;
    } catch (error) {
      if (CONFIG.debugMode) {
        console.error(`[JSS-Memory] Store attempt ${attempt} failed:`, error.message);
      }
      if (attempt < CONFIG.maxRetries) {
        await new Promise(r => setTimeout(r, CONFIG.retryDelay * attempt));
      }
    }
  }
  return false;
}

/**
 * Retrieve memory from Solid Pod
 * @param {string} podPath - Path in pod
 * @param {string} memoryId - Memory ID to retrieve
 * @returns {Object|null} Memory object or null
 */
async function retrieveFromPod(podPath, memoryId) {
  try {
    // Check local cache first
    const cacheFile = path.join(CONFIG.cacheDir, `${memoryId}.jsonld`);
    try {
      const cached = await fs.readFile(cacheFile, 'utf-8');
      return JSON.parse(cached);
    } catch {
      // Cache miss, try pod
    }

    // Try pod storage
    const fullUrl = `${CONFIG.solidPodUrl}${podPath}${memoryId}.jsonld`;
    const headers = CONFIG.nostrToken ? `-H "Authorization: Bearer ${CONFIG.nostrToken}"` : '';

    const result = execSync(
      `curl -s "${fullUrl}" ${headers}`,
      { encoding: 'utf-8', timeout: 10000 }
    );

    return JSON.parse(result);
  } catch (error) {
    if (CONFIG.debugMode) {
      console.error(`[JSS-Memory] Retrieve failed:`, error.message);
    }
    return null;
  }
}

/**
 * List memories from pod path
 * @param {string} podPath - Path to list
 * @returns {Array} List of memory IDs
 */
async function listPodMemories(podPath) {
  try {
    // List from local cache
    const cacheDir = path.join(CONFIG.cacheDir);
    const files = await fs.readdir(cacheDir).catch(() => []);

    return files
      .filter(f => f.endsWith('.jsonld'))
      .map(f => f.replace('.jsonld', ''));
  } catch (error) {
    return [];
  }
}

/**
 * Store memory to claude-flow coordination namespace
 * @param {string} key - Memory key
 * @param {Object} value - Memory value
 * @param {string} namespace - Namespace (default: coordination)
 */
async function storeToClaudeFlow(key, value, namespace = 'coordination') {
  try {
    const valueStr = JSON.stringify(value).replace(/"/g, '\\"');
    execSync(
      `${CONFIG.claudeFlowBin} hooks memory-store --key "${key}" --value "${valueStr}" --namespace "${namespace}"`,
      { stdio: 'pipe', timeout: 10000 }
    );
    return true;
  } catch (error) {
    if (CONFIG.debugMode) {
      console.error(`[JSS-Memory] Claude-flow store failed:`, error.message);
    }
    return false;
  }
}

/**
 * Retrieve memory from claude-flow coordination namespace
 * @param {string} key - Memory key
 * @param {string} namespace - Namespace (default: coordination)
 * @returns {Object|null} Retrieved value
 */
async function retrieveFromClaudeFlow(key, namespace = 'coordination') {
  try {
    const result = execSync(
      `${CONFIG.claudeFlowBin} hooks memory-retrieve --key "${key}" --namespace "${namespace}"`,
      { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000 }
    );
    return JSON.parse(result);
  } catch (error) {
    return null;
  }
}

/**
 * Pre-task hook: Load agent memory from pod
 *
 * Loads relevant memories for the agent before task execution.
 * Restores session context and related episodic memories.
 *
 * @param {Object} context - Task context
 * @param {string} context.taskId - Unique task identifier
 * @param {string} context.agentType - Agent type (coder, tester, etc.)
 * @param {string} context.description - Task description
 * @param {string} context.sessionId - Optional session ID
 * @returns {Object} Loaded memory context
 */
async function preTask(context) {
  const {
    taskId,
    agentType = 'default',
    description = '',
    sessionId = `session-${Date.now()}`
  } = context;

  console.log(`[JSS-Memory] Pre-task: ${taskId} (Agent: ${agentType})`);

  // Execute claude-flow pre-task hook
  await executeClaudeFlowHook('pre-task', {
    description: `[${agentType}] ${description}`,
    'task-id': taskId
  });

  // Load memories from pod
  const podPath = getPodPath(agentType, 'episodic');
  const memoryIds = await listPodMemories(podPath);

  // Load recent memories (last 10)
  const memories = [];
  for (const memId of memoryIds.slice(-10)) {
    const memory = await retrieveFromPod(podPath, memId);
    if (memory) {
      memories.push(memory);
    }
  }

  // Load semantic knowledge
  const semanticPath = getPodPath(agentType, 'semantic');
  const semanticIds = await listPodMemories(semanticPath);
  const semanticMemories = [];
  for (const memId of semanticIds.slice(-5)) {
    const memory = await retrieveFromPod(semanticPath, memId);
    if (memory) {
      semanticMemories.push(memory);
    }
  }

  // Create task context memory
  const taskMemory = createMemoryObject({
    type: MemoryType.EPISODIC,
    agentId: agentType,
    sessionId,
    content: {
      type: 'task_start',
      taskId,
      description,
      timestamp: new Date().toISOString()
    },
    metadata: {
      phase: 'pre-task',
      loadedMemories: memories.length,
      loadedSemantic: semanticMemories.length
    }
  });

  // Store to claude-flow for cross-agent coordination
  await storeToClaudeFlow(
    `swarm/${agentType}/current-task`,
    {
      taskId,
      sessionId,
      status: 'starting',
      timestamp: new Date().toISOString()
    }
  );

  return {
    sessionId,
    taskId,
    agentType,
    loadedMemories: memories,
    semanticKnowledge: semanticMemories,
    taskMemory,
    ready: true
  };
}

/**
 * Post-task hook: Save task results to pod
 *
 * Persists task results as episodic memories and updates
 * procedural knowledge based on successful patterns.
 *
 * @param {Object} context - Task context from pre-task
 * @param {string} context.taskId - Task identifier
 * @param {string} context.agentType - Agent type
 * @param {string} context.sessionId - Session identifier
 * @param {Object} context.result - Task execution result
 * @param {boolean} context.result.success - Whether task succeeded
 * @param {Object} context.result.outputs - Task outputs
 * @param {string} context.result.error - Error if failed
 * @returns {Object} Saved memory references
 */
async function postTask(context) {
  const {
    taskId,
    agentType = 'default',
    sessionId,
    result = {}
  } = context;

  console.log(`[JSS-Memory] Post-task: ${taskId} (Success: ${result.success})`);

  // Execute claude-flow post-task hook
  await executeClaudeFlowHook('post-task', {
    'task-id': taskId,
    success: result.success || false
  });

  // Create episodic memory from task result
  const episodicMemory = createMemoryObject({
    type: MemoryType.EPISODIC,
    agentId: agentType,
    sessionId,
    content: {
      type: 'task_result',
      taskId,
      success: result.success,
      outputs: result.outputs || {},
      error: result.error || null,
      duration: result.duration || 0
    },
    confidence: result.success ? 0.95 : 0.6,
    metadata: {
      phase: 'post-task',
      filesModified: result.filesModified || [],
      testsRun: result.testsRun || 0
    }
  });

  // Store episodic memory to pod
  const episodicPath = getPodPath(agentType, 'episodic');
  await storeToPod(episodicPath, episodicMemory);

  // If successful, create procedural memory for learned pattern
  let proceduralMemory = null;
  if (result.success && result.pattern) {
    proceduralMemory = createMemoryObject({
      type: MemoryType.PROCEDURAL,
      agentId: agentType,
      sessionId,
      content: {
        type: 'learned_pattern',
        taskId,
        pattern: result.pattern,
        steps: result.steps || [],
        applicability: result.applicability || 'general'
      },
      confidence: 0.85,
      metadata: {
        learnedFrom: taskId,
        successCount: 1
      }
    });

    const proceduralPath = getPodPath(agentType, 'procedural');
    await storeToPod(proceduralPath, proceduralMemory);
  }

  // Update claude-flow coordination state
  await storeToClaudeFlow(
    `swarm/${agentType}/task-complete/${taskId}`,
    {
      taskId,
      sessionId,
      success: result.success,
      memoryId: episodicMemory['@id'],
      timestamp: new Date().toISOString()
    }
  );

  // Notify completion
  try {
    execSync(
      `${CONFIG.claudeFlowBin} hooks notify --message "[${agentType}] Task ${taskId} completed (${result.success ? 'success' : 'failed'})"`,
      { stdio: 'pipe' }
    );
  } catch {}

  return {
    taskId,
    episodicMemoryId: episodicMemory['@id'],
    proceduralMemoryId: proceduralMemory?.['@id'] || null,
    stored: true
  };
}

/**
 * Session restore hook: Load session from pod
 *
 * Restores previous session state including memories,
 * context, and learned patterns.
 *
 * @param {Object} context - Session context
 * @param {string} context.sessionId - Session ID to restore
 * @param {string} context.agentType - Agent type
 * @returns {Object} Restored session data
 */
async function sessionRestore(context) {
  const {
    sessionId,
    agentType = 'default'
  } = context;

  console.log(`[JSS-Memory] Session restore: ${sessionId}`);

  // Execute claude-flow session-restore hook
  await executeClaudeFlowHook('session-restore', {
    'session-id': sessionId
  });

  // Load session data from pod
  const sessionPath = getSessionPath(agentType);
  const sessionData = await retrieveFromPod(sessionPath, sessionId);

  // Load related memories
  const memories = [];
  if (sessionData?.memories) {
    for (const memId of sessionData.memories) {
      const episodicPath = getPodPath(agentType, 'episodic');
      const memory = await retrieveFromPod(episodicPath, memId);
      if (memory) {
        memories.push(memory);
      }
    }
  }

  // Check claude-flow for additional context
  const flowContext = await retrieveFromClaudeFlow(
    `swarm/${agentType}/session/${sessionId}`
  );

  // Create session restoration memory
  const restorationMemory = createMemoryObject({
    type: MemoryType.EPISODIC,
    agentId: agentType,
    sessionId,
    content: {
      type: 'session_restore',
      originalSession: sessionId,
      restoredAt: new Date().toISOString(),
      memoriesLoaded: memories.length
    },
    metadata: {
      restored: true,
      previousState: sessionData?.state || 'unknown'
    }
  });

  const episodicPath = getPodPath(agentType, 'episodic');
  await storeToPod(episodicPath, restorationMemory);

  return {
    sessionId,
    agentType,
    restored: true,
    sessionData: sessionData || {},
    memories,
    flowContext,
    restorationMemoryId: restorationMemory['@id']
  };
}

/**
 * Session end hook: Save session summary
 *
 * Creates a session summary with all memories and metrics,
 * then persists to pod for future restoration.
 *
 * @param {Object} context - Session context
 * @param {string} context.sessionId - Session ID
 * @param {string} context.agentType - Agent type
 * @param {boolean} context.exportMetrics - Whether to export metrics
 * @param {Object} context.summary - Session summary data
 * @returns {Object} Session end result
 */
async function sessionEnd(context) {
  const {
    sessionId,
    agentType = 'default',
    exportMetrics = true,
    summary = {}
  } = context;

  console.log(`[JSS-Memory] Session end: ${sessionId}`);

  // Execute claude-flow session-end hook
  if (exportMetrics) {
    await executeClaudeFlowHook('session-end', {
      'session-id': sessionId,
      'export-metrics': 'true'
    });
  }

  // Collect all memories from this session
  const episodicPath = getPodPath(agentType, 'episodic');
  const allMemoryIds = await listPodMemories(episodicPath);

  const sessionMemories = [];
  for (const memId of allMemoryIds) {
    const memory = await retrieveFromPod(episodicPath, memId);
    if (memory?.sessionId === sessionId) {
      sessionMemories.push(memory['@id']);
    }
  }

  // Create session summary
  const sessionSummary = {
    '@context': 'https://visionflow.local/ontology/agent-memory.jsonld',
    '@type': 'SessionSummary',
    '@id': sessionId,
    agentId: `agent:${agentType}`,
    startTime: summary.startTime || new Date().toISOString(),
    endTime: new Date().toISOString(),
    memories: sessionMemories,
    metrics: {
      tasksCompleted: summary.tasksCompleted || 0,
      tasksSuccessful: summary.tasksSuccessful || 0,
      memoriesCreated: sessionMemories.length,
      ...summary.metrics
    },
    state: summary.state || 'completed',
    metadata: {
      version: '1.0.0',
      exportedMetrics: exportMetrics
    }
  };

  // Store session summary to pod
  const sessionPath = getSessionPath(agentType);
  await storeToPod(sessionPath, sessionSummary);

  // Store final state in claude-flow
  await storeToClaudeFlow(
    `swarm/${agentType}/session/${sessionId}`,
    {
      sessionId,
      status: 'completed',
      memoriesCount: sessionMemories.length,
      timestamp: new Date().toISOString()
    }
  );

  // Notify session end
  try {
    execSync(
      `${CONFIG.claudeFlowBin} hooks notify --message "[${agentType}] Session ${sessionId} ended (${sessionMemories.length} memories)"`,
      { stdio: 'pipe' }
    );
  } catch {}

  return {
    sessionId,
    agentType,
    memoriesStored: sessionMemories.length,
    summaryId: sessionId,
    exported: exportMetrics
  };
}

/**
 * Store semantic knowledge
 *
 * Adds factual knowledge to the agent's semantic memory store.
 *
 * @param {Object} context - Knowledge context
 * @param {string} context.agentType - Agent type
 * @param {string} context.domain - Knowledge domain
 * @param {Object} context.knowledge - Knowledge content
 * @param {Array} context.embedding - Optional vector embedding
 * @returns {Object} Stored memory reference
 */
async function storeSemanticKnowledge(context) {
  const {
    agentType = 'default',
    domain,
    knowledge,
    embedding = [],
    relatedMemories = []
  } = context;

  const semanticMemory = createMemoryObject({
    type: MemoryType.SEMANTIC,
    agentId: agentType,
    sessionId: `semantic-${Date.now()}`,
    content: {
      type: 'semantic_knowledge',
      domain,
      knowledge
    },
    embedding,
    relatedMemories,
    confidence: 0.9,
    metadata: {
      domain,
      source: 'explicit_store'
    }
  });

  const semanticPath = getPodPath(agentType, 'semantic');
  const stored = await storeToPod(semanticPath, semanticMemory);

  return {
    memoryId: semanticMemory['@id'],
    stored,
    domain
  };
}

/**
 * Search memories by content
 *
 * Searches agent memories for matching content patterns.
 *
 * @param {Object} context - Search context
 * @param {string} context.agentType - Agent type
 * @param {string} context.query - Search query
 * @param {string} context.memoryType - Memory type to search
 * @param {number} context.limit - Max results
 * @returns {Array} Matching memories
 */
async function searchMemories(context) {
  const {
    agentType = 'default',
    query,
    memoryType = 'episodic',
    limit = 10
  } = context;

  const podPath = getPodPath(agentType, memoryType);
  const memoryIds = await listPodMemories(podPath);

  const matches = [];
  for (const memId of memoryIds) {
    const memory = await retrieveFromPod(podPath, memId);
    if (memory) {
      const contentStr = JSON.stringify(memory.content).toLowerCase();
      if (contentStr.includes(query.toLowerCase())) {
        matches.push(memory);
        if (matches.length >= limit) break;
      }
    }
  }

  return matches;
}

/**
 * Export module with named hooks
 *
 * Hook names match claude-flow expectations:
 * - pre-task
 * - post-task
 * - session-restore
 * - session-end
 */
module.exports = {
  // Standard claude-flow hooks
  'pre-task': preTask,
  'post-task': postTask,
  'session-restore': sessionRestore,
  'session-end': sessionEnd,

  // Additional utilities
  storeSemanticKnowledge,
  searchMemories,

  // Low-level functions
  createMemoryObject,
  storeToPod,
  retrieveFromPod,
  listPodMemories,
  storeToClaudeFlow,
  retrieveFromClaudeFlow,

  // Constants
  MemoryType,
  CONFIG
};

// CLI interface for testing
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];

  (async () => {
    switch (command) {
      case 'test-pre':
        const preResult = await preTask({
          taskId: 'test-task-001',
          agentType: 'coder',
          description: 'Test pre-task hook'
        });
        console.log('Pre-task result:', JSON.stringify(preResult, null, 2));
        break;

      case 'test-post':
        const postResult = await postTask({
          taskId: 'test-task-001',
          agentType: 'coder',
          sessionId: 'test-session',
          result: {
            success: true,
            outputs: { file: 'test.js' },
            pattern: 'unit-test-first'
          }
        });
        console.log('Post-task result:', JSON.stringify(postResult, null, 2));
        break;

      case 'test-session':
        const sessionResult = await sessionEnd({
          sessionId: 'test-session',
          agentType: 'coder',
          exportMetrics: true,
          summary: {
            tasksCompleted: 5,
            tasksSuccessful: 4
          }
        });
        console.log('Session end result:', JSON.stringify(sessionResult, null, 2));
        break;

      case 'search':
        const searchResult = await searchMemories({
          agentType: 'coder',
          query: args[1] || 'test',
          limit: 5
        });
        console.log('Search results:', JSON.stringify(searchResult, null, 2));
        break;

      default:
        console.log(`
JSS Memory Sync - Solid Pod Memory for AI Agents

Usage:
  node jss-memory-sync.js test-pre        # Test pre-task hook
  node jss-memory-sync.js test-post       # Test post-task hook
  node jss-memory-sync.js test-session    # Test session-end hook
  node jss-memory-sync.js search <query>  # Search memories

Environment:
  SOLID_POD_URL          - Solid Pod URL (default: http://localhost:3000)
  NOSTR_SESSION_TOKEN    - Nostr session token for authentication
  JSS_CACHE_DIR          - Local cache directory
  JSS_DEBUG              - Enable debug output (true/false)

Integration:
  const hooks = require('./jss-memory-sync');

  // Before task
  const ctx = await hooks['pre-task']({ taskId, agentType, description });

  // After task
  await hooks['post-task']({ taskId, agentType, sessionId, result });

  // Session management
  await hooks['session-restore']({ sessionId, agentType });
  await hooks['session-end']({ sessionId, agentType, exportMetrics: true });
        `);
    }
  })();
}
