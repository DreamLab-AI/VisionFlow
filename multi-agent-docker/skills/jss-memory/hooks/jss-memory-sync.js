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
  solidPodUrl: process.env.SOLID_POD_URL || 'http://localhost:4000/solid',
  nostrToken: process.env.NOSTR_SESSION_TOKEN || '',
  memoryNamespace: 'agent-memory',
  claudeFlowBin: 'npx claude-flow@v3alpha',
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
 * Get inbox path for agent
 * @param {string} agentType - Type of agent
 * @returns {string} Inbox storage path
 */
function getInboxPath(agentType) {
  return `/agent-${agentType}/inbox/`;
}

/**
 * Get shared path for cross-agent data
 * @param {string} agentType - Type of agent
 * @returns {string} Shared storage path
 */
function getSharedPath(agentType) {
  return `/agent-${agentType}/shared/`;
}

/**
 * Send message to another agent's inbox
 *
 * Creates an inbox message for asynchronous inter-agent communication.
 *
 * @param {Object} context - Message context
 * @param {string} context.fromAgent - Sender agent type
 * @param {string} context.toAgent - Recipient agent type
 * @param {string} context.messageType - Type of message (request, response, notification)
 * @param {Object} context.content - Message content
 * @param {string} context.priority - Message priority (low, normal, high, urgent)
 * @returns {Object} Sent message reference
 */
async function sendInboxMessage(context) {
  const {
    fromAgent,
    toAgent,
    messageType = 'notification',
    content,
    priority = 'normal',
    replyTo = null
  } = context;

  const messageId = `msg-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;

  const message = {
    '@context': 'https://visionflow.local/ontology/agent-memory.jsonld',
    '@type': 'InboxMessage',
    '@id': messageId,
    from: `agent:${fromAgent}`,
    to: `agent:${toAgent}`,
    messageType,
    content,
    priority,
    replyTo,
    timestamp: new Date().toISOString(),
    status: 'unread',
    metadata: {
      version: '1.0.0',
      createdBy: 'jss-memory-sync'
    }
  };

  const inboxPath = getInboxPath(toAgent);
  const stored = await storeToPod(inboxPath, message);

  // Also store to claude-flow for immediate notification
  await storeToClaudeFlow(
    `swarm/${toAgent}/inbox/${messageId}`,
    {
      from: fromAgent,
      messageType,
      priority,
      timestamp: message.timestamp
    }
  );

  if (CONFIG.debugMode) {
    console.log(`[JSS-Memory] Sent message ${messageId} from ${fromAgent} to ${toAgent}`);
  }

  return {
    messageId,
    stored,
    from: fromAgent,
    to: toAgent
  };
}

/**
 * Read messages from agent inbox
 *
 * @param {Object} context - Read context
 * @param {string} context.agentType - Agent type to read inbox for
 * @param {string} context.status - Filter by status (unread, read, all)
 * @param {number} context.limit - Max messages to return
 * @returns {Array} Inbox messages
 */
async function readInbox(context) {
  const {
    agentType,
    status = 'unread',
    limit = 20
  } = context;

  const inboxPath = getInboxPath(agentType);
  const messageIds = await listPodMemories(inboxPath);

  const messages = [];
  for (const msgId of messageIds.slice(-limit)) {
    const message = await retrieveFromPod(inboxPath, msgId);
    if (message) {
      if (status === 'all' || message.status === status) {
        messages.push(message);
      }
    }
  }

  return messages.sort((a, b) => {
    const priorityOrder = { urgent: 0, high: 1, normal: 2, low: 3 };
    return priorityOrder[a.priority] - priorityOrder[b.priority];
  });
}

/**
 * Mark inbox message as read
 *
 * @param {Object} context - Context
 * @param {string} context.agentType - Agent type
 * @param {string} context.messageId - Message ID to mark
 * @returns {boolean} Success status
 */
async function markMessageRead(context) {
  const { agentType, messageId } = context;

  const inboxPath = getInboxPath(agentType);
  const message = await retrieveFromPod(inboxPath, messageId);

  if (message) {
    message.status = 'read';
    message.readAt = new Date().toISOString();
    await storeToPod(inboxPath, message);
    return true;
  }
  return false;
}

/**
 * Share knowledge with other agents
 *
 * Creates shared knowledge accessible by specified agent types.
 *
 * @param {Object} context - Share context
 * @param {string} context.fromAgent - Source agent type
 * @param {Array} context.toAgents - Target agent types (or ['all'] for broadcast)
 * @param {string} context.knowledgeType - Type of knowledge being shared
 * @param {Object} context.knowledge - Knowledge content
 * @returns {Object} Shared knowledge reference
 */
async function shareKnowledge(context) {
  const {
    fromAgent,
    toAgents = ['all'],
    knowledgeType,
    knowledge,
    sessionId = `shared-${Date.now()}`
  } = context;

  const sharedId = `shared-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;

  const sharedKnowledge = {
    '@context': 'https://visionflow.local/ontology/agent-memory.jsonld',
    '@type': 'SharedKnowledge',
    '@id': sharedId,
    sharedBy: `agent:${fromAgent}`,
    sharedWith: toAgents.map(a => a === 'all' ? 'agent:*' : `agent:${a}`),
    knowledgeType,
    content: knowledge,
    timestamp: new Date().toISOString(),
    sessionId,
    accessCount: 0,
    metadata: {
      version: '1.0.0',
      createdBy: 'jss-memory-sync'
    }
  };

  // Store in source agent's shared folder
  const sharedPath = getSharedPath(fromAgent);
  await storeToPod(sharedPath, sharedKnowledge);

  // If broadcasting to all or specific agents, also store reference in coordination
  await storeToClaudeFlow(
    `swarm/shared/${sharedId}`,
    {
      from: fromAgent,
      to: toAgents,
      knowledgeType,
      timestamp: sharedKnowledge.timestamp
    }
  );

  // Notify target agents
  if (!toAgents.includes('all')) {
    for (const targetAgent of toAgents) {
      await sendInboxMessage({
        fromAgent,
        toAgent: targetAgent,
        messageType: 'knowledge-share',
        content: {
          sharedId,
          knowledgeType,
          summary: typeof knowledge === 'object' ? Object.keys(knowledge) : 'data'
        },
        priority: 'normal'
      });
    }
  }

  return {
    sharedId,
    from: fromAgent,
    to: toAgents,
    knowledgeType
  };
}

/**
 * Get shared knowledge by ID
 *
 * @param {Object} context - Context
 * @param {string} context.sharedId - Shared knowledge ID
 * @param {string} context.sourceAgent - Source agent type
 * @returns {Object|null} Shared knowledge or null
 */
async function getSharedKnowledge(context) {
  const { sharedId, sourceAgent } = context;

  const sharedPath = getSharedPath(sourceAgent);
  const knowledge = await retrieveFromPod(sharedPath, sharedId);

  if (knowledge) {
    // Increment access count
    knowledge.accessCount = (knowledge.accessCount || 0) + 1;
    knowledge.lastAccessed = new Date().toISOString();
    await storeToPod(sharedPath, knowledge);
  }

  return knowledge;
}

/**
 * Register agent in coordination registry
 *
 * @param {Object} context - Registration context
 * @param {string} context.agentType - Agent type
 * @param {string} context.agentId - Unique agent instance ID
 * @param {Array} context.capabilities - Agent capabilities
 * @param {string} context.status - Agent status (active, idle, busy)
 * @returns {Object} Registration result
 */
async function registerAgent(context) {
  const {
    agentType,
    agentId = `${agentType}-${Date.now()}`,
    capabilities = [],
    status = 'active'
  } = context;

  const registration = {
    '@context': 'https://visionflow.local/ontology/agent-memory.jsonld',
    '@type': 'AgentRegistry',
    '@id': agentId,
    agentType,
    capabilities,
    status,
    registeredAt: new Date().toISOString(),
    lastHeartbeat: new Date().toISOString(),
    metadata: {
      version: '1.0.0',
      category: AGENT_CATEGORIES[agentType] || 'general'
    }
  };

  // Store in shared agents registry
  await storeToPod('/agents/', registration);

  // Also store in claude-flow
  await storeToClaudeFlow(
    `swarm/registry/${agentId}`,
    {
      agentType,
      status,
      capabilities,
      timestamp: registration.registeredAt
    }
  );

  return {
    agentId,
    agentType,
    registered: true
  };
}

/**
 * Agent heartbeat to update status
 *
 * @param {Object} context - Heartbeat context
 * @param {string} context.agentId - Agent instance ID
 * @param {string} context.status - Current status
 * @param {Object} context.metrics - Optional metrics
 * @returns {boolean} Success status
 */
async function agentHeartbeat(context) {
  const {
    agentId,
    status = 'active',
    metrics = {}
  } = context;

  await storeToClaudeFlow(
    `swarm/heartbeat/${agentId}`,
    {
      status,
      metrics,
      timestamp: new Date().toISOString()
    }
  );

  return true;
}

/**
 * Agent category mappings
 */
const AGENT_CATEGORIES = {
  // Core Development
  'coder': 'core-development',
  'reviewer': 'core-development',
  'tester': 'core-development',
  'planner': 'core-development',
  'researcher': 'core-development',

  // Swarm Coordination
  'hierarchical-coordinator': 'swarm-coordination',
  'mesh-coordinator': 'swarm-coordination',
  'adaptive-coordinator': 'swarm-coordination',
  'collective-intelligence-coordinator': 'swarm-coordination',
  'swarm-memory-manager': 'swarm-coordination',

  // Consensus & Distributed
  'byzantine-coordinator': 'consensus-distributed',
  'raft-manager': 'consensus-distributed',
  'gossip-coordinator': 'consensus-distributed',
  'consensus-builder': 'consensus-distributed',
  'crdt-synchronizer': 'consensus-distributed',
  'quorum-manager': 'consensus-distributed',
  'security-manager': 'consensus-distributed',

  // Performance & Optimization
  'perf-analyzer': 'performance-optimization',
  'performance-benchmarker': 'performance-optimization',
  'task-orchestrator': 'performance-optimization',
  'memory-coordinator': 'performance-optimization',
  'smart-agent': 'performance-optimization',

  // GitHub & Repository
  'github-modes': 'github-repository',
  'pr-manager': 'github-repository',
  'code-review-swarm': 'github-repository',
  'issue-tracker': 'github-repository',
  'release-manager': 'github-repository',
  'workflow-automation': 'github-repository',
  'project-board-sync': 'github-repository',
  'repo-architect': 'github-repository',
  'multi-repo-swarm': 'github-repository',

  // SPARC Methodology
  'sparc-coord': 'sparc-methodology',
  'sparc-coder': 'sparc-methodology',
  'specification': 'sparc-methodology',
  'pseudocode': 'sparc-methodology',
  'architecture': 'sparc-methodology',
  'refinement': 'sparc-methodology',

  // Specialized Development
  'backend-dev': 'specialized-development',
  'mobile-dev': 'specialized-development',
  'ml-developer': 'specialized-development',
  'cicd-engineer': 'specialized-development',
  'api-docs': 'specialized-development',
  'system-architect': 'specialized-development',
  'code-analyzer': 'specialized-development',
  'base-template-generator': 'specialized-development',

  // Testing & Validation
  'tdd-london-swarm': 'testing-validation',
  'production-validator': 'testing-validation',

  // Migration & Planning
  'migration-planner': 'migration-planning',
  'swarm-init': 'migration-planning'
};

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

  // Memory utilities
  storeSemanticKnowledge,
  searchMemories,

  // Inter-agent communication
  sendInboxMessage,
  readInbox,
  markMessageRead,

  // Cross-agent knowledge sharing
  shareKnowledge,
  getSharedKnowledge,

  // Agent registry
  registerAgent,
  agentHeartbeat,

  // Low-level functions
  createMemoryObject,
  storeToPod,
  retrieveFromPod,
  listPodMemories,
  storeToClaudeFlow,
  retrieveFromClaudeFlow,
  getInboxPath,
  getSharedPath,
  getPodPath,
  getSessionPath,

  // Constants
  MemoryType,
  CONFIG,
  AGENT_CATEGORIES
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

      case 'send-message':
        const msgResult = await sendInboxMessage({
          fromAgent: args[1] || 'coder',
          toAgent: args[2] || 'reviewer',
          messageType: 'request',
          content: { request: args[3] || 'Please review my code' },
          priority: 'normal'
        });
        console.log('Message sent:', JSON.stringify(msgResult, null, 2));
        break;

      case 'read-inbox':
        const messages = await readInbox({
          agentType: args[1] || 'reviewer',
          status: 'all',
          limit: 10
        });
        console.log('Inbox messages:', JSON.stringify(messages, null, 2));
        break;

      case 'share-knowledge':
        const shareResult = await shareKnowledge({
          fromAgent: args[1] || 'researcher',
          toAgents: (args[2] || 'coder,tester').split(','),
          knowledgeType: 'best-practice',
          knowledge: { topic: args[3] || 'testing', content: 'Always write tests first' }
        });
        console.log('Knowledge shared:', JSON.stringify(shareResult, null, 2));
        break;

      case 'register':
        const regResult = await registerAgent({
          agentType: args[1] || 'coder',
          capabilities: (args[2] || 'code-generation,refactoring').split(','),
          status: 'active'
        });
        console.log('Agent registered:', JSON.stringify(regResult, null, 2));
        break;

      case 'list-agents':
        console.log('Available agent types (49 total):');
        console.log('');
        const categories = {};
        for (const [agent, category] of Object.entries(AGENT_CATEGORIES)) {
          if (!categories[category]) categories[category] = [];
          categories[category].push(agent);
        }
        for (const [category, agents] of Object.entries(categories)) {
          console.log(`  ${category}:`);
          console.log(`    ${agents.join(', ')}`);
          console.log('');
        }
        break;

      default:
        console.log(`
JSS Memory Sync - Solid Pod Memory for AI Agents (49 agent types)

Usage:
  node jss-memory-sync.js test-pre                    # Test pre-task hook
  node jss-memory-sync.js test-post                   # Test post-task hook
  node jss-memory-sync.js test-session                # Test session-end hook
  node jss-memory-sync.js search <query>              # Search memories

Inter-Agent Communication:
  node jss-memory-sync.js send-message <from> <to> <msg>  # Send inbox message
  node jss-memory-sync.js read-inbox <agent>              # Read agent inbox
  node jss-memory-sync.js share-knowledge <from> <to> <topic>  # Share knowledge

Agent Registry:
  node jss-memory-sync.js register <type> <capabilities>  # Register agent
  node jss-memory-sync.js list-agents                     # List all agent types

Environment:
  SOLID_POD_URL          - Solid Pod URL (default: http://localhost:3000)
  NOSTR_SESSION_TOKEN    - Nostr session token for authentication
  JSS_CACHE_DIR          - Local cache directory
  JSS_DEBUG              - Enable debug output (true/false)

Pod Structure:
  /agents/                    - Shared coordination namespace
    registry.jsonld           - Agent registry
    schemas/                  - JSON-LD contexts
    coordination/             - Swarm state

  /agent-{type}/              - Per-agent-type pods (49 types)
    memories/
      episodic/               - Task memories
      semantic/               - Factual knowledge
      procedural/             - Learned patterns
    inbox/                    - Inter-agent messages
    shared/                   - Cross-agent data
    sessions/                 - Session summaries

Agent Categories (9):
  core-development            - coder, reviewer, tester, planner, researcher
  swarm-coordination          - hierarchical-coordinator, mesh-coordinator, ...
  consensus-distributed       - byzantine-coordinator, raft-manager, ...
  performance-optimization    - perf-analyzer, task-orchestrator, ...
  github-repository           - github-modes, pr-manager, code-review-swarm, ...
  sparc-methodology           - sparc-coord, specification, architecture, ...
  specialized-development     - backend-dev, ml-developer, system-architect, ...
  testing-validation          - tdd-london-swarm, production-validator
  migration-planning          - migration-planner, swarm-init

Integration:
  const hooks = require('./jss-memory-sync');

  // Standard hooks
  await hooks['pre-task']({ taskId, agentType, description });
  await hooks['post-task']({ taskId, agentType, sessionId, result });
  await hooks['session-restore']({ sessionId, agentType });
  await hooks['session-end']({ sessionId, agentType, exportMetrics: true });

  // Inter-agent communication
  await hooks.sendInboxMessage({ fromAgent, toAgent, content });
  const messages = await hooks.readInbox({ agentType });

  // Knowledge sharing
  await hooks.shareKnowledge({ fromAgent, toAgents: ['coder'], knowledge });
  const knowledge = await hooks.getSharedKnowledge({ sharedId, sourceAgent });

  // Agent registry
  await hooks.registerAgent({ agentType, capabilities });
  await hooks.agentHeartbeat({ agentId, status, metrics });
        `);
    }
  })();
}
