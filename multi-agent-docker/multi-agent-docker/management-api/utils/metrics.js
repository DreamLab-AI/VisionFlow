/**
 * Prometheus metrics collector
 */

const client = require('prom-client');

// Create a Registry
const register = new client.Registry();

// Add default metrics (CPU, memory, etc.)
client.collectDefaultMetrics({ register });

// Custom metrics
const httpRequestDuration = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.001, 0.005, 0.015, 0.05, 0.1, 0.5, 1, 5]
});

const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

const activeTasks = new client.Gauge({
  name: 'active_tasks_total',
  help: 'Number of currently active tasks'
});

const completedTasks = new client.Counter({
  name: 'completed_tasks_total',
  help: 'Total number of completed tasks',
  labelNames: ['status']
});

const taskDuration = new client.Histogram({
  name: 'task_duration_seconds',
  help: 'Duration of task execution in seconds',
  labelNames: ['task_type', 'status'],
  buckets: [1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600]
});

const mcpToolInvocations = new client.Counter({
  name: 'mcp_tool_invocations_total',
  help: 'Total number of MCP tool invocations',
  labelNames: ['tool_name', 'status']
});

const mcpToolDuration = new client.Histogram({
  name: 'mcp_tool_duration_seconds',
  help: 'Duration of MCP tool execution in seconds',
  labelNames: ['tool_name'],
  buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60]
});

const workerSessions = new client.Gauge({
  name: 'worker_sessions_total',
  help: 'Number of active worker sessions'
});

const apiErrors = new client.Counter({
  name: 'api_errors_total',
  help: 'Total number of API errors',
  labelNames: ['error_type', 'route']
});

// Register custom metrics
register.registerMetric(httpRequestDuration);
register.registerMetric(httpRequestsTotal);
register.registerMetric(activeTasks);
register.registerMetric(completedTasks);
register.registerMetric(taskDuration);
register.registerMetric(mcpToolInvocations);
register.registerMetric(mcpToolDuration);
register.registerMetric(workerSessions);
register.registerMetric(apiErrors);

// Helper functions
function recordHttpRequest(method, route, statusCode, duration) {
  httpRequestDuration.observe({ method, route, status_code: statusCode }, duration);
  httpRequestsTotal.inc({ method, route, status_code: statusCode });
}

function recordTask(taskType, status, duration) {
  completedTasks.inc({ status });
  if (duration !== undefined) {
    taskDuration.observe({ task_type: taskType, status }, duration);
  }
}

function recordMCPTool(toolName, status, duration) {
  mcpToolInvocations.inc({ tool_name: toolName, status });
  if (duration !== undefined) {
    mcpToolDuration.observe({ tool_name: toolName }, duration);
  }
}

function recordError(errorType, route) {
  apiErrors.inc({ error_type: errorType, route });
}

function setActiveTasks(count) {
  activeTasks.set(count);
}

function setWorkerSessions(count) {
  workerSessions.set(count);
}

module.exports = {
  register,
  recordHttpRequest,
  recordTask,
  recordMCPTool,
  recordError,
  setActiveTasks,
  setWorkerSessions,
  metrics: {
    httpRequestDuration,
    httpRequestsTotal,
    activeTasks,
    completedTasks,
    taskDuration,
    mcpToolInvocations,
    mcpToolDuration,
    workerSessions,
    apiErrors
  }
};
