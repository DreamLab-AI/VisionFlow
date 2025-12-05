/**
 * VisionFlow MCP Skill Types
 *
 * TypeScript types for VisionFlow integration with modernized MCP skills.
 * These types define the contract between VisionFlow's Rust actors and
 * the MCP servers running in the agentic-workstation container.
 */

// =============================================================================
// Skill Capability Discovery
// =============================================================================

export interface SkillCapabilities {
  name: string;
  version: string;
  protocol: 'fastmcp' | 'mcp-sdk' | 'tcp-proxy';
  tools: string[];
  visionflow_compatible: boolean;
  connection?: {
    host: string;
    port: number;
    type: 'tcp' | 'stdio';
  };
  formats_supported?: string[];
}

// =============================================================================
// ImageMagick Skill Types
// =============================================================================

export interface ImageMagickCreateParams {
  output: string;
  width?: number;
  height?: number;
  color?: string;
}

export interface ImageMagickResizeParams {
  input_path: string;
  output_path: string;
  width: number;
  height: number;
  maintain_aspect?: boolean;
  quality?: number;
}

export interface ImageMagickCropParams {
  input_path: string;
  output_path: string;
  width: number;
  height: number;
  x_offset?: number;
  y_offset?: number;
}

export interface ImageMagickBatchParams {
  input_pattern: string;
  output_dir: string;
  operation: 'resize' | 'convert' | 'thumbnail';
  format?: string;
  width?: number;
  height?: number;
}

export interface ImageMagickResult {
  success: boolean;
  stdout?: string;
  stderr?: string;
  command?: string;
  error?: string;
  message?: string;
}

// =============================================================================
// QGIS Skill Types
// =============================================================================

export interface QGISLoadLayerParams {
  path: string;
  name?: string;
  provider?: string;
}

export interface QGISBufferParams {
  layer_name: string;
  distance: number;
  segments?: number;
  output_name?: string;
}

export interface QGISDistanceParams {
  point1: [number, number];
  point2: [number, number];
  crs?: string;
}

export interface QGISTransformParams {
  coordinates: [number, number] | [number, number, number];
  source_crs: string;
  target_crs: string;
}

export interface QGISExportMapParams {
  output_path: string;
  width?: number;
  height?: number;
  dpi?: number;
  extent?: [number, number, number, number];
}

export interface QGISGeoprocessingParams {
  operation: 'intersect' | 'union' | 'difference' | 'dissolve' | 'clip';
  input_layer: string;
  overlay_layer?: string;
  output_name?: string;
}

export interface QGISResult {
  success: boolean;
  result?: unknown;
  error?: string;
}

// =============================================================================
// Playwright Skill Types
// =============================================================================

export interface PlaywrightNavigateParams {
  url: string;
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle';
}

export interface PlaywrightScreenshotParams {
  filename?: string;
  fullPage?: boolean;
  returnBase64?: boolean;
}

export interface PlaywrightClickParams {
  selector: string;
  button?: 'left' | 'right' | 'middle';
  clickCount?: number;
}

export interface PlaywrightTypeParams {
  selector: string;
  text: string;
}

export interface PlaywrightEvaluateParams {
  script: string;
}

export interface PlaywrightWaitParams {
  selector: string;
  state?: 'visible' | 'hidden' | 'attached' | 'detached';
  timeout?: number;
}

export interface PlaywrightResult {
  success: boolean;
  url?: string;
  title?: string;
  path?: string;
  base64?: string;
  content?: string;
  result?: unknown;
  error?: string;
}

// =============================================================================
// Web Summary Skill Types
// =============================================================================

export interface WebSummarySummarizeParams {
  url: string;
  length?: 'short' | 'medium' | 'long';
  include_topics?: boolean;
  format?: 'markdown' | 'plain' | 'logseq' | 'obsidian';
}

export interface WebSummaryTranscriptParams {
  video_id: string;
  language?: string;
}

export interface WebSummaryTopicsParams {
  text: string;
  max_topics?: number;
  format?: 'logseq' | 'obsidian' | 'plain';
}

export interface WebSummaryResult {
  success: boolean;
  url?: string;
  source_type?: 'youtube' | 'webpage';
  summary?: string;
  topics?: string[];
  topics_formatted?: string;
  transcript?: string;
  video_id?: string;
  language?: string;
  segments?: number;
  error?: string;
}

// =============================================================================
// ComfyUI Skill Types
// =============================================================================

export interface ComfyUIWorkflowParams {
  workflow: Record<string, unknown>;
  client_id?: string;
}

export interface ComfyUIGenerateParams {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  seed?: number;
  model?: string;
}

export interface ComfyUIResult {
  success: boolean;
  prompt_id?: string;
  status?: string;
  outputs?: string[];
  images?: Array<{ filename: string; subfolder: string; type: string }>;
  error?: string;
}

// =============================================================================
// VisionFlow MCP Client Types
// =============================================================================

export interface McpTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface McpResource {
  uri: string;
  name: string;
  description: string;
  mimeType: string;
}

export interface McpToolCallRequest {
  jsonrpc: '2.0';
  id: number;
  method: 'tools/call';
  params: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

export interface McpToolCallResponse {
  jsonrpc: '2.0';
  id: number;
  result?: {
    content: Array<{
      type: 'text' | 'image' | 'resource';
      text?: string;
      data?: string;
      mimeType?: string;
    }>;
    isError?: boolean;
  };
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

// =============================================================================
// VisionFlow Actor Message Types
// =============================================================================

export interface SkillInvocationRequest {
  skill_name: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  timeout_ms?: number;
  session_id?: string;
}

export interface SkillInvocationResponse {
  success: boolean;
  skill_name: string;
  tool_name: string;
  result?: unknown;
  error?: string;
  duration_ms: number;
}

export interface SkillDiscoveryRequest {
  skill_name?: string; // If null, discover all skills
}

export interface SkillDiscoveryResponse {
  skills: SkillCapabilities[];
}
