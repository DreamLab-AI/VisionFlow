/**
 * Graph Export API Service
 * Real export functionality for graph data
 */

import { unifiedApiClient } from '../services/api/UnifiedApiClient';

export interface ExportOptions {
  format: string;
  includeMetadata?: boolean;
  compression?: boolean;
  resolution?: string;
  quality?: number;
  embedSettings?: EmbedSettings;
}

export interface EmbedSettings {
  width?: number;
  height?: number;
  interactive?: boolean;
  showControls?: boolean;
  theme?: 'light' | 'dark' | 'auto';
}

export interface ShareOptions {
  description?: string;
  expiry?: string;
  password?: string;
  permissions?: {
    allowDownload?: boolean;
    allowComment?: boolean;
    allowEdit?: boolean;
  };
}

export interface PublishMetadata {
  title: string;
  description: string;
  tags: string[];
  category: string;
  isPublic: boolean;
  license?: string;
}

export interface ExportResponse {
  success: boolean;
  downloadUrl?: string;
  fileName?: string;
  fileSize?: number;
  message?: string;
}

export interface ShareResponse {
  success: boolean;
  shareUrl?: string;
  shareId?: string;
  expiresAt?: string;
  message?: string;
}

export interface PublishResponse {
  success: boolean;
  publishUrl?: string;
  publishId?: string;
  message?: string;
}

/**
 * Export graph in specified format
 */
export const exportGraph = async (
  graphData: any,
  options: ExportOptions
): Promise<ExportResponse> => {
  try {
    const response = await unifiedApiClient.post('/graph/export', {
      graphData,
      options
    });

    if (response.data.downloadUrl) {
      // Trigger download if URL is provided
      const link = document.createElement('a');
      link.href = response.data.downloadUrl;
      link.download = response.data.fileName || `graph.${options.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    return response.data;
  } catch (error) {
    console.error('Export failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Export failed'
    };
  }
};

/**
 * Create shareable link for graph
 */
export const shareGraph = async (
  graphData: any,
  options: ShareOptions
): Promise<ShareResponse> => {
  try {
    const response = await unifiedApiClient.post('/graph/share', {
      graphData,
      options
    });

    return response.data;
  } catch (error) {
    console.error('Share failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Share failed'
    };
  }
};

/**
 * Retrieve shared graph by ID
 */
export const getSharedGraph = async (shareId: string): Promise<any> => {
  try {
    const response = await unifiedApiClient.get(`/graph/shared/${shareId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to retrieve shared graph:', error);
    throw error;
  }
};

/**
 * Delete shared graph
 */
export const deleteSharedGraph = async (shareId: string): Promise<boolean> => {
  try {
    await unifiedApiClient.delete(`/graph/shared/${shareId}`);
    return true;
  } catch (error) {
    console.error('Failed to delete shared graph:', error);
    return false;
  }
};

/**
 * Publish graph to public repository
 */
export const publishGraph = async (
  graphData: any,
  metadata: PublishMetadata
): Promise<PublishResponse> => {
  try {
    const response = await unifiedApiClient.post('/graph/publish', {
      graphData,
      metadata
    });

    return response.data;
  } catch (error) {
    console.error('Publish failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Publish failed'
    };
  }
};

/**
 * Get list of user's shared graphs
 */
export const getUserSharedGraphs = async (): Promise<any[]> => {
  try {
    const response = await unifiedApiClient.get('/graph/shared');
    return response.data.graphs || [];
  } catch (error) {
    console.error('Failed to get shared graphs:', error);
    return [];
  }
};

/**
 * Update share settings for existing shared graph
 */
export const updateShareSettings = async (
  shareId: string,
  options: ShareOptions
): Promise<ShareResponse> => {
  try {
    const response = await unifiedApiClient.put(`/graph/shared/${shareId}`, options);
    return response.data;
  } catch (error) {
    console.error('Failed to update share settings:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Update failed'
    };
  }
};

/**
 * Generate embed code for graph
 */
export const generateEmbedCode = (
  shareId: string,
  settings: EmbedSettings = {}
): string => {
  const {
    width = 800,
    height = 600,
    interactive = true,
    showControls = true,
    theme = 'auto'
  } = settings;

  const params = new URLSearchParams({
    interactive: interactive.toString(),
    controls: showControls.toString(),
    theme
  });

  return `<iframe
  src="${window.location.origin}/embed/${shareId}?${params.toString()}"
  width="${width}"
  height="${height}"
  frameborder="0"
  title="Interactive Graph Visualisation"
  allow="fullscreen">
</iframe>`;
};

/**
 * Generate API endpoint URL for graph data
 */
export const generateApiEndpoint = (shareId: string): string => {
  return `${window.location.origin}/api/graph/shared/${shareId}/data`;
};

/**
 * Download file from blob data
 */
export const downloadFile = (
  data: Blob | string,
  filename: string,
  mimeType: string = 'application/octet-stream'
): void => {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();

  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Copy text to clipboard with fallback
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    } else {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      textArea.style.top = '-999999px';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      const result = document.execCommand('copy');
      document.body.removeChild(textArea);
      return result;
    }
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
};

/**
 * Validate export format support
 */
export const getSupportedFormats = (): string[] => {
  return [
    'json',
    'csv',
    'graphml',
    'gexf',
    'svg',
    'png',
    'pdf',
    'xlsx',
    'dot',
    'adjlist'
  ];
};

/**
 * Get file size estimate for export
 */
export const estimateExportSize = (graphData: any, format: string): number => {
  if (!graphData) return 0;

  const baseSize = JSON.stringify(graphData).length;
  const multipliers: Record<string, number> = {
    json: 1,
    csv: 0.8,
    graphml: 1.5,
    gexf: 1.3,
    svg: 2.5,
    png: 0.3,
    pdf: 1.2,
    xlsx: 0.6
  };

  return Math.round(baseSize * (multipliers[format] || 1));
};