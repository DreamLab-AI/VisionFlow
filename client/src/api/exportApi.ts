

import { unifiedApiClient } from '../services/api/UnifiedApiClient';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('exportApi');

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
      // Validate download URL origin to prevent open redirect
      const allowedOrigins = [window.location.origin];
      const parsedUrl = new URL(response.data.downloadUrl, window.location.origin);
      if (!allowedOrigins.includes(parsedUrl.origin)) {
        throw new Error('Invalid download URL origin');
      }

      const link = document.createElement('a');
      link.href = parsedUrl.href;
      link.download = response.data.fileName || `graph.${options.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    return response.data;
  } catch (error) {
    logger.error('Export failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Export failed'
    };
  }
};


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
    logger.error('Share failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Share failed'
    };
  }
};


export const getSharedGraph = async (shareId: string): Promise<any> => {
  try {
    const response = await unifiedApiClient.get(`/graph/shared/${encodeURIComponent(shareId)}`);
    return response.data;
  } catch (error) {
    logger.error('Failed to retrieve shared graph:', error);
    throw error;
  }
};


export const deleteSharedGraph = async (shareId: string): Promise<boolean> => {
  try {
    await unifiedApiClient.delete(`/graph/shared/${encodeURIComponent(shareId)}`);
    return true;
  } catch (error) {
    logger.error('Failed to delete shared graph:', error);
    return false;
  }
};


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
    logger.error('Publish failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Publish failed'
    };
  }
};


export const getUserSharedGraphs = async (): Promise<any[]> => {
  try {
    const response = await unifiedApiClient.get('/graph/shared');
    return response.data.graphs || [];
  } catch (error) {
    logger.error('Failed to get shared graphs:', error);
    return [];
  }
};


export const updateShareSettings = async (
  shareId: string,
  options: ShareOptions
): Promise<ShareResponse> => {
  try {
    const response = await unifiedApiClient.put(`/graph/shared/${encodeURIComponent(shareId)}`, options);
    return response.data;
  } catch (error) {
    logger.error('Failed to update share settings:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Update failed'
    };
  }
};


/**
 * Sanitize a shareId for safe use in URLs and DOM contexts.
 * Strips anything that is not alphanumeric, hyphen, or underscore.
 */
const sanitizeShareId = (shareId: string): string => {
  return shareId.replace(/[^a-zA-Z0-9_-]/g, '');
};

export const generateEmbedCode = (
  shareId: string,
  settings: EmbedSettings = {}
): string => {
  const safeShareId = sanitizeShareId(shareId);
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
  src="${window.location.origin}/embed/${safeShareId}?${params.toString()}"
  width="${width}"
  height="${height}"
  frameborder="0"
  title="Interactive Graph Visualisation"
  allow="fullscreen">
</iframe>`;
};


export const generateApiEndpoint = (shareId: string): string => {
  return `${window.location.origin}/api/graph/shared/${encodeURIComponent(shareId)}/data`;
};


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


export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    } else {
      
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
    logger.error('Failed to copy to clipboard:', error);
    return false;
  }
};


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