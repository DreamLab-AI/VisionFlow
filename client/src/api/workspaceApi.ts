// Workspace API Client - Complete CRUD operations with error handling and optimistic updates
import { createLogger } from '../utils/loggerConfig';
import { unifiedApiClient, isApiError, ApiError } from '../services/api/UnifiedApiClient';

const API_BASE = '/workspace';
const logger = createLogger('WorkspaceAPI');

// Workspace data types
export interface Workspace {
  id: string;
  name: string;
  description: string;
  type: 'personal' | 'team' | 'public';
  status: 'active' | 'archived';
  memberCount: number;
  lastAccessed: Date;
  createdAt: Date;
  updatedAt: Date;
  favorite: boolean;
  settings?: WorkspaceSettings;
}

export interface WorkspaceSettings {
  autoSave: boolean;
  syncEnabled: boolean;
  collaborationEnabled: boolean;
  backupEnabled: boolean;
  maxMembers: number;
}

export interface CreateWorkspaceRequest {
  name: string;
  description: string;
  type: 'personal' | 'team' | 'public';
  settings?: Partial<WorkspaceSettings>;
}

export interface UpdateWorkspaceRequest {
  name?: string;
  description?: string;
  type?: 'personal' | 'team' | 'public';
  settings?: Partial<WorkspaceSettings>;
}

// API Response types
export interface WorkspaceApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedWorkspaceResponse {
  workspaces: Workspace[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

// Custom error class for workspace operations
export class WorkspaceApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public data?: any
  ) {
    super(message);
    this.name = 'WorkspaceApiError';
  }

  static fromApiError(error: ApiError): WorkspaceApiError {
    return new WorkspaceApiError(error.message, error.status, error.data);
  }
}

// Request configuration with proper headers and error handling
const createRequest = async <T>(url: string, method: string = 'GET', data?: any): Promise<T> => {
  try {
    const response = await unifiedApiClient.request<T>(method, `${API_BASE}${url}`, data, {
      headers: { 'Accept': 'application/json' },
    });

    let responseData = response.data;

    
    if (responseData?.data) {
      responseData.data = transformDates(responseData.data);
    } else if (Array.isArray(responseData)) {
      responseData = responseData.map(transformDates);
    }

    return responseData;
  } catch (error) {
    if (isApiError(error)) {
      logger.error('API request failed', {
        url,
        status: error.status,
        statusText: error.statusText,
        error: error.message,
        data: error.data
      });
      throw WorkspaceApiError.fromApiError(error);
    }

    logger.error('Network error in workspace API', { url, error: error.message });
    throw new WorkspaceApiError(
      `Network error: ${error.message}`,
      0,
      error
    );
  }
};

// Helper to transform date strings to Date objects
const transformDates = (item: any): any => {
  if (!item || typeof item !== 'object') return item;

  const result = { ...item };
  if (result.createdAt && typeof result.createdAt === 'string') {
    result.createdAt = new Date(result.createdAt);
  }
  if (result.updatedAt && typeof result.updatedAt === 'string') {
    result.updatedAt = new Date(result.updatedAt);
  }
  if (result.lastAccessed && typeof result.lastAccessed === 'string') {
    result.lastAccessed = new Date(result.lastAccessed);
  }

  return result;
};

// Workspace API operations
export const workspaceApi = {
  
  async fetchWorkspaces(params?: {
    page?: number;
    limit?: number;
    status?: 'active' | 'archived';
    type?: 'personal' | 'team' | 'public';
    favorite?: boolean;
  }): Promise<PaginatedWorkspaceResponse> {
    const searchParams = new URLSearchParams();

    if (params?.page) searchParams.set('page', params.page.toString());
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    if (params?.status) searchParams.set('status', params.status);
    if (params?.type) searchParams.set('type', params.type);
    if (params?.favorite !== undefined) searchParams.set('favorite', params.favorite.toString());

    const url = `/list${searchParams.toString() ? `?${searchParams}` : ''}`;

    logger.info('Fetching workspaces', { params });

    const response = await createRequest<WorkspaceApiResponse<PaginatedWorkspaceResponse>>(url, 'GET');

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to fetch workspaces');
    }

    return response.data!;
  },

  
  async createWorkspace(data: CreateWorkspaceRequest): Promise<Workspace> {
    
    if (!data.name?.trim()) {
      throw new WorkspaceApiError('Workspace name is required');
    }

    if (data.name.length > 100) {
      throw new WorkspaceApiError('Workspace name must be less than 100 characters');
    }

    logger.info('Creating workspace', { name: data.name, type: data.type });

    const response = await createRequest<WorkspaceApiResponse<Workspace>>('/create', 'POST', data);

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to create workspace');
    }

    logger.info('Workspace created successfully', { id: response.data!.id, name: response.data!.name });
    return response.data!;
  },

  
  async updateWorkspace(id: string, data: UpdateWorkspaceRequest): Promise<Workspace> {
    if (!id) {
      throw new WorkspaceApiError('Workspace ID is required');
    }

    
    if (data.name && !data.name.trim()) {
      throw new WorkspaceApiError('Workspace name cannot be empty');
    }

    if (data.name && data.name.length > 100) {
      throw new WorkspaceApiError('Workspace name must be less than 100 characters');
    }

    logger.info('Updating workspace', { id, updates: Object.keys(data) });

    const response = await createRequest<WorkspaceApiResponse<Workspace>>(`/${id}`, 'PUT', data);

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to update workspace');
    }

    logger.info('Workspace updated successfully', { id, name: response.data!.name });
    return response.data!;
  },

  
  async deleteWorkspace(id: string): Promise<void> {
    if (!id) {
      throw new WorkspaceApiError('Workspace ID is required');
    }

    logger.info('Deleting workspace', { id });

    const response = await createRequest<WorkspaceApiResponse>(`/${id}`, 'DELETE');

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to delete workspace');
    }

    logger.info('Workspace deleted successfully', { id });
  },

  
  async toggleFavorite(id: string): Promise<Workspace> {
    if (!id) {
      throw new WorkspaceApiError('Workspace ID is required');
    }

    logger.info('Toggling workspace favorite status', { id });

    const response = await createRequest<WorkspaceApiResponse<Workspace>>(`/${id}/favorite`, 'POST');

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to toggle favorite status');
    }

    logger.info('Workspace favorite status toggled', { id, favorite: response.data!.favorite });
    return response.data!;
  },

  
  async archiveWorkspace(id: string, archive: boolean = true): Promise<Workspace> {
    if (!id) {
      throw new WorkspaceApiError('Workspace ID is required');
    }

    logger.info('Updating workspace archive status', { id, archive });

    const response = await createRequest<WorkspaceApiResponse<Workspace>>(`/${id}/archive`, 'POST', { archive });

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to update archive status');
    }

    logger.info('Workspace archive status updated', { id, status: response.data!.status });
    return response.data!;
  },

  
  async getWorkspace(id: string): Promise<Workspace> {
    if (!id) {
      throw new WorkspaceApiError('Workspace ID is required');
    }

    logger.info('Fetching workspace details', { id });

    const response = await createRequest<WorkspaceApiResponse<Workspace>>(`/${id}`, 'GET');

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to fetch workspace');
    }

    return response.data!;
  },

  
  async updateWorkspaceSettings(id: string, settings: Partial<WorkspaceSettings>): Promise<Workspace> {
    if (!id) {
      throw new WorkspaceApiError('Workspace ID is required');
    }

    logger.info('Updating workspace settings', { id, settings: Object.keys(settings) });

    const response = await createRequest<WorkspaceApiResponse<Workspace>>(`/${id}/settings`, 'PUT', settings);

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to update workspace settings');
    }

    logger.info('Workspace settings updated', { id });
    return response.data!;
  },

  
  async getWorkspaceMembers(id: string): Promise<any[]> {
    if (!id) {
      throw new WorkspaceApiError('Workspace ID is required');
    }

    logger.info('Fetching workspace members', { id });

    const response = await createRequest<WorkspaceApiResponse<any[]>>(`/${id}/members`, 'GET');

    if (!response.success) {
      throw new WorkspaceApiError(response.error || 'Failed to fetch workspace members');
    }

    return response.data!;
  }
};

export default workspaceApi;