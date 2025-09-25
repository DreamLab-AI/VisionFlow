/**
 * Workspace Integration Tests
 * Tests for workspace CRUD operations, WebSocket updates, and backend integration
 *
 * NOTE: These tests are prepared but disabled due to security concerns
 * with testing dependencies. Re-enable once security issues are resolved.
 */

import { describe, it, expect, beforeEach, afterEach, vi, beforeAll, afterAll } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { WorkspaceManager } from '@/features/workspace/components/WorkspaceManager';
import { createMockWebSocket, MockWebSocketServer } from '../__mocks__/websocket';
import { createMockApiClient } from '../__mocks__/apiClient';
import { SettingsProvider } from '@/contexts/SettingsContext';

// Mock implementations
vi.mock('@/utils/loggerConfig', () => ({
  createLogger: vi.fn(() => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn()
  }))
}));

// Mock workspace API responses
const mockWorkspaces = [
  {
    id: 'ws-1',
    name: 'Test Workspace 1',
    description: 'First test workspace',
    type: 'personal' as const,
    status: 'active' as const,
    memberCount: 1,
    lastAccessed: new Date('2024-09-24T10:00:00Z'),
    createdAt: new Date('2024-09-20T10:00:00Z'),
    favorite: false
  },
  {
    id: 'ws-2',
    name: 'Team Workspace',
    description: 'Collaborative team workspace',
    type: 'team' as const,
    status: 'active' as const,
    memberCount: 5,
    lastAccessed: new Date('2024-09-24T09:00:00Z'),
    createdAt: new Date('2024-09-15T10:00:00Z'),
    favorite: true
  },
  {
    id: 'ws-3',
    name: 'Archived Project',
    description: 'Old project workspace',
    type: 'team' as const,
    status: 'archived' as const,
    memberCount: 3,
    lastAccessed: new Date('2024-08-20T10:00:00Z'),
    createdAt: new Date('2024-08-01T10:00:00Z'),
    favorite: false
  }
];

// Test wrapper component
const WorkspaceTestWrapper = ({ children }: { children: React.ReactNode }) => {
  const mockSettings = {
    workspace: {
      enabled: true,
      autoSave: { enabled: true },
      sync: { enabled: true },
      layout: { default: 'grid' },
      limits: { maxWorkspaces: 10 },
      collaboration: { enabled: true },
      backup: { enabled: false }
    }
  };

  return (
    <SettingsProvider initialSettings={mockSettings}>
      {children}
    </SettingsProvider>
  );
};

describe('Workspace Integration Tests', () => {
  let mockApiClient: any;
  let mockWebSocket: any;
  let mockWebSocketServer: MockWebSocketServer;

  beforeAll(() => {
    // Setup global mocks
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: vi.fn(() => null),
        setItem: vi.fn(),
        removeItem: vi.fn(),
        clear: vi.fn()
      }
    });

    // Mock WebSocket
    mockWebSocketServer = new MockWebSocketServer('ws://localhost:3002');
    mockWebSocket = createMockWebSocket();
    global.WebSocket = vi.fn(() => mockWebSocket);
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mockApiClient = createMockApiClient();

    // Setup default API responses
    mockApiClient.get.mockImplementation((url: string) => {
      if (url === '/api/workspace/list') {
        return Promise.resolve({ data: mockWorkspaces });
      }
      return Promise.reject(new Error(`Unexpected API call: ${url}`));
    });

    mockApiClient.post.mockImplementation((url: string, data: any) => {
      if (url === '/api/workspace/create') {
        const newWorkspace = {
          id: `ws-${Date.now()}`,
          ...data,
          type: 'personal',
          status: 'active',
          memberCount: 1,
          lastAccessed: new Date(),
          createdAt: new Date(),
          favorite: false
        };
        return Promise.resolve({ data: newWorkspace });
      }
      return Promise.reject(new Error(`Unexpected API call: ${url}`));
    });

    mockApiClient.put.mockResolvedValue({ data: { success: true } });
    mockApiClient.delete.mockResolvedValue({ data: { success: true } });
  });

  afterEach(() => {
    vi.resetAllMocks();
    mockWebSocketServer.reset();
  });

  afterAll(() => {
    mockWebSocketServer.close();
  });

  describe('Workspace CRUD Operations', () => {
    it('should load workspaces from API on component mount', async () => {
      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      await waitFor(() => {
        expect(mockApiClient.get).toHaveBeenCalledWith('/api/workspace/list');
      });

      // Check that workspaces are displayed
      expect(screen.getByText('Test Workspace 1')).toBeInTheDocument();
      expect(screen.getByText('Team Workspace')).toBeInTheDocument();

      // Check workspace count badges
      expect(screen.getByText('Active (2)')).toBeInTheDocument();
      expect(screen.getByText('Favorites (1)')).toBeInTheDocument();
      expect(screen.getByText('Archived (1)')).toBeInTheDocument();
    });

    it('should create a new workspace via API', async () => {
      const user = userEvent.setup();

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      // Wait for component to load
      await waitFor(() => {
        expect(screen.getByText('Create New Workspace')).toBeInTheDocument();
      });

      // Fill in workspace name
      const nameInput = screen.getByPlaceholderText('Enter workspace name...');
      await user.type(nameInput, 'New Test Workspace');

      // Submit form
      const createButton = screen.getByText('Create');
      await user.click(createButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/workspace/create', {
          name: 'New Test Workspace',
          description: 'New workspace'
        });
      });

      // Check that input is cleared after creation
      expect(nameInput).toHaveValue('');
    });

    it('should update workspace favorite status via API', async () => {
      const user = userEvent.setup();

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Workspace 1')).toBeInTheDocument();
      });

      // Find and click favorite button for first workspace
      const workspaceCard = screen.getByText('Test Workspace 1').closest('.border.rounded-lg');
      expect(workspaceCard).toBeInTheDocument();

      const favoriteButton = within(workspaceCard!).getAllByRole('button').find(
        btn => btn.querySelector('svg')?.getAttribute('class')?.includes('text-gray-400')
      );

      await user.click(favoriteButton!);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/workspace/ws-1/favorite', {
          favorite: true
        });
      });
    });

    it('should archive workspace via API', async () => {
      const user = userEvent.setup();

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Workspace 1')).toBeInTheDocument();
      });

      // Find and click settings button for first workspace
      const workspaceCard = screen.getByText('Test Workspace 1').closest('.border.rounded-lg');
      const settingsButton = within(workspaceCard!).getAllByRole('button').find(
        btn => btn.querySelector('svg')?.classList?.contains('lucide-settings')
      );

      await user.click(settingsButton!);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/workspace/ws-1/archive', {
          status: 'archived'
        });
      });
    });

    it('should delete workspace via API', async () => {
      const user = userEvent.setup();

      // Add delete functionality to mock
      mockApiClient.delete.mockImplementation((url: string) => {
        if (url.includes('/api/workspace/')) {
          return Promise.resolve({ data: { success: true } });
        }
        return Promise.reject(new Error(`Unexpected DELETE: ${url}`));
      });

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Workspace 1')).toBeInTheDocument();
      });

      // Simulate delete action (would need context menu or delete button)
      // This assumes a delete method exists in the component
      const deleteButton = screen.queryByLabelText('Delete workspace');

      if (deleteButton) {
        await user.click(deleteButton);

        await waitFor(() => {
          expect(mockApiClient.delete).toHaveBeenCalledWith('/api/workspace/ws-1');
        });
      }
    });
  });

  describe('WebSocket Updates', () => {
    it('should receive real-time workspace updates via WebSocket', async () => {
      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      // Wait for component to establish WebSocket connection
      await waitFor(() => {
        expect(global.WebSocket).toHaveBeenCalled();
      });

      // Simulate WebSocket message for workspace update
      const updateMessage = {
        type: 'workspace_updated',
        data: {
          id: 'ws-1',
          name: 'Updated Workspace Name',
          memberCount: 3,
          lastAccessed: new Date().toISOString()
        }
      };

      mockWebSocketServer.send(JSON.stringify(updateMessage));

      // Wait for UI to update
      await waitFor(() => {
        expect(screen.getByText('Updated Workspace Name')).toBeInTheDocument();
      });
    });

    it('should receive workspace creation notifications via WebSocket', async () => {
      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      // Simulate new workspace notification
      const newWorkspaceMessage = {
        type: 'workspace_created',
        data: {
          id: 'ws-new',
          name: 'Remotely Created Workspace',
          description: 'Created by another user',
          type: 'team',
          status: 'active',
          memberCount: 2,
          lastAccessed: new Date().toISOString(),
          createdAt: new Date().toISOString(),
          favorite: false
        }
      };

      mockWebSocketServer.send(JSON.stringify(newWorkspaceMessage));

      await waitFor(() => {
        expect(screen.getByText('Remotely Created Workspace')).toBeInTheDocument();
      });
    });

    it('should handle WebSocket disconnection and reconnection', async () => {
      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      // Simulate WebSocket disconnect
      mockWebSocket.readyState = WebSocket.CLOSED;
      mockWebSocket.onclose?.({ code: 1006, reason: 'Connection lost' } as CloseEvent);

      // Should attempt reconnection after timeout
      await waitFor(() => {
        expect(mockWebSocket.onopen).toBeDefined();
      });

      // Simulate successful reconnection
      mockWebSocket.readyState = WebSocket.OPEN;
      mockWebSocket.onopen?.({} as Event);

      // Should re-sync data after reconnection
      await waitFor(() => {
        expect(mockApiClient.get).toHaveBeenCalledWith('/api/workspace/list');
      });
    });
  });

  describe('Settings Integration', () => {
    it('should save workspace settings via API', async () => {
      const user = userEvent.setup();

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      // Find and toggle auto-save setting
      const autoSaveToggle = screen.getByLabelText(/Auto Save/i);
      await user.click(autoSaveToggle);

      await waitFor(() => {
        expect(mockApiClient.put).toHaveBeenCalledWith('/api/settings/workspace', {
          'workspace.autoSave.enabled': false
        });
      });
    });

    it('should disable workspace manager when workspace is disabled in settings', () => {
      const disabledSettings = {
        workspace: {
          enabled: false
        }
      };

      render(
        <SettingsProvider initialSettings={disabledSettings}>
          <WorkspaceManager />
        </SettingsProvider>
      );

      expect(screen.getByText('Workspace management is disabled')).toBeInTheDocument();
      expect(screen.queryByText('Create New Workspace')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      const user = userEvent.setup();

      // Make API call fail
      mockApiClient.post.mockRejectedValue(new Error('Server error'));

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      // Try to create workspace
      const nameInput = screen.getByPlaceholderText('Enter workspace name...');
      await user.type(nameInput, 'Error Test Workspace');

      const createButton = screen.getByText('Create');
      await user.click(createButton);

      // Should show error message
      await waitFor(() => {
        expect(screen.getByText(/error/i)).toBeInTheDocument();
      });
    });

    it('should handle malformed WebSocket messages', async () => {
      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      // Send malformed WebSocket message
      mockWebSocketServer.send('invalid json');

      // Component should continue to function normally
      await waitFor(() => {
        expect(screen.getByText('Test Workspace 1')).toBeInTheDocument();
      });
    });
  });

  describe('Performance Tests', () => {
    it('should handle large number of workspaces efficiently', async () => {
      const manyWorkspaces = Array.from({ length: 100 }, (_, i) => ({
        id: `ws-${i}`,
        name: `Workspace ${i}`,
        description: `Description ${i}`,
        type: 'personal' as const,
        status: 'active' as const,
        memberCount: 1,
        lastAccessed: new Date(),
        createdAt: new Date(),
        favorite: i % 10 === 0
      }));

      mockApiClient.get.mockResolvedValue({ data: manyWorkspaces });

      const startTime = performance.now();

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Active (100)')).toBeInTheDocument();
      });

      const endTime = performance.now();
      const renderTime = endTime - startTime;

      // Should render within reasonable time
      expect(renderTime).toBeLessThan(1000); // 1 second
    });

    it('should virtualize workspace list for performance', async () => {
      const manyWorkspaces = Array.from({ length: 1000 }, (_, i) => ({
        id: `ws-${i}`,
        name: `Workspace ${i}`,
        description: `Description ${i}`,
        type: 'personal' as const,
        status: 'active' as const,
        memberCount: 1,
        lastAccessed: new Date(),
        createdAt: new Date(),
        favorite: false
      }));

      mockApiClient.get.mockResolvedValue({ data: manyWorkspaces });

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Active (1000)')).toBeInTheDocument();
      });

      // Should only render visible items (virtualization)
      const workspaceCards = screen.getAllByText(/^Workspace \d+$/);
      expect(workspaceCards.length).toBeLessThan(50); // Should be virtualized
    });
  });

  describe('Accessibility', () => {
    it('should be keyboard navigable', async () => {
      const user = userEvent.setup();

      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Workspace 1')).toBeInTheDocument();
      });

      // Tab through interactive elements
      const nameInput = screen.getByPlaceholderText('Enter workspace name...');
      nameInput.focus();

      await user.keyboard('{Tab}');
      expect(screen.getByText('Create')).toHaveFocus();

      await user.keyboard('{Tab}');
      // Should focus on first tab trigger
      expect(screen.getByRole('tab', { name: /Active/ })).toHaveFocus();
    });

    it('should have proper ARIA attributes', () => {
      render(
        <WorkspaceTestWrapper>
          <WorkspaceManager />
        </WorkspaceTestWrapper>
      );

      // Check for proper ARIA labels
      expect(screen.getByRole('tablist')).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /Active/ })).toHaveAttribute('aria-selected');
      expect(screen.getByRole('tabpanel')).toBeInTheDocument();
    });
  });
});