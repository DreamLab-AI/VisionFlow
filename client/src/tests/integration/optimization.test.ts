/**
 * Optimization Integration Tests
 * Tests for optimization triggers, cancellation, result retrieval, and long-running operations
 *
 * NOTE: These tests are prepared but disabled due to security concerns
 * with testing dependencies. Re-enable once security issues are resolved.
 */

import { describe, it, expect, beforeEach, afterEach, vi, beforeAll, afterAll } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GraphOptimisationTab } from '@/features/visualisation/components/tabs/GraphOptimisationTab';
import { createMockApiClient } from '../__mocks__/apiClient';
import { createMockWebSocket } from '../__mocks__/websocket';

// Mock toast notifications
vi.mock('@/features/design-system/components/Toast', () => ({
  toast: vi.fn()
}));

// Mock optimization API responses
const mockOptimizationConfig = {
  algorithms: ['force-directed', 'hierarchical', 'circular', 'grid', 'adaptive'],
  performance: {
    gpuAccelerated: true,
    parallelProcessing: true,
    memoryOptimized: true
  },
  limits: {
    maxIterations: 1000,
    convergenceThreshold: 0.001,
    timeoutMs: 30000
  }
};

const mockOptimizationResult = {
  algorithm: 'Adaptive Force-Directed',
  confidence: 0.87,
  performanceGain: 0.34,
  clusters: 8,
  iterations: 245,
  convergenceReached: true,
  executionTime: 12500,
  recommendations: [
    {
      type: 'layout',
      priority: 'high',
      description: 'Adjust node spacing for better clarity',
      impact: 0.25,
      estimatedImprovement: '25% better readability'
    },
    {
      type: 'clustering',
      priority: 'medium',
      description: 'Group related nodes for improved navigation',
      impact: 0.18,
      estimatedImprovement: '18% faster navigation'
    },
    {
      type: 'performance',
      priority: 'low',
      description: 'Enable GPU acceleration for smoother interactions',
      impact: 0.32,
      estimatedImprovement: '32% performance boost'
    }
  ],
  metrics: {
    stressMajorization: 0.0234,
    edgeCrossings: 45,
    nodeOverlaps: 2,
    symmetryScore: 0.78
  }
};

const mockProgressData = {
  taskId: 'optimization-123',
  status: 'running',
  progress: 0,
  currentIteration: 0,
  estimatedTimeRemaining: 15000,
  memoryUsage: 256 * 1024 * 1024, // 256MB
  gpuUtilization: 85.2
};

describe('Optimization Integration Tests', () => {
  let mockApiClient: any;
  let mockWebSocket: any;

  beforeAll(() => {
    // Setup performance API
    Object.defineProperty(performance, 'now', {
      value: vi.fn(() => Date.now())
    });

    // Mock GPU memory info
    Object.defineProperty(navigator, 'deviceMemory', {
      value: 8 // 8GB RAM
    });
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mockApiClient = createMockApiClient();
    mockWebSocket = createMockWebSocket();

    // Setup optimization API endpoints
    mockApiClient.get.mockImplementation((url: string) => {
      switch (url) {
        case '/api/optimization/config':
          return Promise.resolve({ data: mockOptimizationConfig });
        case '/api/optimization/status':
          return Promise.resolve({ data: { running: false, queue: [] } });
        default:
          return Promise.reject(new Error(`Unexpected GET: ${url}`));
      }
    });

    mockApiClient.post.mockImplementation((url: string, data: any) => {
      switch (url) {
        case '/api/optimization/layout':
          return Promise.resolve({
            data: { taskId: 'layout-123', status: 'queued' }
          });
        case '/api/optimization/clustering':
          return Promise.resolve({
            data: { taskId: 'clustering-123', status: 'queued' }
          });
        case '/api/optimization/stress-majorization':
          return Promise.resolve({
            data: { taskId: 'stress-123', status: 'queued' }
          });
        default:
          return Promise.reject(new Error(`Unexpected POST: ${url}`));
      }
    });

    mockApiClient.delete.mockImplementation((url: string) => {
      if (url.includes('/api/optimization/cancel/')) {
        return Promise.resolve({ data: { cancelled: true } });
      }
      return Promise.reject(new Error(`Unexpected DELETE: ${url}`));
    });
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('Optimization Triggers', () => {
    it('should trigger layout optimization via API', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      // Enable AI insights first
      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      // Trigger layout optimization
      const layoutButton = await screen.findByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/optimization/layout', {
          graphId: 'test-graph',
          algorithm: 'force-directed',
          level: 3,
          autoOptimize: false
        });
      });
    });

    it('should trigger clustering analysis via API', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const clusteringButton = await screen.findByRole('button', { name: /clustering/i });
      await user.click(clusteringButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/optimization/clustering', {
          graphId: 'test-graph',
          algorithm: 'community-detection',
          parameters: expect.any(Object)
        });
      });
    });

    it('should configure optimization parameters before triggering', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      // Enable AI insights
      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      // Change optimization level
      const levelSlider = screen.getByRole('slider');
      fireEvent.change(levelSlider, { target: { value: '5' } });

      // Change algorithm
      const algorithmSelect = screen.getByRole('combobox');
      await user.click(algorithmSelect);
      await user.click(screen.getByText('Adaptive (AI)'));

      // Trigger optimization
      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/optimization/layout', {
          graphId: 'test-graph',
          algorithm: 'adaptive',
          level: 5,
          autoOptimize: false
        });
      });
    });

    it('should handle GPU acceleration settings', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab
          graphId="test-graph"
          onFeatureUpdate={vi.fn()}
        />
      );

      // Change performance mode to extreme
      const performanceSelect = screen.getByDisplayValue('Balanced');
      await user.click(performanceSelect);
      await user.click(screen.getByText('Extreme Performance'));

      // Enable AI insights and trigger optimization
      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/optimization/layout', {
          graphId: 'test-graph',
          algorithm: 'force-directed',
          level: 3,
          autoOptimize: false,
          performance: {
            mode: 'extreme',
            gpuAccelerated: true,
            parallelProcessing: true
          }
        });
      });
    });
  });

  describe('Long-running Operations', () => {
    it('should show progress during optimization', async () => {
      const user = userEvent.setup();

      // Mock progressive API responses
      let callCount = 0;
      mockApiClient.post.mockImplementation(() => {
        callCount++;
        return Promise.resolve({
          data: { taskId: `task-${callCount}`, status: 'queued' }
        });
      });

      // Mock progress endpoint
      mockApiClient.get.mockImplementation((url: string) => {
        if (url.includes('/api/optimization/progress/')) {
          return Promise.resolve({
            data: {
              ...mockProgressData,
              progress: Math.min(callCount * 20, 100)
            }
          });
        }
        return Promise.resolve({ data: mockOptimizationConfig });
      });

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      // Enable AI insights
      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      // Start optimization
      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      // Should show progress bar
      await waitFor(() => {
        expect(screen.getByRole('progressbar')).toBeInTheDocument();
      });

      // Should show progress percentage
      await waitFor(() => {
        expect(screen.getByText(/\d+%/)).toBeInTheDocument();
      });

      // Button should be disabled during optimization
      expect(layoutButton).toBeDisabled();
      expect(screen.getByText('Optimising...')).toBeInTheDocument();
    });

    it('should receive real-time progress updates via WebSocket', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      // Start optimization
      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      // Simulate WebSocket progress updates
      const progressUpdates = [
        { progress: 25, currentIteration: 62, estimatedTimeRemaining: 10000 },
        { progress: 50, currentIteration: 125, estimatedTimeRemaining: 7500 },
        { progress: 75, currentIteration: 187, estimatedTimeRemaining: 3750 },
        { progress: 100, currentIteration: 245, estimatedTimeRemaining: 0 }
      ];

      for (const update of progressUpdates) {
        const progressMessage = {
          type: 'optimization_progress',
          data: {
            taskId: 'task-1',
            ...update
          }
        };

        mockWebSocket.onmessage?.({
          data: JSON.stringify(progressMessage)
        } as MessageEvent);

        await waitFor(() => {
          expect(screen.getByText(`${update.progress}%`)).toBeInTheDocument();
        });
      }

      // Final completion should show results
      await waitFor(() => {
        expect(screen.getByText('Optimisation Complete')).toBeInTheDocument();
      });
    });

    it('should handle optimization timeout', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      // Mock timeout response
      mockApiClient.post.mockImplementation(() =>
        new Promise((resolve) => {
          setTimeout(() => resolve({
            data: { taskId: 'timeout-task', status: 'timeout' }
          }), 100);
        })
      );

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: expect.stringContaining('Timeout'),
            variant: 'destructive'
          })
        );
      }, { timeout: 5000 });
    });
  });

  describe('Cancellation', () => {
    it('should allow cancellation of running optimization', async () => {
      const user = userEvent.setup();

      // Mock long-running task
      mockApiClient.post.mockResolvedValue({
        data: { taskId: 'long-task', status: 'running' }
      });

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      // Should show cancel button during optimization
      await waitFor(() => {
        expect(screen.getByText(/cancel/i)).toBeInTheDocument();
      });

      const cancelButton = screen.getByText(/cancel/i);
      await user.click(cancelButton);

      await waitFor(() => {
        expect(mockApiClient.delete).toHaveBeenCalledWith('/api/optimization/cancel/long-task');
      });

      // Button should be re-enabled after cancellation
      await waitFor(() => {
        expect(layoutButton).not.toBeDisabled();
        expect(screen.getByText('Layout')).toBeInTheDocument();
      });
    });

    it('should confirm cancellation for long-running tasks', async () => {
      const user = userEvent.setup();

      // Mock confirmation dialog
      const mockConfirm = vi.spyOn(window, 'confirm').mockReturnValue(true);

      mockApiClient.post.mockResolvedValue({
        data: { taskId: 'important-task', status: 'running' }
      });

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      const cancelButton = await screen.findByText(/cancel/i);
      await user.click(cancelButton);

      expect(mockConfirm).toHaveBeenCalledWith(
        'Are you sure you want to cancel the optimization? Progress will be lost.'
      );

      mockConfirm.mockRestore();
    });

    it('should handle cancellation failure gracefully', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      mockApiClient.post.mockResolvedValue({
        data: { taskId: 'stuck-task', status: 'running' }
      });

      // Make cancellation fail
      mockApiClient.delete.mockRejectedValue(new Error('Cancellation failed'));

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      const cancelButton = await screen.findByText(/cancel/i);
      await user.click(cancelButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: expect.stringContaining('Cancellation Failed'),
            variant: 'destructive'
          })
        );
      });
    });
  });

  describe('Result Retrieval', () => {
    it('should retrieve and display optimization results', async () => {
      const user = userEvent.setup();

      // Mock successful optimization
      mockApiClient.post.mockResolvedValue({
        data: { taskId: 'success-task', status: 'queued' }
      });

      mockApiClient.get.mockImplementation((url: string) => {
        if (url.includes('/api/optimization/result/')) {
          return Promise.resolve({ data: mockOptimizationResult });
        }
        return Promise.resolve({ data: mockOptimizationConfig });
      });

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      // Simulate completion via WebSocket
      const completionMessage = {
        type: 'optimization_complete',
        data: {
          taskId: 'success-task',
          result: mockOptimizationResult
        }
      };

      mockWebSocket.onmessage?.({
        data: JSON.stringify(completionMessage)
      } as MessageEvent);

      // Should display results
      await waitFor(() => {
        expect(screen.getByText('87%')).toBeInTheDocument(); // Confidence
        expect(screen.getByText('+34%')).toBeInTheDocument(); // Performance gain
        expect(screen.getByText('8')).toBeInTheDocument(); // Clusters
      });
    });

    it('should display AI recommendations from optimization results', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      // Simulate completion with recommendations
      const completionMessage = {
        type: 'optimization_complete',
        data: {
          taskId: 'success-task',
          result: mockOptimizationResult
        }
      };

      mockWebSocket.onmessage?.({
        data: JSON.stringify(completionMessage)
      } as MessageEvent);

      // Should show recommendations
      await waitFor(() => {
        expect(screen.getByText('AI Recommendations')).toBeInTheDocument();
        expect(screen.getByText('Adjust node spacing for better clarity')).toBeInTheDocument();
        expect(screen.getByText('high')).toBeInTheDocument(); // Priority badge
      });

      // Should allow applying recommendations
      const applyButtons = screen.getAllByText('Apply');
      expect(applyButtons).toHaveLength(3); // One for each recommendation

      await user.click(applyButtons[0]);

      // Should trigger recommendation application
      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/optimization/apply-recommendation', {
          type: 'layout',
          description: 'Adjust node spacing for better clarity'
        });
      });
    });

    it('should cache optimization results for reuse', async () => {
      const user = userEvent.setup();

      mockApiClient.get.mockImplementation((url: string) => {
        if (url.includes('/api/optimization/result/')) {
          return Promise.resolve({ data: mockOptimizationResult });
        }
        return Promise.resolve({ data: mockOptimizationConfig });
      });

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      // First optimization
      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(1);
      });

      // Complete optimization
      const completionMessage = {
        type: 'optimization_complete',
        data: { taskId: 'task-1', result: mockOptimizationResult }
      };

      mockWebSocket.onmessage?.({
        data: JSON.stringify(completionMessage)
      } as MessageEvent);

      // Results should be cached and immediately available
      await waitFor(() => {
        expect(screen.getByText('87%')).toBeInTheDocument();
      });

      // Second identical optimization should use cached results
      await user.click(layoutButton);

      // Should not make another API call
      expect(mockApiClient.post).toHaveBeenCalledTimes(1);

      // Results should still be displayed
      expect(screen.getByText('87%')).toBeInTheDocument();
    });
  });

  describe('Performance Tests', () => {
    it('should handle multiple concurrent optimizations', async () => {
      const user = userEvent.setup();

      mockApiClient.post.mockImplementation((url: string) => {
        const taskId = `task-${Date.now()}-${Math.random()}`;
        return Promise.resolve({
          data: { taskId, status: 'queued' }
        });
      });

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      // Start multiple optimizations rapidly
      const layoutButton = screen.getByRole('button', { name: /layout/i });
      const clusteringButton = screen.getByRole('button', { name: /clustering/i });

      await user.click(layoutButton);
      await user.click(clusteringButton);

      // Should handle concurrent requests
      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(2);
      });
    });

    it('should monitor memory usage during optimization', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      // Simulate high memory usage update
      const memoryUpdate = {
        type: 'optimization_memory',
        data: {
          taskId: 'task-1',
          memoryUsage: 1024 * 1024 * 1024, // 1GB
          memoryLimit: 2 * 1024 * 1024 * 1024 // 2GB
        }
      };

      mockWebSocket.onmessage?.({
        data: JSON.stringify(memoryUpdate)
      } as MessageEvent);

      await waitFor(() => {
        expect(screen.getByText(/1\.0 GB/)).toBeInTheDocument();
        expect(screen.getByText(/50%/)).toBeInTheDocument(); // 1GB/2GB = 50%
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle optimization API errors', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      mockApiClient.post.mockRejectedValue(new Error('GPU compute unavailable'));

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: expect.stringContaining('Optimization Failed'),
            variant: 'destructive'
          })
        );
      });
    });

    it('should recover from WebSocket disconnection during optimization', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      // Simulate WebSocket disconnection
      mockWebSocket.readyState = WebSocket.CLOSED;
      mockWebSocket.onclose?.({ code: 1006, reason: 'Connection lost' } as CloseEvent);

      // Should attempt to reconnect and poll for progress
      await waitFor(() => {
        expect(mockApiClient.get).toHaveBeenCalledWith('/api/optimization/progress/task-1');
      });
    });
  });

  describe('Accessibility', () => {
    it('should announce optimization progress to screen readers', async () => {
      const user = userEvent.setup();

      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      const aiInsightsToggle = screen.getByLabelText('Enable AI Insights');
      await user.click(aiInsightsToggle);

      const layoutButton = screen.getByRole('button', { name: /layout/i });
      await user.click(layoutButton);

      // Should have aria-live region for progress updates
      await waitFor(() => {
        expect(screen.getByRole('status')).toBeInTheDocument();
      });

      // Should announce progress changes
      const progressMessage = {
        type: 'optimization_progress',
        data: { taskId: 'task-1', progress: 50 }
      };

      mockWebSocket.onmessage?.({
        data: JSON.stringify(progressMessage)
      } as MessageEvent);

      await waitFor(() => {
        expect(screen.getByText(/50%/)).toBeInTheDocument();
      });
    });

    it('should have proper ARIA labels for optimization controls', () => {
      render(
        <GraphOptimisationTab graphId="test-graph" />
      );

      expect(screen.getByLabelText('Enable AI Insights')).toBeInTheDocument();
      expect(screen.getByLabelText('Optimisation Level')).toBeInTheDocument();
      expect(screen.getByRole('slider')).toHaveAttribute('aria-valuemin', '1');
      expect(screen.getByRole('slider')).toHaveAttribute('aria-valuemax', '5');
    });
  });
});