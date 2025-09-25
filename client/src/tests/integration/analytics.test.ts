/**
 * Analytics Integration Tests
 * Tests for analytics API calls, progress updates, result caching, and GPU metrics display
 *
 * NOTE: These tests are prepared but disabled due to security concerns
 * with testing dependencies. Re-enable once security issues are resolved.
 */

import { describe, it, expect, beforeEach, afterEach, vi, beforeAll, afterAll } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GraphAnalysisTab } from '@/features/visualisation/components/tabs/GraphAnalysisTab';
import { createMockApiClient } from '../__mocks__/apiClient';
import { createMockWebSocket } from '../__mocks__/websocket';
import { SettingsProvider } from '@/contexts/SettingsContext';

// Mock toast notifications
vi.mock('@/features/design-system/components/Toast', () => ({
  toast: vi.fn()
}));

// Mock analytics API responses
const mockAnalyticsParams = {
  visualAnalytics: {
    enabled: true,
    algorithm: 'stress_majorization',
    iterations: 100,
    convergenceThreshold: 0.01
  },
  gpuAcceleration: {
    enabled: true,
    device: 'CUDA:0',
    memoryUsage: 0.65,
    computeCapability: '8.6'
  }
};

const mockAnalyticsStats = {
  performance: {
    avgFrameTime: 16.7,
    gpuUtilization: 78.2,
    memoryUsage: 1024 * 1024 * 512, // 512MB
    activeOperations: 3
  },
  graph: {
    nodeCount: 1500,
    edgeCount: 3200,
    clusterCount: 8,
    centralityMetrics: {
      betweenness: 0.34,
      closeness: 0.62,
      eigenvector: 0.51
    }
  }
};

const mockStructuralAnalysis = {
  similarity: {
    overall: 0.73,
    structural: 0.68,
    semantic: 0.78
  },
  matches: 127,
  differences: 42,
  clusters: 8,
  centrality: {
    betweenness: 0.34,
    closeness: 0.62,
    eigenvector: 0.51
  },
  communities: [
    { id: 0, size: 45, density: 0.8, modularity: 0.42 },
    { id: 1, size: 32, density: 0.7, modularity: 0.38 },
    { id: 2, size: 28, density: 0.9, modularity: 0.51 }
  ]
};

const mockGraphData = {
  nodes: [
    { id: '1', label: 'Node 1', x: 0, y: 0, size: 10 },
    { id: '2', label: 'Node 2', x: 100, y: 100, size: 15 },
    { id: '3', label: 'Node 3', x: -50, y: 75, size: 8 }
  ],
  edges: [
    { id: 'e1', source: '1', target: '2', weight: 1.0 },
    { id: 'e2', source: '2', target: '3', weight: 0.8 }
  ]
};

describe('Analytics Integration Tests', () => {
  let mockApiClient: any;
  let mockWebSocket: any;

  beforeAll(() => {
    // Setup performance mocks
    Object.defineProperty(performance, 'now', {
      value: vi.fn(() => Date.now())
    });

    // Mock GPU monitoring
    global.navigator.gpu = {
      requestAdapter: vi.fn(() => Promise.resolve({
        requestDevice: vi.fn(() => Promise.resolve({
          createBuffer: vi.fn(),
          queue: { writeBuffer: vi.fn() }
        }))
      }))
    };
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mockApiClient = createMockApiClient();
    mockWebSocket = createMockWebSocket();

    // Setup analytics API endpoints
    mockApiClient.get.mockImplementation((url: string) => {
      switch (url) {
        case '/api/analytics/params':
          return Promise.resolve({ data: mockAnalyticsParams });
        case '/api/analytics/stats':
          return Promise.resolve({ data: mockAnalyticsStats });
        case '/api/analytics/constraints':
          return Promise.resolve({ data: { constraints: [] } });
        default:
          return Promise.reject(new Error(`Unexpected GET: ${url}`));
      }
    });

    mockApiClient.post.mockImplementation((url: string, data: any) => {
      switch (url) {
        case '/api/analytics/structural':
          return Promise.resolve({
            data: mockStructuralAnalysis,
            status: 'completed',
            progress: 100
          });
        case '/api/analytics/semantic':
          return Promise.resolve({
            data: { semanticSimilarity: 0.78, contentAnalysis: {} },
            status: 'completed',
            progress: 100
          });
        case '/api/analytics/clustering':
          return Promise.resolve({
            data: { clusters: mockStructuralAnalysis.communities },
            status: 'completed',
            progress: 100
          });
        case '/api/analytics/focus':
          return Promise.resolve({ data: { success: true } });
        default:
          return Promise.reject(new Error(`Unexpected POST: ${url}`));
      }
    });
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('Analytics API Integration', () => {
    it('should load analytics parameters on component mount', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      await waitFor(() => {
        expect(mockApiClient.get).toHaveBeenCalledWith('/api/analytics/params');
      });
    });

    it('should fetch performance statistics periodically', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      // Wait for initial load
      await waitFor(() => {
        expect(mockApiClient.get).toHaveBeenCalledWith('/api/analytics/stats');
      });

      // Advance time to trigger periodic updates
      vi.advanceTimersByTime(5000);

      await waitFor(() => {
        expect(mockApiClient.get).toHaveBeenCalledTimes(2); // Initial + periodic
      });
    });

    it('should trigger structural analysis via API', async () => {
      const user = userEvent.setup();

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      // Find and click structural analysis button
      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/analytics/structural', {
          graphData: mockGraphData,
          analysisType: 'comprehensive',
          includeMetrics: true
        });
      });

      // Check that results are displayed
      await waitFor(() => {
        expect(screen.getByText('73.0%')).toBeInTheDocument(); // Overall similarity
        expect(screen.getByText('127')).toBeInTheDocument(); // Node matches
      });
    });

    it('should trigger semantic analysis via API', async () => {
      const user = userEvent.setup();

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      const semanticButton = await screen.findByRole('button', { name: /semantic/i });
      await user.click(semanticButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/analytics/semantic', {
          graphData: mockGraphData,
          analysisType: 'content',
          includeEmbeddings: true
        });
      });
    });

    it('should handle API errors gracefully', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      // Make API call fail
      mockApiClient.post.mockRejectedValue(new Error('GPU compute unavailable'));

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: expect.stringContaining('Error'),
            variant: 'destructive'
          })
        );
      });
    });
  });

  describe('GPU Metrics Display', () => {
    it('should display GPU utilization metrics', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      await waitFor(() => {
        expect(screen.getByText('78.2%')).toBeInTheDocument(); // GPU utilization
      });
    });

    it('should show memory usage in readable format', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      await waitFor(() => {
        expect(screen.getByText('512 MB')).toBeInTheDocument(); // Memory usage
      });
    });

    it('should update GPU metrics in real-time', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      // Initial metrics
      await waitFor(() => {
        expect(screen.getByText('78.2%')).toBeInTheDocument();
      });

      // Update mock to return different metrics
      const updatedStats = {
        ...mockAnalyticsStats,
        performance: {
          ...mockAnalyticsStats.performance,
          gpuUtilization: 85.5
        }
      };
      mockApiClient.get.mockResolvedValue({ data: updatedStats });

      // Trigger update
      vi.advanceTimersByTime(1000);

      await waitFor(() => {
        expect(screen.getByText('85.5%')).toBeInTheDocument();
      });
    });

    it('should warn when GPU utilization is high', async () => {
      const highUtilizationStats = {
        ...mockAnalyticsStats,
        performance: {
          ...mockAnalyticsStats.performance,
          gpuUtilization: 95.0
        }
      };
      mockApiClient.get.mockResolvedValue({ data: highUtilizationStats });

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      await waitFor(() => {
        expect(screen.getByText(/high utilization/i)).toBeInTheDocument();
      });
    });
  });

  describe('Progress Updates', () => {
    it('should show progress during long-running analysis', async () => {
      const user = userEvent.setup();

      // Mock progressive responses
      let progressCallCount = 0;
      mockApiClient.post.mockImplementation(() => {
        progressCallCount++;
        const progress = Math.min(progressCallCount * 25, 100);

        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              data: progress === 100 ? mockStructuralAnalysis : null,
              status: progress === 100 ? 'completed' : 'processing',
              progress
            });
          }, 100);
        });
      });

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      // Should show progress indicator
      await waitFor(() => {
        expect(screen.getByText(/analysing/i)).toBeInTheDocument();
      });

      // Wait for completion
      await waitFor(() => {
        expect(screen.getByText('Analysis Complete')).toBeInTheDocument();
      }, { timeout: 5000 });
    });

    it('should allow cancellation of running analysis', async () => {
      const user = userEvent.setup();

      // Mock long-running operation
      mockApiClient.post.mockImplementation(() => {
        return new Promise((resolve) => {
          setTimeout(() => resolve({
            data: mockStructuralAnalysis,
            status: 'cancelled',
            progress: 50
          }), 5000);
        });
      });

      mockApiClient.delete.mockResolvedValue({ data: { cancelled: true } });

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      // Should show cancel button during processing
      await waitFor(() => {
        expect(screen.getByText(/cancel/i)).toBeInTheDocument();
      });

      const cancelButton = screen.getByText(/cancel/i);
      await user.click(cancelButton);

      await waitFor(() => {
        expect(mockApiClient.delete).toHaveBeenCalledWith('/api/analytics/cancel');
      });
    });
  });

  describe('Result Caching', () => {
    it('should cache analysis results to avoid redundant API calls', async () => {
      const user = userEvent.setup();

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      // First analysis call
      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(1);
      });

      // Second identical call should use cache
      await user.click(structuralButton);

      // Should not make another API call
      expect(mockApiClient.post).toHaveBeenCalledTimes(1);

      // Results should still be displayed
      expect(screen.getByText('73.0%')).toBeInTheDocument();
    });

    it('should invalidate cache when graph data changes', async () => {
      const user = userEvent.setup();
      const { rerender } = render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      // First analysis
      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(1);
      });

      // Change graph data
      const newGraphData = {
        ...mockGraphData,
        nodes: [...mockGraphData.nodes, { id: '4', label: 'Node 4', x: 200, y: 200, size: 12 }]
      };

      rerender(<GraphAnalysisTab graphData={newGraphData} />);

      // Second analysis with new data should make new API call
      await user.click(structuralButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(2);
      });
    });

    it('should respect cache TTL settings', async () => {
      const user = userEvent.setup();

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      // First analysis
      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(1);
      });

      // Fast forward past cache TTL (assume 5 minutes)
      vi.advanceTimersByTime(5 * 60 * 1000 + 1);

      // Second call should make new API call after TTL expiry
      await user.click(structuralButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Real-time Analytics WebSocket', () => {
    it('should connect to analytics WebSocket for real-time updates', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      await waitFor(() => {
        expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:3002/analytics');
      });
    });

    it('should receive real-time GPU metrics via WebSocket', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      // Simulate WebSocket message
      const metricsUpdate = {
        type: 'gpu_metrics',
        data: {
          utilization: 82.1,
          temperature: 65,
          memoryUsage: 1024 * 1024 * 600,
          powerDraw: 180
        }
      };

      mockWebSocket.onmessage?.({
        data: JSON.stringify(metricsUpdate)
      } as MessageEvent);

      await waitFor(() => {
        expect(screen.getByText('82.1%')).toBeInTheDocument();
        expect(screen.getByText('65°C')).toBeInTheDocument();
        expect(screen.getByText('600 MB')).toBeInTheDocument();
      });
    });

    it('should receive analysis progress updates via WebSocket', async () => {
      const user = userEvent.setup();

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      // Start analysis
      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      // Simulate progress updates via WebSocket
      const progressUpdates = [25, 50, 75, 100];

      for (const progress of progressUpdates) {
        const progressMessage = {
          type: 'analysis_progress',
          data: { taskId: 'structural-analysis', progress }
        };

        mockWebSocket.onmessage?.({
          data: JSON.stringify(progressMessage)
        } as MessageEvent);

        await waitFor(() => {
          expect(screen.getByText(`${progress}%`)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Performance Tests', () => {
    it('should handle high-frequency GPU metrics updates efficiently', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      const startTime = performance.now();

      // Send 100 rapid WebSocket updates
      for (let i = 0; i < 100; i++) {
        const metricsUpdate = {
          type: 'gpu_metrics',
          data: { utilization: 75 + i % 20 }
        };

        mockWebSocket.onmessage?.({
          data: JSON.stringify(metricsUpdate)
        } as MessageEvent);
      }

      const endTime = performance.now();
      const processingTime = endTime - startTime;

      // Should handle updates efficiently
      expect(processingTime).toBeLessThan(100); // 100ms
    });

    it('should throttle API calls to prevent overload', async () => {
      const user = userEvent.setup();

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      const structuralButton = await screen.findByRole('button', { name: /structural/i });

      // Rapid clicking should be throttled
      await user.click(structuralButton);
      await user.click(structuralButton);
      await user.click(structuralButton);

      // Should only make one API call due to throttling
      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('Accessibility', () => {
    it('should announce analysis progress to screen readers', async () => {
      const user = userEvent.setup();

      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      const structuralButton = await screen.findByRole('button', { name: /structural/i });
      await user.click(structuralButton);

      // Should have aria-live region for progress updates
      expect(screen.getByRole('status')).toBeInTheDocument();

      await waitFor(() => {
        expect(screen.getByText(/analysis complete/i)).toBeInTheDocument();
      });
    });

    it('should have proper ARIA labels for metrics', async () => {
      render(
        <GraphAnalysisTab graphData={mockGraphData} />
      );

      await waitFor(() => {
        expect(screen.getByLabelText(/gpu utilization/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/memory usage/i)).toBeInTheDocument();
      });
    });
  });
});