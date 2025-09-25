/**
 * Export Integration Tests
 * Tests for all export formats, share link generation, download functionality, and file format verification
 *
 * NOTE: These tests are prepared but disabled due to security concerns
 * with testing dependencies. Re-enable once security issues are resolved.
 */

import { describe, it, expect, beforeEach, afterEach, vi, beforeAll, afterAll } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GraphExportTab } from '@/features/visualisation/components/tabs/GraphExportTab';
import { createMockApiClient } from '../__mocks__/apiClient';

// Mock toast notifications
vi.mock('@/features/design-system/components/Toast', () => ({
  toast: vi.fn()
}));

// Mock file downloads
const mockCreateObjectURL = vi.fn();
const mockRevokeObjectURL = vi.fn();
const mockAppendChild = vi.fn();
const mockRemoveChild = vi.fn();

// Mock graph data for export
const mockGraphData = {
  nodes: [
    { id: '1', label: 'Node A', x: 100, y: 150, size: 10, color: '#ff0000' },
    { id: '2', label: 'Node B', x: 200, y: 250, size: 15, color: '#00ff00' },
    { id: '3', label: 'Node C', x: 300, y: 350, size: 8, color: '#0000ff' }
  ],
  edges: [
    { id: 'e1', source: '1', target: '2', weight: 1.2, color: '#666666' },
    { id: 'e2', source: '2', target: '3', weight: 0.8, color: '#333333' }
  ],
  metadata: {
    title: 'Test Graph',
    description: 'A sample graph for testing export functionality',
    created: new Date('2024-09-20T10:00:00Z'),
    nodeCount: 3,
    edgeCount: 2
  }
};

// Mock export API responses
const mockExportResponse = {
  taskId: 'export-123',
  status: 'processing',
  format: 'json',
  downloadUrl: null,
  progress: 0
};

const mockShareResponse = {
  shareId: 'shared-abc123',
  url: 'https://example.com/shared/shared-abc123',
  expires: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
  public: true,
  passwordProtected: false
};

describe('Export Integration Tests', () => {
  let mockApiClient: any;

  beforeAll(() => {
    // Setup URL and DOM mocks
    global.URL.createObjectURL = mockCreateObjectURL;
    global.URL.revokeObjectURL = mockRevokeObjectURL;

    // Mock document methods
    const originalAppendChild = document.body.appendChild;
    const originalRemoveChild = document.body.removeChild;

    document.body.appendChild = mockAppendChild;
    document.body.removeChild = mockRemoveChild;

    // Mock link element
    const mockLink = {
      href: '',
      download: '',
      click: vi.fn(),
      style: {},
      className: ''
    };

    vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
      if (tagName === 'a') {
        return mockLink as any;
      }
      return originalCreateElement.call(document, tagName);
    });

    // Mock Blob
    global.Blob = vi.fn().mockImplementation((content, options) => ({
      size: content[0].length,
      type: options?.type || 'text/plain'
    }));
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mockApiClient = createMockApiClient();

    // Setup export API endpoints
    mockApiClient.post.mockImplementation((url: string, data: any) => {
      switch (url) {
        case '/api/export/graph':
          return Promise.resolve({
            data: {
              ...mockExportResponse,
              format: data.format,
              includeMetadata: data.includeMetadata
            }
          });
        case '/api/export/share':
          return Promise.resolve({
            data: {
              ...mockShareResponse,
              description: data.description,
              expires: data.expires
            }
          });
        case '/api/export/embed':
          return Promise.resolve({
            data: {
              embedId: 'embed-xyz789',
              url: `https://example.com/embed/embed-xyz789`,
              html: `<iframe src="https://example.com/embed/embed-xyz789" width="800" height="600"></iframe>`
            }
          });
        default:
          return Promise.reject(new Error(`Unexpected POST: ${url}`));
      }
    });

    mockApiClient.get.mockImplementation((url: string) => {
      if (url.includes('/api/export/status/')) {
        const taskId = url.split('/').pop();
        return Promise.resolve({
          data: {
            taskId,
            status: 'completed',
            progress: 100,
            downloadUrl: `https://api.example.com/download/${taskId}`,
            fileSize: 2048
          }
        });
      }
      if (url.includes('/api/export/download/')) {
        // Simulate file download
        return Promise.resolve({
          data: JSON.stringify(mockGraphData),
          headers: {
            'content-type': 'application/json',
            'content-disposition': 'attachment; filename=graph-export.json'
          }
        });
      }
      return Promise.reject(new Error(`Unexpected GET: ${url}`));
    });

    mockCreateObjectURL.mockReturnValue('blob:mock-url');
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('Data Export Formats', () => {
    it('should export graph as JSON format', async () => {
      const user = userEvent.setup();
      const onExport = vi.fn();

      render(
        <GraphExportTab
          graphId="test-graph"
          graphData={mockGraphData}
          onExport={onExport}
        />
      );

      // Select JSON format (should be default)
      const formatSelect = screen.getByDisplayValue(/JSON Data/i);
      expect(formatSelect).toBeInTheDocument();

      // Click export button
      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/graph', {
          graphId: 'test-graph',
          format: 'json',
          includeMetadata: true,
          compression: false,
          graphData: mockGraphData
        });
      });

      expect(onExport).toHaveBeenCalledWith('json', {
        format: 'json',
        includeMetadata: true,
        compression: false
      });
    });

    it('should export graph as CSV format', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab
          graphId="test-graph"
          graphData={mockGraphData}
        />
      );

      // Change format to CSV
      const formatSelect = screen.getByRole('combobox');
      await user.click(formatSelect);
      await user.click(screen.getByText('CSV Spreadsheet'));

      const exportButton = screen.getByRole('button', { name: /export csv/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/graph', {
          graphId: 'test-graph',
          format: 'csv',
          includeMetadata: true,
          compression: false,
          graphData: mockGraphData
        });
      });
    });

    it('should export graph as GraphML format', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab
          graphId="test-graph"
          graphData={mockGraphData}
        />
      );

      // Change format to GraphML
      const formatSelect = screen.getByRole('combobox');
      await user.click(formatSelect);
      await user.click(screen.getByText('GraphML'));

      const exportButton = screen.getByRole('button', { name: /export graphml/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/graph', {
          graphId: 'test-graph',
          format: 'graphml',
          includeMetadata: true,
          compression: false,
          graphData: mockGraphData
        });
      });
    });

    it('should export graph as PNG image with custom resolution', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab
          graphId="test-graph"
          graphData={mockGraphData}
        />
      );

      // Change format to PNG
      const formatSelect = screen.getByRole('combobox');
      await user.click(formatSelect);
      await user.click(screen.getByText('PNG Image'));

      // Change resolution to 4K
      const resolutionSelect = screen.getByDisplayValue('1920×1080 (Full HD)');
      await user.click(resolutionSelect);
      await user.click(screen.getByText('3840×2160 (4K)'));

      const exportButton = screen.getByRole('button', { name: /export png/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/graph', {
          graphId: 'test-graph',
          format: 'png',
          includeMetadata: true,
          compression: false,
          graphData: mockGraphData,
          resolution: '3840x2160'
        });
      });
    });

    it('should handle export options correctly', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab
          graphId="test-graph"
          graphData={mockGraphData}
        />
      );

      // Disable metadata inclusion
      const metadataToggle = screen.getByLabelText(/include metadata/i);
      await user.click(metadataToggle);

      // Enable compression
      const compressionToggle = screen.getByLabelText(/enable compression/i);
      await user.click(compressionToggle);

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/graph', {
          graphId: 'test-graph',
          format: 'json',
          includeMetadata: false,
          compression: true,
          graphData: mockGraphData
        });
      });
    });
  });

  describe('File Format Verification', () => {
    it('should verify JSON export format is valid', async () => {
      const user = userEvent.setup();

      // Mock successful export with JSON validation
      mockApiClient.get.mockImplementation((url: string) => {
        if (url.includes('/api/export/download/')) {
          const jsonData = JSON.stringify(mockGraphData, null, 2);
          return Promise.resolve({
            data: jsonData,
            headers: {
              'content-type': 'application/json',
              'content-disposition': 'attachment; filename=graph-export.json'
            }
          });
        }
        return Promise.resolve({
          data: { status: 'completed', progress: 100, downloadUrl: 'mock-url' }
        });
      });

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      // Wait for export completion
      await waitFor(() => {
        expect(mockApiClient.get).toHaveBeenCalledWith('/api/export/status/export-123');
      });

      // Verify JSON structure
      const downloadedData = JSON.parse(mockApiClient.get.mock.results[0].value.data);
      expect(downloadedData).toHaveProperty('nodes');
      expect(downloadedData).toHaveProperty('edges');
      expect(downloadedData).toHaveProperty('metadata');
      expect(downloadedData.nodes).toHaveLength(3);
      expect(downloadedData.edges).toHaveLength(2);
    });

    it('should verify GraphML export format structure', async () => {
      const user = userEvent.setup();

      // Mock GraphML export
      const graphMLContent = `<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="label" for="node" attr.name="label" attr.type="string"/>
  <key id="size" for="node" attr.name="size" attr.type="int"/>
  <graph id="test-graph" edgedefault="undirected">
    <node id="1"><data key="label">Node A</data><data key="size">10</data></node>
    <node id="2"><data key="label">Node B</data><data key="size">15</data></node>
    <node id="3"><data key="label">Node C</data><data key="size">8</data></node>
    <edge source="1" target="2"/>
    <edge source="2" target="3"/>
  </graph>
</graphml>`;

      mockApiClient.get.mockImplementation((url: string) => {
        if (url.includes('/api/export/download/')) {
          return Promise.resolve({
            data: graphMLContent,
            headers: {
              'content-type': 'application/xml',
              'content-disposition': 'attachment; filename=graph-export.graphml'
            }
          });
        }
        return Promise.resolve({
          data: { status: 'completed', progress: 100, downloadUrl: 'mock-url' }
        });
      });

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      // Change to GraphML format
      const formatSelect = screen.getByRole('combobox');
      await user.click(formatSelect);
      await user.click(screen.getByText('GraphML'));

      const exportButton = screen.getByRole('button', { name: /export graphml/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(mockApiClient.get).toHaveBeenCalledWith('/api/export/status/export-123');
      });

      // Verify GraphML structure
      const downloadedContent = mockApiClient.get.mock.results[0].value.data;
      expect(downloadedContent).toContain('<?xml version="1.0"');
      expect(downloadedContent).toContain('<graphml');
      expect(downloadedContent).toContain('<node id="1">');
      expect(downloadedContent).toContain('<edge source="1" target="2"/>');
    });

    it('should handle invalid export format gracefully', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      // Mock API error for invalid format
      mockApiClient.post.mockRejectedValue(new Error('Unsupported export format'));

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: 'Export Failed',
            variant: 'destructive'
          })
        );
      });
    });
  });

  describe('Share Link Generation', () => {
    it('should create share link with default settings', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const shareButton = screen.getByRole('button', { name: /create share link/i });
      await user.click(shareButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/share', {
          graphId: 'test-graph',
          graphData: mockGraphData,
          description: '',
          expires: '7days',
          password: null,
          public: true
        });
      });

      // Should display the share link
      await waitFor(() => {
        expect(screen.getByText('Share Link Created')).toBeInTheDocument();
        expect(screen.getByDisplayValue(mockShareResponse.url)).toBeInTheDocument();
      });
    });

    it('should create password-protected share link', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      // Set description
      const descriptionInput = screen.getByPlaceholderText(/describe what this graph shows/i);
      await user.type(descriptionInput, 'Test graph for sharing');

      // Set password
      const passwordInput = screen.getByPlaceholderText(/security password/i);
      await user.type(passwordInput, 'secret123');

      // Change expiry
      const expirySelect = screen.getByDisplayValue('7 Days');
      await user.click(expirySelect);
      await user.click(screen.getByText('30 Days'));

      const shareButton = screen.getByRole('button', { name: /create share link/i });
      await user.click(shareButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/share', {
          graphId: 'test-graph',
          graphData: mockGraphData,
          description: 'Test graph for sharing',
          expires: '30days',
          password: 'secret123',
          public: true
        });
      });
    });

    it('should copy share link to clipboard', async () => {
      const user = userEvent.setup();
      const mockWriteText = vi.fn();
      Object.assign(navigator, {
        clipboard: {
          writeText: mockWriteText
        }
      });

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const shareButton = screen.getByRole('button', { name: /create share link/i });
      await user.click(shareButton);

      // Wait for share link to be created
      await waitFor(() => {
        expect(screen.getByDisplayValue(mockShareResponse.url)).toBeInTheDocument();
      });

      // Click copy button
      const copyButton = screen.getByRole('button', { name: /copy/i });
      await user.click(copyButton);

      expect(mockWriteText).toHaveBeenCalledWith(mockShareResponse.url);
    });

    it('should handle share link creation errors', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      mockApiClient.post.mockRejectedValue(new Error('Share service unavailable'));

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const shareButton = screen.getByRole('button', { name: /create share link/i });
      await user.click(shareButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: 'Share Link Failed',
            variant: 'destructive'
          })
        );
      });
    });
  });

  describe('Download Functionality', () => {
    it('should trigger file download after export completion', async () => {
      const user = userEvent.setup();
      const mockClick = vi.fn();
      const mockLink = { href: '', download: '', click: mockClick };

      vi.spyOn(document, 'createElement').mockReturnValue(mockLink as any);

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      // Wait for export completion and download
      await waitFor(() => {
        expect(mockCreateObjectURL).toHaveBeenCalled();
        expect(mockLink.href).toBe('blob:mock-url');
        expect(mockLink.download).toContain('.json');
        expect(mockClick).toHaveBeenCalled();
      });

      // Verify cleanup
      expect(mockAppendChild).toHaveBeenCalledWith(mockLink);
      expect(mockRemoveChild).toHaveBeenCalledWith(mockLink);
      expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });

    it('should handle large file downloads efficiently', async () => {
      const user = userEvent.setup();

      // Mock large file response
      const largeGraphData = {
        ...mockGraphData,
        nodes: Array.from({ length: 10000 }, (_, i) => ({
          id: `node-${i}`,
          label: `Node ${i}`,
          x: Math.random() * 1000,
          y: Math.random() * 1000,
          size: Math.random() * 20
        })),
        edges: Array.from({ length: 50000 }, (_, i) => ({
          id: `edge-${i}`,
          source: `node-${Math.floor(Math.random() * 10000)}`,
          target: `node-${Math.floor(Math.random() * 10000)}`
        }))
      };

      mockApiClient.get.mockImplementation((url: string) => {
        if (url.includes('/api/export/download/')) {
          return Promise.resolve({
            data: JSON.stringify(largeGraphData),
            headers: {
              'content-type': 'application/json',
              'content-length': '10485760' // 10MB
            }
          });
        }
        return Promise.resolve({
          data: { status: 'completed', progress: 100, downloadUrl: 'mock-url' }
        });
      });

      render(
        <GraphExportTab graphId="test-graph" graphData={largeGraphData} />
      );

      const startTime = performance.now();

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(mockCreateObjectURL).toHaveBeenCalled();
      });

      const endTime = performance.now();
      const processingTime = endTime - startTime;

      // Should handle large files efficiently (under 5 seconds)
      expect(processingTime).toBeLessThan(5000);
    });

    it('should show download progress for large exports', async () => {
      const user = userEvent.setup();

      // Mock progressive export status
      let statusCallCount = 0;
      mockApiClient.get.mockImplementation((url: string) => {
        if (url.includes('/api/export/status/')) {
          statusCallCount++;
          const progress = Math.min(statusCallCount * 20, 100);
          const status = progress === 100 ? 'completed' : 'processing';

          return Promise.resolve({
            data: {
              status,
              progress,
              downloadUrl: status === 'completed' ? 'mock-url' : null
            }
          });
        }
        if (url.includes('/api/export/download/')) {
          return Promise.resolve({
            data: JSON.stringify(mockGraphData)
          });
        }
        return Promise.reject(new Error(`Unexpected GET: ${url}`));
      });

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      // Should show progress indicator
      await waitFor(() => {
        expect(screen.getByText('Exporting...')).toBeInTheDocument();
      });

      // Should complete and download
      await waitFor(() => {
        expect(screen.getByText(/export/i)).not.toHaveTextContent('Exporting...');
      });
    });
  });

  describe('Web Integration', () => {
    it('should generate embed code', async () => {
      const user = userEvent.setup();
      const mockWriteText = vi.fn();
      Object.assign(navigator, {
        clipboard: { writeText: mockWriteText }
      });

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const embedButton = screen.getByRole('button', { name: /embed code/i });
      await user.click(embedButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/embed', {
          graphId: 'test-graph',
          graphData: mockGraphData
        });
      });

      // Should display embed code
      const embedTextarea = screen.getByDisplayValue(/iframe/);
      expect(embedTextarea).toBeInTheDocument();
      expect(embedTextarea.value).toContain('https://example.com/embed/embed-xyz789');

      // Should auto-copy to clipboard
      expect(mockWriteText).toHaveBeenCalledWith(
        expect.stringContaining('<iframe')
      );
    });

    it('should generate API endpoint', async () => {
      const user = userEvent.setup();
      const mockWriteText = vi.fn();
      Object.assign(navigator, {
        clipboard: { writeText: mockWriteText }
      });

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const apiButton = screen.getByRole('button', { name: /api endpoint/i });
      await user.click(apiButton);

      // Should display API endpoint
      const endpointInput = screen.getByDisplayValue(/\/api\/graphs\/test-graph\/data/);
      expect(endpointInput).toBeInTheDocument();

      // Should copy to clipboard
      expect(mockWriteText).toHaveBeenCalledWith(
        expect.stringContaining('/api/graphs/test-graph/data')
      );
    });

    it('should open API endpoint in new tab', async () => {
      const user = userEvent.setup();
      const mockOpen = vi.fn();
      vi.stubGlobal('open', mockOpen);

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      // Generate API endpoint first
      const apiButton = screen.getByRole('button', { name: /api endpoint/i });
      await user.click(apiButton);

      // Click external link button
      const externalButton = screen.getByRole('button', { name: /external/i });
      await user.click(externalButton);

      expect(mockOpen).toHaveBeenCalledWith(
        expect.stringContaining('/api/graphs/test-graph/data'),
        '_blank'
      );
    });
  });

  describe('Quick Actions', () => {
    it('should save image via quick action', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const saveImageButton = screen.getByRole('button', { name: /save image/i });
      await user.click(saveImageButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/graph', {
          graphId: 'test-graph',
          format: 'png',
          includeMetadata: true,
          compression: false,
          graphData: mockGraphData
        });
      });
    });

    it('should export data via quick action', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportDataButton = screen.getByRole('button', { name: /export data/i });
      await user.click(exportDataButton);

      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledWith('/api/export/graph', {
          graphId: 'test-graph',
          format: 'json',
          includeMetadata: true,
          compression: false,
          graphData: mockGraphData
        });
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle export timeout', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      // Mock timeout response
      mockApiClient.get.mockImplementation((url: string) => {
        if (url.includes('/api/export/status/')) {
          return Promise.resolve({
            data: { status: 'timeout', progress: 50, error: 'Export timed out' }
          });
        }
        return Promise.reject(new Error('Timeout'));
      });

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: expect.stringContaining('Timeout'),
            variant: 'destructive'
          })
        );
      });
    });

    it('should handle network errors during export', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      mockApiClient.post.mockRejectedValue(new Error('Network error'));

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: 'Export Failed',
            variant: 'destructive'
          })
        );
      });
    });

    it('should handle missing graph data gracefully', async () => {
      const user = userEvent.setup();
      const { toast } = await import('@/features/design-system/components/Toast');

      render(
        <GraphExportTab graphId="test-graph" />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: expect.stringContaining('No Graph Data'),
            variant: 'destructive'
          })
        );
      });

      // Should not make API call without data
      expect(mockApiClient.post).not.toHaveBeenCalled();
    });
  });

  describe('Performance Tests', () => {
    it('should handle multiple concurrent export requests', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      // Trigger multiple exports rapidly
      const exportButton = screen.getByRole('button', { name: /export json/i });
      const saveImageButton = screen.getByRole('button', { name: /save image/i });

      await user.click(exportButton);
      await user.click(saveImageButton);

      // Should handle concurrent requests gracefully
      await waitFor(() => {
        expect(mockApiClient.post).toHaveBeenCalledTimes(2);
      });
    });

    it('should cleanup resources properly after export', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      await waitFor(() => {
        expect(mockRevokeObjectURL).toHaveBeenCalled();
      });

      // Memory should be released
      expect(mockAppendChild).toHaveBeenCalled();
      expect(mockRemoveChild).toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('should be keyboard navigable', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      // Tab through export controls
      const formatSelect = screen.getByRole('combobox');
      formatSelect.focus();

      await user.keyboard('{Tab}');
      expect(screen.getByLabelText(/include metadata/i)).toHaveFocus();

      await user.keyboard('{Tab}');
      expect(screen.getByLabelText(/enable compression/i)).toHaveFocus();

      await user.keyboard('{Tab}');
      expect(screen.getByRole('button', { name: /export json/i })).toHaveFocus();
    });

    it('should have proper ARIA attributes', () => {
      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      // Export section should have proper labeling
      expect(screen.getByLabelText(/export format/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/include metadata/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/enable compression/i)).toBeInTheDocument();

      // Share section should be accessible
      expect(screen.getByLabelText(/share description/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/expires in/i)).toBeInTheDocument();
    });

    it('should announce export progress to screen readers', async () => {
      const user = userEvent.setup();

      render(
        <GraphExportTab graphId="test-graph" graphData={mockGraphData} />
      );

      const exportButton = screen.getByRole('button', { name: /export json/i });
      await user.click(exportButton);

      // Should have aria-live region for status updates
      await waitFor(() => {
        expect(screen.getByText(/exporting/i)).toBeInTheDocument();
      });

      // Should announce completion
      await waitFor(() => {
        expect(screen.getByText(/complete/i)).toBeInTheDocument();
      });
    });
  });
});