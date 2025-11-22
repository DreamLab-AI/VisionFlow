/**
 * Top menu bar component with visualization controls
 */

import { useGraphStore } from '../../stores/useGraphStore';
import { useUIStore } from '../../stores/useUIStore';
import { exportSVG, downloadSVG, exportPNG, downloadPNG } from '../../utils/export';

export function TopMenuBar() {
  const { statistics, clear, nodes, edges } = useGraphStore();
  const { toggleSidebar, sidebarOpen, toggleViewMode, viewport, updateSettings, settings, addNotification } = useUIStore();

  const handleNewFile = () => {
    if (confirm('Clear current ontology and load a new one?')) {
      clear();
      // App will automatically show FileDropZone
    }
  };

  const handleExportSVG = async () => {
    try {
      addNotification({
        type: 'info',
        message: 'Generating SVG export...',
        duration: 2000
      });

      const svg = exportSVG(nodes, edges, {
        width: 1920,
        height: 1080,
        showLabels: settings.showLabels
      });

      downloadSVG(svg, 'ontology-graph.svg');

      addNotification({
        type: 'success',
        message: 'SVG exported successfully!',
        duration: 3000
      });
    } catch (error) {
      console.error('SVG export failed:', error);
      addNotification({
        type: 'error',
        message: 'Failed to export SVG',
        duration: 5000
      });
    }
  };

  const handleExportPNG = async () => {
    try {
      addNotification({
        type: 'info',
        message: 'Generating PNG export...',
        duration: 2000
      });

      const blob = await exportPNG(nodes, edges, {
        width: 1920,
        height: 1080,
        showLabels: settings.showLabels,
        quality: 0.95
      });

      downloadPNG(blob, 'ontology-graph.png');

      addNotification({
        type: 'success',
        message: 'PNG exported successfully!',
        duration: 3000
      });
    } catch (error) {
      console.error('PNG export failed:', error);
      addNotification({
        type: 'error',
        message: 'Failed to export PNG',
        duration: 5000
      });
    }
  };

  const handleZoomIn = () => {
    useUIStore.getState().setZoom(viewport.zoom * 1.2);
  };

  const handleZoomOut = () => {
    useUIStore.getState().setZoom(viewport.zoom / 1.2);
  };

  const handleResetView = () => {
    useUIStore.getState().setZoom(1);
    useUIStore.getState().setRotation([0, 0, 0]);
    useUIStore.getState().setTarget([0, 0, 0]);
  };

  return (
    <div className="flex justify-between items-center px-4 py-3 bg-background border-b border-border shadow-sm sticky top-0 z-50 gap-4">
      {/* Left section - File controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleNewFile}
          className="flex items-center gap-2 px-3 py-2 bg-transparent border border-gray-300 dark:border-gray-700 rounded-md text-gray-700 dark:text-gray-200 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-800 hover:border-gray-400 active:bg-gray-100 dark:active:bg-gray-700 active:scale-98 transition-all whitespace-nowrap"
          title="Load new ontology"
        >
          <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span className="max-md:hidden">New</span>
        </button>

        <button
          onClick={handleExportSVG}
          className="flex items-center gap-2 px-3 py-2 bg-transparent border border-gray-300 dark:border-gray-700 rounded-md text-gray-700 dark:text-gray-200 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-800 hover:border-gray-400 active:bg-gray-100 dark:active:bg-gray-700 active:scale-98 transition-all whitespace-nowrap"
          title="Export as SVG"
        >
          <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          <span className="max-md:hidden">Export SVG</span>
        </button>

        <button
          onClick={handleExportPNG}
          className="flex items-center gap-2 px-3 py-2 bg-transparent border border-gray-300 dark:border-gray-700 rounded-md text-gray-700 dark:text-gray-200 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-800 hover:border-gray-400 active:bg-gray-100 dark:active:bg-gray-700 active:scale-98 transition-all whitespace-nowrap"
          title="Export as PNG"
        >
          <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <span className="max-md:hidden">Export PNG</span>
        </button>
      </div>

      {/* Center section - Statistics */}
      <div className="flex items-center gap-6 px-4 border-l border-r border-gray-200 dark:border-gray-800 max-md:w-full max-md:justify-center max-md:border-t max-md:border-l-0 max-md:border-r-0 max-md:py-2 max-md:mt-2">
        {statistics && (
          <>
            <div className="flex items-baseline gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Nodes:</span>
              <span className="text-base font-bold text-gray-800 dark:text-gray-100">{statistics.nodeCount}</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Edges:</span>
              <span className="text-base font-bold text-gray-800 dark:text-gray-100">{statistics.edgeCount}</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Max Degree:</span>
              <span className="text-base font-bold text-gray-800 dark:text-gray-100">{statistics.maxDegree}</span>
            </div>
          </>
        )}
      </div>

      {/* Right section - View controls */}
      <div className="flex items-center gap-2">
        {/* Hierarchy Depth Control */}
        <div className="flex items-center gap-2 px-2 border-r border-gray-200 dark:border-gray-800 mr-2">
          <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Depth:</span>
          <input
            type="number"
            min="1"
            max="10"
            value={settings.hierarchyDepth}
            onChange={(e) => updateSettings({ hierarchyDepth: Math.max(1, parseInt(e.target.value) || 2) })}
            className="w-12 px-1 py-1 text-sm border rounded bg-transparent border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-200"
            title="Hierarchy Folding Depth"
          />
        </div>

        <button
          onClick={handleZoomOut}
          className="p-2 bg-transparent border border-gray-300 dark:border-gray-700 rounded-md text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800 hover:border-gray-400 active:bg-gray-100 dark:active:bg-gray-700 active:scale-98 transition-all"
          title="Zoom out"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
          </svg>
        </button>

        <button
          onClick={handleZoomIn}
          className="p-2 bg-transparent border border-gray-300 dark:border-gray-700 rounded-md text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800 hover:border-gray-400 active:bg-gray-100 dark:active:bg-gray-700 active:scale-98 transition-all"
          title="Zoom in"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
          </svg>
        </button>

        <button
          onClick={handleResetView}
          className="p-2 bg-transparent border border-gray-300 dark:border-gray-700 rounded-md text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800 hover:border-gray-400 active:bg-gray-100 dark:active:bg-gray-700 active:scale-98 transition-all"
          title="Reset view"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>

        <button
          onClick={toggleViewMode}
          className={`flex items-center gap-2 p-2 border rounded-md text-sm font-medium transition-all ${viewport.mode === '3d'
              ? 'bg-blue-600 border-blue-600 text-white hover:bg-blue-700 hover:border-blue-700'
              : 'bg-transparent border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800'
            }`}
          title={viewport.mode === '2d' ? 'Switch to 3D' : 'Switch to 2D'}
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
          <span className="text-xs font-semibold">{viewport.mode.toUpperCase()}</span>
        </button>

        <button
          onClick={() => updateSettings({ showLabels: !settings.showLabels })}
          className={`p-2 border rounded-md transition-all ${settings.showLabels
              ? 'bg-blue-600 border-blue-600 text-white hover:bg-blue-700 hover:border-blue-700'
              : 'bg-transparent border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800'
            }`}
          title="Toggle labels"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
          </svg>
        </button>

        <button
          onClick={toggleSidebar}
          className={`p-2 border rounded-md transition-all ${sidebarOpen
              ? 'bg-blue-600 border-blue-600 text-white hover:bg-blue-700 hover:border-blue-700'
              : 'bg-transparent border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800'
            }`}
          title="Toggle sidebar"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
    </div>
  );
}
