/**
 * Debug panel for force simulation debugging
 * Migrated to shadcn/ui with Collapsible and Slider
 */

import { Bug, X } from 'lucide-react';
import { useDebugControls } from '../../hooks/useDebugControls';
import { Button } from '../ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import { Slider } from '../ui/slider';

export function DebugPanel() {
  const {
    isOpen,
    setIsOpen,
    config,
    updateConfig,
    enableAll,
    disableAll,
    logCurrentDistribution,
    exportPositions,
  } = useDebugControls();

  if (!isOpen) {
    return (
      <Button
        onClick={() => setIsOpen(true)}
        variant="secondary"
        size="sm"
        className="fixed top-20 right-5 z-[1000] shadow-lg"
        title="Open debug panel"
      >
        <Bug className="h-4 w-4 mr-2" />
        Debug
      </Button>
    );
  }

  return (
    <div className="fixed top-20 right-5 w-[350px] max-h-[calc(100vh-100px)] bg-slate-900 text-slate-100 rounded-xl shadow-2xl z-[1000] overflow-hidden flex flex-col border border-slate-700">
      <div className="flex justify-between items-center p-4 bg-slate-800 border-b border-slate-700">
        <h3 className="text-base font-semibold flex items-center gap-2">
          <Bug className="h-5 w-5" />
          Force Simulation Debug
        </h3>
        <Button variant="ghost" size="sm" onClick={() => setIsOpen(false)} className="h-6 w-6 p-0">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Quick Actions</h4>
          <div className="grid grid-cols-2 gap-2">
            <Button onClick={enableAll} size="sm" variant="default">Enable All</Button>
            <Button onClick={disableAll} size="sm" variant="secondary">Disable All</Button>
            <Button onClick={logCurrentDistribution} size="sm" variant="outline" className="col-span-2">Log Distribution</Button>
            <Button onClick={exportPositions} size="sm" variant="outline" className="col-span-2">Export Positions</Button>
          </div>
        </div>

        <Collapsible defaultOpen>
          <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded hover:bg-slate-800">
            <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Force Logging</h4>
            <span className="text-xs text-slate-500">▼</span>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-2 mt-2">
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logForces} onChange={(e) => updateConfig({ logForces: e.target.checked })} />
              <span>Log Forces</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logRepulsion} onChange={(e) => updateConfig({ logRepulsion: e.target.checked })} />
              <span>Log Repulsion</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logAttraction} onChange={(e) => updateConfig({ logAttraction: e.target.checked })} />
              <span>Log Attraction</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logCentering} onChange={(e) => updateConfig({ logCentering: e.target.checked })} />
              <span>Log Centering</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logBarnesHut} onChange={(e) => updateConfig({ logBarnesHut: e.target.checked })} />
              <span>Log Barnes-Hut</span>
            </label>
          </CollapsibleContent>
        </Collapsible>

        <Collapsible defaultOpen>
          <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded hover:bg-slate-800">
            <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Position & Velocity</h4>
            <span className="text-xs text-slate-500">▼</span>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-2 mt-2">
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logPositions} onChange={(e) => updateConfig({ logPositions: e.target.checked })} />
              <span>Log Positions</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logVelocities} onChange={(e) => updateConfig({ logVelocities: e.target.checked })} />
              <span>Log Velocities</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logPositionHistory} onChange={(e) => updateConfig({ logPositionHistory: e.target.checked })} />
              <span>Track Position History</span>
            </label>
          </CollapsibleContent>
        </Collapsible>

        <Collapsible defaultOpen>
          <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded hover:bg-slate-800">
            <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Performance</h4>
            <span className="text-xs text-slate-500">▼</span>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-2 mt-2">
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logFPS} onChange={(e) => updateConfig({ logFPS: e.target.checked })} />
              <span>Log FPS</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logIterations} onChange={(e) => updateConfig({ logIterations: e.target.checked })} />
              <span>Log Iterations</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.logAlpha} onChange={(e) => updateConfig({ logAlpha: e.target.checked })} />
              <span>Log Alpha</span>
            </label>
          </CollapsibleContent>
        </Collapsible>

        <Collapsible defaultOpen>
          <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded hover:bg-slate-800">
            <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Visualization</h4>
            <span className="text-xs text-slate-500">▼</span>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-2 mt-2">
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.showQuadtree} onChange={(e) => updateConfig({ showQuadtree: e.target.checked })} />
              <span>Show Quadtree</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.showForceVectors} onChange={(e) => updateConfig({ showForceVectors: e.target.checked })} />
              <span>Show Force Vectors</span>
            </label>
            <label className="flex items-center space-x-2 text-sm cursor-pointer hover:text-slate-300">
              <input type="checkbox" checked={config.highlightClusters} onChange={(e) => updateConfig({ highlightClusters: e.target.checked })} />
              <span>Highlight Clusters</span>
            </label>
          </CollapsibleContent>
        </Collapsible>

        <div className="space-y-3 pt-2">
          <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Settings</h4>
          <div className="space-y-2">
            <label className="flex flex-col space-y-2">
              <span className="text-sm">Log Interval: {config.logInterval}</span>
              <Slider value={[config.logInterval]} onValueChange={(value) => updateConfig({ logInterval: value[0] })} min={1} max={100} step={1} />
            </label>
          </div>
        </div>

        <div className="bg-slate-800 p-3 rounded-lg text-xs text-slate-400 space-y-1 border border-slate-700">
          <p>💡 Open browser console to see logs</p>
          <p>🔍 Use window.wasmvowlDebug for advanced debugging</p>
        </div>
      </div>
    </div>
  );
}
