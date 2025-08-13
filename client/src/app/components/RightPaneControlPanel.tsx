// client/src/app/components/RightPaneControlPanel.tsx
import React, { useState } from 'react';
import { ChevronDown, Terminal, Code, Bug, Database, Cpu, Network } from 'lucide-react';
import { SettingsPanelRedesign } from '../../features/settings/components/panels/SettingsPanelRedesign';
import ConversationPane from './ConversationPane';
import NarrativeGoldminePanel from './NarrativeGoldminePanel';
import { ProgrammaticMonitorControl } from '../../features/bots/components/ProgrammaticMonitorControl';
import { SpacePilotButtonPanel } from '../../features/visualisation/components/SpacePilotButtonPanel';
import { SystemHealthPanel, ActivityLogPanel, AgentDetailPanel } from '../../features/bots/components';
import GraphFeaturesPanel from '../../features/graph/components/GraphFeaturesPanel';
import { useSettingsStore } from '../../store/settingsStore';
import { Button } from '@/features/design-system/components/Button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/features/design-system/components/Collapsible';

const RightPaneControlPanel: React.FC = () => {
  const isPowerUser = useSettingsStore(state => state.isPowerUser);
  const [showDevTools, setShowDevTools] = useState(false);

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Scrollable content area */}
      <div className="flex-1 overflow-y-auto">
        {/* Settings Panel - Always visible at the top */}
        <div className="flex-shrink-0">
          <SettingsPanelRedesign />
        </div>

        {/* 1. Graph Features - Innovative graph controls */}
        <Collapsible defaultOpen={true} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">1. Graph Features</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="p-2">
              <GraphFeaturesPanel />
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 2. Conversation - Chat and AI interaction */}
        <Collapsible defaultOpen={true} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">2. Conversation</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="max-h-96 overflow-y-auto">
              <ConversationPane />
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 3. AI Agents - Agent monitoring and control */}
        <Collapsible defaultOpen={false} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">3. AI Agents</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="p-2">
              <AgentDetailPanel />
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 4. System Monitor - Performance and health tracking */}
        <Collapsible defaultOpen={false} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">4. System Monitor</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="p-2 space-y-2">
              <SystemHealthPanel />
              <div className="border-t pt-2">
                <ProgrammaticMonitorControl />
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 5. Narrative - Story generation and insights */}
        <Collapsible defaultOpen={false} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">5. Narrative</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="max-h-96 overflow-y-auto">
              <NarrativeGoldminePanel />
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 6. Controller - 6DOF SpacePilot control */}
        <Collapsible defaultOpen={false} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">6. Controller</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="p-2">
              <SpacePilotButtonPanel compact={true} showLabels={true} />
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 7. Activity Log - System activity tracking */}
        <Collapsible defaultOpen={false} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">7. Activity Log</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="p-2">
              <ActivityLogPanel />
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 8. Performance - Real-time performance metrics */}
        <Collapsible defaultOpen={false} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">8. Performance</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="p-4 space-y-3">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">FPS</span>
                    <span className="font-mono font-semibold">60</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Memory</span>
                    <span className="font-mono font-semibold">245 MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">GPU</span>
                    <span className="font-mono font-semibold">32%</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Network</span>
                    <span className="font-mono font-semibold">12 KB/s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Nodes</span>
                    <span className="font-mono font-semibold">1,284</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Edges</span>
                    <span className="font-mono font-semibold">3,421</span>
                  </div>
                </div>
              </div>
              <div className="pt-2 border-t">
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" className="flex-1 text-xs">
                    <Cpu className="h-3 w-3 mr-1" />
                    Profile
                  </Button>
                  <Button variant="outline" size="sm" className="flex-1 text-xs">
                    <Network className="h-3 w-3 mr-1" />
                    Network
                  </Button>
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* 9. Quick Actions - Shortcuts and commands */}
        <Collapsible defaultOpen={false} className="flex-shrink-0 border-t">
          <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50 transition-colors">
            <h3 className="text-sm font-semibold">9. Quick Actions</h3>
            <ChevronDown className="h-4 w-4 transition-transform duration-200 data-[state=closed]:-rotate-90" />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden">
            <div className="p-4 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <Button variant="outline" size="sm" className="text-xs">
                  Refresh Graph
                </Button>
                <Button variant="outline" size="sm" className="text-xs">
                  Reset View
                </Button>
                <Button variant="outline" size="sm" className="text-xs">
                  Export Data
                </Button>
                <Button variant="outline" size="sm" className="text-xs">
                  Take Snapshot
                </Button>
              </div>
              <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t">
                <p className="font-semibold">Keyboard Shortcuts:</p>
                <p>• 1-9: Toggle panels</p>
                <p>• Shift+1-9: Expand/collapse</p>
                <p>• Ctrl+S: Save settings</p>
                <p>• Ctrl+R: Refresh graph</p>
                <p>• Space: Pause/resume</p>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
      
      {/* Power User Developer Tools - Shown at bottom for power users only */}
      {isPowerUser && (
        <div className="mt-auto border-t bg-muted/20">
          {!showDevTools ? (
            <Button
              variant="ghost"
              size="sm"
              className="w-full rounded-none justify-start gap-2 py-3"
              onClick={() => setShowDevTools(true)}
            >
              <Terminal className="h-4 w-4" />
              Developer Tools
              <span className="ml-auto text-xs text-muted-foreground">Power User</span>
            </Button>
          ) : (
            <div className="p-3 space-y-2">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <Terminal className="h-4 w-4" />
                  Developer Tools
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowDevTools(false)}
                  className="h-6 w-6 p-0"
                >
                  ×
                </Button>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Button variant="outline" size="sm" className="text-xs">
                  <Code className="h-3 w-3" />
                  Console
                </Button>
                <Button variant="outline" size="sm" className="text-xs">
                  <Bug className="h-3 w-3" />
                  Debug
                </Button>
                <Button variant="outline" size="sm" className="text-xs">
                  <Database className="h-3 w-3" />
                  State
                </Button>
              </div>
              <div className="text-xs text-muted-foreground space-y-1 mt-2 p-2 bg-background rounded">
                <p className="font-semibold mb-1">Developer Commands:</p>
                <p>• Ctrl+D: Toggle debug mode</p>
                <p>• Ctrl+Shift+I: Open DevTools</p>
                <p>• Ctrl+Shift+P: Command palette</p>
                <p>• Ctrl+Alt+R: Reload modules</p>
                <p>• Ctrl+Alt+M: Memory profiler</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RightPaneControlPanel;