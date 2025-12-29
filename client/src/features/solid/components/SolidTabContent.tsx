/**
 * SolidTabContent Component
 * Combined view for Solid Pod features in the control panel
 */

import React, { useState } from 'react';
import {
  Database,
  Folder,
  Settings,
  Info,
  Lock,
} from 'lucide-react';
import { cn } from '@/utils/classNameUtils';
import { PodSettings } from './PodSettings';
import { PodBrowser } from './PodBrowser';
import { ResourceEditor } from './ResourceEditor';
import { useSettingsStore } from '@/store/settingsStore';

type SolidSubTab = 'settings' | 'browser';

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

const TabButton: React.FC<TabButtonProps> = ({ active, onClick, icon, label }) => (
  <button
    onClick={onClick}
    className={cn(
      'flex items-center gap-2 px-3 py-2 text-xs font-medium rounded-md transition-colors',
      active
        ? 'bg-primary/20 text-primary'
        : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
    )}
  >
    {icon}
    {label}
  </button>
);

export interface SolidTabContentProps {
  className?: string;
}

export const SolidTabContent: React.FC<SolidTabContentProps> = ({ className }) => {
  const [activeSubTab, setActiveSubTab] = useState<SolidSubTab>('settings');
  const [selectedResourceUrl, setSelectedResourceUrl] = useState<string | null>(null);

  // Check if user is authenticated via Nostr
  const isAuthenticated = useSettingsStore(state => state.authenticated);

  // Handle resource selection from browser
  const handleResourceSelect = (resourceUrl: string) => {
    setSelectedResourceUrl(resourceUrl);
  };

  // If not authenticated, show login prompt
  if (!isAuthenticated) {
    return (
      <div className={cn('flex flex-col items-center justify-center py-12 px-4', className)}>
        <div className="flex flex-col items-center text-center max-w-sm">
          <div className="p-4 rounded-full bg-muted/50 mb-4">
            <Lock className="h-8 w-8 text-muted-foreground" />
          </div>
          <h3 className="text-sm font-medium mb-2">Login Required</h3>
          <p className="text-xs text-muted-foreground mb-4">
            Login with Nostr to access your Solid Pod and manage your decentralized data storage.
          </p>
          <div className="p-3 rounded-md bg-blue-500/10 border border-blue-500/20 text-xs text-blue-400">
            <Info className="h-4 w-4 inline-block mr-2" />
            Use the Auth/Nostr tab to connect your Nostr identity.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Sub-tab navigation */}
      <div className="flex items-center gap-2 p-2 border-b border-border/50">
        <TabButton
          active={activeSubTab === 'settings'}
          onClick={() => setActiveSubTab('settings')}
          icon={<Settings className="h-3.5 w-3.5" />}
          label="Pod Settings"
        />
        <TabButton
          active={activeSubTab === 'browser'}
          onClick={() => setActiveSubTab('browser')}
          icon={<Folder className="h-3.5 w-3.5" />}
          label="Browse Files"
        />
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-hidden p-2">
        {activeSubTab === 'settings' && (
          <div className="space-y-4 overflow-y-auto max-h-full pr-1">
            <PodSettings />

            {/* Info section about Solid */}
            <div className="p-3 rounded-md bg-muted/30 border border-border/50">
              <div className="flex items-start gap-2">
                <Database className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                <div className="text-xs">
                  <p className="font-medium text-foreground mb-1">About Solid Pods</p>
                  <p className="text-muted-foreground">
                    Your Solid Pod is a personal data store that you control.
                    Data is stored as Linked Data (JSON-LD) and can be shared
                    with other applications using standard web protocols.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeSubTab === 'browser' && (
          <div className="h-full flex flex-col gap-2">
            <div className="flex-1 min-h-0">
              <PodBrowser
                onResourceSelect={handleResourceSelect}
                className="h-full"
              />
            </div>

            {/* Resource preview/editor */}
            {selectedResourceUrl && (
              <div className="h-48 border-t border-border/50 pt-2">
                <ResourceEditor
                  resourceUrl={selectedResourceUrl}
                  onClose={() => setSelectedResourceUrl(null)}
                  className="h-full"
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SolidTabContent;
