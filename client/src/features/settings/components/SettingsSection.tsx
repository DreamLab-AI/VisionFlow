import React, { useState } from 'react'; 
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/features/design-system/components/Collapsible';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { ChevronDown, ChevronUp, Minimize, Maximize } from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
// Removed import for SettingsSectionProps from types
// Removed import for SettingsSubsection
import Draggable from 'react-draggable';
import { useControlPanelContext } from './control-panel-context';
import { UISettingDefinition } from '../config/settingsUIDefinition'; 
import { SettingControlComponent } from './SettingControlComponent'; 
import { useSettingsStore } from '@/store/settingsStore'; 

// Define props locally
interface SettingsSectionProps {
  id: string;
  title: string;
  subsectionSettings: Record<string, UISettingDefinition>;
}

export function SettingsSection({ id, title, subsectionSettings }: SettingsSectionProps) {
  const [isOpen, setIsOpen] = useState(true);
  const [isDetached, setIsDetached] = useState(false);
  const { advancedMode } = useControlPanelContext();
  const settingsStore = useSettingsStore.getState(); 
  const setByPath = useSettingsStore(state => state.setByPath); 

  

  

  const handleDetach = () => {
    setIsDetached(!isDetached);
  };

  const renderSettings = () => (
    <div className="space-y-4">
      {Object.entries(subsectionSettings).map(([settingKey, settingDef]) => {
        
        if (settingDef.isAdvanced && !advancedMode) {
          return null;
        }

        
        const isPowerUser = useSettingsStore.getState().isPowerUser;
        if (settingDef.isPowerUserOnly && !isPowerUser) {
          
          
          return null;
        }

        
        const value = settingsStore.get(settingDef.path);
        const handleChange = (newValue: any) => {
          
          setByPath(settingDef.path, newValue);
        };

        return (
          <SettingControlComponent
            key={settingKey}
            path={settingDef.path}
            settingDef={settingDef}
            value={value}
            onChange={handleChange}
          />
        );
      })}
    </div>
  );

  if (isDetached) {
    return (
      <DetachedSection
        title={title}
        onReattach={handleDetach}
        sectionId={id}
      >
        <div className="p-2"> {}
          {renderSettings()}
        </div>
      </DetachedSection>
    );
  }

  return (
    <Card className="settings-section bg-card border border-border"> {}
      <CardHeader className="py-2 px-4">
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <div className="flex items-center justify-between">
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 p-0 hover:bg-muted/50"> {}
                <CardTitle className="text-sm font-medium text-card-foreground">{title}</CardTitle>
                {isOpen ? <ChevronUp className="ml-2 h-4 w-4 text-muted-foreground" /> : <ChevronDown className="ml-2 h-4 w-4 text-muted-foreground" />}
              </Button>
            </CollapsibleTrigger>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-muted-foreground hover:text-card-foreground hover:bg-muted/50" 
              onClick={handleDetach}
              title="Detach section"
            >
              <Maximize className="h-3 w-3" />
            </Button>
          </div>

          <CollapsibleContent>
            <CardContent className="p-4 pt-3"> {}
              {renderSettings()}
            </CardContent>
          </CollapsibleContent>
        </Collapsible>
      </CardHeader>
    </Card>
  );
}

// Detached floating section component (Keep as is, but ensure it uses the new renderSettings)
function DetachedSection({
  children,
  title,
  onReattach,
  sectionId
}: {
  children: React.ReactNode;
  title: string;
  onReattach: () => void;
  sectionId: string;
}) {
  const [position, setPosition] = useState({ x: 100, y: 100 });

  const handleDrag = (e: any, data: { x: number; y: number }) => {
    setPosition({ x: data.x, y: data.y });
  };

  
  

  return (
    <Draggable
      handle=".drag-handle" 
      position={position}
      onDrag={handleDrag}
      bounds="body" 
    >
      <div
        className="detached-panel absolute z-[3000] min-w-[300px] bg-card rounded-lg shadow-lg border border-border" 
        data-section-id={sectionId}
      >
        <div className="drag-handle flex items-center justify-between border-b border-border p-2 cursor-move bg-muted/50 rounded-t-lg"> {}
          <div className="text-sm font-medium text-card-foreground">
            {title}
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-muted-foreground hover:text-card-foreground hover:bg-muted/50" 
            onClick={onReattach}
            title="Reattach section"
          >
            <Minimize className="h-3 w-3" />
          </Button>
        </div>
        <div className="p-4 max-h-[400px] overflow-y-auto custom-scrollbar"> {}
          {children}
        </div>
      </div>
    </Draggable>
  );
}