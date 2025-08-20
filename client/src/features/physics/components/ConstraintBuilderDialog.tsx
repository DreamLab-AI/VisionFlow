import React, { useState, useCallback } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/features/design-system/components/Dialog';
import { Button } from '@/features/design-system/components/Button';
import { Input } from '@/features/design-system/components/Input';
import { Label } from '@/features/design-system/components/Label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Slider } from '@/features/design-system/components/Slider';
import { Badge } from '@/features/design-system/components/Badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import Plus from 'lucide-react/dist/esm/icons/plus';
import X from 'lucide-react/dist/esm/icons/x';
import Save from 'lucide-react/dist/esm/icons/save';
import Copy from 'lucide-react/dist/esm/icons/copy';
import Upload from 'lucide-react/dist/esm/icons/upload';
import { useToast } from '@/features/design-system/components/Toast';

interface ConstraintRule {
  id: string;
  type: string;
  name: string;
  params: Record<string, any>;
  nodes: string[];
  active: boolean;
}

interface ConstraintBuilderDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (constraint: ConstraintRule) => void;
  existingConstraint?: ConstraintRule;
}

export function ConstraintBuilderDialog({
  isOpen,
  onClose,
  onSave,
  existingConstraint,
}: ConstraintBuilderDialogProps) {
  const { toast } = useToast();
  const [constraint, setConstraint] = useState<ConstraintRule>(
    existingConstraint || {
      id: `constraint_${Date.now()}`,
      type: 'separation',
      name: '',
      params: {},
      nodes: [],
      active: true,
    }
  );
  
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [nodeSelectionMode, setNodeSelectionMode] = useState<'manual' | 'query' | 'group'>('manual');

  const constraintTypes = [
    {
      id: 'separation',
      name: 'Separation',
      description: 'Maintain minimum distance between nodes',
      params: [
        { key: 'minDistance', label: 'Minimum Distance', type: 'slider', min: 10, max: 500, default: 100 },
        { key: 'strength', label: 'Strength', type: 'slider', min: 0, max: 1, step: 0.1, default: 0.5 },
      ],
    },
    {
      id: 'alignment',
      name: 'Alignment',
      description: 'Align nodes along an axis',
      params: [
        { key: 'axis', label: 'Axis', type: 'select', options: ['x', 'y', 'z'], default: 'x' },
        { key: 'tolerance', label: 'Tolerance', type: 'slider', min: 0, max: 50, default: 10 },
        { key: 'strength', label: 'Strength', type: 'slider', min: 0, max: 1, step: 0.1, default: 0.7 },
      ],
    },
    {
      id: 'cluster',
      name: 'Cluster',
      description: 'Group nodes together',
      params: [
        { key: 'radius', label: 'Cluster Radius', type: 'slider', min: 50, max: 500, default: 200 },
        { key: 'centerX', label: 'Center X', type: 'number', default: 0 },
        { key: 'centerY', label: 'Center Y', type: 'number', default: 0 },
        { key: 'centerZ', label: 'Center Z', type: 'number', default: 0 },
        { key: 'strength', label: 'Strength', type: 'slider', min: 0, max: 1, step: 0.1, default: 0.5 },
      ],
    },
    {
      id: 'fixed',
      name: 'Fixed Position',
      description: 'Lock nodes to specific positions',
      params: [
        { key: 'x', label: 'X Position', type: 'number', default: 0 },
        { key: 'y', label: 'Y Position', type: 'number', default: 0 },
        { key: 'z', label: 'Z Position', type: 'number', default: 0 },
        { key: 'allowRotation', label: 'Allow Rotation', type: 'boolean', default: false },
      ],
    },
    {
      id: 'boundary',
      name: 'Boundary',
      description: 'Keep nodes within bounds',
      params: [
        { key: 'minX', label: 'Min X', type: 'number', default: -1000 },
        { key: 'maxX', label: 'Max X', type: 'number', default: 1000 },
        { key: 'minY', label: 'Min Y', type: 'number', default: -1000 },
        { key: 'maxY', label: 'Max Y', type: 'number', default: 1000 },
        { key: 'minZ', label: 'Min Z', type: 'number', default: -1000 },
        { key: 'maxZ', label: 'Max Z', type: 'number', default: 1000 },
        { key: 'bounce', label: 'Bounce', type: 'slider', min: 0, max: 1, step: 0.1, default: 0.5 },
      ],
    },
    {
      id: 'radial',
      name: 'Radial',
      description: 'Arrange nodes in circles',
      params: [
        { key: 'centerX', label: 'Center X', type: 'number', default: 0 },
        { key: 'centerY', label: 'Center Y', type: 'number', default: 0 },
        { key: 'radius', label: 'Radius', type: 'slider', min: 50, max: 1000, default: 300 },
        { key: 'angleOffset', label: 'Angle Offset', type: 'slider', min: 0, max: 360, default: 0 },
        { key: 'strength', label: 'Strength', type: 'slider', min: 0, max: 1, step: 0.1, default: 0.7 },
      ],
    },
    {
      id: 'tree',
      name: 'Tree Layout',
      description: 'Hierarchical tree structure',
      params: [
        { key: 'direction', label: 'Direction', type: 'select', options: ['top', 'bottom', 'left', 'right'], default: 'top' },
        { key: 'levelGap', label: 'Level Gap', type: 'slider', min: 50, max: 300, default: 150 },
        { key: 'siblingGap', label: 'Sibling Gap', type: 'slider', min: 20, max: 200, default: 80 },
        { key: 'strength', label: 'Strength', type: 'slider', min: 0, max: 1, step: 0.1, default: 0.8 },
      ],
    },
    {
      id: 'layer',
      name: 'Layer',
      description: 'Organize nodes in layers',
      params: [
        { key: 'layerIndex', label: 'Layer Index', type: 'number', default: 0 },
        { key: 'layerSpacing', label: 'Layer Spacing', type: 'slider', min: 50, max: 500, default: 200 },
        { key: 'direction', label: 'Direction', type: 'select', options: ['x', 'y', 'z'], default: 'z' },
        { key: 'strength', label: 'Strength', type: 'slider', min: 0, max: 1, step: 0.1, default: 0.6 },
      ],
    },
  ];

  const currentType = constraintTypes.find(t => t.id === constraint.type);

  const handleTypeChange = (type: string) => {
    const typeConfig = constraintTypes.find(t => t.id === type);
    if (typeConfig) {
      const defaultParams: Record<string, any> = {};
      typeConfig.params.forEach(param => {
        defaultParams[param.key] = param.default;
      });
      setConstraint({
        ...constraint,
        type,
        params: defaultParams,
      });
    }
  };

  const handleParamChange = (key: string, value: any) => {
    setConstraint({
      ...constraint,
      params: {
        ...constraint.params,
        [key]: value,
      },
    });
  };

  const handleNodeSelection = (nodeId: string) => {
    setSelectedNodes(prev => {
      if (prev.includes(nodeId)) {
        return prev.filter(id => id !== nodeId);
      }
      return [...prev, nodeId];
    });
  };

  const handleSave = () => {
    if (!constraint.name) {
      toast({
        title: 'Error',
        description: 'Please provide a name for the constraint',
        variant: 'destructive',
      });
      return;
    }

    const finalConstraint = {
      ...constraint,
      nodes: selectedNodes,
    };

    onSave(finalConstraint);
    onClose();
    
    toast({
      title: 'Constraint Saved',
      description: `${constraint.name} has been created`,
    });
  };

  const renderParamControl = (param: any) => {
    const value = constraint.params[param.key] ?? param.default;

    switch (param.type) {
      case 'slider':
        return (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label>{param.label}</Label>
              <span className="text-sm text-muted-foreground">{value}</span>
            </div>
            <Slider
              min={param.min}
              max={param.max}
              step={param.step || 1}
              value={[value]}
              onValueChange={([v]) => handleParamChange(param.key, v)}
            />
          </div>
        );

      case 'number':
        return (
          <div className="space-y-2">
            <Label>{param.label}</Label>
            <Input
              type="number"
              value={value}
              onChange={(e) => handleParamChange(param.key, parseFloat(e.target.value))}
            />
          </div>
        );

      case 'select':
        return (
          <div className="space-y-2">
            <Label>{param.label}</Label>
            <Select value={value} onValueChange={(v) => handleParamChange(param.key, v)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {param.options.map((option: string) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        );

      case 'boolean':
        return (
          <div className="flex items-center justify-between">
            <Label>{param.label}</Label>
            <Button
              variant={value ? 'default' : 'outline'}
              size="sm"
              onClick={() => handleParamChange(param.key, !value)}
            >
              {value ? 'Enabled' : 'Disabled'}
            </Button>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle>Constraint Builder</DialogTitle>
          <DialogDescription>
            Create custom constraints to control node layout behavior
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-auto">
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="basic">Basic Settings</TabsTrigger>
              <TabsTrigger value="parameters">Parameters</TabsTrigger>
              <TabsTrigger value="nodes">Node Selection</TabsTrigger>
            </TabsList>

            <TabsContent value="basic" className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Constraint Name</Label>
                <Input
                  id="name"
                  placeholder="Enter a descriptive name"
                  value={constraint.name}
                  onChange={(e) => setConstraint({ ...constraint, name: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label>Constraint Type</Label>
                <Select value={constraint.type} onValueChange={handleTypeChange}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {constraintTypes.map(type => (
                      <SelectItem key={type.id} value={type.id}>
                        <div className="flex flex-col">
                          <span>{type.name}</span>
                          <span className="text-xs text-muted-foreground">{type.description}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {currentType && (
                <div className="p-4 rounded-lg bg-muted/50">
                  <p className="text-sm">{currentType.description}</p>
                </div>
              )}
            </TabsContent>

            <TabsContent value="parameters" className="space-y-4">
              {currentType ? (
                <ScrollArea className="h-[400px] pr-4">
                  <div className="space-y-4">
                    {currentType.params.map(param => (
                      <div key={param.key}>
                        {renderParamControl(param)}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="flex items-center justify-center h-[200px]">
                  <p className="text-muted-foreground">Select a constraint type first</p>
                </div>
              )}
            </TabsContent>

            <TabsContent value="nodes" className="space-y-4">
              <div className="space-y-2">
                <Label>Node Selection Mode</Label>
                <Tabs value={nodeSelectionMode} onValueChange={(v: any) => setNodeSelectionMode(v)}>
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="manual">Manual</TabsTrigger>
                    <TabsTrigger value="query">Query</TabsTrigger>
                    <TabsTrigger value="group">Group</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>

              {nodeSelectionMode === 'manual' && (
                <div>
                  <Label>Selected Nodes</Label>
                  <ScrollArea className="h-[200px] w-full rounded-md border p-4">
                    <div className="flex flex-wrap gap-2">
                      {selectedNodes.length > 0 ? (
                        selectedNodes.map(nodeId => (
                          <Badge
                            key={nodeId}
                            variant="secondary"
                            className="cursor-pointer"
                            onClick={() => handleNodeSelection(nodeId)}
                          >
                            {nodeId}
                            <X className="ml-1 h-3 w-3" />
                          </Badge>
                        ))
                      ) : (
                        <p className="text-sm text-muted-foreground">
                          Click nodes in the graph to select them
                        </p>
                      )}
                    </div>
                  </ScrollArea>
                  <p className="text-xs text-muted-foreground mt-2">
                    {selectedNodes.length} nodes selected
                  </p>
                </div>
              )}

              {nodeSelectionMode === 'query' && (
                <div className="space-y-2">
                  <Label htmlFor="node-query">Node Query</Label>
                  <Input
                    id="node-query"
                    placeholder="e.g., type:person OR label:important"
                    />
                  <p className="text-xs text-muted-foreground">
                    Use queries to select nodes dynamically
                  </p>
                </div>
              )}

              {nodeSelectionMode === 'group' && (
                <div className="space-y-2">
                  <Label>Select Group</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Choose a node group" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Nodes</SelectItem>
                      <SelectItem value="visible">Visible Nodes</SelectItem>
                      <SelectItem value="selected">Selected Nodes</SelectItem>
                      <SelectItem value="pinned">Pinned Nodes</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSave}>
            <Save className="mr-2 h-4 w-4" />
            Save Constraint
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}