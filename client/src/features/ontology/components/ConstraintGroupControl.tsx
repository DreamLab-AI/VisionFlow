import React from 'react';
import { Card } from '../../design-system/components/Card';
import { Switch } from '../../design-system/components/Switch';
import { Slider } from '../../design-system/components/Slider';
import { Label } from '../../design-system/components/Label';
import { Badge } from '../../design-system/components/Badge';
import { useOntologyStore, ConstraintGroup } from '../store/useOntologyStore';
import { Tooltip, TooltipContent, TooltipTrigger } from '../../design-system/components/Tooltip';
import { Info } from 'lucide-react';

interface ConstraintGroupControlProps {
  group: ConstraintGroup;
}

export function ConstraintGroupControl({ group }: ConstraintGroupControlProps) {
  const { toggleConstraintGroup, updateStrength } = useOntologyStore();

  return (
    <Card className={`p-4 transition-all ${group.enabled ? 'border-blue-500' : 'border-gray-200'}`}>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Switch
              checked={group.enabled}
              onCheckedChange={() => toggleConstraintGroup(group.id)}
              id={`constraint-${group.id}`}
            />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <Label htmlFor={`constraint-${group.id}`} className="text-base font-medium">
                  {group.name}
                </Label>
                <Tooltip>
                  <TooltipTrigger>
                    <Info className="w-4 h-4 text-gray-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">{group.description}</p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <p className="text-sm text-gray-500">{group.description}</p>
            </div>
          </div>
          <Badge variant={group.enabled ? 'default' : 'secondary'}>
            {group.constraintCount} constraints
          </Badge>
        </div>

        {group.enabled && (
          <div className="space-y-2 pl-10">
            <div className="flex items-center justify-between">
              <Label htmlFor={`strength-${group.id}`} className="text-sm">
                Strength
              </Label>
              <span className="text-sm font-medium text-gray-700">
                {(group.strength * 100).toFixed(0)}%
              </span>
            </div>
            <Slider
              id={`strength-${group.id}`}
              min={0}
              max={1}
              step={0.05}
              value={[group.strength]}
              onValueChange={(values) => updateStrength(group.id, values[0])}
              className="w-full"
            />
          </div>
        )}
      </div>
    </Card>
  );
}
