import React from 'react';
import { Badge } from '../../design-system/components/Badge';
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '../../design-system/components/Collapsible';
import { useOntologyStore } from '../store/useOntologyStore';
import { AlertCircle, CheckCircle, Clock, ChevronDown } from 'lucide-react';
import { Button } from '../../design-system/components/Button';

export function ValidationStatus() {
  const { validating, violations } = useOntologyStore();
  const [isOpen, setIsOpen] = React.useState(false);

  const status = validating ? 'processing' : violations.length === 0 ? 'valid' : 'invalid';

  const statusConfig = {
    valid: {
      icon: CheckCircle,
      label: 'Valid',
      variant: 'default' as const,
      color: 'text-green-600'
    },
    invalid: {
      icon: AlertCircle,
      label: `${violations.length} Violations`,
      variant: 'destructive' as const,
      color: 'text-red-600'
    },
    processing: {
      icon: Clock,
      label: 'Validating...',
      variant: 'secondary' as const,
      color: 'text-blue-600'
    }
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <Button variant="ghost" className="flex items-center gap-2">
          <Badge variant={config.variant} className="flex items-center gap-1">
            <Icon className={`w-4 h-4 ${config.color}`} />
            {config.label}
          </Badge>
          {violations.length > 0 && (
            <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
          )}
        </Button>
      </CollapsibleTrigger>

      {violations.length > 0 && (
        <CollapsibleContent className="absolute right-0 mt-2 w-96 max-h-96 overflow-y-auto bg-white border border-gray-200 rounded-lg shadow-lg z-50">
          <div className="p-4 space-y-3">
            <h3 className="font-semibold text-sm">Validation Violations</h3>
            {violations.map((violation, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded-md space-y-1">
                <div className="flex items-center gap-2">
                  <Badge variant={violation.severity === 'error' ? 'destructive' : 'secondary'}>
                    {violation.severity}
                  </Badge>
                  <span className="text-sm font-medium">{violation.axiomType}</span>
                </div>
                <p className="text-sm text-gray-600">{violation.description}</p>
                {violation.affectedEntities.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs text-gray-500">Affected entities:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {violation.affectedEntities.slice(0, 3).map((entity, i) => (
                        <Badge key={i} variant="outline" className="text-xs">
                          {entity}
                        </Badge>
                      ))}
                      {violation.affectedEntities.length > 3 && (
                        <Badge variant="outline" className="text-xs">
                          +{violation.affectedEntities.length - 3} more
                        </Badge>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CollapsibleContent>
      )}
    </Collapsible>
  );
}
