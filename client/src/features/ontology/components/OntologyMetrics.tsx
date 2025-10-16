import React from 'react';
import { Card } from '../../design-system/components/Card';
import { Badge } from '../../design-system/components/Badge';
import { Progress } from '../../design-system/components/Progress';
import { useOntologyStore } from '../store/useOntologyStore';
import { FileText, Box, Link2, Users, Gauge, Clock } from 'lucide-react';

interface OntologyMetricsProps {
  detailed?: boolean;
}

export function OntologyMetrics({ detailed = false }: OntologyMetricsProps) {
  const { metrics } = useOntologyStore();

  const formatTimestamp = (timestamp?: number) => {
    if (!timestamp) return 'Never';
    return new Date(timestamp).toLocaleString();
  };

  const metricCards = [
    {
      icon: FileText,
      label: 'Axioms',
      value: metrics.axiomCount,
      color: 'text-blue-600'
    },
    {
      icon: Box,
      label: 'Classes',
      value: metrics.classCount,
      color: 'text-green-600'
    },
    {
      icon: Link2,
      label: 'Properties',
      value: metrics.propertyCount,
      color: 'text-purple-600'
    },
    {
      icon: Users,
      label: 'Individuals',
      value: metrics.individualCount,
      color: 'text-orange-600'
    }
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metricCards.map((metric) => {
          const Icon = metric.icon;
          return (
            <Card key={metric.label} className="p-4 space-y-2">
              <div className="flex items-center gap-2">
                <Icon className={`w-5 h-5 ${metric.color}`} />
                <span className="text-sm text-gray-600">{metric.label}</span>
              </div>
              <p className="text-2xl font-bold">{metric.value.toLocaleString()}</p>
            </Card>
          );
        })}
      </div>

      {detailed && (
        <>
          <Card className="p-4 space-y-3">
            <h4 className="font-semibold flex items-center gap-2">
              <Gauge className="w-4 h-4" />
              Performance Metrics
            </h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Cache Hit Rate</span>
                <Badge variant="secondary">
                  {(metrics.cacheHitRate * 100).toFixed(1)}%
                </Badge>
              </div>
              <Progress value={metrics.cacheHitRate * 100} className="h-2" />
            </div>
            <div className="flex items-center justify-between pt-2">
              <span className="text-sm text-gray-600">Validation Time</span>
              <Badge variant="outline">
                {metrics.validationTimeMs.toFixed(0)}ms
              </Badge>
            </div>
          </Card>

          <Card className="p-4 space-y-3">
            <h4 className="font-semibold">Constraints by Type</h4>
            <div className="space-y-2">
              {Object.entries(metrics.constraintsByType).length === 0 ? (
                <p className="text-sm text-gray-500">No constraints loaded</p>
              ) : (
                Object.entries(metrics.constraintsByType).map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">{type}</span>
                    <Badge variant="secondary">{count}</Badge>
                  </div>
                ))
              )}
            </div>
          </Card>

          {metrics.lastValidated && (
            <Card className="p-4">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Clock className="w-4 h-4" />
                <span>Last validated: {formatTimestamp(metrics.lastValidated)}</span>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
