import React from 'react';
import { Card } from '../../design-system/components/Card';
import { Button } from '../../design-system/components/Button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../design-system/components/Tabs';
import { ValidationStatus } from './ValidationStatus';
import { ConstraintGroupControl } from './ConstraintGroupControl';
import { OntologyMetrics } from './OntologyMetrics';
import { useOntologyStore } from '../store/useOntologyStore';
import { useOntologyWebSocket } from '../hooks/useOntologyWebSocket';
import { Upload, RefreshCw } from 'lucide-react';

export function OntologyPanel() {
  const { loaded, validating, constraintGroups, loadOntology, validateOntology } = useOntologyStore();
  const [selectedFile, setSelectedFile] = React.useState<string>('');
  const [loading, setLoading] = React.useState(false);

  useOntologyWebSocket();

  const handleLoadOntology = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      await loadOntology(selectedFile);
    } catch (error) {
      console.error('Failed to load ontology:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleValidate = async () => {
    try {
      await validateOntology();
    } catch (error) {
      console.error('Failed to validate:', error);
    }
  };

  return (
    <Card className="w-full max-w-4xl p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Ontology Management</h2>
        <ValidationStatus />
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="constraints">Constraints</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="space-y-4">
            <div className="flex gap-4">
              <input
                type="text"
                placeholder="Ontology file URL or path"
                value={selectedFile}
                onChange={(e) => setSelectedFile(e.target.value)}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <Button
                onClick={handleLoadOntology}
                disabled={!selectedFile || loading || validating}
              >
                <Upload className="w-4 h-4 mr-2" />
                Load Ontology
              </Button>
            </div>

            {loaded && (
              <div className="flex gap-4">
                <Button
                  onClick={handleValidate}
                  disabled={validating}
                  variant="outline"
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${validating ? 'animate-spin' : ''}`} />
                  Validate
                </Button>
              </div>
            )}

            <OntologyMetrics />
          </div>
        </TabsContent>

        <TabsContent value="constraints" className="space-y-4">
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">Constraint Groups</h3>
            <p className="text-sm text-gray-600">
              Enable or disable constraint groups and adjust their strength
            </p>
          </div>

          <div className="space-y-3">
            {constraintGroups.map((group) => (
              <ConstraintGroupControl key={group.id} group={group} />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          <OntologyMetrics detailed />
        </TabsContent>
      </Tabs>
    </Card>
  );
}
