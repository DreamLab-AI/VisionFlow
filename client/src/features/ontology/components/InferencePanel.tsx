/**
 * Inference Panel
 *
 * Provides ontology reasoning tools via Phase 7 Inference API:
 * - Run inference on loaded ontologies
 * - Validate ontology consistency
 * - View inference results and explanations
 * - Classification and consistency reports
 * - Cache management
 */

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Label } from '@/features/design-system/components/Label';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Input } from '@/features/design-system/components/Input';
import { Switch } from '@/features/design-system/components/Switch';
import { useToast } from '@/features/design-system/components/Toast';
import { Brain, CheckCircle, AlertTriangle, AlertCircle, Trash2, FileText } from 'lucide-react';
import { useInferenceService, RunInferenceResponse, OntologyClassification } from '../hooks/useInferenceService';

interface InferencePanelProps {
  ontologyId?: string;
  className?: string;
}

export function InferencePanel({ ontologyId: propOntologyId, className }: InferencePanelProps) {
  const { toast } = useToast();
  const {
    loading,
    error,
    runInference,
    validateOntology,
    getClassification,
    getConsistencyReport,
    invalidateCache,
  } = useInferenceService();

  const [ontologyId, setOntologyId] = useState(propOntologyId || '');
  const [force, setForce] = useState(false);
  const [inferenceResults, setInferenceResults] = useState<RunInferenceResponse | null>(null);
  const [classification, setClassification] = useState<OntologyClassification | null>(null);
  const [validationResult, setValidationResult] = useState<{
    success: boolean;
    consistent: boolean;
    message: string;
  } | null>(null);

  const handleRunInference = async () => {
    if (!ontologyId) {
      toast({
        title: 'Invalid Input',
        description: 'Please enter an ontology ID',
        variant: 'destructive',
      });
      return;
    }

    try {
      const result = await runInference({ ontology_id: ontologyId, force });
      setInferenceResults(result);

      if (result.success) {
        toast({
          title: 'Inference Complete',
          description: `Inferred ${result.inferred_axioms_count} axioms in ${result.inference_time_ms}ms`,
          variant: 'default',
        });
      } else {
        toast({
          title: 'Inference Failed',
          description: result.error || 'Unknown error',
          variant: 'destructive',
        });
      }
    } catch (err: any) {
      toast({
        title: 'Inference Failed',
        description: err.message || 'Failed to run inference',
        variant: 'destructive',
      });
    }
  };

  const handleValidate = async () => {
    if (!ontologyId) {
      toast({
        title: 'Invalid Input',
        description: 'Please enter an ontology ID',
        variant: 'destructive',
      });
      return;
    }

    try {
      const result = await validateOntology({ ontology_id: ontologyId });
      setValidationResult(result);

      if (result.consistent) {
        toast({
          title: 'Validation Complete',
          description: 'Ontology is consistent',
          variant: 'default',
        });
      } else {
        toast({
          title: 'Validation Failed',
          description: 'Ontology has inconsistencies',
          variant: 'destructive',
        });
      }
    } catch (err: any) {
      toast({
        title: 'Validation Failed',
        description: err.message || 'Failed to validate ontology',
        variant: 'destructive',
      });
    }
  };

  const handleGetClassification = async () => {
    if (!ontologyId) {
      toast({
        title: 'Invalid Input',
        description: 'Please enter an ontology ID',
        variant: 'destructive',
      });
      return;
    }

    try {
      const result = await getClassification(ontologyId);
      setClassification(result);
      toast({
        title: 'Classification Retrieved',
        description: `Found ${result.classes} classes, ${result.properties} properties`,
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Classification Failed',
        description: err.message || 'Failed to get classification',
        variant: 'destructive',
      });
    }
  };

  const handleInvalidateCache = async () => {
    if (!ontologyId) {
      toast({
        title: 'Invalid Input',
        description: 'Please enter an ontology ID',
        variant: 'destructive',
      });
      return;
    }

    try {
      await invalidateCache(ontologyId);
      toast({
        title: 'Cache Cleared',
        description: 'Inference cache has been invalidated',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Clear Failed',
        description: err.message || 'Failed to invalidate cache',
        variant: 'destructive',
      });
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Ontology Inference
        </CardTitle>
        <CardDescription>Run reasoning and validation on ontologies</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Ontology ID Input */}
        <div className="space-y-2">
          <Label htmlFor="ontology-id">Ontology ID</Label>
          <Input
            id="ontology-id"
            placeholder="Enter ontology identifier"
            value={ontologyId}
            onChange={(e) => setOntologyId(e.target.value)}
          />
        </div>

        {/* Force Option */}
        <div className="flex items-center justify-between">
          <Label htmlFor="force">Force Re-Inference (ignore cache)</Label>
          <Switch id="force" checked={force} onCheckedChange={setForce} />
        </div>

        {/* Actions */}
        <div className="grid grid-cols-2 gap-3">
          <Button onClick={handleRunInference} disabled={loading || !ontologyId} className="w-full">
            <Brain className="mr-2 h-4 w-4" />
            Run Inference
          </Button>
          <Button onClick={handleValidate} disabled={loading || !ontologyId} variant="outline" className="w-full">
            <CheckCircle className="mr-2 h-4 w-4" />
            Validate
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Button
            onClick={handleGetClassification}
            disabled={loading || !ontologyId}
            variant="outline"
            className="w-full"
          >
            <FileText className="mr-2 h-4 w-4" />
            Classification
          </Button>
          <Button
            onClick={handleInvalidateCache}
            disabled={loading || !ontologyId}
            variant="outline"
            className="w-full"
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Clear Cache
          </Button>
        </div>

        {/* Inference Results */}
        {inferenceResults && (
          <div className="border-t pt-4">
            <div className="rounded-lg border p-4 space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                {inferenceResults.success ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Inference Complete
                  </>
                ) : (
                  <>
                    <AlertTriangle className="h-4 w-4 text-destructive" />
                    Inference Failed
                  </>
                )}
              </h4>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Ontology ID:</span>
                  <Badge>{inferenceResults.ontology_id}</Badge>
                </div>
                {inferenceResults.success && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Inferred Axioms:</span>
                      <span className="font-medium">{inferenceResults.inferred_axioms_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Inference Time:</span>
                      <span className="font-medium">{inferenceResults.inference_time_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Reasoner:</span>
                      <span className="font-medium">{inferenceResults.reasoner_version}</span>
                    </div>
                  </>
                )}
                {inferenceResults.error && (
                  <div className="rounded-lg border border-destructive bg-destructive/10 p-2 text-sm text-destructive">
                    {inferenceResults.error}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Validation Results */}
        {validationResult && (
          <div className="border-t pt-4">
            <div className="rounded-lg border p-4 space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                {validationResult.consistent ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Consistent
                  </>
                ) : (
                  <>
                    <AlertTriangle className="h-4 w-4 text-destructive" />
                    Inconsistent
                  </>
                )}
              </h4>

              <p className="text-sm text-muted-foreground">{validationResult.message}</p>
            </div>
          </div>
        )}

        {/* Classification Results */}
        {classification && (
          <div className="border-t pt-4">
            <div className="rounded-lg border p-4 space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Classification
              </h4>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Classes:</span>
                  <Badge>{classification.classes}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Properties:</span>
                  <Badge>{classification.properties}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Individuals:</span>
                  <Badge>{classification.individuals}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Axioms:</span>
                  <Badge>{classification.axioms}</Badge>
                </div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-destructive bg-destructive/10 p-3 flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-destructive">Error</p>
              <p className="text-sm text-destructive/90">{error}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
