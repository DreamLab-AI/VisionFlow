import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { Button } from '@/features/design-system/components/Button';
import { Input } from '@/features/design-system/components/Input';
import { Label } from '@/features/design-system/components/Label';
import { Textarea } from '@/features/design-system/components/Textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Badge } from '@/features/design-system/components/Badge';
import { useToast } from '@/features/design-system/components/Toast';
import { Plus, Save, Send, Layers, Link2, Tag, AlertCircle } from 'lucide-react';
import {
  useOntologyContributionStore,
  OntologyClass,
  OntologyProperty,
  OntologyAnnotation,
  ProposalType
} from '../hooks/useOntologyStore';

interface OntologyContributionProps {
  className?: string;
  onProposalCreated?: (proposalId: string) => void;
}

export function OntologyContribution({ className, onProposalCreated }: OntologyContributionProps) {
  const { toast } = useToast();
  const {
    classes,
    properties,
    proposals,
    loading,
    error,
    createProposal,
    clearError
  } = useOntologyContributionStore();

  const [activeTab, setActiveTab] = useState<ProposalType>('class');

  // Class form state
  const [classForm, setClassForm] = useState<Partial<OntologyClass>>({
    iri: '',
    label: '',
    parentClass: '',
    description: ''
  });

  // Property form state
  const [propertyForm, setPropertyForm] = useState<Partial<OntologyProperty>>({
    iri: '',
    label: '',
    domain: '',
    range: '',
    propertyType: 'object',
    description: ''
  });

  // Annotation form state
  const [annotationForm, setAnnotationForm] = useState<Partial<OntologyAnnotation>>({
    targetIri: '',
    predicate: '',
    value: '',
    language: 'en'
  });

  const pendingProposals = proposals.filter(p => p.status === 'pending' || p.status === 'draft');

  const resetForms = useCallback(() => {
    setClassForm({ iri: '', label: '', parentClass: '', description: '' });
    setPropertyForm({ iri: '', label: '', domain: '', range: '', propertyType: 'object', description: '' });
    setAnnotationForm({ targetIri: '', predicate: '', value: '', language: 'en' });
  }, []);

  const handleCreateClassProposal = async (isDraft: boolean = true) => {
    if (!classForm.iri || !classForm.label) {
      toast({
        title: 'Validation Error',
        description: 'IRI and Label are required',
        variant: 'destructive'
      });
      return;
    }

    try {
      const proposal = await createProposal('class', classForm as OntologyClass);

      toast({
        title: isDraft ? 'Draft Saved' : 'Proposal Created',
        description: `Class proposal "${classForm.label}" has been ${isDraft ? 'saved as draft' : 'created'}`,
        variant: 'default'
      });

      resetForms();
      onProposalCreated?.(proposal.id);
    } catch (err: any) {
      toast({
        title: 'Error',
        description: err.message || 'Failed to create proposal',
        variant: 'destructive'
      });
    }
  };

  const handleCreatePropertyProposal = async (isDraft: boolean = true) => {
    if (!propertyForm.iri || !propertyForm.label) {
      toast({
        title: 'Validation Error',
        description: 'IRI and Label are required',
        variant: 'destructive'
      });
      return;
    }

    try {
      const proposal = await createProposal('property', propertyForm as OntologyProperty);

      toast({
        title: isDraft ? 'Draft Saved' : 'Proposal Created',
        description: `Property proposal "${propertyForm.label}" has been ${isDraft ? 'saved as draft' : 'created'}`,
        variant: 'default'
      });

      resetForms();
      onProposalCreated?.(proposal.id);
    } catch (err: any) {
      toast({
        title: 'Error',
        description: err.message || 'Failed to create proposal',
        variant: 'destructive'
      });
    }
  };

  const handleCreateAnnotationProposal = async (isDraft: boolean = true) => {
    if (!annotationForm.targetIri || !annotationForm.predicate || !annotationForm.value) {
      toast({
        title: 'Validation Error',
        description: 'Target, Predicate, and Value are required',
        variant: 'destructive'
      });
      return;
    }

    try {
      const proposal = await createProposal('annotation', annotationForm as OntologyAnnotation);

      toast({
        title: isDraft ? 'Draft Saved' : 'Proposal Created',
        description: `Annotation proposal has been ${isDraft ? 'saved as draft' : 'created'}`,
        variant: 'default'
      });

      resetForms();
      onProposalCreated?.(proposal.id);
    } catch (err: any) {
      toast({
        title: 'Error',
        description: err.message || 'Failed to create proposal',
        variant: 'destructive'
      });
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5" />
              Contribute to Ontology
            </CardTitle>
            <CardDescription>
              Propose additions or modifications to the public ontology
            </CardDescription>
          </div>
          {pendingProposals.length > 0 && (
            <Badge variant="secondary">
              {pendingProposals.length} pending
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {error && (
          <div className="rounded-lg border border-destructive bg-destructive/10 p-3 flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-destructive">{error}</p>
              <Button
                variant="ghost"
                size="sm"
                className="mt-1 h-6 px-2 text-xs"
                onClick={clearError}
              >
                Dismiss
              </Button>
            </div>
          </div>
        )}

        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as ProposalType)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="class">
              <Layers className="h-4 w-4 mr-1" />
              Class
            </TabsTrigger>
            <TabsTrigger value="property">
              <Link2 className="h-4 w-4 mr-1" />
              Property
            </TabsTrigger>
            <TabsTrigger value="annotation">
              <Tag className="h-4 w-4 mr-1" />
              Annotation
            </TabsTrigger>
          </TabsList>

          {/* Class Definition Form */}
          <TabsContent value="class" className="space-y-4">
            <div className="space-y-3">
              <div>
                <Label htmlFor="class-iri">IRI *</Label>
                <Input
                  id="class-iri"
                  placeholder="e.g., http://example.org/ontology#MyClass"
                  value={classForm.iri}
                  onChange={(e) => setClassForm({ ...classForm, iri: e.target.value })}
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Unique identifier for the class
                </p>
              </div>

              <div>
                <Label htmlFor="class-label">Label *</Label>
                <Input
                  id="class-label"
                  placeholder="e.g., My Class"
                  value={classForm.label}
                  onChange={(e) => setClassForm({ ...classForm, label: e.target.value })}
                />
              </div>

              <div>
                <Label htmlFor="class-parent">Parent Class</Label>
                <Select
                  value={classForm.parentClass}
                  onValueChange={(v) => setClassForm({ ...classForm, parentClass: v })}
                >
                  <SelectTrigger id="class-parent">
                    <SelectValue placeholder="Select parent class (optional)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">None (root class)</SelectItem>
                    {classes.map((cls) => (
                      <SelectItem key={cls.iri} value={cls.iri}>
                        {cls.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="class-description">Description</Label>
                <Textarea
                  id="class-description"
                  placeholder="Describe the purpose of this class..."
                  value={classForm.description}
                  onChange={(e) => setClassForm({ ...classForm, description: e.target.value })}
                  rows={3}
                />
              </div>

              <div className="flex gap-2 pt-2">
                <Button
                  variant="outline"
                  onClick={() => handleCreateClassProposal(true)}
                  disabled={loading}
                >
                  <Save className="h-4 w-4 mr-2" />
                  Save Draft
                </Button>
                <Button
                  onClick={() => handleCreateClassProposal(false)}
                  disabled={loading}
                >
                  <Send className="h-4 w-4 mr-2" />
                  Submit Proposal
                </Button>
              </div>
            </div>
          </TabsContent>

          {/* Property Definition Form */}
          <TabsContent value="property" className="space-y-4">
            <div className="space-y-3">
              <div>
                <Label htmlFor="prop-iri">IRI *</Label>
                <Input
                  id="prop-iri"
                  placeholder="e.g., http://example.org/ontology#hasProperty"
                  value={propertyForm.iri}
                  onChange={(e) => setPropertyForm({ ...propertyForm, iri: e.target.value })}
                />
              </div>

              <div>
                <Label htmlFor="prop-label">Label *</Label>
                <Input
                  id="prop-label"
                  placeholder="e.g., has property"
                  value={propertyForm.label}
                  onChange={(e) => setPropertyForm({ ...propertyForm, label: e.target.value })}
                />
              </div>

              <div>
                <Label htmlFor="prop-type">Property Type</Label>
                <Select
                  value={propertyForm.propertyType}
                  onValueChange={(v) => setPropertyForm({ ...propertyForm, propertyType: v as 'object' | 'data' | 'annotation' })}
                >
                  <SelectTrigger id="prop-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="object">Object Property</SelectItem>
                    <SelectItem value="data">Data Property</SelectItem>
                    <SelectItem value="annotation">Annotation Property</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="prop-domain">Domain</Label>
                <Select
                  value={propertyForm.domain}
                  onValueChange={(v) => setPropertyForm({ ...propertyForm, domain: v })}
                >
                  <SelectTrigger id="prop-domain">
                    <SelectValue placeholder="Select domain class (optional)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">owl:Thing (any)</SelectItem>
                    {classes.map((cls) => (
                      <SelectItem key={cls.iri} value={cls.iri}>
                        {cls.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="prop-range">Range</Label>
                {propertyForm.propertyType === 'object' ? (
                  <Select
                    value={propertyForm.range}
                    onValueChange={(v) => setPropertyForm({ ...propertyForm, range: v })}
                  >
                    <SelectTrigger id="prop-range">
                      <SelectValue placeholder="Select range class (optional)" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">owl:Thing (any)</SelectItem>
                      {classes.map((cls) => (
                        <SelectItem key={cls.iri} value={cls.iri}>
                          {cls.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Select
                    value={propertyForm.range}
                    onValueChange={(v) => setPropertyForm({ ...propertyForm, range: v })}
                  >
                    <SelectTrigger id="prop-range">
                      <SelectValue placeholder="Select datatype" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="xsd:string">String</SelectItem>
                      <SelectItem value="xsd:integer">Integer</SelectItem>
                      <SelectItem value="xsd:float">Float</SelectItem>
                      <SelectItem value="xsd:boolean">Boolean</SelectItem>
                      <SelectItem value="xsd:dateTime">DateTime</SelectItem>
                      <SelectItem value="xsd:date">Date</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              </div>

              <div>
                <Label htmlFor="prop-description">Description</Label>
                <Textarea
                  id="prop-description"
                  placeholder="Describe the purpose of this property..."
                  value={propertyForm.description}
                  onChange={(e) => setPropertyForm({ ...propertyForm, description: e.target.value })}
                  rows={3}
                />
              </div>

              <div className="flex gap-2 pt-2">
                <Button
                  variant="outline"
                  onClick={() => handleCreatePropertyProposal(true)}
                  disabled={loading}
                >
                  <Save className="h-4 w-4 mr-2" />
                  Save Draft
                </Button>
                <Button
                  onClick={() => handleCreatePropertyProposal(false)}
                  disabled={loading}
                >
                  <Send className="h-4 w-4 mr-2" />
                  Submit Proposal
                </Button>
              </div>
            </div>
          </TabsContent>

          {/* Annotation Form */}
          <TabsContent value="annotation" className="space-y-4">
            <div className="space-y-3">
              <div>
                <Label htmlFor="ann-target">Target IRI *</Label>
                <Input
                  id="ann-target"
                  placeholder="e.g., http://example.org/ontology#MyClass"
                  value={annotationForm.targetIri}
                  onChange={(e) => setAnnotationForm({ ...annotationForm, targetIri: e.target.value })}
                />
                <p className="text-xs text-muted-foreground mt-1">
                  The class or property to annotate
                </p>
              </div>

              <div>
                <Label htmlFor="ann-predicate">Predicate *</Label>
                <Select
                  value={annotationForm.predicate}
                  onValueChange={(v) => setAnnotationForm({ ...annotationForm, predicate: v })}
                >
                  <SelectTrigger id="ann-predicate">
                    <SelectValue placeholder="Select annotation type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rdfs:label">Label</SelectItem>
                    <SelectItem value="rdfs:comment">Comment</SelectItem>
                    <SelectItem value="rdfs:seeAlso">See Also</SelectItem>
                    <SelectItem value="rdfs:isDefinedBy">Defined By</SelectItem>
                    <SelectItem value="owl:deprecated">Deprecated</SelectItem>
                    <SelectItem value="skos:prefLabel">Preferred Label</SelectItem>
                    <SelectItem value="skos:altLabel">Alternative Label</SelectItem>
                    <SelectItem value="skos:definition">Definition</SelectItem>
                    <SelectItem value="skos:example">Example</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="ann-value">Value *</Label>
                <Textarea
                  id="ann-value"
                  placeholder="Enter annotation value..."
                  value={annotationForm.value}
                  onChange={(e) => setAnnotationForm({ ...annotationForm, value: e.target.value })}
                  rows={3}
                />
              </div>

              <div>
                <Label htmlFor="ann-language">Language</Label>
                <Select
                  value={annotationForm.language}
                  onValueChange={(v) => setAnnotationForm({ ...annotationForm, language: v })}
                >
                  <SelectTrigger id="ann-language">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="de">German</SelectItem>
                    <SelectItem value="fr">French</SelectItem>
                    <SelectItem value="es">Spanish</SelectItem>
                    <SelectItem value="it">Italian</SelectItem>
                    <SelectItem value="pt">Portuguese</SelectItem>
                    <SelectItem value="ja">Japanese</SelectItem>
                    <SelectItem value="zh">Chinese</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex gap-2 pt-2">
                <Button
                  variant="outline"
                  onClick={() => handleCreateAnnotationProposal(true)}
                  disabled={loading}
                >
                  <Save className="h-4 w-4 mr-2" />
                  Save Draft
                </Button>
                <Button
                  onClick={() => handleCreateAnnotationProposal(false)}
                  disabled={loading}
                >
                  <Send className="h-4 w-4 mr-2" />
                  Submit Proposal
                </Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
