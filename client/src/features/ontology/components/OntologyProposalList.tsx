import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/features/design-system/components/Collapsible';
import { useToast } from '@/features/design-system/components/Toast';
import FileText from 'lucide-react/dist/esm/icons/file-text';
import ChevronDown from 'lucide-react/dist/esm/icons/chevron-down';
import ChevronRight from 'lucide-react/dist/esm/icons/chevron-right';
import Send from 'lucide-react/dist/esm/icons/send';
import Trash2 from 'lucide-react/dist/esm/icons/trash-2';
import RotateCcw from 'lucide-react/dist/esm/icons/rotate-ccw';
import CheckCircle from 'lucide-react/dist/esm/icons/check-circle';
import XCircle from 'lucide-react/dist/esm/icons/x-circle';
import Clock from 'lucide-react/dist/esm/icons/clock';
import Edit from 'lucide-react/dist/esm/icons/edit';
import GitMerge from 'lucide-react/dist/esm/icons/git-merge';
import AlertCircle from 'lucide-react/dist/esm/icons/alert-circle';
import FileCode from 'lucide-react/dist/esm/icons/file-code';
import Plus from 'lucide-react/dist/esm/icons/plus';
import Minus from 'lucide-react/dist/esm/icons/minus';
import {
  useOntologyContributionStore,
  OntologyProposal,
  ProposalStatus,
  OntologyClass,
  OntologyProperty,
  OntologyAnnotation
} from '../hooks/useOntologyStore';

interface OntologyProposalListProps {
  className?: string;
  onEditProposal?: (proposalId: string) => void;
}

const statusConfig: Record<ProposalStatus, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline'; icon: React.ElementType }> = {
  draft: { label: 'Draft', variant: 'secondary', icon: Edit },
  pending: { label: 'Pending Review', variant: 'default', icon: Clock },
  approved: { label: 'Approved', variant: 'default', icon: CheckCircle },
  rejected: { label: 'Rejected', variant: 'destructive', icon: XCircle },
  withdrawn: { label: 'Withdrawn', variant: 'outline', icon: RotateCcw }
};

function ProposalDiff({ diff }: { diff: OntologyProposal['diff'] }) {
  if (!diff) return null;

  return (
    <div className="mt-3 rounded-md border bg-muted/30 p-3 font-mono text-xs space-y-2">
      <div className="font-semibold text-sm mb-2 flex items-center gap-2">
        <FileCode className="h-4 w-4" />
        Changes
      </div>
      {diff.added.length > 0 && (
        <div className="space-y-1">
          {diff.added.map((line, i) => (
            <div key={`add-${i}`} className="flex items-start gap-2 text-green-600 dark:text-green-400">
              <Plus className="h-3 w-3 mt-0.5 flex-shrink-0" />
              <span className="break-all">{line}</span>
            </div>
          ))}
        </div>
      )}
      {diff.removed.length > 0 && (
        <div className="space-y-1">
          {diff.removed.map((line, i) => (
            <div key={`rem-${i}`} className="flex items-start gap-2 text-red-600 dark:text-red-400">
              <Minus className="h-3 w-3 mt-0.5 flex-shrink-0" />
              <span className="break-all">{line}</span>
            </div>
          ))}
        </div>
      )}
      {diff.modified.length > 0 && (
        <div className="space-y-1">
          {diff.modified.map((line, i) => (
            <div key={`mod-${i}`} className="flex items-start gap-2 text-yellow-600 dark:text-yellow-400">
              <Edit className="h-3 w-3 mt-0.5 flex-shrink-0" />
              <span className="break-all">{line}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ProposalCard({
  proposal,
  onSubmit,
  onWithdraw,
  onDelete,
  onEdit,
  loading
}: {
  proposal: OntologyProposal;
  onSubmit: (id: string) => void;
  onWithdraw: (id: string) => void;
  onDelete: (id: string) => void;
  onEdit: (id: string) => void;
  loading: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const status = statusConfig[proposal.status];
  const StatusIcon = status.icon;

  const getProposalTitle = () => {
    switch (proposal.type) {
      case 'class':
        return (proposal.data as OntologyClass).label || 'Unnamed Class';
      case 'property':
        return (proposal.data as OntologyProperty).label || 'Unnamed Property';
      case 'annotation':
        const ann = proposal.data as OntologyAnnotation;
        return `${ann.predicate} on ${ann.targetIri.split(/[#/]/).pop()}`;
      default:
        return 'Unknown Proposal';
    }
  };

  const getProposalDetails = () => {
    switch (proposal.type) {
      case 'class': {
        const cls = proposal.data as OntologyClass;
        return (
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-[100px_1fr] gap-2">
              <span className="text-muted-foreground">IRI:</span>
              <span className="font-mono text-xs break-all">{cls.iri}</span>
            </div>
            {cls.parentClass && (
              <div className="grid grid-cols-[100px_1fr] gap-2">
                <span className="text-muted-foreground">Parent:</span>
                <span className="font-mono text-xs break-all">{cls.parentClass}</span>
              </div>
            )}
            {cls.description && (
              <div className="grid grid-cols-[100px_1fr] gap-2">
                <span className="text-muted-foreground">Description:</span>
                <span>{cls.description}</span>
              </div>
            )}
          </div>
        );
      }
      case 'property': {
        const prop = proposal.data as OntologyProperty;
        return (
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-[100px_1fr] gap-2">
              <span className="text-muted-foreground">IRI:</span>
              <span className="font-mono text-xs break-all">{prop.iri}</span>
            </div>
            <div className="grid grid-cols-[100px_1fr] gap-2">
              <span className="text-muted-foreground">Type:</span>
              <Badge variant="outline" className="w-fit">
                {prop.propertyType}
              </Badge>
            </div>
            {prop.domain && (
              <div className="grid grid-cols-[100px_1fr] gap-2">
                <span className="text-muted-foreground">Domain:</span>
                <span className="font-mono text-xs">{prop.domain}</span>
              </div>
            )}
            {prop.range && (
              <div className="grid grid-cols-[100px_1fr] gap-2">
                <span className="text-muted-foreground">Range:</span>
                <span className="font-mono text-xs">{prop.range}</span>
              </div>
            )}
          </div>
        );
      }
      case 'annotation': {
        const ann = proposal.data as OntologyAnnotation;
        return (
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-[100px_1fr] gap-2">
              <span className="text-muted-foreground">Target:</span>
              <span className="font-mono text-xs break-all">{ann.targetIri}</span>
            </div>
            <div className="grid grid-cols-[100px_1fr] gap-2">
              <span className="text-muted-foreground">Predicate:</span>
              <span className="font-mono text-xs">{ann.predicate}</span>
            </div>
            <div className="grid grid-cols-[100px_1fr] gap-2">
              <span className="text-muted-foreground">Value:</span>
              <span>{ann.value}</span>
            </div>
            {ann.language && (
              <div className="grid grid-cols-[100px_1fr] gap-2">
                <span className="text-muted-foreground">Language:</span>
                <span>{ann.language}</span>
              </div>
            )}
          </div>
        );
      }
    }
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
      <div className="rounded-lg border bg-card p-4">
        <CollapsibleTrigger asChild>
          <div className="flex items-center justify-between cursor-pointer">
            <div className="flex items-center gap-3">
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-medium">{getProposalTitle()}</span>
                  <Badge variant="outline" className="text-xs">
                    {proposal.type}
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground mt-0.5">
                  Created {formatDate(proposal.createdAt)}
                </div>
              </div>
            </div>
            <Badge variant={status.variant as 'default' | 'secondary' | 'destructive' | 'outline'} className="flex items-center gap-1">
              {/* @ts-ignore - StatusIcon is dynamically assigned from lucide-react */}
              <StatusIcon className="h-3 w-3" />
              {status.label}
            </Badge>
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="mt-4 pt-4 border-t space-y-4">
            {getProposalDetails()}

            {proposal.diff && <ProposalDiff diff={proposal.diff} />}

            {proposal.reviewNotes && (
              <div className="rounded-md border border-yellow-500/30 bg-yellow-500/10 p-3">
                <div className="text-sm font-medium text-yellow-600 dark:text-yellow-400 mb-1">
                  Review Notes
                </div>
                <p className="text-sm">{proposal.reviewNotes}</p>
              </div>
            )}

            {proposal.mergeCommit && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <GitMerge className="h-4 w-4" />
                Merged: <code className="font-mono text-xs">{proposal.mergeCommit}</code>
              </div>
            )}

            <div className="flex items-center gap-2 pt-2">
              {proposal.status === 'draft' && (
                <>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onEdit(proposal.id)}
                    disabled={loading}
                  >
                    <Edit className="h-4 w-4 mr-1" />
                    Edit
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => onSubmit(proposal.id)}
                    disabled={loading}
                  >
                    <Send className="h-4 w-4 mr-1" />
                    Submit
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => onDelete(proposal.id)}
                    disabled={loading}
                  >
                    <Trash2 className="h-4 w-4 mr-1" />
                    Delete
                  </Button>
                </>
              )}
              {proposal.status === 'pending' && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => onWithdraw(proposal.id)}
                  disabled={loading}
                >
                  <RotateCcw className="h-4 w-4 mr-1" />
                  Withdraw
                </Button>
              )}
            </div>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

export function OntologyProposalList({ className, onEditProposal }: OntologyProposalListProps) {
  const { toast } = useToast();
  const {
    proposals,
    loading,
    error,
    submitProposal,
    withdrawProposal,
    deleteProposal,
    clearError
  } = useOntologyContributionStore();

  const [statusFilter, setStatusFilter] = useState<ProposalStatus | 'all'>('all');

  const filteredProposals = useMemo(() => {
    let filtered = [...proposals];

    if (statusFilter !== 'all') {
      filtered = filtered.filter(p => p.status === statusFilter);
    }

    // Sort by updatedAt descending
    return filtered.sort((a, b) => b.updatedAt - a.updatedAt);
  }, [proposals, statusFilter]);

  const handleSubmit = async (id: string) => {
    try {
      await submitProposal(id);
      toast({
        title: 'Proposal Submitted',
        description: 'Your proposal has been submitted for review',
        variant: 'default'
      });
    } catch (err: any) {
      toast({
        title: 'Error',
        description: err.message || 'Failed to submit proposal',
        variant: 'destructive'
      });
    }
  };

  const handleWithdraw = async (id: string) => {
    try {
      await withdrawProposal(id);
      toast({
        title: 'Proposal Withdrawn',
        description: 'Your proposal has been withdrawn',
        variant: 'default'
      });
    } catch (err: any) {
      toast({
        title: 'Error',
        description: err.message || 'Failed to withdraw proposal',
        variant: 'destructive'
      });
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteProposal(id);
      toast({
        title: 'Proposal Deleted',
        description: 'Your draft proposal has been deleted',
        variant: 'default'
      });
    } catch (err: any) {
      toast({
        title: 'Error',
        description: err.message || 'Failed to delete proposal',
        variant: 'destructive'
      });
    }
  };

  const handleEdit = (id: string) => {
    onEditProposal?.(id);
  };

  const statusCounts = useMemo(() => {
    const counts: Record<ProposalStatus | 'all', number> = {
      all: proposals.length,
      draft: 0,
      pending: 0,
      approved: 0,
      rejected: 0,
      withdrawn: 0
    };

    for (const p of proposals) {
      counts[p.status]++;
    }

    return counts;
  }, [proposals]);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              My Proposals
            </CardTitle>
            <CardDescription>
              Manage your ontology contribution proposals
            </CardDescription>
          </div>
          <Select value={statusFilter} onValueChange={(v) => setStatusFilter(v as ProposalStatus | 'all')}>
            <SelectTrigger className="w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All ({statusCounts.all})</SelectItem>
              <SelectItem value="draft">Drafts ({statusCounts.draft})</SelectItem>
              <SelectItem value="pending">Pending ({statusCounts.pending})</SelectItem>
              <SelectItem value="approved">Approved ({statusCounts.approved})</SelectItem>
              <SelectItem value="rejected">Rejected ({statusCounts.rejected})</SelectItem>
              <SelectItem value="withdrawn">Withdrawn ({statusCounts.withdrawn})</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>

      <CardContent>
        {error && (
          <div className="rounded-lg border border-destructive bg-destructive/10 p-3 mb-4 flex items-start gap-2">
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

        {filteredProposals.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p className="font-medium">No proposals found</p>
            <p className="text-sm mt-1">
              {statusFilter === 'all'
                ? 'Create your first ontology contribution proposal'
                : `No ${statusFilter} proposals`}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredProposals.map((proposal) => (
              <ProposalCard
                key={proposal.id}
                proposal={proposal}
                onSubmit={handleSubmit}
                onWithdraw={handleWithdraw}
                onDelete={handleDelete}
                onEdit={handleEdit}
                loading={loading}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
