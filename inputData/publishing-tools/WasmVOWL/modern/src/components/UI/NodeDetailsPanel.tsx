/**
 * Node Details Panel - Displays detailed information about selected node
 * Migrated to shadcn/ui with Sheet and Accordion
 */

import { useGraphStore } from '../../stores/useGraphStore';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '../ui/sheet';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../ui/accordion';
import { Button } from '../ui/button';
import { cn } from '@/lib/utils';

export function NodeDetailsPanel() {
  const { selectedNode, nodes, selectNode } = useGraphStore();

  if (!selectedNode) {
    return null;
  }

  const node = nodes.get(selectedNode);

  if (!node) {
    return null;
  }

  // Extract metadata from node properties
  const metadata = {
    term_id: node.id,
    label: node.label,
    type: node.type,
    iri: node.iri || 'N/A',
    // Mock data for ontology metadata (will be populated from actual data in Phase 4)
    preferred_term: node.label,
    domain: node.properties?.domain || 'General',
    maturity: node.properties?.maturity || 'Stable',
    status: node.properties?.status || 'Active',
    authority_score: node.properties?.authority_score || 0.85,
    instances: node.properties?.instances || 0,
  };

  const handleClose = () => {
    selectNode(null);
  };

  const handleViewPage = () => {
    // Convert node label to Logseq page URL format
    const pageName = encodeURIComponent(node.label.toLowerCase());
    const logseqUrl = `https://narrativegoldmine.com/#/page/${pageName}`;
    window.open(logseqUrl, '_blank');
  };

  // Domain color mapping
  const getDomainColor = (domain: string): string => {
    const colors: Record<string, string> = {
      'AI/ML': 'bg-purple-500',
      'Mathematics': 'bg-blue-500',
      'Physics': 'bg-cyan-500',
      'Biology': 'bg-green-500',
      'Chemistry': 'bg-amber-500',
      'General': 'bg-gray-500',
    };
    return colors[domain] || colors['General'];
  };

  // Maturity level indicator
  const getMaturityColor = (maturity: string): string => {
    const colors: Record<string, string> = {
      'Experimental': 'bg-amber-500',
      'Beta': 'bg-blue-500',
      'Stable': 'bg-green-500',
      'Deprecated': 'bg-red-500',
    };
    return colors[maturity] || colors['Stable'];
  };

  return (
    <Sheet open={!!selectedNode} onOpenChange={(open) => !open && handleClose()}>
      <SheetContent
        side="right"
        className="w-[400px] sm:w-[540px] overflow-y-auto bg-gradient-to-br from-slate-900 to-slate-800 text-white border-l border-slate-700"
      >
        <SheetHeader>
          <SheetTitle className="text-white">Node Details</SheetTitle>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* Node Label and Type */}
          <div className="space-y-2">
            <h3 className="text-xl font-semibold">{metadata.label}</h3>
            <p className="text-sm text-slate-400 uppercase tracking-wide">
              {metadata.type}
            </p>
          </div>

          {/* Accordion for organized sections */}
          <Accordion type="multiple" defaultValue={["metadata", "authority", "relations"]} className="space-y-2">
            {/* Metadata Section */}
            <AccordionItem value="metadata" className="border-slate-700">
              <AccordionTrigger className="text-white hover:text-slate-300">
                Metadata
              </AccordionTrigger>
              <AccordionContent className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Term ID:</span>
                  <span className="text-sm">{metadata.term_id}</span>
                </div>

                <div className="flex justify-between items-start gap-3">
                  <span className="text-sm text-slate-400 flex-shrink-0">IRI:</span>
                  <span className="text-sm font-mono text-slate-300 break-all text-right">
                    {metadata.iri}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Domain:</span>
                  <span
                    className={cn(
                      "px-3 py-1 rounded-full text-xs font-semibold text-white",
                      getDomainColor(metadata.domain)
                    )}
                  >
                    {metadata.domain}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Maturity:</span>
                  <span
                    className={cn(
                      "px-3 py-1 rounded-full text-xs font-semibold text-white",
                      getMaturityColor(metadata.maturity)
                    )}
                  >
                    {metadata.maturity}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Status:</span>
                  <span className="text-sm text-green-400 font-medium">
                    {metadata.status}
                  </span>
                </div>

                {metadata.instances > 0 && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-400">Instances:</span>
                    <span className="text-sm">{metadata.instances}</span>
                  </div>
                )}
              </AccordionContent>
            </AccordionItem>

            {/* Authority Score Section */}
            <AccordionItem value="authority" className="border-slate-700">
              <AccordionTrigger className="text-white hover:text-slate-300">
                Authority Score
              </AccordionTrigger>
              <AccordionContent className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Score:</span>
                  <span className="text-sm font-semibold">
                    {(metadata.authority_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="relative h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 bg-gradient-to-r from-green-500 to-green-600 rounded-full shadow-lg shadow-green-500/50 transition-all duration-300"
                    style={{ width: `${metadata.authority_score * 100}%` }}
                  />
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* Relations Section (Placeholder) */}
            <AccordionItem value="relations" className="border-slate-700">
              <AccordionTrigger className="text-white hover:text-slate-300">
                Relations
              </AccordionTrigger>
              <AccordionContent className="space-y-2">
                <p className="text-sm text-slate-400">
                  Relations data will be displayed here in future versions.
                </p>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <Button
              onClick={handleViewPage}
              className="flex-1 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white shadow-lg shadow-blue-500/30"
            >
              View Full Page
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
