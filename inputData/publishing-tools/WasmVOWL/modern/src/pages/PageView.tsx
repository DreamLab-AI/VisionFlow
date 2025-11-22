import { useParams } from 'react-router-dom';
import { usePage } from '../hooks/usePage';
import { MarkdownRenderer } from '../components/PageRenderer/MarkdownRenderer';
import { BacklinksPanel } from '../components/PageRenderer/BacklinksPanel';
import { MiniGraph } from '../components/PageRenderer/MiniGraph';
import { LoadingSpinner } from '../components/UI/LoadingSpinner';
import './PageView.css';

export default function PageView() {
  const { pageName } = useParams<{ pageName: string }>();
  const { data: page, isLoading, error } = usePage(pageName || '');

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <div className="text-center py-16">
        <h1 className="text-3xl font-bold text-foreground mb-4">Error Loading Page</h1>
        <p className="text-lg text-muted-foreground">{error.message}</p>
      </div>
    );
  }

  if (!page) {
    return (
      <div className="text-center py-16">
        <h1 className="text-3xl font-bold text-foreground mb-4">404 - Page Not Found</h1>
        <p className="text-lg text-muted-foreground">The page "{pageName}" could not be found.</p>
      </div>
    );
  }

  return (
    <div className={`grid gap-8 mx-auto px-4 py-6 ${page.ontology ? 'grid-cols-1 lg:grid-cols-[1fr_350px] max-w-7xl' : 'grid-cols-1 max-w-4xl'}`}>
      <article className="min-w-0">
        <header className="mb-8 pb-6 border-b-2 border-border">
          <h1 className="text-4xl font-bold text-foreground mb-4 leading-tight">{page.title}</h1>
          {page.properties && Object.keys(page.properties).length > 0 && (
            <div className="flex flex-wrap gap-4 mt-4">
              {Object.entries(page.properties).map(([key, value]) => (
                <div key={key} className="flex gap-2 px-4 py-2 bg-muted rounded-md text-sm">
                  <span className="font-semibold text-muted-foreground">{key}:</span>
                  <span className="text-foreground">{value}</span>
                </div>
              ))}
            </div>
          )}
        </header>

        <div className="mb-12">
          <MarkdownRenderer content={page.content} />
        </div>

        <footer>
          <BacklinksPanel backlinks={page.backlinks} />
        </footer>
      </article>

      {page.ontology && (
        <aside className="sticky top-8 h-fit">
          <MiniGraph nodeId={page.ontology.term_id} />
          <div className="bg-muted rounded-lg p-6 shadow-sm">
            <h3 className="font-semibold text-foreground mb-4">Ontology Information</h3>
            <dl className="grid gap-3">
              <div>
                <dt className="font-semibold text-muted-foreground text-xs">Term ID:</dt>
                <dd className="mt-1 text-foreground font-mono text-sm">{page.ontology.term_id}</dd>
              </div>
              <div>
                <dt className="font-semibold text-muted-foreground text-xs">Domain:</dt>
                <dd className={`mt-1 font-mono text-sm domain-${page.ontology.source_domain}`}>
                  {page.ontology.source_domain.toUpperCase()}
                </dd>
              </div>
              {page.ontology.maturity_level && (
                <div>
                  <dt className="font-semibold text-muted-foreground text-xs">Maturity:</dt>
                  <dd className="mt-1 text-foreground font-mono text-sm">{page.ontology.maturity_level}</dd>
                </div>
              )}
              {page.ontology.authority_score && (
                <div>
                  <dt className="font-semibold text-muted-foreground text-xs">Authority Score:</dt>
                  <dd className="mt-1 text-foreground font-mono text-sm">{(page.ontology.authority_score * 100).toFixed(0)}%</dd>
                </div>
              )}
            </dl>
          </div>
        </aside>
      )}
    </div>
  );
}
