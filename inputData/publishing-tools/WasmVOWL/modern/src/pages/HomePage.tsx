import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { loadOntologyStatistics, formatNumber, type OntologyStatistics } from '../services/statisticsService';
import './HomePage.css';

export default function HomePage() {
  const [stats, setStats] = useState<OntologyStatistics | null>(null);
  const [searchPageCount, setSearchPageCount] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadStats = async () => {
      try {
        // Load ontology statistics
        const ontologyStats = await loadOntologyStatistics();
        setStats(ontologyStats);

        // Load search page count
        try {
          const searchResponse = await fetch('./api/search-index.json');
          if (searchResponse.ok) {
            const searchIndex = await searchResponse.json();
            setSearchPageCount(searchIndex.length);
          }
        } catch {
          // Search index might not be available yet
          console.warn('Search index not available');
        }
      } finally {
        setIsLoading(false);
      }
    };

    loadStats();
  }, []);

  // Fallback values while loading
  const classCount = stats?.totalClasses || 1663;
  const propertyCount = stats?.totalProperties || 70;
  const tripleCount = stats?.estimatedTriples || 16630;
  const pageCount = searchPageCount || 2040;
  const domainCount = stats ? Object.keys(stats.domains).length : 6;

  // Get domain stats with fallback descriptions
  const getDomainInfo = (domain: string) => {
    const stat = stats?.domains[domain];
    const count = stat?.count || 0;

    // Fallback descriptions if no real data
    const descriptions: Record<string, string> = {
      'AI': 'machine learning, neural networks, and AI governance',
      'BC': 'distributed ledger technology and smart contracts',
      'RB': 'robot hardware, control, and manipulation',
      'DT': 'cross-cutting technological concepts',
      'MV': 'virtual worlds and spatial computing',
      'TELE': 'telepresence, virtual collaboration, and remote presence',
    };

    return {
      label: stats?.domainLabels[domain] || domain,
      count,
      description: descriptions[domain] || 'domain entities',
    };
  };

  const domainOrder = ['AI', 'BC', 'TELE', 'RB', 'DT', 'MV'];
  const activeDomains = stats
    ? domainOrder.filter(d => stats.domains[d])
    : ['AI', 'BC', 'RB', 'DT', 'MV'];

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <header className="text-center mb-12 py-8">
        <h1 className="text-4xl font-bold text-foreground mb-4">Welcome to Narrative Goldmine</h1>
        <p className="text-xl text-muted-foreground">
          Explore the unified knowledge graph of disruptive technologies
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
        <Link to="/ontology" className="home-card bg-card border border-border rounded-xl p-8 no-underline shadow-sm transition-all hover:-translate-y-1 hover:shadow-lg hover:border-accent">
          <div className="text-5xl mb-4">🌐</div>
          <h2 className="text-2xl font-semibold text-foreground mb-4">Ontology Graph</h2>
          <p className="text-muted-foreground leading-relaxed mb-6">
            Interactive 3D visualization of {formatNumber(classCount)} classes across{' '}
            {domainCount} domains (AI, Blockchain, Robotics, Metaverse, Disruptive Tech, and more).
          </p>
          <span className="card-action inline-block text-accent font-semibold transition-transform">Explore Graph →</span>
        </Link>

        <Link to="/search" className="home-card bg-card border border-border rounded-xl p-8 no-underline shadow-sm transition-all hover:-translate-y-1 hover:shadow-lg hover:border-accent">
          <div className="text-5xl mb-4">🔍</div>
          <h2 className="text-2xl font-semibold text-foreground mb-4">Search</h2>
          <p className="text-muted-foreground leading-relaxed mb-6">
            Search across {formatNumber(pageCount)} Logseq pages with unified full-text search
            powered by Fuse.js.
          </p>
          <span className="card-action inline-block text-accent font-semibold transition-transform">Start Searching →</span>
        </Link>

        <div className="info-card home-card bg-card border border-border rounded-xl p-8 shadow-sm">
          <div className="text-5xl mb-4">📊</div>
          <h2 className="text-2xl font-semibold text-foreground mb-4">Statistics</h2>
          {isLoading ? (
            <p className="text-muted-foreground italic py-4">Loading statistics...</p>
          ) : (
            <ul className="stats-list list-none p-0">
              <li className="py-2 text-foreground"><strong>{formatNumber(tripleCount)}</strong> RDF triples</li>
              <li className="py-2 text-foreground"><strong>{formatNumber(classCount)}</strong> OWL classes</li>
              <li className="py-2 text-foreground"><strong>{formatNumber(propertyCount)}</strong> properties</li>
              <li className="py-2 text-foreground"><strong>{domainCount}</strong> knowledge domains</li>
            </ul>
          )}
        </div>
      </div>

      <section className="mt-16">
        <h2 className="text-center text-3xl font-bold text-foreground mb-8">Knowledge Domains</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {activeDomains.map((domain) => {
            const info = getDomainInfo(domain);
            const colorClass = domain.toLowerCase();
            return (
              <div key={domain} className={`domain-card bg-card border-2 rounded-xl p-6 transition-all hover:-translate-y-0.5 hover:shadow-md ${colorClass}`}>
                <h3 className="text-xl font-semibold text-foreground mb-3">
                  {domain} ({info.label})
                </h3>
                <p className="text-muted-foreground leading-normal">
                  {info.count > 0 ? `${formatNumber(info.count)} entities` : 'Entities'} covering{' '}
                  {info.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
