import { Link, useLocation } from 'react-router-dom';
import { SearchBar } from '../Search/SearchBar';

export function Navbar() {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  return (
    <nav className="flex items-center gap-8 border-b border-border bg-background px-8 py-4 shadow-sm">
      <Link to="/" className="flex items-center gap-2 text-xl font-bold text-foreground transition-opacity hover:opacity-80">
        <span className="text-2xl">🧠</span>
        Narrative Goldmine
      </Link>

      <div className="flex flex-1 gap-6">
        <Link
          to="/"
          className={`rounded-md px-4 py-2 font-medium transition-all ${isActive('/') && location.pathname === '/'
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
            }`}
        >
          Home
        </Link>
        <Link
          to="/ontology"
          className={`rounded-md px-4 py-2 font-medium transition-all ${isActive('/ontology')
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
            }`}
        >
          Ontology Graph
        </Link>
        <Link
          to="/search"
          className={`rounded-md px-4 py-2 font-medium transition-all ${isActive('/search')
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
            }`}
        >
          Search
        </Link>
      </div>

      <SearchBar />
    </nav>
  );
}
