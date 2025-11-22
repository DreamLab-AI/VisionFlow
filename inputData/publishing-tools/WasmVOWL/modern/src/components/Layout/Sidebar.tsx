import { useState } from 'react';
import { Link } from 'react-router-dom';

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [pages] = useState<string[]>([
    'AI Alignment',
    'Smart Contracts',
    'Robotics',
    'Virtual Worlds',
    'Disruptive Technologies',
  ]);

  return (
    <aside className={`${isCollapsed ? 'w-12' : 'w-64'} flex-shrink-0 border-r border-border bg-background transition-all duration-300`}>
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h3 className={`text-lg font-semibold text-foreground ${isCollapsed ? 'hidden' : 'block'}`}>
          Pages
        </h3>
        <button
          className="rounded p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
          onClick={() => setIsCollapsed(!isCollapsed)}
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            {isCollapsed ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            )}
          </svg>
        </button>
      </div>
      {!isCollapsed && (
        <nav className="p-2">
          <ul className="space-y-1">
            {pages.map((page) => (
              <li key={page}>
                <Link
                  to={`/page/${encodeURIComponent(page)}`}
                  className="block rounded px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                >
                  {page}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </aside>
  );
}
