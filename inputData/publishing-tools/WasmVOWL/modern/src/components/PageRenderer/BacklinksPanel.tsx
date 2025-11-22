import { Link } from 'react-router-dom';

interface BacklinksPanelProps {
  backlinks: string[];
}

export function BacklinksPanel({ backlinks }: BacklinksPanelProps) {
  // Safely handle backlinks that might not be an array
  const backlinksArray = Array.isArray(backlinks) ? backlinks : [];

  if (backlinksArray.length === 0) {
    return null;
  }

  return (
    <div className="mt-12 pt-8 border-t-2 border-gray-200 dark:border-gray-700">
      <h3 className="flex items-center gap-2 m-0 mb-4 text-lg font-semibold text-gray-800 dark:text-gray-100">
        <span className="text-xl text-blue-600 dark:text-blue-400">←</span>
        Linked References ({backlinksArray.length})
      </h3>
      <ul className="list-none p-0 m-0 grid gap-2">
        {backlinksArray.map((backlink) => (
          <li
            key={backlink}
            className="p-3 px-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 hover:translate-x-1 transition-all"
          >
            <Link
              to={`/page/${encodeURIComponent(backlink)}`}
              className="text-gray-800 dark:text-gray-200 no-underline font-medium hover:text-blue-600 dark:hover:text-blue-400"
            >
              {backlink}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
