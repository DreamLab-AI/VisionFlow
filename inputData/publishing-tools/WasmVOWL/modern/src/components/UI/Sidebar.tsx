/**
 * Sidebar component with node details, filters, and statistics
 * Migrated to Tailwind CSS with tab-based navigation
 */

import { useGraphStore } from '../../stores/useGraphStore';
import { useUIStore } from '../../stores/useUIStore';

export function Sidebar() {
  const { sidebarOpen, sidebarTab, setSidebarTab } = useUIStore();
  const {
    selectedNode,
    nodes,
    edges,
    statistics,
    activeFilters,
    addFilter,
    removeFilter,
    clearFilters
  } = useGraphStore();

  if (!sidebarOpen) {
    return null;
  }

  const node = selectedNode ? nodes.get(selectedNode) : null;

  return (
    <aside className="fixed right-0 top-0 bottom-0 z-50 flex w-[350px] flex-col border-l border-border bg-background shadow-lg animate-in slide-in-from-right duration-300">
      {/* Tab navigation */}
      <div className="flex border-b border-gray-200 bg-gray-50">
        <button
          className={`flex-1 border-b-2 px-4 py-3 text-sm font-semibold transition-colors ${
            sidebarTab === 'details'
              ? 'border-blue-500 bg-card text-blue-600'
              : 'border-transparent text-gray-600 hover:bg-accent hover:text-gray-900'
          }`}
          onClick={() => setSidebarTab('details')}
        >
          Details
        </button>
        <button
          className={`flex-1 border-b-2 px-4 py-3 text-sm font-semibold transition-colors ${
            sidebarTab === 'filters'
              ? 'border-blue-500 bg-card text-blue-600'
              : 'border-transparent text-gray-600 hover:bg-accent hover:text-gray-900'
          }`}
          onClick={() => setSidebarTab('filters')}
        >
          Filters
        </button>
        <button
          className={`flex-1 border-b-2 px-4 py-3 text-sm font-semibold transition-colors ${
            sidebarTab === 'statistics'
              ? 'border-blue-500 bg-card text-blue-600'
              : 'border-transparent text-gray-600 hover:bg-accent hover:text-gray-900'
          }`}
          onClick={() => setSidebarTab('statistics')}
        >
          Statistics
        </button>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-6">
        {sidebarTab === 'details' && (
          <div className="animate-in fade-in duration-200">
            {node ? (
              <div className="space-y-4">
                <h3 className="text-lg font-bold text-gray-900">Node Details</h3>

                <div className="flex items-start justify-between border-b border-gray-100 py-3">
                  <span className="text-sm font-semibold text-gray-600">ID:</span>
                  <span className="text-sm text-gray-900">{node.id}</span>
                </div>

                <div className="flex items-start justify-between border-b border-gray-100 py-3">
                  <span className="text-sm font-semibold text-gray-600">Type:</span>
                  <span className="inline-block rounded bg-blue-500 px-2 py-1 text-xs font-semibold uppercase text-white">
                    {node.type}
                  </span>
                </div>

                <div className="flex items-start justify-between border-b border-gray-100 py-3">
                  <span className="text-sm font-semibold text-gray-600">Label:</span>
                  <span className="text-sm text-gray-900">{node.label}</span>
                </div>

                {node.iri && (
                  <div className="flex items-start justify-between border-b border-gray-100 py-3">
                    <span className="text-sm font-semibold text-gray-600">IRI:</span>
                    <span className="break-all text-xs font-mono text-gray-600">{node.iri}</span>
                  </div>
                )}

                {node.properties.instances !== undefined && (
                  <div className="flex items-start justify-between border-b border-gray-100 py-3">
                    <span className="text-sm font-semibold text-gray-600">Instances:</span>
                    <span className="text-sm text-gray-900">{node.properties.instances}</span>
                  </div>
                )}

                <h4 className="mt-6 text-xs font-semibold uppercase tracking-wider text-gray-600">
                  Connections
                </h4>

                <div className="flex items-start justify-between border-b border-gray-100 py-3">
                  <span className="text-sm font-semibold text-gray-600">Incoming:</span>
                  <span className="text-sm text-gray-900">
                    {Array.from(edges.values()).filter(e => e.target === node.id).length}
                  </span>
                </div>

                <div className="flex items-start justify-between border-b border-gray-100 py-3">
                  <span className="text-sm font-semibold text-gray-600">Outgoing:</span>
                  <span className="text-sm text-gray-900">
                    {Array.from(edges.values()).filter(e => e.source === node.id).length}
                  </span>
                </div>

                {node.properties.attributes && node.properties.attributes.length > 0 && (
                  <>
                    <h4 className="mt-6 text-xs font-semibold uppercase tracking-wider text-gray-600">
                      Attributes
                    </h4>
                    <ul className="space-y-2">
                      {node.properties.attributes.map((attr, idx) => (
                        <li key={idx} className="rounded bg-gray-50 p-2 text-sm text-gray-900">
                          {attr}
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center text-gray-400">
                <svg className="mb-4 h-16 w-16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-sm">Select a node to view details</p>
              </div>
            )}
          </div>
        )}

        {sidebarTab === 'filters' && (
          <div className="animate-in fade-in duration-200 space-y-4">
            <h3 className="text-lg font-bold text-gray-900">Active Filters</h3>

            {activeFilters.length > 0 ? (
              <div className="space-y-4">
                <div className="space-y-2">
                  {activeFilters.map((filter, index) => (
                    <div key={index} className="flex items-center justify-between rounded-md border border-gray-200 bg-gray-50 p-3">
                      <span className="text-sm font-medium text-gray-900">
                        {filter.type === 'nodeType' && `Type: ${filter.config.nodeType}`}
                        {filter.type === 'degree' && `Degree: ${filter.config.min || 0}-${filter.config.max || '∞'}`}
                        {filter.type === 'edgeType' && `Edge: ${filter.config.edgeType}`}
                      </span>
                      <button
                        onClick={() => removeFilter(index)}
                        className="flex items-center justify-center text-gray-400 transition-colors hover:text-red-500"
                        aria-label="Remove filter"
                      >
                        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
                <button
                  onClick={clearFilters}
                  className="w-full rounded-md bg-red-500 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-red-600"
                >
                  Clear All Filters
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center text-gray-400">
                <svg className="mb-4 h-16 w-16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                </svg>
                <p className="text-sm">No active filters</p>
              </div>
            )}

            <h4 className="mt-6 text-xs font-semibold uppercase tracking-wider text-gray-600">
              Add Filter
            </h4>

            <div className="space-y-2">
              <button
                onClick={() => addFilter({ type: 'nodeType', config: { nodeType: 'class' } })}
                className="flex w-full items-center gap-2 rounded-md border-2 border-dashed border-gray-300 bg-card p-3 text-left text-sm font-medium text-gray-600 transition-all hover:border-blue-500 hover:bg-accent hover:text-blue-600"
              >
                <svg className="h-5 w-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>Filter by Class</span>
              </button>

              <button
                onClick={() => addFilter({ type: 'degree', config: { min: 2 } })}
                className="flex w-full items-center gap-2 rounded-md border-2 border-dashed border-gray-300 bg-card p-3 text-left text-sm font-medium text-gray-600 transition-all hover:border-blue-500 hover:bg-accent hover:text-blue-600"
              >
                <svg className="h-5 w-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>Filter by Degree (min: 2)</span>
              </button>

              <button
                onClick={() => addFilter({ type: 'edgeType', config: { edgeType: 'objectProperty' } })}
                className="flex w-full items-center gap-2 rounded-md border-2 border-dashed border-gray-300 bg-card p-3 text-left text-sm font-medium text-gray-600 transition-all hover:border-blue-500 hover:bg-accent hover:text-blue-600"
              >
                <svg className="h-5 w-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>Filter by Object Properties</span>
              </button>
            </div>
          </div>
        )}

        {sidebarTab === 'statistics' && statistics && (
          <div className="animate-in fade-in duration-200 space-y-4">
            <h3 className="text-lg font-bold text-gray-900">Graph Statistics</h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg bg-gradient-to-br from-purple-500 to-purple-700 p-5 text-center text-white">
                <div className="text-3xl font-bold">{statistics.nodeCount}</div>
                <div className="text-xs font-medium opacity-90">Total Nodes</div>
              </div>

              <div className="rounded-lg bg-gradient-to-br from-pink-500 to-rose-600 p-5 text-center text-white">
                <div className="text-3xl font-bold">{statistics.edgeCount}</div>
                <div className="text-xs font-medium opacity-90">Total Edges</div>
              </div>

              <div className="rounded-lg bg-gradient-to-br from-blue-400 to-cyan-500 p-5 text-center text-white">
                <div className="text-3xl font-bold">{statistics.maxDegree}</div>
                <div className="text-xs font-medium opacity-90">Max Degree</div>
              </div>

              <div className="rounded-lg bg-gradient-to-br from-emerald-400 to-teal-500 p-5 text-center text-white">
                <div className="text-3xl font-bold">{statistics.avgDegree.toFixed(2)}</div>
                <div className="text-xs font-medium opacity-90">Avg Degree</div>
              </div>
            </div>

            <h4 className="mt-6 text-xs font-semibold uppercase tracking-wider text-gray-600">
              Node Types
            </h4>
            <div className="space-y-2">
              {Object.entries(statistics.classCounts).map(([type, count]) => (
                <div key={type} className="flex items-center justify-between rounded-md bg-gray-50 p-3">
                  <span className="text-sm font-medium text-gray-900">{type}</span>
                  <span className="text-base font-bold text-blue-500">{count}</span>
                </div>
              ))}
            </div>

            <h4 className="mt-6 text-xs font-semibold uppercase tracking-wider text-gray-600">
              Property Types
            </h4>
            <div className="space-y-2">
              {Object.entries(statistics.propertyCounts).map(([type, count]) => (
                <div key={type} className="flex items-center justify-between rounded-md bg-gray-50 p-3">
                  <span className="text-sm font-medium text-gray-900">{type}</span>
                  <span className="text-base font-bold text-blue-500">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
