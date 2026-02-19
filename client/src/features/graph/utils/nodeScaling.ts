import type { GraphVisualMode } from '../hooks/useGraphVisualState';
import type { Node as GraphNode } from '../managers/graphDataManager';
import type { GraphTypeVisualsSettings } from '../../settings/config/settings';

// Default scaling parameters matching the original hardcoded values.
// These are used when settings are not yet loaded or fields are missing.
const KG_DEFAULTS = { authorityScaleFactor: 0.5, connectionInfluence: 0.4, globalScaleMultiplier: 2.5 };
const ONTO_DEFAULTS = { hierarchyScaleFactor: 0.15, minScale: 0.4, instanceCountInfluence: 0.1 };
const AGENT_DEFAULTS = { workloadInfluence: 0.3, tokenRateInfluence: 100, tokenRateCap: 0.5 };

/**
 * Compute visual scale for a graph node based on its metadata and the active graph mode.
 *
 * Single source of truth for node scaling — both GemNodes (instanced mesh)
 * and GraphManager (edge offset computation) must call this function to stay in sync.
 *
 * Scaling parameters are read from `graphTypeVisuals` settings with hardcoded
 * fallback defaults that preserve the original visual behavior.
 */
export const computeNodeScale = (
  node: GraphNode,
  connectionCountMap: Map<string, number>,
  graphMode: GraphVisualMode,
  hierarchyMap?: Map<string, any>,
  visuals?: GraphTypeVisualsSettings,
): number => {
  const base = node.metadata?.size || 1.0;
  const id = String(node.id);

  if (graphMode === 'ontology') {
    const cfg = visuals?.ontology;
    const depth = hierarchyMap?.get(id)?.depth ?? (node.metadata?.depth ?? 0);
    const ic = parseInt(node.metadata?.instanceCount || '0', 10);
    const hsFactor = cfg?.hierarchyScaleFactor ?? ONTO_DEFAULTS.hierarchyScaleFactor;
    const minS = cfg?.minScale ?? ONTO_DEFAULTS.minScale;
    const icInfluence = cfg?.instanceCountInfluence ?? ONTO_DEFAULTS.instanceCountInfluence;
    return base * Math.max(minS, 1.0 - depth * hsFactor) * (1 + Math.log(ic + 1) * icInfluence);
  }

  if (graphMode === 'agent') {
    const cfg = visuals?.agent;
    const w = node.metadata?.workload ?? 0;
    const t = node.metadata?.tokenRate ?? 0;
    const wInfluence = cfg?.workloadInfluence ?? AGENT_DEFAULTS.workloadInfluence;
    const tDivisor = cfg?.tokenRateInfluence ?? AGENT_DEFAULTS.tokenRateInfluence;
    const tCap = cfg?.tokenRateCap ?? AGENT_DEFAULTS.tokenRateCap;
    return base * (1 + w * wInfluence + Math.min(t / tDivisor, tCap));
  }

  // Knowledge graph (default)
  const cfg = visuals?.knowledgeGraph;
  const auth = node.metadata?.authority ?? node.metadata?.authorityScore ?? 0;
  const cc = connectionCountMap.get(id) || 0;
  const connInfl = cfg?.connectionInfluence ?? KG_DEFAULTS.connectionInfluence;
  const authFactor = cfg?.authorityScaleFactor ?? KG_DEFAULTS.authorityScaleFactor;
  const globalMult = cfg?.globalScaleMultiplier ?? KG_DEFAULTS.globalScaleMultiplier;
  return base * (1 + Math.log(cc + 1) * connInfl) * (1 + auth * authFactor) * globalMult;
};
