'use strict';

/**
 * jev-questions.cjs — the four question sets of the Jev option, as pure
 * builders. No I/O. Each returns { version, state, questions, ids } for
 * `jev.decide()`; the caller maps the validated answers back to its own items.
 *
 * Question IDs are bookkeeping and are not shown to the model, so every
 * `instructions` string carries the question's complete meaning. Versions are
 * stable strings, bumped on any wording or criteria change, and are logged.
 *
 * Every set is ADVISORY: it annotates or orders. None changes a pass/fail.
 */

const MAX_EXCERPT_LINES = 14;

/** VF-01-S1-01: is one architecture claim supported by its cited evidence? Choice per claim. */
function claimCheck(claims) {
  const questions = {};
  const ids = [];
  const stateClaims = {};
  claims.forEach((c, i) => {
    const id = `claim_${i}`;
    ids.push({ id, key: c.key });
    stateClaims[id] = { text: c.text, assigned_status: c.status, cited_excerpts: c.excerpts };
    questions[id] = {
      type: 'choice',
      instructions: `Judge \`state.claims.${id}\`. Its \`text\` is a statement about a software component, and its \`assigned_status\` is a grade where "partial" means a mechanism exists but the condition is not measurable from it and "measured" means a runtime quantity or receipt exists. Using ONLY \`cited_excerpts\` (source lines at the cited locations), decide whether the evidence supports the claim as worded and graded. Do not use outside knowledge. Prefer "insufficient" when the excerpts do not show the thing claimed.`,
      criteria: {
        supported: 'The excerpts directly show what the text claims, at the assigned status.',
        overstated: 'The excerpts show something related but weaker than the claim or the assigned status (for example a mechanism without the measurement, or a stub described as live).',
        contradicted: 'The excerpts show the opposite of the claim, or show it is absent or disabled.',
        insufficient: 'The excerpts are unrelated, blank, too short or otherwise cannot settle the claim either way.',
      },
    };
  });
  return {
    version: 'vf-01-s1-01/claim-check/v1',
    state: { claims: stateClaims },
    questions,
    ids,
  };
}

/** VF-06-S1-01: is one diagram consistent with the source it cites? Noul + Choice. */
function diagramReview(diagram) {
  return {
    version: 'vf-06-s1-01/diagram-review/v1',
    state: { diagram_id: diagram.id, diagram_source: diagram.source, cited_excerpts: diagram.excerpts },
    questions: {
      supported: {
        type: 'noul',
        instructions: 'Every relationship, ordering or control-flow claim made by `diagram_source` that carries a path:line citation is stated by the matching entry of `cited_excerpts`. Answer yes only if you found no claim the excerpts fail to support.',
      },
      verdict: {
        type: 'choice',
        instructions: 'Compare the labelled relationships and control flow in `diagram_source` with the source excerpts in `cited_excerpts`. Choose "supported" if the excerpts back the diagram, "contradicted" if at least one cited claim is the opposite of, or inconsistent with, what the excerpt shows, and "uncheckable" if the excerpts are too short or unrelated to judge. Do not use outside knowledge.',
        criteria: {
          supported: 'The cited excerpts back every cited relationship in the diagram.',
          contradicted: 'At least one cited relationship or control-flow step is inconsistent with its excerpt.',
          uncheckable: 'The excerpts are missing, unrelated or too short to judge the diagram.',
        },
      },
    },
  };
}

const IMPACT_LEVELS = [
  'No user impact: informational, a retired or experimental surface, or a failure that affects no published artefact.',
  'Low: one non-critical workflow is red or a repository is unreadable, but published surfaces and the main build still work.',
  'Moderate: a core build or verification workflow is red, so changes cannot be trusted, but published surfaces still respond.',
  'High: a published surface is down, or a deploy/build workflow that produces a published surface is failing.',
];

/** VF-03-S1-01: rank exact failed/unreadable observations by likely user impact. Score per item. */
function healthPriority(items) {
  const questions = {};
  const ids = [];
  const stateItems = {};
  items.forEach((it, i) => {
    const id = `impact_${i}`;
    ids.push({ id, index: i });
    stateItems[id] = it;
    questions[id] = {
      type: 'score',
      instructions: `Rate the likely end-user impact of the observation in \`state.items.${id}\`. Each item is one exact, already-measured fact from a nightly estate health snapshot (a repository workflow outcome, an unreadable repository, or a published surface that did not respond). Do not doubt the measurement; judge only how much it matters to users of the estate's published surfaces.`,
      criteria: IMPACT_LEVELS,
    };
  });
  return { version: 'vf-03-s1-01/health-priority/v1', state: { items: stateItems }, questions, ids };
}

const TRIAGE_CATEGORIES = {
  'source-build': 'The failure is in compiling or testing this repository\'s own source in CI.',
  'stale-bind-path': 'A host/container build used stale or wrong bind-mounted paths, so the image baked old code.',
  'pin-or-schema': 'A pinned dependency, submodule pointer, lockfile or schema contract is out of step.',
  dependency: 'An external dependency, registry, action or network resource is unavailable or changed.',
  'gpu-xr': 'The failure needs GPU, display or XR hardware that the CI runner lacks.',
  'runtime-evidence-gap': 'The build passed but a deployment or runtime receipt is missing or unverified.',
  insufficient: 'The name and outcome of the workflow alone do not indicate where to look.',
};

/** ES-09-S1-01: which build context should a failed run be investigated in first? Choice per item. */
function failureTriage(items) {
  const questions = {};
  const ids = [];
  const stateItems = {};
  items.forEach((it, i) => {
    const id = `triage_${i}`;
    ids.push({ id, index: i });
    stateItems[id] = it;
    questions[id] = {
      type: 'choice',
      instructions: `For the failed workflow run in \`state.items.${id}\` (only its repository, workflow name and conclusion are known; no log text), choose the build context an engineer should investigate first. Choose "insufficient" whenever the workflow name and conclusion do not clearly point to one context; do not guess.`,
      criteria: TRIAGE_CATEGORIES,
    };
  });
  return { version: 'es-09-s1-01/failure-triage/v1', state: { items: stateItems }, questions, ids };
}

/** Excerpt of `lines` around a 1-based citation a[-b], bounded; `null` when out of range. */
function excerpt(lines, a, b) {
  const start = Math.max(1, a - 1);
  const end = Math.min(lines.length, Math.min((b || a) + 1, start + MAX_EXCERPT_LINES - 1));
  if (!lines.length || a > lines.length || end < start) return null;
  return lines.slice(start - 1, end).map((t, k) => `${start + k}: ${t.slice(0, 240)}`).join('\n');
}

module.exports = { claimCheck, diagramReview, healthPriority, failureTriage, excerpt, IMPACT_LEVELS, TRIAGE_CATEGORIES };
