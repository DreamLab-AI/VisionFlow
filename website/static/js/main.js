import { initMesh } from './mesh-webgl.js';

const MOBILE = () => matchMedia('(max-width: 768px)').matches;

// One source of truth for the sixteen mobile sections. The index rows and the
// rail pips are two renderings of this list, so they can never drift.
const SECTIONS = [
  ['hero', 'The offer'], ['questions', 'Straight answers'], ['problem', 'Problem'],
  ['substrates', 'Six substrates'], ['guarantees', 'Guarantees'], ['immersive', 'Immersive'],
  ['broker', 'Judgment Broker'], ['economic', 'Economics'], ['loom', 'Ontology Loom'],
  ['cases', 'Case studies'], ['competitive', 'Landscape'], ['scaling', 'Scaling'],
  ['doors', 'Your seat'], ['status', 'What ships'], ['estate', 'Nightly check'],
  ['repos', 'Repositories']
];

// Assigned by initSheet(); called from the sticky bar and the index footer.
let openSheet = () => {};

async function initMeshBackdrop() {
  const canvas = document.getElementById('mesh-gl');
  if (!canvas) return;
  try {
    initMesh(canvas);
    canvas.classList.add('ready');
  } catch (e) {
    console.warn('Mesh backdrop unavailable, page renders without it:', e.message);
  }
}

function initScrollReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
  );
  document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
}

// Inline SVG figures: drawn in once when they enter the viewport (the same
// observer pattern as .reveal), then eased by scroll. --p runs 0 → 1 as the
// figure crosses the viewport; the CSS drifts the boxes a few pixels and walks
// the pulse along the governed path, so the figure reads as part of the page's
// motion rather than a looping animation. Under reduced motion the figures are
// static (CSS) and --p is never written.
function initFigures() {
  const figs = Array.prototype.slice.call(document.querySelectorAll('svg.fig'));
  if (!figs.length) return;
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) { en.target.classList.add('visible'); io.unobserve(en.target); }
    });
  }, { threshold: 0.35 });
  figs.forEach((f) => io.observe(f));

  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  let ticking = false;
  function ease() {
    const vh = window.innerHeight;
    figs.forEach((f) => {
      const r = f.getBoundingClientRect();
      if (r.bottom < -40 || r.top > vh + 40) return;
      const p = (vh - r.top) / (vh + r.height);
      f.style.setProperty('--p', Math.max(0, Math.min(1, p)).toFixed(3));
    });
    ticking = false;
  }
  window.addEventListener('scroll', () => {
    if (!ticking) { requestAnimationFrame(ease); ticking = true; }
  }, { passive: true });
  ease();
}

// Three doors: a WAI-ARIA tab set. Arrow keys, Home and End move between
// tabs; the panels are plain `hidden` toggles so the reading switch can still
// measure and flip the copy inside them.
function initDoors() {
  const list = document.querySelector('.door-tabs[role="tablist"]');
  if (!list) return;
  const tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));
  function select(tab, focus) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      const panel = document.getElementById(t.getAttribute('aria-controls'));
      if (panel) panel.hidden = !on;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(t, false));
    t.addEventListener('keydown', (e) => {
      let j = null;
      if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
      if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
      if (e.key === 'Home') j = 0;
      if (e.key === 'End') j = tabs.length - 1;
      if (j !== null) { e.preventDefault(); select(tabs[j], true); }
    });
  });
}

function initNavScroll() {
  const nav = document.getElementById('nav');
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        nav.classList.toggle('scrolled', window.scrollY > 50);
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });
}

// scrollIntoView({block:'start'}) lands section headings underneath the fixed
// nav. Scroll to offsetTop minus the live nav height instead (fixes both surfaces).
function scrollToTarget(target) {
  const nav = document.getElementById('nav');
  const navH = nav ? nav.offsetHeight : 64;
  const y = target.getBoundingClientRect().top + window.pageYOffset - navH - 8;
  window.scrollTo({ top: Math.max(0, y), behavior: 'smooth' });
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        scrollToTarget(target);
        history.replaceState(null, '', href);
        const navLinks = document.getElementById('nav-links');
        if (navLinks) navLinks.classList.remove('open');
      }
    });
  });
}

function initBackgroundVideo() {
  if (MOBILE()) return; // the ambient video is dropped below 768px (CSS hides it; skip the decode)
  const video = document.getElementById('bg-video');
  const hero = document.getElementById('hero');
  if (!video || !hero) return;
  let active = false;
  const observer = new IntersectionObserver(
    ([entry]) => {
      const heroVisible = entry.isIntersecting;
      if (heroVisible && active) {
        video.classList.remove('active');
        video.pause();
        active = false;
      } else if (!heroVisible && !active) {
        video.classList.add('active');
        video.play().catch(() => {});
        active = true;
      }
    },
    { threshold: 0.3 }
  );
  observer.observe(hero);
}

// Plain-English versions of each section intro (heading + lead). Sharp and
// executive, not simplified to death — the same claim, in fewer moving parts.
const PLAIN = {
  questions: { title: 'What this is, on one screen', lead: 'Everything after this block is detail. This is the whole thing.' },
  doors: { title: 'Read the version written for you', lead: 'One system, three readers: the person who will use it, the engineer who will inherit it, and the person who has to sign it off.' },
  status: { title: 'What is real today', lead: 'Every claim on this page is either running, part-built or deliberately parked. This table says which, checked against the engineering record on 6 September 2026.' },
  hero: { lead: 'AI agents now do real work, but most organisations can&rsquo;t say who authorised an action, what it drew on, or who owns the result. VisionFlow gives people and AI a shared, accountable way to work: data stays with its owner, key claims are checked, every decision is recorded, and anything consequential goes to a person.' },
  problem: { title: 'More AI means more coordination, not less', lead: 'Teams are wiring AI tools together faster than anyone can govern them: duplicated effort, invisible risk, and decisions no one can explain. VisionFlow wraps identity, ownership and approval around the work itself, so independent tools cooperate without a new central bottleneck.', callout: '<strong>It is already happening.</strong> Most frontline AI use starts without management sign-off. Your people are stitching agents together and automating shortcuts on their own. The organisation is becoming an agentic mesh whether you planned it or not. The only real choice is whether it is governed.' },
  evolution: { title: 'AI stopped answering and started acting', lead: 'You can check a chatbot one reply at a time. You can&rsquo;t check a swarm of agents making hundreds of linked decisions. Plausible text is no longer a control. VisionFlow gives every agent the same checked vocabulary and rules, and rejects contradictions before they become accepted facts.' },
  substrates: { title: 'Six focused services, one accountable workflow', lead: 'All-in-one AI platforms are convenient until you need to change a model, move your data, or work with another organisation. VisionFlow keeps knowledge, agents, private data, human approval, delivery and grounding as separate services, joined by one shared identity, so no single part holds all the control.', callout: '<strong>The knowledge is public.</strong> An open corpus of more than 8,000 pages doubles as a formal, machine-checked vocabulary: readable on the web, rendered in 3D, and released for anyone to inspect. It is a testbed built to prove the method, not an encyclopaedia.' },
  guarantees: { title: 'Important actions need proof, not trust', lead: 'A policy in a handbook won&rsquo;t stop an automated system acting outside it. VisionFlow checks permissions as the work happens, verifies who is asking, records the evidence, and routes the decision to a person whenever real authority is required.' },
  immersive: { title: 'Make complex relationships something people can see', lead: 'Dense knowledge networks are hard to grasp on a flat dashboard, and harder still when specialists need to inspect the same evidence together. VisionFlow turns connected information into a shared space you can walk through, compare, and point at in the same session.', callout: '<strong>From lab to headset.</strong> The immersive view is moving from room-sized projection labs to a standalone Meta Quest 3 headset: same data, same physics, same identity, now portable. The lab proved it at scale; the headset takes it anywhere.' },
  broker: { title: 'Let AI prepare the decision; keep the authority with people', lead: 'Human sign-off becomes a rubber stamp when the reviewer gets too little, too late. Here an agent submits a clear proposal with its evidence and the authority it is asking for; the right person approves, rejects or revises. That signed decision is what actually runs.' },
  economic: { title: 'Spend compute where a better decision is worth it', lead: 'AI cost is easy to count. The cost of a missed insight, a repeated experiment, or an ungoverned decision usually isn&rsquo;t. VisionFlow puts deeper analysis on the problems that justify it, and cuts the waste of agents redoing work or acting on conflicting information.' },
  loom: { title: 'Give any model the facts before it answers', lead: 'A strong model still invents details when it lacks your organisation&rsquo;s knowledge. The Ontology Loom hands it a compact set of checked, relevant facts before each answer, and you can swap the underlying model without touching what depends on it.', callout: '<strong>One honest caveat.</strong> The recall numbers come from a mostly synthetic test corpus, which is what makes them a fair read of the method rather than any one model. Everything else runs in the deployed system today: swapping models behind one interface, the checked fact store, and sending each model only the facts it needs.' },
  cases: { title: 'Coordinate work no single team can own', lead: 'Climate research, drug discovery and large productions all depend on organisations that must cooperate but can&rsquo;t pool every dataset or hand over control. Each keeps its own data and rules while sharing signed requests, approved findings and evidence across the group.' },
  competitive: { title: 'Connecting agents is common. Accountable coordination isn&rsquo;t.', lead: 'Most agent products can pass messages between tools. Few can prove identity, keep data with its owner, enforce human decisions and check shared facts at the same time, in the open, across organisations. That combination is the whole point.', callout: '<strong>The closest parallel.</strong> Block&rsquo;s Buzz is an open, Nostr-native platform where people and agents share channels and every action is signed. A serious team reaching the same foundation is a good sign for the direction. The piece it does not have is formal, checked reasoning. With owner-held data and immersive views, that is where VisionFlow is different.' },
  scaling: { title: 'Start with one operator; grow without changing the rules', lead: 'Pilots that work for one expert often need a costly rebuild when a team or partner joins. VisionFlow uses the same identity, ownership and approval model for an individual, a team, and a federation of organisations.' },
  repos: { title: 'Inspect, run, or replace every major part', lead: 'A governance promise means little if you can&rsquo;t see how it&rsquo;s enforced or leave the supplier. VisionFlow is built from open repositories with clear responsibilities: audit the controls, deploy what you need, and keep a real exit route.' }
};

// Plain-English versions of the detail panes (cards, callouts with diagrams).
// Matched to each section's panes in document order; a missing entry is left as-is.
const PANES = [
  { id: 'questions', sel: '.q-card p', plains: [
    'Six open-source services that turn what your organisation knows into something checked, searchable and visible. One cryptographic identity runs through all of them, so the person who signs a request is the same one who owns the data and approves the outcome.',
    'Ask the shared model a question mid-task. Watch the knowledge graph move as work happens. Review what an AI proposes and approve it with your own signature. Walk through the same graph with colleagues in a shared 3D space.',
    'AI can now produce work faster than anyone can check it. This keeps the speed and puts a person, with a signature, at every decision that matters.',
    'Changes nobody approved, decisions nobody can trace, AI tools with no limits on what they may touch or spend, and knowledge locked in documents and in people&rsquo;s heads.',
    'Someone writes a note. The system notices it matters, a person reviews the proposed change, a consistency check confirms it breaks nothing, it is approved and merged, and the shared picture updates. <a href="#guarantees">See the six stages.</a>',
    'Research that regulators audit, handing a system from one team to another, leadership questions over a live knowledge base, and teaching complex subjects in a shared space. <a href="#cases">Three worked scenarios.</a>',
    'You run it on your own hardware with a graphics card. Each part is open source and free to run; DreamLab AI offers the engineering. <a href="#repos">The repositories.</a>',
    'DreamLab AI, a UK studio, building on fifteen years of immersive data research at the University of Salford. Everything is developed in the open.'
  ] },
  { id: 'guarantees', sel: '.fig-lead', plains: [
    'Every AI task can consult the shared model as it works: a small hint arrives automatically with each step, and a fuller answer is available on request within a fixed budget. Reading is always allowed. Writing never is: a change only lands after a consistency check and a signed approval.',
    'There are two kinds of knowledge here: working notes, which change freely, and the agreed vocabulary, which changes only through review. When a note earns its place, it moves from one to the other in six steps.'
  ] },
  { id: 'broker', sel: '.fig-lead', plains: [
    'Not every agreement needs the same level of trust, so each one says which level it has. The first two levels ship today: signed records an operator cannot quietly alter, and shared custody where several keys must agree. The stronger levels wait for an independent audit.'
  ] },
  { id: 'doors', sel: '.door-dev', plains: [
    'Six services, one job each: the knowledge engine, the agent runtime, the data-store library, the human decision forum, the public edge deployment and the fact-supplier. This repository describes how they fit together.',
    'Engine and runtime run side by side and find each other by name. Almost nothing is exposed to the network, and the one door that is demands a signed request. Browser and headset read the same compact live feed. Knowledge lives in storage that survives any container being thrown away.',
    'Start with the diagrams, which cite the exact code they describe. Then the decision records, which are never rewritten, only added to. Then follow one action end to end.',
    'What an agent may do is written down and enforced as it runs. Secrets never ship inside images. Opening anything new to the network needs a reason and a review.'
  ] },
  { id: 'status', sel: '.st-what p', plains: [
    'Running, with its checks. A change that would contradict the shared model is refused before it lands.',
    'Running in the browser, with more than thirteen thousand nodes on screen in the latest check.',
    'Being built. A room-scale lab version proved the idea; the portable one is on its way.',
    'People approve or reject AI proposals with a signature today, for one kind of change: adding to the shared vocabulary. The final confirmation step after a merge is not yet wired.',
    'Running, with a fixed list of what agents may do and one guarded way in from the network.',
    'Running and answering, but without automated checks on a hosted build yet.',
    'A new fact has gone the whole way through: proposed by an agent, approved by a person, published the same day.',
    'Tamper-evident records and shared custody are available now. The stronger trust levels wait for an independent audit.',
    'Balances, charges, exchange and token issuing work today on Bitcoin&rsquo;s test network. The estate runs its own Bitcoin node, and the Lightning rail is chosen and being wired in.',
    'Organisations can connect their own instances, but a full live test between two of them has not been run yet, so a single instance remains the supported setup.',
    'Each night the code proposes one improvement to itself and a person decides whether to merge it. That runs here; extending it across every part is in progress.',
    'There is no contact form yet. The button on this page takes you straight to DreamLab AI.'
  ] },
  { id: 'substrates', sel: '.substrate-card ul', plains: [
    'The knowledge engine. It holds the shared, machine-checked model of your field, shows it as a 3D graph you can explore, and runs the reasoning and physics that keep it consistent, running fast on the graphics card.',
    'Where the AI agents live. Each runs in its own sealed workspace with its own identity, a library of skills and a shared memory, and everything it does can be checked and approved.',
    'Personal data storage that stays yours. Each person or organisation keeps its own data privately, with fine-grained access control and a tamper-evident record of every change.',
    'Where people make the calls. Agents post proposals here and the right person approves, rejects or revises. Every decision is cryptographically signed.',
    'The public-facing product: the company site, built from the same forum kit and running at the edge: proof the parts are reusable in a real deployment.',
    'The fact-supplier. It hands any AI model a compact set of checked, relevant facts before it answers, and lets you swap the model behind one stable interface.'
  ] },
  { id: 'substrates', sel: '.identity-callout p', plains: [
    'One key, one identity. The same cryptographic key is a person&rsquo;s login, their permissions, the signature on what they do and their payment account, so identity is never guessed or re-issued as work moves between systems.'
  ] },
  { id: 'guarantees', sel: '.ledger-card p', plains: [
    'Every claim is checked against the rules as it is written. Anything malformed is turned away at the door, not caught later in review.',
    'Every recorded decision keeps a searchable trail: who claimed what, and when, is a simple lookup rather than a dig through logs.',
    'The network checks identity: each participant proves who they are before they can publish, so trust holds at the edges as well as the centre.',
    'Organisations connect without a middleman. Trust travels on each actor&rsquo;s own key, so two organisations can work together while each keeps its own data.',
    'New knowledge reaches the shared model through one governed door: an agent proposes, a person approves, and only then is it published, checked for contradictions on the way in.',
    'The picture moves the way the work does: as agents act, related ideas are physically drawn together on screen, so activity is something you can see, not just log.'
  ] },
  { id: 'immersive', sel: '.immersive-card p', plains: [
    'Data becomes objects you can reach into, and relationships become spaces you walk through, surfacing patterns you would miss on a flat screen.',
    'Hand tracking and physical controllers: you handle data the way you handle real objects.',
    'Real places, rebuilt as walk-through spaces at room scale, useful for surveys, heritage and environmental monitoring.',
    'Remote colleagues appear life-size, so gesture and gaze carry meaning, and a standalone headset extends it to anyone, anywhere.'
  ] },
  { id: 'broker', sel: '.identity-callout p', plains: [
    'The whole loop, end to end: someone speaks, an agent acts as itself and writes to that person&rsquo;s own store, the action appears live in the shared 3D view, and anything worth keeping is proposed for a person to approve.',
    'Value moves on the same rails as everything else. The identity that signs the work can also be paid. Balances are kept in satoshis, access is charged through the web&rsquo;s own payment-required response, and the real-money rail today is Lightning, backed by the estate&rsquo;s own Bitcoin node; stablecoin rails can follow later. The same machinery can issue tokens on Bitcoin; DREAM, the operator&rsquo;s own token, has been minted experimentally on the test network. High-value records borrow Bitcoin&rsquo;s security, on the test network by default.'
  ] },
  { id: 'economic', sel: '.econ-card > p', plains: [
    'Uncoordinated agents waste most of their effort rediscovering context, repeating reasoning and contradicting each other. Every session starts from cold.',
    'A shared model and record means agents stop re-deriving vocabulary, re-checking conclusions and stalling on decisions they cannot make, so each one adds signal, not noise.',
    'On problems worth millions (a drug, a climate model, a franchise), spending thousands on well-governed AI is a rounding error, if it stops one wrong conclusion spreading.',
    'Every decision is already an auditable record. In regulated industries, reconstructing that after the fact costs far more than capturing it as you go.'
  ] },
  { id: 'loom', sel: '.gap-card p', plains: [
    'Extra facts only help when they are relevant. On a weak match, feeding in context can crowd out what the model already knows, so the Loom sends the full set on a strong match, a little on a weak one, and nothing when the question is off-topic.',
    'The Loom keeps one checked, queryable store of facts behind its interface: the trusted source, never the messy working draft.',
    'New facts enter through a single governed door: an agent proposes, a person approves, and only then does it publish. Proven end to end.'
  ] },
  { id: 'loom', sel: '.identity-callout p', plains: [
    'The model is just a setting. Every application talks to one stable interface; whether a local or hosted model answers is carried in the result, not wired into each app, so you can change the model without changing anything that uses it.'
  ] },
  { id: 'cases', sel: '.case-card > p', plains: [
    'Several universities, agencies and an NGO each run their own copy with their own definitions, and shared reasoning makes sure a term like &ldquo;sea-surface temperature anomaly&rdquo; means the same thing to all of them. Data stays put; only agreed findings cross.',
    'A biotech, a research partner and a regulatory consultancy each keep their own agents inside their own boundary. The partner never sees the biotech&rsquo;s proprietary targets; findings and evidence move only under signed permission.',
    'A 12-episode series across five time zones, mapped from episodes down to individual assets. When a shot depends on an unapproved asset, the rule propagates and the blocked dependency becomes visually obvious.'
  ] },
  { id: 'competitive', sel: '.gap-card p', plains: [
    'Everyone else has their AI guess from patterns. VisionFlow checks conclusions against a formal model and rejects contradictions before they land: the one column no competitor fills.',
    'Rivals each match one piece: Buzz on identity and federation, Palantir on governance. Only VisionFlow carries all of it at once, bound to a single key.'
  ] },
  { id: 'scaling', sel: '.scale-card p', plains: [
    'One workspace, running on its own: local storage, local decisions, privacy on by default. One key, minutes to deploy.',
    'Engine, forum and agents on a shared relay, with a common model and human oversight. Changes are gated by signed approvals.',
    'Independent instances trusting each other over the network: each organisation keeps and hardens its own, with trust carried by identity rather than shared infrastructure.'
  ] }
];

function initReadingSwitch() {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const panels = [];

  function measureFace(face, sibling) {
    const prev = face.style.cssText;
    sibling.style.display = 'none';
    face.style.cssText = 'position:static;transform:none;backface-visibility:visible;display:flex;';
    const h = face.offsetHeight;
    face.style.cssText = prev;
    sibling.style.display = '';
    return h;
  }

  function makePanel(anchor, extra, plainHTML) {
    const panel = document.createElement('div');
    panel.className = 'rl-panel';
    const inner = document.createElement('div');
    inner.className = 'rl-inner';
    const tech = document.createElement('div');
    tech.className = 'rl-face rl-face-tech';
    const plain = document.createElement('div');
    plain.className = 'rl-face rl-face-plain';
    anchor.replaceWith(panel);
    tech.appendChild(anchor);
    if (extra) tech.appendChild(extra);
    plain.innerHTML = plainHTML;
    inner.append(tech, plain);
    panel.appendChild(inner);
    const i = panels.length;
    const r = Math.abs(Math.sin((i + 1) * 12.9898) * 43758.5453) % 1; // stable pseudo-random
    panel.style.setProperty('--d', (r * 0.32).toFixed(3) + 's');
    panel.style.setProperty('--dur', (0.92 + r * 0.42).toFixed(3) + 's');
    panels.push({ panel, inner, tech, plain, techH: 0, plainH: 0 });
  }

  const heroSub = document.querySelector('#hero .hero-sub');
  if (heroSub) makePanel(heroSub, null, `<p class="hero-sub">${PLAIN.hero.lead}</p>`);
  Object.entries(PLAIN).forEach(([id, copy]) => {
    if (id === 'hero') return;
    const section = document.getElementById(id);
    if (!section) return;
    const h2 = section.querySelector('h2');
    if (!h2) return;
    const lead = section.querySelector('.section-lead');
    const plainHTML = `<h2>${copy.title}</h2>` + (lead && copy.lead ? `<p class="section-lead">${copy.lead}</p>` : '');
    makePanel(h2, lead && copy.lead ? lead : null, plainHTML);
    if (copy.callout) {
      const callP = section.querySelector('.callout p');
      if (callP) makePanel(callP, null, `<p>${copy.callout}</p>`);
    }
  });

  PANES.forEach(({ id, sel, plains }) => {
    const section = document.getElementById(id);
    if (!section) return;
    section.querySelectorAll(sel).forEach((el, i) => {
      if (plains[i] && !el.closest('.rl-panel')) makePanel(el, null, `<p>${plains[i]}</p>`);
    });
  });

  function sizeAll(active) {
    if (MOBILE()) return; // no 3D flip on mobile — faces flow statically, CSS forces height:auto
    // Panels inside a hidden tab (the three doors) measure as zero height, so the
    // hidden tab panels are shown invisibly for the duration of the measurement.
    const hid = Array.prototype.slice.call(document.querySelectorAll('[role="tabpanel"][hidden]'));
    hid.forEach((el) => { el.hidden = false; el.style.visibility = 'hidden'; });
    panels.forEach((p) => {
      p.techH = measureFace(p.tech, p.plain);
      p.plainH = measureFace(p.plain, p.tech);
      p.inner.style.height = (active === 'plain' ? p.plainH : p.techH) + 'px';
    });
    hid.forEach((el) => { el.hidden = true; el.style.visibility = ''; });
  }

  let liveT;
  function setMode(plain, animate) {
    const mob = MOBILE();
    document.body.classList.toggle('reading-plain', plain);
    panels.forEach((p) => {
      if (!mob) p.inner.style.height = (plain ? p.plainH : p.techH) + 'px';
      p.tech.setAttribute('aria-hidden', plain ? 'true' : 'false');
      p.plain.setAttribute('aria-hidden', plain ? 'false' : 'true');
      if (animate && !reduced && !mob) p.panel.classList.add('rl-live');
    });
    if (animate && !reduced && !mob) {
      clearTimeout(liveT);
      liveT = setTimeout(() => panels.forEach((p) => p.panel.classList.remove('rl-live')), 1700);
    }
    try { localStorage.setItem('vf-reading', plain ? 'plain' : 'tech'); } catch (e) { /* ignore */ }
  }

  // floating switch
  const sw = document.createElement('div');
  sw.id = 'reading-switch';
  // region (a labelled landmark): the pill is appended to <body> outside any
  // landmark, and axe's region rule requires all content to live inside one
  sw.setAttribute('role', 'region');
  sw.setAttribute('aria-label', 'Reading level');
  sw.innerHTML = '<span class="rl-thumb"></span><button type="button" data-plain="false">Technical</button><button type="button" data-plain="true">Plain English</button>';
  document.body.appendChild(sw);
  const btns = Array.prototype.slice.call(sw.querySelectorAll('button'));
  const thumb = sw.querySelector('.rl-thumb');
  function paintSwitch(plain) {
    const active = plain ? btns[1] : btns[0];
    thumb.style.width = active.offsetWidth + 'px';
    thumb.style.transform = 'translateX(' + active.offsetLeft + 'px)';
    btns.forEach((b) => b.setAttribute('aria-pressed', String((b.dataset.plain === 'true') === plain)));
  }
  // nav knob — the reading switch's mobile home (the floating pill is CSS-hidden <768px)
  const knob = document.createElement('button');
  knob.type = 'button';
  knob.id = 'm-reading';
  knob.className = 'm-reading';
  // Accessible name derives from the visible "Tech" / "Plain" text (WCAG 2.5.3 Label in Name);
  // the explanation rides as a description instead of overriding the name.
  knob.title = 'Reading level: technical or plain English';
  knob.innerHTML = '<span class="m-rl-tech">Tech</span><span class="m-rl-track"><span class="m-rl-knob"></span></span><span class="m-rl-plain">Plain</span>';
  const navInner = document.querySelector('.nav .nav-inner');
  const navToggleEl = document.getElementById('nav-toggle');
  if (navInner && navToggleEl) navInner.insertBefore(knob, navToggleEl);

  function applyMode(plain) {
    setMode(plain, true);
    paintSwitch(plain);
    knob.setAttribute('aria-pressed', String(plain));
  }
  knob.addEventListener('click', () => applyMode(!document.body.classList.contains('reading-plain')));
  btns.forEach((b) => b.addEventListener('click', () => applyMode(b.dataset.plain === 'true')));

  let startPlain = false;
  try { startPlain = localStorage.getItem('vf-reading') === 'plain'; } catch (e) { /* ignore */ }
  document.body.classList.add('rl-init');
  sizeAll(startPlain ? 'plain' : 'tech');
  setMode(startPlain, false);
  paintSwitch(startPlain);
  knob.setAttribute('aria-pressed', String(startPlain));
  requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.remove('rl-init')));

  let rt;
  addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(() => {
      sizeAll(document.body.classList.contains('reading-plain') ? 'plain' : 'tech');
      paintSwitch(document.body.classList.contains('reading-plain'));
    }, 160);
  });
}

// Full-screen index overlay (mobile). Replaces the hamburger dropdown: the same
// nav-toggle button now opens a flat, numbered 01–12 list built from SECTIONS.
function initIndex() {
  const overlay = document.getElementById('m-index');
  const toggle = document.getElementById('nav-toggle');
  const list = overlay && overlay.querySelector('.m-index-list');
  if (!overlay || !toggle || !list) return;
  const main = document.getElementById('main-content');

  list.innerHTML = SECTIONS.map(([id, label], i) =>
    `<a class="m-index-row" href="#${id}"><span class="n">${String(i + 1).padStart(2, '0')}</span>` +
    `<span class="l">${label}</span><span class="c" aria-hidden="true">&rarr;</span></a>`
  ).join('');

  let lastFocus = null;
  function open() {
    lastFocus = document.activeElement;
    overlay.classList.add('open');
    toggle.setAttribute('aria-expanded', 'true');
    if (main) main.setAttribute('inert', '');
    document.body.style.overflow = 'hidden';
    // defer focus: the overlay is visibility:hidden until the .open style applies,
    // and a visibility:hidden element cannot receive focus.
    const first = overlay.querySelector('button, a');
    if (first) requestAnimationFrame(() => requestAnimationFrame(() => first.focus()));
    document.addEventListener('keydown', onKey);
  }
  function close() {
    overlay.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
    if (main) main.removeAttribute('inert');
    document.body.style.overflow = '';
    document.removeEventListener('keydown', onKey);
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  function onKey(e) {
    if (e.key === 'Escape') { close(); return; }
    if (e.key !== 'Tab') return;
    const f = overlay.querySelectorAll('a[href], button');
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  toggle.addEventListener('click', (e) => {
    if (!MOBILE()) return; // the toggle is only visible <768px; desktop keeps the full nav
    e.preventDefault();
    overlay.classList.contains('open') ? close() : open();
  });
  overlay.querySelector('.m-index-close').addEventListener('click', close);

  // rows own their navigation (close, then offset-scroll on the next frame so the
  // body's overflow lock is lifted first) — the global smooth-scroll handler is
  // not attached to them because initIndex runs after initSmoothScroll.
  list.querySelectorAll('.m-index-row').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault();
    const id = a.getAttribute('href');
    close();
    const target = document.querySelector(id);
    if (target) requestAnimationFrame(() => { scrollToTarget(target); history.replaceState(null, '', id); });
  }));

  const contact = overlay.querySelector('.m-index-contact');
  if (contact) contact.addEventListener('click', () => { close(); openSheet(); });
}

// Scroll progress: a top bar (fraction scrolled) + a right-edge rail whose active
// pip is the section crossing the viewport centre.
function initProgress() {
  const bar = document.querySelector('.m-progress > span');
  const rail = document.querySelector('.m-rail');
  if (!bar || !rail) return;

  rail.innerHTML = SECTIONS.map(() => '<span></span>').join('');
  const pips = Array.prototype.slice.call(rail.children);
  const secs = SECTIONS.map(([id]) => document.getElementById(id)).filter(Boolean);

  let ticking = false;
  function updateBar() {
    const h = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0).toFixed(2) + '%';
    ticking = false;
  }
  window.addEventListener('scroll', () => {
    if (!ticking) { requestAnimationFrame(updateBar); ticking = true; }
  }, { passive: true });
  updateBar();

  // -50%/-50% collapses the root to a line at the viewport centre, so a section
  // "intersects" exactly while it spans the middle — robust even for sections
  // taller than the viewport (where a ratio threshold never fires).
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      const idx = secs.indexOf(en.target);
      if (idx >= 0) pips.forEach((p, i) => p.classList.toggle('on', i === idx));
    });
  }, { rootMargin: '-50% 0px -50% 0px', threshold: 0 });
  secs.forEach((s) => io.observe(s));
}

// Contact bottom sheet — the one primary action. No form submission exists in
// this repo (the contact endpoint is deferred), so the sheet routes to the real
// front door rather than faking a send.
function initSheet() {
  const sheet = document.getElementById('m-sheet');
  const scrim = document.querySelector('.m-scrim');
  const bar = document.getElementById('m-contact');
  const main = document.getElementById('main-content');
  if (!sheet || !scrim) return;
  const dismiss = sheet.querySelector('.m-sheet-dismiss');

  function open() {
    scrim.classList.add('open');
    sheet.classList.add('open');
    if (main) main.setAttribute('inert', '');
    const first = sheet.querySelector('a, button');
    if (first) requestAnimationFrame(() => first.focus());
    document.addEventListener('keydown', onKey);
  }
  function close() {
    scrim.classList.remove('open');
    sheet.classList.remove('open');
    if (main) main.removeAttribute('inert');
    document.removeEventListener('keydown', onKey);
  }
  function onKey(e) { if (e.key === 'Escape') close(); }

  if (bar) bar.addEventListener('click', open);
  scrim.addEventListener('click', close);
  if (dismiss) dismiss.addEventListener('click', close);
  sheet.querySelectorAll('a').forEach((a) => a.addEventListener('click', close));
  openSheet = open;
}

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ── Nightly estate check ───────────────────────────────────────────────────
// Renders website/static/data/estate-health.json: the snapshot CI committed at
// 02:30 UTC, not a live query. Two rules govern this code.
//   1. The section's .container.reveal wrapper is already observed by
//      initScrollReveal(); replacing it would detach the observed node and
//      leave the section permanently at opacity 0. Everything here writes into
//      #estate-body, a plain child, and never touches the wrapper.
//   2. Every string below comes from the GitHub API by way of the collector, so
//      nothing reaches innerHTML without passing through esc().

const CI_PILLS = {
  green: ['st-green', 'green'],
  red: ['st-red', 'red'],
  amber: ['st-amber', 'amber'],
  none: ['st-none', 'no workflows'],
  unknown: ['st-none', 'unreadable']
};

function esc(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// esc() neutralises markup but not a URL scheme: a "javascript:" href survives
// escaping intact. Every link below is built from collector data, so href values
// go through this allowlist and anything that is not http(s) renders as plain
// text rather than as a link.
function safeUrl(value) {
  if (!value) return '';
  try {
    const url = new URL(value, document.baseURI);
    return (url.protocol === 'https:' || url.protocol === 'http:') ? url.href : '';
  } catch (error) {
    return '';
  }
}

// Relative times are computed here rather than baked into the JSON, so a stale
// deploy reads as stale ("29 h ago") instead of quietly claiming freshness.
function relTime(iso) {
  if (!iso) return '';
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '';
  const secs = Math.round((Date.now() - then) / 1000);
  if (secs < 0) return 'just now';
  if (secs < 90) return 'just now';
  const mins = Math.round(secs / 60);
  if (mins < 90) return mins + ' min ago';
  const hours = Math.round(mins / 60);
  if (hours < 36) return hours + ' h ago';
  const days = Math.round(hours / 24);
  if (days < 60) return days + ' d ago';
  return Math.round(days / 30) + ' mo ago';
}

function num(value) {
  return (typeof value === 'number' && Number.isFinite(value)) ? String(value) : '—';
}

function estateTile(value, label, tone) {
  return `<div class="glass-card estate-tile ${tone}">
    <span class="metric-value">${esc(num(value))}</span>
    <span class="metric-label">${esc(label)}</span>
  </div>`;
}

function estateRun(run) {
  const verdict = run.conclusion || run.status || 'unknown';
  let tone = '';
  if (verdict === 'failure' || verdict === 'timed_out') {
    tone = ' r-red';
  } else if (verdict !== 'success' && verdict !== 'skipped') {
    tone = ' r-amber';   // cancelled, action_required, queued, or still running
  }
  const name = esc(run.workflow || 'workflow');
  const text = `${name} · ${esc(verdict)}`;
  const href = safeUrl(run.url);
  if (!href) return `<li><span class="${tone.trim()}">${text}</span></li>`;
  return `<li><a class="${tone.trim()}" href="${esc(href)}" target="_blank" rel="noopener">${text}</a></li>`;
}

// GitHub reports no build status for a Pages site deployed by a workflow, so a
// null status with build_type "workflow" means live-but-unreported, not broken.
// Only "errored" is a defect and only it is painted red; an unreported status
// stays dim so the page never invents a failure the estate does not have.
function estatePages(pages) {
  if (!pages || !pages.url) return '';
  let label = 'status not reported';
  let tone = '';
  if (pages.status === 'built') {
    label = 'built';
  } else if (pages.status === 'errored') {
    label = 'errored';
    tone = 'r-red';
  } else if (pages.status === 'building') {
    label = 'building';
    tone = 'r-amber';
  } else if (pages.build_type === 'workflow') {
    label = 'deployed by workflow, status not reported';
  }

  let shown = pages.url;
  try {
    const parsed = new URL(pages.url);
    shown = parsed.host + (parsed.pathname === '/' ? '' : parsed.pathname.replace(/\/$/, ''));
  } catch (error) {
    shown = pages.url;
  }
  const href = safeUrl(pages.url);
  const link = href
    ? `<a href="${esc(href)}" target="_blank" rel="noopener">${esc(shown)}</a>`
    : esc(shown);
  return `<span class="estate-meta estate-pages">Pages: ${link} · <span class="${tone}">${esc(label)}</span></span>`;
}

function estateRepoRow(repo) {
  const ci = repo.ci || {};
  const pill = CI_PILLS[ci.state] || CI_PILLS.unknown;
  const runs = Array.isArray(ci.runs) ? ci.runs : [];
  const provenance = [repo.provenance, repo.visibility, repo.archived ? 'archived' : null]
    .filter(Boolean).join(' · ');

  const repoHref = safeUrl(repo.url);
  const name = repoHref
    ? `<a href="${esc(repoHref)}" target="_blank" rel="noopener">${esc(repo.name)}</a>`
    : esc(repo.name);

  const releaseHref = repo.release ? safeUrl(repo.release.url) || repoHref : '';
  const release = repo.release && repo.release.tag && releaseHref
    ? `<a class="estate-link" href="${esc(releaseHref)}" target="_blank" rel="noopener">${esc(repo.release.tag)}</a>
       <span class="estate-meta">${esc(relTime(repo.release.date))}</span>`
    : '<span class="estate-dash">no release</span>';

  const headHref = repo.head ? safeUrl(repo.head.url) || repoHref : '';
  const head = repo.head && repo.head.short && headHref
    ? `<a class="estate-link" href="${esc(headHref)}" target="_blank" rel="noopener">${esc(repo.head.short)}</a>
       <span class="estate-meta">${esc(relTime(repo.head.date))}${repo.head.message ? ' · ' + esc(repo.head.message) : ''}</span>`
    : '<span class="estate-dash">—</span>';

  const prs = typeof repo.open_prs === 'number' && repo.open_prs > 0
    ? `<span class="estate-num on">${esc(num(repo.open_prs))}</span>`
    : `<span class="estate-num">${esc(num(repo.open_prs))}</span>`;

  const notes = (Array.isArray(repo.notes) ? repo.notes : [])
    .map((note) => `<span class="estate-meta estate-note-line">${esc(note)}</span>`).join('');

  return `<tr>
    <td class="estate-repo" data-label="Repository">${name}<span class="estate-meta">${esc(provenance)}</span>${estatePages(repo.pages)}${notes}</td>
    <td data-label="CI on ${esc(repo.default_branch || 'default branch')}">
      <span class="st-pill ${pill[0]}">${esc(pill[1])}</span>
      ${runs.length ? `<ul class="estate-runs">${runs.map(estateRun).join('')}</ul>` : ''}
    </td>
    <td data-label="Open PRs">${prs}</td>
    <td data-label="Latest release">${release}</td>
    <td data-label="Last commit">${head}</td>
  </tr>`;
}

function estateSurface(surface) {
  const ok = surface.ok === true;
  const detail = [
    surface.status == null ? 'no response' : String(surface.status),
    typeof surface.latency_ms === 'number' ? surface.latency_ms + ' ms' : null,
    surface.content_type || null,
    surface.note || null
  ].filter(Boolean).join(' · ');
  const href = safeUrl(surface.url);
  const name = href
    ? `<a href="${esc(href)}" target="_blank" rel="noopener">${esc(surface.name)}</a>`
    : esc(surface.name);
  return `<li>${name}
    <span class="st-pill ${ok ? 'st-green' : 'st-red'}">${ok ? 'up' : 'down'}</span>
    <span class="spacer"></span><span class="estate-meta">${esc(detail)}</span></li>`;
}

function estateRegistry(entry) {
  const href = safeUrl(entry.url);
  const name = href
    ? `<a href="${esc(href)}" target="_blank" rel="noopener">${esc(entry.name)}</a>`
    : esc(entry.name);
  const detail = [
    entry.published_at ? 'published ' + relTime(entry.published_at) : null,
    entry.note || null
  ].filter(Boolean).join(' · ');
  return `<li>${name} <span class="estate-meta">${esc(entry.registry)}</span>
    <span class="spacer"></span>
    <span class="estate-num">${esc(entry.version || '—')}</span>
    <span class="estate-meta">${detail ? '· ' + esc(detail) : ''}</span></li>`;
}

function estateMarkup(data) {
  const summary = data.summary || {};
  const repos = Array.isArray(data.repos) ? data.repos : [];
  const surfaces = Array.isArray(data.surfaces) ? data.surfaces : [];
  const registries = Array.isArray(data.registries) ? data.registries : [];
  const generator = data.generator || {};

  const surfacesUp = (typeof summary.surfaces_ok === 'number' && typeof summary.surfaces_total === 'number')
    ? `${summary.surfaces_ok}/${summary.surfaces_total}`
    : '—';

  const stamp = [
    data.generated_at ? 'collected ' + relTime(data.generated_at) : 'collection time unknown',
    data.generated_at || null,
    generator.revision ? 'revision ' + generator.revision : null
  ].filter(Boolean).map(esc).join(' · ');

  const collectorHref = safeUrl(generator.workflow_run);
  const runLink = collectorHref
    ? ` · <a href="${esc(collectorHref)}" target="_blank" rel="noopener">collector run</a>`
    : '';

  return `<div class="estate-summary-grid">
      ${estateTile(summary.green, 'repositories green', 't-green')}
      ${estateTile(summary.red, 'red', 't-red')}
      ${estateTile(summary.amber, 'amber', 't-amber')}
      ${estateTile(summary.unreadable, 'unreadable', 't-dim')}
      ${estateTile(summary.open_prs, 'open pull requests', 't-dim')}
      <div class="glass-card estate-tile t-green">
        <span class="metric-value">${esc(surfacesUp)}</span>
        <span class="metric-label">public surfaces up</span>
      </div>
    </div>
    <p class="estate-stamp">${stamp}${runLink}</p>

    <div class="status-wrap" tabindex="0" role="region" aria-label="Nightly estate check: repository table">
      <table class="status-table estate-table">
        <thead><tr>
          <th scope="col">Repository</th>
          <th scope="col">CI on default branch</th>
          <th scope="col">Open PRs</th>
          <th scope="col">Latest release</th>
          <th scope="col">Last commit</th>
        </tr></thead>
        <tbody>${repos.map(estateRepoRow).join('')}</tbody>
      </table>
    </div>

    <h3 class="estate-sub">Public surfaces</h3>
    <ul class="estate-list">${surfaces.map(estateSurface).join('')}</ul>

    <h3 class="estate-sub">Published registry versions</h3>
    <ul class="estate-list">${registries.map(estateRegistry).join('')}</ul>`;
}

async function initEstateHealth() {
  const body = document.getElementById('estate-body');
  if (!body) return;
  try {
    const response = await fetch('data/estate-health.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const data = await response.json();
    body.innerHTML = estateMarkup(data);
  } catch (error) {
    // Failing loudly beats a blank section: the file is the record, so say so.
    body.innerHTML = '<p class="estate-note">The nightly snapshot could not be read in this browser. ' +
      'It is committed at <a href="data/estate-health.json">data/estate-health.json</a>.</p>';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initSmoothScroll();     // binds #-anchors before initIndex builds its own rows
  initScrollReveal();
  initFigures();
  initDoors();            // before the reading switch, so the tab panels exist to be measured
  initReadingSwitch();
  initIndex();
  initProgress();
  initSheet();
  initEstateHealth();     // async; the section renders at rest without it
  initMeshBackdrop();     // renders one calm frame under reduced-motion; full flight otherwise
  if (!prefersReducedMotion) {
    initBackgroundVideo(); // early-returns on mobile
  }
});
