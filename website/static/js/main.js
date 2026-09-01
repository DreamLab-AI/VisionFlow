import { initMesh } from './mesh-webgl.js';

const MOBILE = () => matchMedia('(max-width: 768px)').matches;

// One source of truth for the twelve mobile sections. The index rows and the
// rail pips are two renderings of this list, so they can never drift.
const SECTIONS = [
  ['hero', 'The offer'], ['problem', 'Problem'], ['substrates', 'Six substrates'],
  ['guarantees', 'Guarantees'], ['immersive', 'Immersive'], ['broker', 'Judgment Broker'],
  ['economic', 'Economics'], ['loom', 'Ontology Loom'], ['cases', 'Case studies'],
  ['competitive', 'Landscape'], ['scaling', 'Scaling'], ['repos', 'Repositories']
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
    'Value moves on the same rails as everything else. The identity that signs the work can also be paid; settlement is in US-dollar stablecoins, and high-value records borrow Bitcoin&rsquo;s security, with no token of our own.'
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
    panels.forEach((p) => {
      p.techH = measureFace(p.tech, p.plain);
      p.plainH = measureFace(p.plain, p.tech);
      p.inner.style.height = (active === 'plain' ? p.plainH : p.techH) + 'px';
    });
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
  knob.setAttribute('aria-label', 'Reading level: technical or plain English');
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

document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initSmoothScroll();     // binds #-anchors before initIndex builds its own rows
  initScrollReveal();
  initReadingSwitch();
  initIndex();
  initProgress();
  initSheet();
  initMeshBackdrop();     // renders one calm frame under reduced-motion; full flight otherwise
  if (!prefersReducedMotion) {
    initBackgroundVideo(); // early-returns on mobile
  }
});
