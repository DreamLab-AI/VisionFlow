import { initMesh } from './mesh-webgl.js';

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

function initNavToggle() {
  const toggle = document.getElementById('nav-toggle');
  const navLinks = document.getElementById('nav-links');
  if (toggle && navLinks) {
    toggle.addEventListener('click', () => {
      const open = navLinks.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.replaceState(null, '', href);
        const navLinks = document.getElementById('nav-links');
        if (navLinks) navLinks.classList.remove('open');
      }
    });
  });
}

function initBackgroundVideo() {
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

const plainLanguageSections = {
  hero: {
    label: 'What VisionFlow does',
    title: 'Make AI teams accountable, even across organisations',
    problem: 'AI agents can now do useful work, but most organisations cannot see who authorised an action, which information it used, or who owns the result.',
    solution: 'VisionFlow gives people and AI agents a shared way to work. It keeps data with its owner, checks important claims, records decisions and sends risky actions to a person.',
    outcomes: ['Use several AI tools without handing control to one vendor', 'Keep a clear record of who decided what', 'Work with partners while each organisation keeps its own data']
  },
  problem: {
    label: 'Business problem',
    title: 'More AI creates more coordination work',
    problem: 'Teams are connecting AI tools faster than leaders can set rules for them. The result is duplicated work, invisible risk and decisions nobody can properly explain.',
    solution: 'VisionFlow puts identity, ownership and approval around the work itself, so independent tools can cooperate without creating another central bottleneck.',
    outcomes: ['See which person or agent took each action', 'Set approval points for consequential decisions', 'Replace shadow workflows with an auditable process']
  },
  evolution: {
    label: 'Why this matters now',
    title: 'AI has moved from answering questions to taking action',
    problem: 'A chatbot can be checked one answer at a time. A group of agents can make hundreds of linked decisions, so plausible-sounding text is no longer a sufficient control.',
    solution: 'VisionFlow gives every agent the same checked vocabulary and business rules. The system can reject contradictions before they become accepted facts.',
    outcomes: ['Apply the same definitions across teams and tools', 'Catch incompatible claims automatically', 'Change AI models without rebuilding the control layer']
  },
  substrates: {
    label: 'Practical solution',
    title: 'Six focused services, one accountable workflow',
    problem: 'Large all-in-one AI platforms are convenient until a team needs to change a model, move its data or work with an outside organisation.',
    solution: 'VisionFlow separates knowledge, agent work, private data, human approval, delivery and model grounding. A shared identity joins the services without giving any one of them total control.',
    outcomes: ['Replace one part without replacing the whole system', 'Keep sensitive data in the right organisation', 'Trace work across the full process']
  },
  guarantees: {
    label: 'Operational controls',
    title: 'Important actions require proof, not trust',
    problem: 'Policies written in a handbook do not stop an automated system from acting outside them.',
    solution: 'VisionFlow checks permissions when work happens, verifies the sender, records the evidence and routes decisions to a person when authority is required.',
    outcomes: ['Block unauthorised changes before they land', 'Produce an audit trail from existing operations', 'Give external partners access without creating a shared master account']
  },
  immersive: {
    label: 'Human understanding',
    title: 'Make complex relationships visible and discussable',
    problem: 'Large knowledge networks are hard to understand on a flat dashboard, especially when several specialists need to inspect the same evidence together.',
    solution: 'VisionFlow turns connected information into a shared spatial view. People can explore relationships, compare interpretations and point to the same structure in the same session.',
    outcomes: ['Find important connections in dense information', 'Review complex evidence with remote colleagues', 'Give non-specialists a usable view of the system']
  },
  broker: {
    label: 'Decision governance',
    title: 'Let AI prepare the decision; keep authority with people',
    problem: 'Human approval often becomes a rubber stamp because the reviewer receives too little context, too late.',
    solution: 'An agent submits a clear proposal with its evidence and requested authority. The right person can approve, reject or revise it, and the signed decision controls what happens next.',
    outcomes: ['Send only material decisions for review', 'Give reviewers the evidence behind each request', 'Connect approval directly to the permitted action']
  },
  economic: {
    label: 'Business case',
    title: 'Spend computing power where a better decision is worth more',
    problem: 'AI cost is easy to count; the cost of a missed interaction, a repeated experiment or a poorly governed decision is usually hidden.',
    solution: 'VisionFlow lets organisations use deeper analysis on high-value problems while reducing waste from agents repeating work or acting on conflicting information.',
    outcomes: ['Measure value against the decision, not token volume', 'Reuse checked knowledge across future work', 'Reduce the cost of rework and preventable mistakes']
  },
  loom: {
    label: 'Reliable answers',
    title: 'Give any AI model the facts it needs before it answers',
    problem: 'A capable model can still invent details when it lacks the right organisational knowledge.',
    solution: 'The Ontology Loom supplies a compact set of checked, relevant facts before each answer. The underlying model can change without changing the applications that depend on it.',
    outcomes: ['Improve factual recall with measured grounding', 'Run the same service with local or hosted models', 'Keep contradictory information out of the trusted knowledge base']
  },
  cases: {
    label: 'Where it helps',
    title: 'Coordinate work no single team can own',
    problem: 'Climate research, drug discovery and large creative programmes depend on organisations that need to cooperate but cannot pool every dataset or surrender authority.',
    solution: 'Each participant keeps its own information and operating rules while sharing signed requests, approved findings and evidence across the group.',
    outcomes: ['Collaborate without copying all data into one platform', 'Keep ownership clear across institutional boundaries', 'Build a shared result from independently governed work']
  },
  competitive: {
    label: 'Why it is different',
    title: 'Agent communication is common; accountable coordination is not',
    problem: 'Most agent products can connect tools and pass messages. Few can prove identity, preserve data ownership, enforce human decisions and check shared facts together.',
    solution: 'VisionFlow combines those controls in an open, federated system. Organisations can inspect the implementation and keep operating their own part of the network.',
    outcomes: ['Avoid dependence on one platform owner', 'Carry governance across organisational boundaries', 'Check claims against formal business knowledge']
  },
  scaling: {
    label: 'Adoption path',
    title: 'Start with one operator and grow without changing the rules',
    problem: 'Pilot systems often work for one expert, then need a costly redesign when a team or partner organisation joins.',
    solution: 'VisionFlow uses the same identity, data ownership and approval model for an individual, a team and a federation of organisations.',
    outcomes: ['Begin in a single controlled environment', 'Add team governance when it becomes useful', 'Connect independent organisations without centralising them']
  },
  repos: {
    label: 'Open implementation',
    title: 'Inspect, run or replace every major part',
    problem: 'A governance promise is weak when customers cannot see how it is enforced or move away from its supplier.',
    solution: 'VisionFlow is built from open repositories with clear responsibilities. Technical teams can audit the controls, deploy the services they need and integrate them with existing systems.',
    outcomes: ['Review the code behind security and governance claims', 'Adopt the system in stages', 'Retain a practical exit route']
  }
};

function technicalIcon() {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 9-3 3 3 3M16 9l3 3-3 3M14 5l-4 14"/></svg>`;
}

function createBlockTranslation(original, text, label, index) {
  const block = document.createElement('div');
  const technicalId = `technical-block-${index}`;
  original.id = technicalId;
  original.classList.add('technical-copy');
  block.className = 'translation-block';
  block.innerHTML = `
    <div class="plain-copy">
      <span class="plain-label">${label}</span>
      <p>${text}</p>
    </div>
    <button class="detail-toggle" type="button" aria-expanded="false" aria-controls="${technicalId}">
      ${technicalIcon()}<span>Technical detail</span>
    </button>
  `;
  original.before(block);
  block.append(original);
  return block;
}

function initPlainLanguagePanels() {
  const blocks = [];
  let index = 0;

  Object.entries(plainLanguageSections).forEach(([id, copy]) => {
    const section = document.getElementById(id);
    if (!section) return;

    const candidates = [];
    if (id === 'hero') {
      const heroCopy = section.querySelector('.hero-sub');
      if (heroCopy) candidates.push({ element: heroCopy, text: copy.solution, label: copy.label });
    } else {
      section.querySelectorAll('.section-lead').forEach((element, leadIndex) => {
        candidates.push({
          element,
          text: leadIndex === 0 ? copy.problem : copy.solution,
          label: leadIndex === 0 ? 'In plain language' : 'What this means'
        });
      });
      section.querySelectorAll('.glass-card > p, .substrate-card > p, .ledger-card > p, .case-card > p, .scale-card > p, .gap-card > p').forEach((element, cardIndex) => {
        candidates.push({
          element,
          text: copy.outcomes[cardIndex % copy.outcomes.length],
          label: 'Business value'
        });
      });
      section.querySelectorAll('.callout > p').forEach((element) => {
        candidates.push({ element, text: copy.solution, label: 'Practical answer' });
      });
    }

    candidates.forEach(({ element, text, label }) => {
      if (element.closest('.translation-block')) return;
      blocks.push(createBlockTranslation(element, text, label, index++));
    });
  });

  blocks.forEach((block) => {
    const button = block.querySelector('.detail-toggle');
    button.addEventListener('click', () => {
      const technical = block.classList.toggle('show-technical');
      button.setAttribute('aria-expanded', String(technical));
      button.querySelector('span').textContent = technical ? 'Plain-language view' : 'Technical detail';
    });
  });

  const resetObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting || !entry.target.classList.contains('show-technical')) return;
      const button = entry.target.querySelector('.detail-toggle');
      entry.target.classList.remove('show-technical');
      button?.setAttribute('aria-expanded', 'false');
      const label = button?.querySelector('span');
      if (label) label.textContent = 'Technical detail';
    });
  }, { threshold: 0 });

  blocks.forEach((block) => resetObserver.observe(block));
}

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initNavToggle();
  initSmoothScroll();
  initScrollReveal();
  initPlainLanguagePanels();
  initMeshBackdrop(); // renders one calm frame under reduced-motion; full flight otherwise
  if (!prefersReducedMotion) {
    initBackgroundVideo();
  }
});
