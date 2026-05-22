const WASM_BASE = '/wasm';

function debounce(fn, ms) {
  let id;
  return (...args) => { clearTimeout(id); id = setTimeout(() => fn(...args), ms); };
}

async function initMeshHero() {
  const canvas = document.getElementById('mesh-canvas');
  if (!canvas) return;

  try {
    const mod = await import(`${WASM_BASE}/mesh-hero/wasm_mesh_hero.js`);
    await mod.default();

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const hero = new mod.MeshHero(rect.width, rect.height);

    canvas.addEventListener('mousemove', (e) => {
      const r = canvas.getBoundingClientRect();
      hero.set_mouse(e.clientX - r.left, e.clientY - r.top);
    });
    canvas.addEventListener('mouseleave', () => hero.set_mouse(-1000, -1000));

    let lastTime = performance.now();
    function animate(now) {
      const dt = Math.min(now - lastTime, 50);
      lastTime = now;
      hero.tick(dt);
      hero.render(ctx);
      requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);

    window.addEventListener('resize', debounce(() => {
      const r = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = r.width * dpr;
      canvas.height = r.height * dpr;
      canvas.style.width = r.width + 'px';
      canvas.style.height = r.height + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      hero.resize(r.width, r.height);
    }, 150));
  } catch (e) {
    console.warn('Mesh hero WASM not available, skipping animation:', e.message);
  }
}

async function initParticleFields() {
  const canvases = document.querySelectorAll('[id^="particle-canvas-"]');
  if (!canvases.length) return;

  try {
    const mod = await import(`${WASM_BASE}/particle-field/wasm_particle_field.js`);
    await mod.default();

    const fields = [];

    canvases.forEach((canvas) => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      const field = new mod.ParticleField(rect.width, rect.height);
      fields.push({ canvas, ctx, field, rect });
    });

    let lastTime = performance.now();
    function animate(now) {
      const dt = Math.min(now - lastTime, 50);
      lastTime = now;
      const scrollY = window.scrollY;

      fields.forEach(({ canvas, ctx, field }) => {
        const r = canvas.getBoundingClientRect();
        if (r.bottom > -100 && r.top < window.innerHeight + 100) {
          field.tick(dt, scrollY);
          field.render(ctx);
        }
      });

      requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);

    window.addEventListener('resize', debounce(() => {
      fields.forEach(({ canvas, ctx, field }) => {
        const r = canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = r.width * dpr;
        canvas.height = r.height * dpr;
        canvas.style.width = r.width + 'px';
        canvas.style.height = r.height + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        field.resize(r.width, r.height);
      });
    }, 150));
  } catch (e) {
    console.warn('Particle field WASM not available, skipping:', e.message);
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
  });
}

function initNavToggle() {
  const toggle = document.getElementById('nav-toggle');
  const navLinks = document.getElementById('nav-links');
  if (toggle && navLinks) {
    toggle.addEventListener('click', () => navLinks.classList.toggle('open'));
  }
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
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

  let videoActive = false;

  const observer = new IntersectionObserver(
    ([entry]) => {
      const heroVisible = entry.isIntersecting;
      if (heroVisible && videoActive) {
        video.classList.remove('active');
        video.pause();
        videoActive = false;
      } else if (!heroVisible && !videoActive) {
        video.classList.add('active');
        video.play().catch(() => {});
        videoActive = true;
      }
    },
    { threshold: 0.3 }
  );

  observer.observe(hero);
}

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initNavToggle();
  initSmoothScroll();
  initScrollReveal();

  if (!prefersReducedMotion) {
    initMeshHero();
    initParticleFields();
    initBackgroundVideo();
  }
});
