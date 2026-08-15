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

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initNavToggle();
  initSmoothScroll();
  initScrollReveal();
  initMeshBackdrop(); // renders one calm frame under reduced-motion; full flight otherwise
  if (!prefersReducedMotion) {
    initBackgroundVideo();
  }
});
