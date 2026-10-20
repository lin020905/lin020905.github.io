(() => {
  'use strict';

  if (document.querySelector('.pixel-trail')) return;
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return;
  canvas.className = 'pixel-trail';
  canvas.setAttribute('aria-hidden', 'true');
  canvas.hidden = true;
  document.body.appendChild(canvas);

  // Fail open if the stylesheet is unavailable: no unstyled canvas in the page.
  if (getComputedStyle(canvas).getPropertyValue('--pixel-trail-ready').trim() !== '1') return;

  const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const forcedColors = matchMedia('(forced-colors: active)');
  const colorScheme = matchMedia('(prefers-color-scheme: light)');
  const print = matchMedia('print');
  const lifetime = 620;
  const spacing = 14;
  let particles = [];
  let last = null;
  let frame = 0;
  let width = 0;
  let height = 0;
  let active = false;
  let color = '#ffffff';

  function clear() {
    cancelAnimationFrame(frame);
    frame = 0;
    particles = [];
    last = null;
    context.clearRect(0, 0, width, height);
  }

  function configure() {
    clear();
    active = finePointer.matches && !reducedMotion.matches && !forcedColors.matches && !print.matches;
    canvas.hidden = !active;
    width = active ? window.innerWidth : 1;
    height = active ? window.innerHeight : 1;
    const ratio = active ? Math.min(window.devicePixelRatio || 1, 2) : 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    color = getComputedStyle(canvas).color;
  }

  function emit(x, y, now) {
    particles.push({ x, y, born: now, size: 2 + Math.random() * 2,
      vx: (Math.random() - .5) * 30, vy: 12 + Math.random() * 20 });
    if (particles.length > 180) particles.shift();
  }

  function draw(now) {
    frame = 0;
    particles = particles.filter((particle) => now - particle.born < lifetime);
    context.clearRect(0, 0, width, height);
    context.fillStyle = color;
    for (const particle of particles) {
      const age = Math.max(0, now - particle.born);
      const fade = 1 - age / lifetime;
      const seconds = age / 1000;
      const size = Math.max(1, Math.round(particle.size * fade));
      context.globalAlpha = fade * .75;
      context.fillRect(
        Math.round(particle.x + particle.vx * seconds),
        Math.round(particle.y + particle.vy * seconds + 18 * seconds * seconds),
        size, size
      );
    }
    context.globalAlpha = 1;
    // No permanent animation loop: stop as soon as the last fragment fades.
    if (particles.length) frame = requestAnimationFrame(draw);
  }

  function move(event) {
    if (!active) return;
    if (event.pointerType === 'touch' || document.hidden ||
        document.documentElement.classList.contains('entrance-active') ||
        event.target.closest('input, textarea, select, [contenteditable], iframe')) {
      clear();
      return;
    }
    const x = event.clientX;
    const y = event.clientY;
    const now = performance.now();
    if (!last || now - last.time > 120) {
      emit(x, y, now);
      last = { x, y, time: now };
    } else {
      const dx = x - last.x;
      const dy = y - last.y;
      const distance = Math.hypot(dx, dy);
      const count = Math.min(80, Math.floor(distance / spacing));
      if (count) {
        for (let i = 1; i <= count; i += 1) {
          emit(last.x + dx * i * spacing / distance, last.y + dy * i * spacing / distance, now);
        }
        last = { x: last.x + dx * count * spacing / distance, y: last.y + dy * count * spacing / distance, time: now };
      } else {
        last.time = now;
      }
    }
    if (!frame) frame = requestAnimationFrame(draw);
  }

  document.addEventListener('pointermove', move, { passive: true });
  document.documentElement.addEventListener('pointerleave', clear);
  document.addEventListener('pointercancel', clear);
  document.addEventListener('visibilitychange', clear);
  document.addEventListener('keydown', clear);
  document.addEventListener('focusin', clear);
  window.addEventListener('blur', clear);
  window.addEventListener('pagehide', clear);
  window.addEventListener('scroll', clear, { passive: true, capture: true });
  window.addEventListener('resize', configure);
  [finePointer, reducedMotion, forcedColors, colorScheme, print].forEach((query) => query.addEventListener('change', configure));
  configure();
})();
