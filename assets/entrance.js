(function () {
  'use strict';

  var root = document.documentElement;
  var script = document.currentScript;
  var storageKey = 'lin-entrance-seen:' + (script.getAttribute('data-home') || '/');
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var replay = new URLSearchParams(window.location.search).get('intro') === 'replay';
  var seen = false;
  var navigation = window.performance.getEntriesByType('navigation')[0];

  try {
    seen = window.sessionStorage.getItem(storageKey) === '1';
  } catch (error) {
    // Storage is optional. The animation must still finish normally without it.
  }

  if (reducedMotion.matches || (!replay && (seen || window.location.hash ||
      (navigation && navigation.type === 'back_forward')))) return;

  var command = '\\enter';
  var idleDelay = 800;
  var keyDelay = 120;
  var holdDelay = 300;
  var fadeDelay = 400;
  var overlay;
  var content;
  var skipButton;
  var sequenceTimer;
  var fadeTimer;
  var finished = false;
  var leaving = false;

  function finish() {
    if (finished) return;
    finished = true;
    window.clearTimeout(sequenceTimer);
    window.clearTimeout(fadeTimer);
    window.clearTimeout(safetyTimer);

    var moveFocus = overlay && overlay.contains(document.activeElement);
    root.classList.remove('entrance-active', 'entrance-revealing');
    if (overlay) overlay.hidden = true;
    if (content) {
      content.removeAttribute('inert');
      content.removeAttribute('aria-hidden');
    }

    document.removeEventListener('keydown', onKeyDown, true);
    window.removeEventListener('pagehide', finish);
    reducedMotion.removeEventListener('change', onMotionChange);
    if (skipButton) skipButton.removeEventListener('click', finish);

    if (moveFocus) {
      var main = document.querySelector('main');
      if (main) {
        main.setAttribute('tabindex', '-1');
        main.focus({ preventScroll: true });
        main.addEventListener('blur', function () { main.removeAttribute('tabindex'); }, { once: true });
      }
    }
  }

  function reveal() {
    if (finished || leaving) return;
    leaving = true;
    root.classList.add('entrance-revealing');
    fadeTimer = window.setTimeout(finish, fadeDelay);
  }

  function onMotionChange(event) {
    if (event.matches) finish();
  }

  function onKeyDown(event) {
    if (event.key === 'Enter' || event.key === 'Escape') {
      event.preventDefault();
      finish();
    } else if (event.key === 'Tab') {
      // There is one control in the dialog; never tab into the covered page.
      event.preventDefault();
      skipButton.focus({ preventScroll: true });
    }
  }

  function start() {
    if (finished) return;
    overlay = document.getElementById('site-entrance');
    content = document.querySelector('.container');
    skipButton = overlay && overlay.querySelector('[data-entrance-skip]');
    var text = overlay && overlay.querySelector('[data-entrance-command]');

    // Fail open if markup or the dedicated stylesheet could not be loaded.
    if (!overlay || !content || !skipButton || !text ||
        window.getComputedStyle(overlay).getPropertyValue('--entrance-ready').trim() !== '1') {
      finish();
      return;
    }

    try {
      window.sessionStorage.setItem(storageKey, '1');
    } catch (error) {
      // Private browsing or blocked storage must not prevent entering the site.
    }

    content.setAttribute('inert', '');
    content.setAttribute('aria-hidden', 'true');
    overlay.hidden = false;
    overlay.setAttribute('data-phase', 'idle');
    overlay.focus({ preventScroll: true });
    skipButton.addEventListener('click', finish);
    document.addEventListener('keydown', onKeyDown, true);
    reducedMotion.addEventListener('change', onMotionChange);
    window.addEventListener('pagehide', finish);

    var position = 0;
    function typeNextCharacter() {
      if (finished) return;
      text.textContent = command.slice(0, ++position);
      if (position < command.length) {
        sequenceTimer = window.setTimeout(typeNextCharacter, keyDelay);
      } else {
        overlay.setAttribute('data-phase', 'holding');
        sequenceTimer = window.setTimeout(reveal, holdDelay);
      }
    }

    sequenceTimer = window.setTimeout(function () {
      if (finished) return;
      overlay.setAttribute('data-phase', 'typing');
      sequenceTimer = window.setTimeout(typeNextCharacter, keyDelay);
    }, idleDelay);
  }

  // Set up the escape hatch before hiding anything. Never leave a blank page.
  var safetyTimer = window.setTimeout(finish, 5000);
  root.classList.add('entrance-active');
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }
})();
