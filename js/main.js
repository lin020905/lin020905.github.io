const loadingScreen = document.getElementById('loading-screen');
const loadingBar = document.getElementById('loading-bar');
const loadingPercent = document.getElementById('loading-percent');
const header = document.querySelector('.site-header');
const revealItems = document.querySelectorAll('.reveal-up');
const menuToggle = document.getElementById('menu-toggle');
const navLinks = document.getElementById('nav-links');
const langSwitch = document.getElementById('lang-switch');

function runFakeLoading() {
  let progress = 0;
  const timer = setInterval(() => {
    progress += Math.floor(Math.random() * 10) + 3;
    if (progress > 100) progress = 100;

    if (loadingBar) loadingBar.style.width = `${progress}%`;
    if (loadingPercent) loadingPercent.textContent = `${progress}%`;

    if (progress >= 100) {
      clearInterval(timer);
      setTimeout(() => {
        loadingScreen?.classList.add('hidden');
      }, 450);
    }
  }, 90);
}

function setupHeaderScroll() {
  window.addEventListener('scroll', () => {
    if (window.scrollY > 28) {
      header?.classList.add('scrolled');
    } else {
      header?.classList.remove('scrolled');
    }
  });
}

function setupRevealAnimation() {
  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.2 }
  );

  revealItems.forEach(item => observer.observe(item));
}

function setupMobileMenu() {
  menuToggle?.addEventListener('click', () => {
    navLinks?.classList.toggle('open');
  });

  navLinks?.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => navLinks.classList.remove('open'));
  });
}

function setupLanguageSwitch() {
  const updateBtnText = lang => {
    langSwitch.textContent = lang === 'zh' ? 'EN / 中文' : '中文 / EN';
  };

  let current = window.__i18n.getCurrent();
  window.__i18n.applyLanguage(current);
  updateBtnText(current);

  langSwitch?.addEventListener('click', () => {
    current = current === 'zh' ? 'en' : 'zh';
    window.__i18n.applyLanguage(current);
    updateBtnText(current);
  });
}

function setYear() {
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();
}

runFakeLoading();
setupHeaderScroll();
setupRevealAnimation();
setupMobileMenu();
setupLanguageSwitch();
setYear();