const translations = {
  zh: {
    skipLink: "跳转到主要内容",
    navAbout: "简介",
    navProjects: "项目",
    navNotes: "随笔",
    navContact: "联系",
    aboutLead:
      "一个用来收纳个人项目、创作实验和短随笔的轻量索引。先保持简单，再让内容慢慢长出来。",
    aboutLabel: "About",
    aboutTitle: "关于这个空间",
    aboutText:
      "这里不是一份厚重简历，更像一个持续更新的小站框架：项目可以沉淀成作品，随笔可以先保留成一句灵感。",
    projectsLabel: "Projects",
    projectsTitle: "项目索引",
    notesLabel: "Notes",
    notesTitle: "随笔片段",
    externalLink: "外部链接",
    openProject: "打开项目",
    contactLabel: "Contact",
    contactText: "欢迎交流项目、想法，或者只是打个招呼。",
    footerText: "© 2026 lin. Built with Jekyll.",
    backTop: "返回顶部",
  },
  en: {
    skipLink: "Skip to main content",
    navAbout: "About",
    navProjects: "Projects",
    navNotes: "Notes",
    navContact: "Contact",
    aboutLead:
      "A lightweight index for personal projects, creative experiments, and short notes. Keep it simple first, then let the work grow.",
    aboutLabel: "About",
    aboutTitle: "About This Space",
    aboutText:
      "This is not a heavy resume. It is a small site framework that can keep projects, notes, and sparks in one place.",
    projectsLabel: "Projects",
    projectsTitle: "Project Index",
    notesLabel: "Notes",
    notesTitle: "Note Fragments",
    externalLink: "External Link",
    openProject: "Open Project",
    contactLabel: "Contact",
    contactText: "Open to project notes, ideas, or a simple hello.",
    footerText: "© 2026 lin. Built with Jekyll.",
    backTop: "Back To Top",
  },
};

const quotes = {
  zh: ["爱能穿越时间和空间", "过去无可挽回，未来可以改变。"],
  en: [
    "Love can transcend time and space.",
    "The past cannot be undone, but the future can be changed.",
  ],
};

const storageKey = "lin020905-site-language";
const buildRefreshKey = "lin020905-build-refresh";
const legacyCacheCleanKey = "lin020905-legacy-cache-cleaned";
const currentBuildVersion = document.documentElement.dataset.buildVersion || "";
const languageButtons = document.querySelectorAll("[data-language]");
const translatableNodes = document.querySelectorAll("[data-i18n]");
const quoteNode = document.querySelector("[data-quote-rotator]");
const quoteFrame = document.querySelector("[data-quote-frame]");
const sectionNodes = document.querySelectorAll("[data-section]");
const sectionLinks = document.querySelectorAll("[data-section-link]");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let currentLanguage = "zh";
let currentQuoteIndex = 0;
let quoteAnimationTimer;
let buildRefreshAttempted = false;
let legacyCacheReloadAttempted = false;

function readSavedLanguage() {
  try {
    return localStorage.getItem(storageKey);
  } catch {
    return null;
  }
}

function saveLanguage(language) {
  try {
    localStorage.setItem(storageKey, language);
  } catch {
    return;
  }
}

function getInitialLanguage() {
  const savedLanguage = readSavedLanguage();
  return savedLanguage === "en" || savedLanguage === "zh" ? savedLanguage : "zh";
}

function animateLanguageChange() {
  if (prefersReducedMotion.matches) {
    return;
  }

  document.body.classList.remove("is-language-changing");
  void document.body.offsetWidth;
  document.body.classList.add("is-language-changing");

  window.setTimeout(() => {
    document.body.classList.remove("is-language-changing");
  }, 260);
}

function updateQuote({ animate = false } = {}) {
  if (!quoteNode) {
    return;
  }

  const quoteList = quotes[currentLanguage] || quotes.zh;
  const nextQuote = quoteList[currentQuoteIndex % quoteList.length];

  if (!animate || prefersReducedMotion.matches || !quoteFrame) {
    quoteNode.textContent = nextQuote;
    return;
  }

  window.clearTimeout(quoteAnimationTimer);
  quoteFrame.classList.remove("is-quote-entering", "is-quote-leaving", "is-quote-underlined");
  void quoteFrame.offsetWidth;
  quoteFrame.classList.add("is-quote-leaving", "is-quote-underlined");

  quoteAnimationTimer = window.setTimeout(() => {
    quoteNode.textContent = nextQuote;
    quoteFrame.classList.remove("is-quote-leaving");
    quoteFrame.classList.add("is-quote-entering");

    quoteAnimationTimer = window.setTimeout(() => {
      quoteFrame.classList.remove("is-quote-entering", "is-quote-underlined");
    }, 420);
  }, 220);
}

function setLanguage(language) {
  const dictionary = translations[language] || translations.zh;
  currentLanguage = language === "en" || language === "zh" ? language : "zh";

  translatableNodes.forEach((node) => {
    const key = node.dataset.i18n;
    if (Object.prototype.hasOwnProperty.call(dictionary, key)) {
      node.textContent = dictionary[key];
    }
  });

  languageButtons.forEach((button) => {
    const isActive = button.dataset.language === language;
    button.setAttribute("aria-pressed", String(isActive));
  });

  document.documentElement.lang = currentLanguage === "zh" ? "zh-CN" : "en";
  document.documentElement.dataset.language = currentLanguage;
  updateQuote();
  saveLanguage(currentLanguage);
  animateLanguageChange();
}

function setupLanguageSwitch() {
  languageButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setLanguage(button.dataset.language);
    });
  });

  setLanguage(getInitialLanguage());
}

function setupQuoteRotator() {
  if (!quoteNode) {
    return;
  }

  updateQuote();

  window.setInterval(() => {
    currentQuoteIndex += 1;
    updateQuote({ animate: true });
  }, 15000);
}

function setActiveSection(sectionId) {
  sectionNodes.forEach((section) => {
    section.classList.toggle("is-section-active", section.id === sectionId);
  });

  sectionLinks.forEach((link) => {
    const isActive = link.dataset.sectionLink === sectionId;

    if (isActive) {
      link.setAttribute("aria-current", "location");
    } else {
      link.removeAttribute("aria-current");
    }
  });
}

function setupSectionSwitching() {
  if (!sectionNodes.length) {
    return;
  }

  document.documentElement.classList.add("section-switch-ready");
  setActiveSection(sectionNodes[0].id);

  if (!("IntersectionObserver" in window) || prefersReducedMotion.matches) {
    sectionNodes.forEach((section) => section.classList.add("is-section-active"));
    return;
  }

  const sectionRatios = new Map(
    Array.from(sectionNodes, (section) => [section.id, section === sectionNodes[0] ? 1 : 0]),
  );

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        sectionRatios.set(entry.target.id, entry.intersectionRatio);
      });

      const activeSection = Array.from(sectionNodes).sort(
        (first, second) => sectionRatios.get(second.id) - sectionRatios.get(first.id),
      )[0];

      if (activeSection && sectionRatios.get(activeSection.id) > 0) {
        setActiveSection(activeSection.id);
      }
    },
    {
      rootMargin: "-28% 0px -42% 0px",
      threshold: [0, 0.15, 0.35, 0.6, 0.85],
    },
  );

  sectionNodes.forEach((section) => sectionObserver.observe(section));
}

function setupReveal() {
  const revealNodes = document.querySelectorAll("[data-reveal]");

  if (!revealNodes.length) {
    return;
  }

  document.documentElement.classList.add("reveal-ready");

  if (!("IntersectionObserver" in window) || prefersReducedMotion.matches) {
    revealNodes.forEach((node) => node.classList.add("is-visible"));
    return;
  }

  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      rootMargin: "0px 0px -70px 0px",
      threshold: 0.12,
    },
  );

  revealNodes.forEach((node) => revealObserver.observe(node));
}

function setupFreshBuildCheck() {
  if (!currentBuildVersion || !("fetch" in window)) {
    return;
  }

  const versionUrl = new URL("/site-version.json", window.location.origin);
  versionUrl.searchParams.set("t", String(Date.now()));

  fetch(versionUrl, { cache: "no-store" })
    .then((response) => (response.ok ? response.json() : null))
    .then((siteVersion) => {
      const latestVersion = siteVersion && siteVersion.version;

      if (!latestVersion || latestVersion === currentBuildVersion) {
        return;
      }

      const refreshToken = `${currentBuildVersion}:${latestVersion}`;

      if (buildRefreshAttempted) {
        return;
      }

      try {
        if (sessionStorage.getItem(buildRefreshKey) === refreshToken) {
          return;
        }

        sessionStorage.setItem(buildRefreshKey, refreshToken);
      } catch {
        // Continue with the in-memory guard if sessionStorage is unavailable.
      }

      buildRefreshAttempted = true;
      const freshUrl = new URL(window.location.href);
      freshUrl.searchParams.set("build", latestVersion);
      window.location.replace(freshUrl.toString());
    })
    .catch(() => {
      return;
    });
}

function clearLegacyServiceWorkerCache() {
  if (!("serviceWorker" in navigator)) {
    return;
  }

  navigator.serviceWorker
    .getRegistrations()
    .then((registrations) => {
      const rootScope = `${window.location.origin}/`;
      const rootRegistrations = registrations.filter((registration) => registration.scope === rootScope);

      if (!rootRegistrations.length && !navigator.serviceWorker.controller) {
        return false;
      }

      return Promise.all(rootRegistrations.map((registration) => registration.unregister())).then(() => true);
    })
    .then((hadLegacyServiceWorker) => {
      if (!hadLegacyServiceWorker) {
        return;
      }

      const clearCaches =
        "caches" in window
          ? caches
              .keys()
              .then((cacheNames) => Promise.all(cacheNames.map((cacheName) => caches.delete(cacheName))))
          : Promise.resolve();

      clearCaches
        .then(() => {
          if (legacyCacheReloadAttempted) {
            return;
          }

          legacyCacheReloadAttempted = true;

          try {
            if (sessionStorage.getItem(legacyCacheCleanKey) === currentBuildVersion) {
              return;
            }

            sessionStorage.setItem(legacyCacheCleanKey, currentBuildVersion);
          } catch {
            // Continue with the in-memory guard if sessionStorage is unavailable.
          }

          if (navigator.serviceWorker.controller) {
            window.location.reload();
          }
        })
        .catch(() => {
          return;
        });
    })
    .catch(() => {
      return;
    });
}

window.addEventListener("pageshow", setupFreshBuildCheck);
window.addEventListener("focus", setupFreshBuildCheck);

clearLegacyServiceWorkerCache();
setupLanguageSwitch();
setupQuoteRotator();
setupSectionSwitching();
setupReveal();
