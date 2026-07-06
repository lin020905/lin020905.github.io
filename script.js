const translations = {
  zh: {
    skipLink: "跳转到主要内容",
    siteDescription: "项目 / 随笔 / 折腾记录",
    navAbout: "简介",
    navProjects: "项目",
    navNotes: "随笔",
    navContact: "联系",
    homeKicker: "Projects / Notes / Tinkering",
    funQuote: "我问你：今天有什么值得折腾？",
    homeIntro:
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
    siteDescription: "Projects / Notes / Tinkering",
    navAbout: "About",
    navProjects: "Projects",
    navNotes: "Notes",
    navContact: "Contact",
    homeKicker: "Projects / Notes / Tinkering",
    funQuote: "Question: what is worth tinkering with today?",
    homeIntro:
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

const storageKey = "lin020905-site-language";
const languageButtons = document.querySelectorAll("[data-language]");
const translatableNodes = document.querySelectorAll("[data-i18n]");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

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

function setLanguage(language) {
  const dictionary = translations[language] || translations.zh;

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

  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.documentElement.dataset.language = language;
  saveLanguage(language);
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

setupLanguageSwitch();
setupReveal();
