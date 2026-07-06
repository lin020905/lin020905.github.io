const translations = {
  zh: {
    skipLink: "跳转到主要内容",
    navAbout: "简介",
    navWork: "作品",
    navContact: "联系",
    heroEyebrow: "Portfolio Framework",
    heroTitleLine1: "Your Name",
    heroTitleLine2: "个人作品集",
    heroText:
      "一个为开发者、设计者或创作者准备的极简个人网站框架。替换这里的文字，就能把它变成你的线上名片。",
    heroPrimary: "查看作品",
    heroSecondary: "联系我",
    panelLabel: "Index",
    statOneLabel: "Focus",
    statOneValue: "Portfolio",
    statTwoLabel: "Style",
    statTwoValue: "Editorial",
    statThreeLabel: "Status",
    statThreeValue: "Replaceable",
    aboutTitle: "简介占位",
    aboutLead:
      "在这里写下你的身份、兴趣方向和正在探索的问题。保持简短、有力量，让访客快速知道你是谁。",
    aboutNoteOne: "可替换为：学校 / 专业 / 城市 / 研究方向 / 当前目标。",
    aboutNoteTwo: "建议控制在 2-3 句话，避免把首页写成完整简历。",
    workTitle: "精选作品",
    projectTypeOne: "Web / App",
    projectTypeTwo: "Research / Tool",
    projectTypeThree: "Creative / Study",
    projectOneTitle: "项目标题占位",
    projectOneText: "用一句话说明项目解决了什么问题，以及你负责的核心部分。",
    projectTwoTitle: "项目标题占位",
    projectTwoText: "可以放课程项目、开源项目、实验作品或一段值得展示的实践。",
    projectThreeTitle: "项目标题占位",
    projectThreeText: "如果暂时没有项目，可以先保留为占位，后续替换为真实内容。",
    projectLink: "查看链接",
    skillsTitle: "方向与技能",
    skillOne: "Frontend",
    skillTwo: "Design Systems",
    skillThree: "Data Visualization",
    skillFour: "Creative Coding",
    skillFive: "Open Source",
    contactTitle: "保持联系",
    contactText:
      "把这里替换成邮箱、GitHub、LinkedIn、作品集文档或任何你希望访客点击的入口。",
    contactLinkOne: "Email",
    contactLinkTwo: "Resume",
    footerText: "© 2026 lin020905. Built for GitHub Pages.",
    backTop: "返回顶部",
  },
  en: {
    skipLink: "Skip to main content",
    navAbout: "About",
    navWork: "Work",
    navContact: "Contact",
    heroEyebrow: "Portfolio Framework",
    heroTitleLine1: "Your Name",
    heroTitleLine2: "Portfolio",
    heroText:
      "A minimal personal site framework for developers, designers, and makers. Replace the copy here to turn it into your online calling card.",
    heroPrimary: "View Work",
    heroSecondary: "Contact",
    panelLabel: "Index",
    statOneLabel: "Focus",
    statOneValue: "Portfolio",
    statTwoLabel: "Style",
    statTwoValue: "Editorial",
    statThreeLabel: "Status",
    statThreeValue: "Replaceable",
    aboutTitle: "About Placeholder",
    aboutLead:
      "Write a short note about who you are, what you care about, and what you are exploring now.",
    aboutNoteOne: "Replace with: school, major, city, research direction, or current goal.",
    aboutNoteTwo: "Keep it to two or three strong sentences instead of a full resume.",
    workTitle: "Selected Work",
    projectTypeOne: "Web / App",
    projectTypeTwo: "Research / Tool",
    projectTypeThree: "Creative / Study",
    projectOneTitle: "Project Title Placeholder",
    projectOneText:
      "Describe the problem this project solved and the core contribution you made.",
    projectTwoTitle: "Project Title Placeholder",
    projectTwoText:
      "Use this slot for a course project, open-source work, experiment, or polished practice.",
    projectThreeTitle: "Project Title Placeholder",
    projectThreeText:
      "If you do not have enough projects yet, keep this as a placeholder for later.",
    projectLink: "View Link",
    skillsTitle: "Direction & Skills",
    skillOne: "Frontend",
    skillTwo: "Design Systems",
    skillThree: "Data Visualization",
    skillFour: "Creative Coding",
    skillFive: "Open Source",
    contactTitle: "Keep In Touch",
    contactText:
      "Replace this section with an email, GitHub, LinkedIn, portfolio document, or any link you want visitors to open.",
    contactLinkOne: "Email",
    contactLinkTwo: "Resume",
    footerText: "© 2026 lin020905. Built for GitHub Pages.",
    backTop: "Back To Top",
  },
};

const storageKey = "lin020905-site-language";
const languageButtons = document.querySelectorAll("[data-language]");
const translatableNodes = document.querySelectorAll("[data-i18n]");

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

function setLanguage(language) {
  const dictionary = translations[language] || translations.zh;

  translatableNodes.forEach((node) => {
    const key = node.dataset.i18n;
    if (dictionary[key]) {
      node.textContent = dictionary[key];
    }
  });

  languageButtons.forEach((button) => {
    const isActive = button.dataset.language === language;
    button.setAttribute("aria-pressed", String(isActive));
  });

  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  saveLanguage(language);
}

languageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setLanguage(button.dataset.language);
  });
});

setLanguage(getInitialLanguage());
