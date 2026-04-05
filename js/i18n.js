const translations = {
  zh: {
    'loading.subtitle': '正在同步展示数据...',
    'nav.intro': '引导',
    'nav.story': '背景',
    'nav.projects': '项目',
    'nav.news': '动态',
    'nav.contact': '联系',
    'hero.title': '电影感项目首页',
    'hero.desc': '这里作为你的主视觉区，可替换为项目视频、海报图与开场文案。',
    'hero.cta': '进入档案',
    'intro.title': '项目导语',
    'intro.desc': '请在这里填写你的个人定位、项目使命和核心价值。',
    'story.title': '背景与愿景',
    'story.desc': '这一段用于承载你的项目故事线，形式参考动画官网的剧情介绍分区。',
    'projects.title': '主要作品',
    'projects.items.p1.title': '项目名称',
    'projects.items.p1.desc': '项目摘要占位内容，请替换为你的真实信息。',
    'projects.items.p2.title': '项目名称',
    'projects.items.p2.desc': '项目摘要占位内容，请替换为你的真实信息。',
    'projects.items.p3.title': '项目名称',
    'projects.items.p3.desc': '项目摘要占位内容，请替换为你的真实信息。',
    'projects.items.p4.title': '项目名称',
    'projects.items.p4.desc': '项目摘要占位内容，请替换为你的真实信息。',
    'news.title': '资讯 / 时间轴',
    'news.n1': '网站基础框架已发布。',
    'news.n2': '在这里填写你的项目里程碑。',
    'news.n3': '在这里填写你的项目里程碑。',
    'contact.title': '加入 / 联系',
    'contact.desc': '在这里放置你的邮箱、GitHub 和社交账号。',
    'footer.rights': '保留所有权利。'
  },
  en: {
    'loading.subtitle': 'Synchronizing showcase data...',
    'nav.intro': 'Intro',
    'nav.story': 'Story',
    'nav.projects': 'Projects',
    'nav.news': 'News',
    'nav.contact': 'Contact',
    'hero.title': 'Cinematic Project Landing',
    'hero.desc': 'Use this key visual area for your project title video/image and opening statement.',
    'hero.cta': 'Enter Archive',
    'intro.title': 'Project Introduction',
    'intro.desc': 'Replace this paragraph with your personal positioning, project mission, and core value statement.',
    'story.title': 'Background & Vision',
    'story.desc': 'This section is designed to mimic official anime promo-page storytelling blocks. Put your long-form narrative here.',
    'projects.title': 'Main Works',
    'projects.items.p1.title': 'Project Name',
    'projects.items.p1.desc': 'Project summary placeholder. Replace with your own content.',
    'projects.items.p2.title': 'Project Name',
    'projects.items.p2.desc': 'Project summary placeholder. Replace with your own content.',
    'projects.items.p3.title': 'Project Name',
    'projects.items.p3.desc': 'Project summary placeholder. Replace with your own content.',
    'projects.items.p4.title': 'Project Name',
    'projects.items.p4.desc': 'Project summary placeholder. Replace with your own content.',
    'news.title': 'News / Timeline',
    'news.n1': 'Website framework released.',
    'news.n2': 'Add your project milestone here.',
    'news.n3': 'Add your project milestone here.',
    'contact.title': 'Join / Contact',
    'contact.desc': 'Add your e-mail, GitHub, and social accounts here.',
    'footer.rights': 'All rights reserved.'
  }
};

function applyLanguage(lang) {
  const dict = translations[lang] || translations.en;
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (dict[key]) el.textContent = dict[key];
  });
  localStorage.setItem('lang', lang);
}

window.__i18n = {
  applyLanguage,
  getCurrent: () => localStorage.getItem('lang') || 'zh'
};