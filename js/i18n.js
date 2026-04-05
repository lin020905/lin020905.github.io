const translations = {
  zh: {
    'loading.subtitle': '正在加载电影感界面...',
    'nav.about': '关于',
    'nav.projects': '项目',
    'nav.skills': '技能',
    'nav.gallery': '画廊',
    'nav.contact': '联系',
    'hero.title': '你的下一个代表作',
    'hero.desc': '这里替换成你的主视觉和项目宣言。',
    'hero.cta': '查看作品',
    'about.title': '关于我',
    'about.desc': '这里预留给你的个人介绍、目标与项目背景。',
    'projects.title': '项目展示',
    'projects.items.p1.title': '项目 01',
    'projects.items.p1.desc': '你的项目简介占位内容。',
    'projects.items.p2.title': '项目 02',
    'projects.items.p2.desc': '你的项目简介占位内容。',
    'projects.items.p3.title': '项目 03',
    'projects.items.p3.desc': '你的项目简介占位内容。',
    'projects.items.p4.title': '项目 04',
    'projects.items.p4.desc': '你的项目简介占位内容。',
    'projects.items.p5.title': '项目 05',
    'projects.items.p5.desc': '你的项目简介占位内容。',
    'projects.items.p6.title': '项目 06',
    'projects.items.p6.desc': '你的项目简介占位内容。',
    'skills.title': '技能与技术栈',
    'skills.desc': '把这些标签替换成你真实的技术栈与熟练度。',
    'gallery.title': '作品画廊',
    'gallery.g1': '视觉占位 01',
    'gallery.g2': '视觉占位 02',
    'gallery.g3': '视觉占位 03',
    'gallery.g4': '视觉占位 04',
    'contact.title': '联系我',
    'contact.desc': '在这里放邮箱、GitHub、社交账号。',
    'footer.rights': '保留所有权利。'
  },
  en: {
    'loading.subtitle': 'Loading cinematic interface...',
    'nav.about': 'About',
    'nav.projects': 'Projects',
    'nav.skills': 'Skills',
    'nav.gallery': 'Gallery',
    'nav.contact': 'Contact',
    'hero.title': 'Your Next Big Project',
    'hero.desc': 'Replace this section with your own main visual and project statement.',
    'hero.cta': 'Explore Works',
    'about.title': 'About Me',
    'about.desc': 'This area is reserved for your self-introduction, mission, and project background.',
    'projects.title': 'Projects',
    'projects.items.p1.title': 'Project 01',
    'projects.items.p1.desc': 'Placeholder content for your project summary.',
    'projects.items.p2.title': 'Project 02',
    'projects.items.p2.desc': 'Placeholder content for your project summary.',
    'projects.items.p3.title': 'Project 03',
    'projects.items.p3.desc': 'Placeholder content for your project summary.',
    'projects.items.p4.title': 'Project 04',
    'projects.items.p4.desc': 'Placeholder content for your project summary.',
    'projects.items.p5.title': 'Project 05',
    'projects.items.p5.desc': 'Placeholder content for your project summary.',
    'projects.items.p6.title': 'Project 06',
    'projects.items.p6.desc': 'Placeholder content for your project summary.',
    'skills.title': 'Skills & Stack',
    'skills.desc': 'Swap these tags with your real stack and proficiency details.',
    'gallery.title': 'Gallery',
    'gallery.g1': 'Visual Placeholder 01',
    'gallery.g2': 'Visual Placeholder 02',
    'gallery.g3': 'Visual Placeholder 03',
    'gallery.g4': 'Visual Placeholder 04',
    'contact.title': 'Contact',
    'contact.desc': 'Put your email, GitHub links, and social accounts here.',
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