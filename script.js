const translations = {
  zh: {
    skipLink: "跳转到主要内容",
    navAbout: "简介",
    navProjects: "项目",
    navNotes: "随笔",
    navContact: "联系",
    musicLabel: "never see me again \u2014 Kanye West",
    musicPlay: "\u64ad\u653e\u97f3\u4e50",
    musicPause: "\u6682\u505c\u97f3\u4e50",
    musicSeek: "\u64ad\u653e\u8fdb\u5ea6",
    musicVolume: "\u97f3\u91cf",
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
    footerText: "© 2026 dreamworld. Built with Jekyll.",
    backTop: "返回顶部",
  },
  en: {
    skipLink: "Skip to main content",
    navAbout: "About",
    navProjects: "Projects",
    navNotes: "Notes",
    navContact: "Contact",
    musicLabel: "never see me again \u2014 Kanye West",
    musicPlay: "Play music",
    musicPause: "Pause music",
    musicSeek: "Seek music",
    musicVolume: "Music volume",
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
    footerText: "© 2026 dreamworld. Built with Jekyll.",
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
const mainNode = document.querySelector("#main");
const languageButtons = document.querySelectorAll("[data-language]");
let translatableNodes = document.querySelectorAll("[data-i18n]");
const quoteNode = document.querySelector("[data-quote-rotator]");
const quoteFrame = document.querySelector("[data-quote-frame]");
let sectionNodes = document.querySelectorAll("[data-section]");
const sectionLinks = document.querySelectorAll("[data-section-link]");
const musicPlayer = document.querySelector("[data-music-player]");
const musicToggle = document.querySelector("[data-music-toggle]");
const musicSeek = document.querySelector("[data-music-seek]");
const musicVolume = document.querySelector("[data-music-volume]");
const musicCurrent = document.querySelector("[data-music-current]");
const musicDurationNode = document.querySelector("[data-music-duration]");
const musicAudio = document.querySelector("[data-music-audio]");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let currentLanguage = "zh";
let currentQuoteIndex = 0;
let quoteAnimationTimer;
let buildRefreshAttempted = false;
let legacyCacheReloadAttempted = false;
let sectionObserver;
let revealObserver;
let musicContext;
let musicFilter;
let musicMaster;
let musicAnimationFrame;
let isMusicPlaying = false;
let musicStartedAt = 0;
let musicPausedAt = 0;
let lastMusicBeat = -1;
let isMusicAudioActive = false;
let isMusicAudioUnavailable = false;

const musicDuration = 48;
const musicBeatLength = 0.75;
const musicBeatCount = Math.round(musicDuration / musicBeatLength);
const musicChords = [
  { root: 196.0, tones: [196.0, 246.94, 293.66, 392.0] },
  { root: 174.61, tones: [174.61, 220.0, 261.63, 349.23] },
  { root: 130.81, tones: [130.81, 196.0, 261.63, 329.63] },
  { root: 146.83, tones: [146.83, 220.0, 293.66, 369.99] },
];
const musicMelody = [
  392.0,
  440.0,
  null,
  392.0,
  493.88,
  440.0,
  null,
  329.63,
  349.23,
  392.0,
  null,
  293.66,
  329.63,
  392.0,
  440.0,
  null,
];

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

function refreshPageNodes() {
  translatableNodes = document.querySelectorAll("[data-i18n]");
  sectionNodes = document.querySelectorAll("[data-section]");
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
  refreshPageNodes();

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
  updateMusicCopy();
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

function getMusicDictionary() {
  return translations[currentLanguage] || translations.zh;
}

function updateMusicCopy() {
  if (!musicToggle) {
    return;
  }

  const dictionary = getMusicDictionary();
  const actionLabel = isMusicPlaying ? dictionary.musicPause : dictionary.musicPlay;

  musicToggle.setAttribute("aria-label", actionLabel);
  musicToggle.setAttribute("title", actionLabel);

  if (musicSeek) {
    musicSeek.setAttribute("aria-label", dictionary.musicSeek);
  }

  if (musicVolume) {
    musicVolume.setAttribute("aria-label", dictionary.musicVolume);

    if (musicVolume.parentElement) {
      musicVolume.parentElement.setAttribute("title", dictionary.musicVolume);
    }
  }
}

function formatMusicTime(seconds) {
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainder = String(totalSeconds % 60).padStart(2, "0");

  return `${minutes}:${remainder}`;
}

function updateRangeProgress(range, ratio) {
  if (!range) {
    return;
  }

  const progress = `${Math.max(0, Math.min(1, ratio)) * 100}%`;
  range.style.setProperty("--progress", progress);
}

function getMusicLength() {
  if (
    musicAudio &&
    !isMusicAudioUnavailable &&
    Number.isFinite(musicAudio.duration) &&
    musicAudio.duration > 0
  ) {
    return musicAudio.duration;
  }

  return musicDuration;
}

function getMusicPosition() {
  if (isMusicAudioActive && musicAudio) {
    return musicAudio.currentTime || 0;
  }

  if (!isMusicPlaying) {
    return musicPausedAt;
  }

  const elapsed = performance.now() / 1000 - musicStartedAt;
  return ((elapsed % musicDuration) + musicDuration) % musicDuration;
}

function updateMusicProgress() {
  const position = getMusicPosition();
  const duration = getMusicLength();

  if (musicSeek) {
    musicSeek.max = String(duration);
    musicSeek.value = String(position);
    updateRangeProgress(musicSeek, position / duration);
  }

  if (musicCurrent) {
    musicCurrent.textContent = formatMusicTime(position);
  }

  if (musicDurationNode) {
    musicDurationNode.textContent = formatMusicTime(duration);
  }
}

function setMusicPosition(position) {
  const numericPosition = Number(position);
  const duration = getMusicLength();
  const normalizedPosition =
    ((Number.isFinite(numericPosition) ? numericPosition : 0) % duration + duration) % duration;

  if (isMusicAudioActive && musicAudio) {
    musicAudio.currentTime = normalizedPosition;
    updateMusicProgress();
    return;
  }

  musicPausedAt = normalizedPosition;

  if (musicAudio && Number.isFinite(musicAudio.duration)) {
    musicAudio.currentTime = Math.min(normalizedPosition, Math.max(0, musicAudio.duration - 0.05));
  }

  if (isMusicPlaying) {
    musicStartedAt = performance.now() / 1000 - normalizedPosition;
    lastMusicBeat = Math.floor(normalizedPosition / musicBeatLength) - 1;
  }

  updateMusicProgress();
}

function createMusicContext() {
  if (musicContext) {
    return true;
  }

  const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;

  if (!AudioContextConstructor) {
    return false;
  }

  musicContext = new AudioContextConstructor();
  musicFilter = musicContext.createBiquadFilter();
  musicMaster = musicContext.createGain();

  const delay = musicContext.createDelay(1.2);
  const feedback = musicContext.createGain();

  musicFilter.type = "lowpass";
  musicFilter.frequency.value = 1850;
  musicFilter.Q.value = 0.7;
  delay.delayTime.value = 0.32;
  feedback.gain.value = 0.18;
  musicMaster.gain.value = 0;

  musicFilter.connect(musicMaster);
  musicFilter.connect(delay);
  delay.connect(feedback);
  feedback.connect(delay);
  delay.connect(musicMaster);
  musicMaster.connect(musicContext.destination);

  return true;
}

function playMusicTone(frequency, startTime, duration, gainValue, type = "sine") {
  if (!musicContext || !musicFilter) {
    return;
  }

  const oscillator = musicContext.createOscillator();
  const gain = musicContext.createGain();
  const releaseTime = startTime + duration;

  oscillator.type = type;
  oscillator.frequency.setValueAtTime(frequency, startTime);
  gain.gain.setValueAtTime(0.0001, startTime);
  gain.gain.exponentialRampToValueAtTime(gainValue, startTime + 0.045);
  gain.gain.exponentialRampToValueAtTime(0.0001, releaseTime);

  oscillator.connect(gain);
  gain.connect(musicFilter);
  oscillator.start(startTime);
  oscillator.stop(releaseTime + 0.04);
}

function playMusicBeat(beat) {
  if (!musicContext) {
    return;
  }

  const safeBeat = ((beat % musicBeatCount) + musicBeatCount) % musicBeatCount;
  const chord = musicChords[Math.floor(safeBeat / 16) % musicChords.length];
  const melodyNote = musicMelody[safeBeat % musicMelody.length];
  const now = musicContext.currentTime + 0.025;

  if (safeBeat % 8 === 0) {
    chord.tones.forEach((tone, index) => {
      playMusicTone(tone, now + index * 0.014, 2.8, 0.033, "sine");
    });
  }

  if (safeBeat % 4 === 0) {
    playMusicTone(chord.root / 2, now, 1.9, 0.045, "triangle");
  }

  if (melodyNote) {
    playMusicTone(melodyNote, now, 0.52, 0.036, "sine");
  }

  if (safeBeat % 2 === 1) {
    const echoTone = melodyNote ? melodyNote * 2 : chord.tones[2] * 2;
    playMusicTone(echoTone, now + 0.18, 0.16, 0.012, "triangle");
  }
}

function runMusicLoop() {
  if (isMusicAudioActive) {
    updateMusicProgress();
    musicAnimationFrame = window.requestAnimationFrame(runMusicLoop);
    return;
  }

  const position = getMusicPosition();
  const currentBeat = Math.floor(position / musicBeatLength);

  if (currentBeat < lastMusicBeat) {
    lastMusicBeat = -1;
  }

  if (currentBeat - lastMusicBeat > 2) {
    lastMusicBeat = currentBeat - 1;
  }

  for (let beat = lastMusicBeat + 1; beat <= currentBeat; beat += 1) {
    playMusicBeat(beat);
  }

  lastMusicBeat = currentBeat;
  updateMusicProgress();
  musicAnimationFrame = window.requestAnimationFrame(runMusicLoop);
}

function setMusicUiPlaying(playing) {
  isMusicPlaying = playing;

  if (musicPlayer) {
    musicPlayer.classList.toggle("is-playing", playing);
  }

  if (musicToggle) {
    musicToggle.setAttribute("aria-pressed", String(playing));
  }

  updateMusicCopy();
}

async function playMusicAudio() {
  if (!musicAudio || isMusicAudioUnavailable) {
    return false;
  }

  if (Number.isFinite(musicAudio.duration) && musicPausedAt > 0) {
    musicAudio.currentTime = Math.min(musicPausedAt, Math.max(0, musicAudio.duration - 0.05));
  }

  if (musicVolume) {
    musicAudio.volume = Number(musicVolume.value);
  }

  try {
    await musicAudio.play();
    isMusicAudioActive = true;
    musicPausedAt = musicAudio.currentTime || 0;
    return true;
  } catch {
    isMusicAudioActive = false;
    isMusicAudioUnavailable = true;
    return false;
  }
}

async function playMusic() {
  if (!musicPlayer || !musicToggle) {
    if (musicToggle) {
      musicToggle.disabled = true;
    }

    return;
  }

  const canUseGeneratedAudio = createMusicContext();

  if (canUseGeneratedAudio) {
    try {
      await musicContext.resume();
    } catch {
      // The file-backed audio path can still work if Web Audio is blocked.
    }
  }

  if (await playMusicAudio()) {
    setMusicUiPlaying(true);
    window.cancelAnimationFrame(musicAnimationFrame);
    runMusicLoop();
    return;
  }

  if (!canUseGeneratedAudio) {
    musicToggle.disabled = true;
    return;
  }

  isMusicAudioActive = false;
  musicStartedAt = performance.now() / 1000 - musicPausedAt;
  lastMusicBeat = Math.floor(musicPausedAt / musicBeatLength) - 1;

  if (musicMaster && musicVolume) {
    musicMaster.gain.cancelScheduledValues(musicContext.currentTime);
    musicMaster.gain.setTargetAtTime(Number(musicVolume.value), musicContext.currentTime, 0.04);
  }

  setMusicUiPlaying(true);
  window.cancelAnimationFrame(musicAnimationFrame);
  runMusicLoop();
}

function pauseMusic() {
  musicPausedAt = getMusicPosition();
  if (isMusicAudioActive && musicAudio) {
    musicAudio.pause();
  }

  isMusicAudioActive = false;
  window.cancelAnimationFrame(musicAnimationFrame);

  if (musicMaster && musicContext) {
    musicMaster.gain.cancelScheduledValues(musicContext.currentTime);
    musicMaster.gain.setTargetAtTime(0.0001, musicContext.currentTime, 0.04);
  }

  setMusicUiPlaying(false);
  updateMusicProgress();
}

function setupMusicPlayer() {
  if (!musicPlayer || !musicToggle || !musicSeek || !musicVolume) {
    return;
  }

  musicSeek.max = String(musicDuration);
  if (musicDurationNode) {
    musicDurationNode.textContent = formatMusicTime(musicDuration);
  }
  updateRangeProgress(musicSeek, 0);
  updateRangeProgress(musicVolume, Number(musicVolume.value));
  updateMusicCopy();
  updateMusicProgress();

  musicToggle.addEventListener("click", () => {
    if (isMusicPlaying) {
      pauseMusic();
      return;
    }

    playMusic();
  });

  musicSeek.addEventListener("input", () => {
    setMusicPosition(musicSeek.value);
  });

  musicVolume.addEventListener("input", () => {
    const volume = Number(musicVolume.value);
    updateRangeProgress(musicVolume, volume);

    if (musicAudio) {
      musicAudio.volume = volume;
    }

    if (musicMaster && musicContext && isMusicPlaying) {
      musicMaster.gain.setTargetAtTime(volume, musicContext.currentTime, 0.04);
    }
  });

  if (musicAudio) {
    musicAudio.volume = Number(musicVolume.value);
    musicAudio.addEventListener("loadedmetadata", updateMusicProgress);
    musicAudio.addEventListener("error", () => {
      isMusicAudioActive = false;
      isMusicAudioUnavailable = true;
      updateMusicProgress();
    });
  }
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
  if (sectionObserver) {
    sectionObserver.disconnect();
  }

  document.documentElement.classList.remove("section-switch-ready");

  if (!sectionNodes.length) {
    sectionLinks.forEach((link) => link.removeAttribute("aria-current"));
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

  sectionObserver = new IntersectionObserver(
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
  if (revealObserver) {
    revealObserver.disconnect();
  }

  const revealNodes = document.querySelectorAll("[data-reveal]");

  if (!revealNodes.length) {
    return;
  }

  document.documentElement.classList.add("reveal-ready");

  if (!("IntersectionObserver" in window) || prefersReducedMotion.matches) {
    revealNodes.forEach((node) => node.classList.add("is-visible"));
    return;
  }

  revealObserver = new IntersectionObserver(
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

function shouldHandlePageLink(link) {
  if (!link || !mainNode || link.target || link.hasAttribute("download")) {
    return false;
  }

  const nextUrl = new URL(link.href, window.location.href);
  const currentUrl = new URL(window.location.href);

  if (nextUrl.origin !== currentUrl.origin) {
    return false;
  }

  if (nextUrl.pathname === currentUrl.pathname && nextUrl.search === currentUrl.search && nextUrl.hash) {
    return false;
  }

  if (/\.(?:avif|gif|jpe?g|mp3|pdf|png|svg|webp|zip)$/i.test(nextUrl.pathname)) {
    return false;
  }

  return true;
}

function scrollAfterPageSwap(url, shouldScroll) {
  if (!shouldScroll) {
    return;
  }

  if (url.hash) {
    const target = document.querySelector(url.hash);

    if (target) {
      target.scrollIntoView({ behavior: prefersReducedMotion.matches ? "auto" : "smooth" });
      return;
    }
  }

  window.scrollTo({
    top: 0,
    behavior: prefersReducedMotion.matches ? "auto" : "smooth",
  });
}

function applyFetchedPage(html, url, { replace = false, scroll = true } = {}) {
  const nextDocument = new DOMParser().parseFromString(html, "text/html");
  const nextMain = nextDocument.querySelector("#main");

  if (!nextMain || !mainNode) {
    window.location.assign(url.href);
    return;
  }

  mainNode.innerHTML = nextMain.innerHTML;

  const nextTitle = nextDocument.querySelector("title");

  if (nextTitle) {
    document.title = nextTitle.textContent;
  }

  if (replace) {
    window.history.replaceState(null, "", url.href);
  } else {
    window.history.pushState(null, "", url.href);
  }

  refreshPageNodes();
  setLanguage(currentLanguage);
  setupSectionSwitching();
  setupReveal();
  scrollAfterPageSwap(url, scroll);
}

function navigateWithinSite(url, options = {}) {
  if (!("fetch" in window) || !("DOMParser" in window) || !mainNode) {
    window.location.assign(url.href);
    return;
  }

  mainNode.setAttribute("aria-busy", "true");

  fetch(url.href, {
    cache: "no-store",
    credentials: "same-origin",
    headers: {
      "X-Requested-With": "fetch",
    },
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Navigation failed with ${response.status}`);
      }

      return response.text();
    })
    .then((html) => {
      applyFetchedPage(html, url, options);
    })
    .catch(() => {
      window.location.assign(url.href);
    })
    .finally(() => {
      mainNode.removeAttribute("aria-busy");
    });
}

function setupPersistentNavigation() {
  if (!mainNode) {
    return;
  }

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const link = target ? target.closest("a[href]") : null;

    if (!shouldHandlePageLink(link)) {
      return;
    }

    event.preventDefault();
    navigateWithinSite(new URL(link.href, window.location.href));
  });

  window.addEventListener("popstate", () => {
    navigateWithinSite(new URL(window.location.href), {
      replace: true,
      scroll: false,
    });
  });
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
setupMusicPlayer();
setupSectionSwitching();
setupReveal();
setupPersistentNavigation();
