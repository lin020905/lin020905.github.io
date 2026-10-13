(function () {
  'use strict';

  var storageKey = 'lin-language';
  var storedLanguage = null;

  try {
    storedLanguage = window.localStorage.getItem(storageKey);
  } catch (error) {
    storedLanguage = null;
  }

  var initialLanguage = storedLanguage === 'en' ? 'en' : 'zh';

  function setDocumentLanguage(language) {
    document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
    document.documentElement.setAttribute('data-language', language);
  }

  function applyLanguage(language, persist) {
    language = language === 'en' ? 'en' : 'zh';
    setDocumentLanguage(language);

    document.querySelectorAll('[data-language-switch]').forEach(function (button) {
      var active = button.getAttribute('data-language-switch') === language;
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });

    if (document.body) {
      var title = language === 'en' ? document.body.getAttribute('data-title-en') : document.body.getAttribute('data-title-zh');
      if (title) document.title = title;
    }

    if (persist) {
      try {
        window.localStorage.setItem(storageKey, language);
      } catch (error) {
        // The language still switches when storage is unavailable.
      }
    }
  }

  setDocumentLanguage(initialLanguage);

  window.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-language-switch]').forEach(function (button) {
      button.addEventListener('click', function () {
        applyLanguage(button.getAttribute('data-language-switch'), true);
      });
    });

    applyLanguage(initialLanguage, false);
  });
})();
