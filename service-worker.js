const CACHE_CLEANUP_VERSION = "2026-07-07";

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) => Promise.all(cacheNames.map((cacheName) => caches.delete(cacheName))))
      .then(() => self.registration.unregister())
      .then(() => self.clients.matchAll({ type: "window" }))
      .then((clients) => {
        clients.forEach((client) => {
          client.navigate(client.url);
        });
      }),
  );
});

self.addEventListener("fetch", (event) => {
  event.respondWith(fetch(event.request));
});
