// Service worker: lets the page raise lock-screen notifications, and handles their buttons.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  const momentId = e.notification.data && e.notification.data.moment_id;
  e.waitUntil((async () => {
    if (e.action === "not-it" && momentId) {
      await fetch(`/api/moments/${momentId}/not-it`, { method: "POST", credentials: "include" });
      return;
    }
    const tabs = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    if (tabs.length) return tabs[0].focus();
    return self.clients.openWindow(e.action === "more" ? "/#moments" : "/");
  })());
});
