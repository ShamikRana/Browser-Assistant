// Keeps a lightweight badge in sync with backend availability so the user can
// see at a glance whether the local server is up.
const HEALTH_URL = 'http://127.0.0.1:5000/health';
const ALARM = 'health-check';

async function updateBadge() {
  try {
    const response = await fetch(HEALTH_URL, { cache: 'no-store' });
    const healthy = response.ok && (await response.json()).model_loaded;
    chrome.action.setBadgeText({ text: healthy ? '' : '…' });
    chrome.action.setBadgeBackgroundColor({ color: '#f59e0b' });
  } catch {
    chrome.action.setBadgeText({ text: '!' });
    chrome.action.setBadgeBackgroundColor({ color: '#ef4444' });
  }
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create(ALARM, { periodInMinutes: 1 });
  updateBadge();
});

chrome.runtime.onStartup.addListener(updateBadge);
chrome.alarms?.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM) updateBadge();
});
