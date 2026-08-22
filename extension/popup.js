const BASE_URL = 'http://127.0.0.1:5000';
const HEALTH_URL = `${BASE_URL}/health`;
const STREAM_URL = `${BASE_URL}/query/stream`;
const MAX_HTML_CHARS = 1_500_000;

const el = {
  question: document.getElementById('question'),
  answer: document.getElementById('answer'),
  askBtn: document.getElementById('askBtn'),
  stopBtn: document.getElementById('stopBtn'),
  copyBtn: document.getElementById('copyBtn'),
  helpBtn: document.getElementById('helpBtn'),
  statusPill: document.getElementById('statusPill'),
  statusText: document.getElementById('statusText'),
  pageTitle: document.getElementById('pageTitle'),
  meta: document.getElementById('meta'),
  device: document.getElementById('deviceBadge'),
  chips: document.getElementById('chips'),
  spinner: document.querySelector('.spinner'),
  btnLabel: document.querySelector('.btn-label'),
};

let controller = null;
let answerText = '';

/* ---------------- status ---------------- */

function setStatus(state, text) {
  el.statusPill.className = `pill pill--${state}`;
  el.statusText.textContent = text;
}

async function checkServer() {
  try {
    const response = await fetch(HEALTH_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error('bad status');
    const health = await response.json();
    if (health.model_loaded) {
      setStatus('ok', 'ready');
      el.device.textContent = health.device || '';
      return true;
    }
    setStatus('warn', 'loading model');
    return false;
  } catch {
    setStatus('err', 'offline');
    el.device.textContent = '';
    return false;
  }
}

/* ---------------- page capture ---------------- */

// Reads the *rendered* DOM, so JavaScript-heavy, logged-in and paywalled pages
// are handled correctly. Falls back to a server-side fetch when injection is
// blocked (e.g. on chrome:// pages).
async function capturePage() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return { url: '', html: null, title: '' };

  const page = { url: tab.url || '', html: null, title: tab.title || '' };
  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (limit) => ({
        html: document.documentElement.outerHTML.slice(0, limit),
        title: document.title,
      }),
      args: [MAX_HTML_CHARS],
    });
    if (injection?.result) {
      page.html = injection.result.html;
      page.title = injection.result.title || page.title;
    }
  } catch {
    // Injection not permitted on this page; the backend will fetch the URL.
  }
  return page;
}

/* ---------------- rendering ---------------- */

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}

// Minimal, safe markdown: escape first, then re-introduce a few inline styles.
function renderMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function paintAnswer(streaming) {
  el.answer.classList.remove('answer--empty', 'answer--error');
  el.answer.innerHTML = renderMarkdown(answerText) + (streaming ? '<span class="caret">&nbsp;</span>' : '');
  el.answer.scrollTop = el.answer.scrollHeight;
}

function showError(message) {
  el.answer.classList.remove('answer--empty');
  el.answer.classList.add('answer--error');
  el.answer.textContent = message;
}

function setBusy(busy) {
  el.askBtn.disabled = busy;
  el.spinner.hidden = !busy;
  el.btnLabel.textContent = busy ? 'Thinking' : 'Ask';
  el.stopBtn.hidden = !busy;
  el.chips.querySelectorAll('.chip').forEach((chip) => { chip.disabled = busy; });
}

/* ---------------- ask ---------------- */

async function ask() {
  const question = el.question.value.trim();
  if (!question) {
    el.question.focus();
    return;
  }

  if (!(await checkServer())) {
    showError('Local server is not reachable. Start the backend, then try again.');
    return;
  }

  answerText = '';
  el.meta.textContent = '';
  el.copyBtn.hidden = true;
  el.answer.classList.remove('answer--error');
  el.answer.textContent = 'Reading the page…';
  setBusy(true);

  const started = performance.now();
  controller = new AbortController();

  try {
    const page = await capturePage();
    el.pageTitle.textContent = page.title || page.url;

    const response = await fetch(STREAM_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: page.url, question, html: page.html }),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`Server responded with ${response.status}`);
    }

    await consumeStream(response.body);

    if (answerText) {
      el.copyBtn.hidden = false;
      const seconds = ((performance.now() - started) / 1000).toFixed(1);
      el.meta.textContent = `Answered in ${seconds}s`;
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      el.meta.textContent = 'Stopped.';
    } else {
      showError(`Error: ${error.message}`);
    }
  } finally {
    setBusy(false);
    controller = null;
    paintAnswerIfNeeded();
  }
}

function paintAnswerIfNeeded() {
  if (answerText) paintAnswer(false);
}

async function consumeStream(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let first = true;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';

    for (const frame of frames) {
      const event = /^event:\s*(.+)$/m.exec(frame)?.[1]?.trim();
      const rawData = /^data:\s*(.*)$/m.exec(frame)?.[1];
      if (!event || rawData === undefined) continue;

      let payload;
      try {
        payload = JSON.parse(rawData);
      } catch {
        continue;
      }

      if (event === 'token') {
        if (first) {
          answerText = '';
          first = false;
        }
        answerText += payload;
        paintAnswer(true);
      } else if (event === 'meta' && payload.device) {
        el.device.textContent = payload.device;
      } else if (event === 'error') {
        throw new Error(payload.message || 'Unknown server error');
      }
    }
  }
}

/* ---------------- wiring ---------------- */

el.askBtn.addEventListener('click', ask);

el.stopBtn.addEventListener('click', () => controller?.abort());

el.copyBtn.addEventListener('click', async () => {
  await navigator.clipboard.writeText(answerText);
  el.copyBtn.textContent = 'Copied';
  setTimeout(() => { el.copyBtn.textContent = 'Copy'; }, 1200);
});

el.question.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) ask();
});

el.chips.addEventListener('click', (event) => {
  const chip = event.target.closest('.chip');
  if (!chip) return;
  el.question.value = chip.dataset.prompt;
  ask();
});

el.helpBtn.addEventListener('click', () => {
  showError(
    'Start the backend:\n\n' +
    '  cd Browser-Assistant-Phi4/backend\n' +
    '  run_server.bat          (Windows)\n' +
    '  ./run_server.sh         (Linux/Mac)\n\n' +
    'To start it automatically at login, run register_startup.ps1 once.'
  );
});

(async function init() {
  await checkServer();
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  el.pageTitle.textContent = tab?.title || tab?.url || '';
  el.question.focus();
})();
