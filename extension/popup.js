// popup.js
const askBtn = document.getElementById('askBtn');
const startHelpBtn = document.getElementById('startHelpBtn');
const serverState = document.getElementById('serverState');
const answerBox = document.getElementById('answer');
const questionBox = document.getElementById('question');

const SERVER_URL = 'http://localhost:5000/query';

async function checkServer() {
  try {
    const res = await fetch(SERVER_URL, { method: 'OPTIONS' }).catch(() => null);
    if (res && (res.status === 200 || res.status === 204 || res.status === 405)) {
      serverState.textContent = 'running';
      return true;
    }
  } catch (e) {}
  serverState.textContent = 'not running';
  return false;
}

// Get the current active tab's URL
async function getActiveTabUrl() {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].url) {
        resolve(tabs[0].url);
      } else {
        resolve('');
      }
    });
  });
}

async function askQuestion() {
  answerBox.value = '';
  serverState.textContent = 'checking...';
  const serverReady = await checkServer();
  if (!serverReady) {
    serverState.textContent = 'not running';
    answerBox.value = "Local server not running. Click 'Start server?' for instructions.";
    return;
  }

  const url = await getActiveTabUrl();
  const question = questionBox.value.trim();

  if (!url) {
    answerBox.value = 'Could not detect page URL.';
    return;
  }
  if (!question) {
    answerBox.value = 'Please type a question.';
    return;
  }

  answerBox.value = 'Fetching and analyzing page content...';

  try {
    const resp = await fetch(SERVER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, question })
    });

    if (!resp.ok) {
      const txt = await resp.text().catch(() => null);
      answerBox.value = `Server error: ${resp.status} ${resp.statusText}\n${txt || ''}`;
      return;
    }

    const j = await resp.json();
    answerBox.value = j.answer || 'No answer returned.';
  } catch (err) {
    answerBox.value = 'Error contacting server: ' + err.message;
  }
}

function showStartInstructions() {
  const instructions = `Start the backend locally before using the extension:

1) Open terminal and change directory to the backend folder:
   cd <path-to>/RAG Chrome Extension/backend

2) Run the helper to download models and start the server:
   Windows: run_server.bat
   Linux/Mac: ./run_server.sh

This runs download_models.py (may take long initially) and starts server.py on port 5000.
`;
  const ok = confirm(instructions + '\n\nClick OK to copy these instructions to clipboard.');
  if (ok) {
    navigator.clipboard.writeText(instructions).then(() => alert('Instructions copied to clipboard.'));
  }
}

askBtn.addEventListener('click', askQuestion);
startHelpBtn.addEventListener('click', showStartInstructions);

// initial check
checkServer();
