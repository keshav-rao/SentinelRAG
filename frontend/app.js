/**
 * SentinelRAG — Frontend Application Logic
 * Pure Vanilla JavaScript communicating with SentinelRAG FastAPI backend.
 * Features:
 *  1. Live SSE Token Streaming (1-sec TTFT word-by-word streaming)
 *  2. Interactive Citation Deep Peek Modal
 *  3. Drag & Drop PDF/DOCX/TXT Uploader with instant Qdrant vectorization
 *  4. Conversational Multi-Turn Memory with Contextual Follow-ups
 */

const API_BASE_URL = window.location.origin.includes('http') && !window.location.origin.includes('file://')
  ? window.location.origin
  : 'http://localhost:8000';

// Global Conversational State
let threadId = 'thread_' + Math.random().toString(36).substring(2, 9);
let chatHistory = []; // [{role: 'user', content: '...'}, {role: 'assistant', content: '...'}]
let currentCitations = []; // Cached citations for Deep Peek in current turn

// DOM Elements: Header & Status
const serverStatusDot = document.getElementById('server-status-dot');
const serverStatusText = document.getElementById('server-status-text');
const btnReindex = document.getElementById('btn-reindex');
const btnOpenUpload = document.getElementById('btn-open-upload');
const btnNewThread = document.getElementById('btn-new-thread');

// DOM Elements: Form & Query
const queryForm = document.getElementById('rag-query-form');
const queryInput = document.getElementById('query-input');
const btnSubmit = document.getElementById('btn-submit-query');
const threadBanner = document.getElementById('thread-banner');
const threadTurnCounter = document.getElementById('thread-turn-counter');
const btnClearThread = document.getElementById('btn-clear-thread');

// DOM Elements: Pipeline Progress
const progressSection = document.getElementById('pipeline-progress');
const progressTitle = document.getElementById('progress-step-title');
const streamingTtftTag = document.getElementById('streaming-ttft-tag');
const stepGuardrail = document.getElementById('step-guardrail');
const stepRetrieve = document.getElementById('step-retrieve');
const stepRerank = document.getElementById('step-rerank');
const stepGenerate = document.getElementById('step-generate');
const stepCritic = document.getElementById('step-critic');

// DOM Elements: Response Showcase
const responseContainer = document.getElementById('response-container');
const bannerBadge = document.getElementById('banner-status-badge');
const bannerLatency = document.getElementById('banner-latency-meta');
const bannerAuditScore = document.getElementById('banner-audit-score');
const answerContent = document.getElementById('answer-content');
const citationsList = document.getElementById('citations-list');
const btnCopyAnswer = document.getElementById('btn-copy-answer');
const liveStreamBadge = document.getElementById('live-stream-badge');

// DOM Elements: Critic Card
const criticCard = document.getElementById('critic-card');
const criticDecisionBadge = document.getElementById('critic-decision-badge');
const criticMeter = document.getElementById('critic-meter');
const criticScoreNumber = document.getElementById('critic-score-number');
const checkIconGrounded = document.getElementById('check-icon-grounded');
const checkValGrounded = document.getElementById('check-val-grounded');
const checkIconHallucination = document.getElementById('check-icon-hallucination');
const checkValHallucination = document.getElementById('check-val-hallucination');
const checkValRetries = document.getElementById('check-val-retries');
const criticReasoningText = document.getElementById('critic-reasoning-text');

// DOM Elements: Dialogue & History
const dialogueList = document.getElementById('dialogue-list');
const emptyDialogueMsg = document.getElementById('empty-dialogue-msg');
const btnClearDialogue = document.getElementById('btn-clear-dialogue');
const historyTurnPill = document.getElementById('history-turn-pill');
const historyTags = document.getElementById('history-tags');
const btnClearHistory = document.getElementById('btn-clear-history');

// DOM Elements: Upload Modal
const uploadModal = document.getElementById('upload-modal');
const btnCloseUpload = document.getElementById('btn-close-upload');
const btnDoneUpload = document.getElementById('btn-done-upload');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const uploadProgressCard = document.getElementById('upload-progress-card');
const uploadFileIcon = document.getElementById('upload-file-icon');
const uploadFileName = document.getElementById('upload-file-name');
const uploadStatusText = document.getElementById('upload-status-text');
const uploadBarFill = document.getElementById('upload-bar-fill');
const uploadResultBanner = document.getElementById('upload-result-banner');

// DOM Elements: Citation Deep Peek Modal
const citationModal = document.getElementById('citation-modal');
const btnClosePeek = document.getElementById('btn-close-peek');
const btnDismissPeek = document.getElementById('btn-dismiss-peek');
const btnCopyPeek = document.getElementById('btn-copy-peek');
const peekDocName = document.getElementById('peek-doc-name');
const peekScoreBadge = document.getElementById('peek-score-badge');
const peekChunkId = document.getElementById('peek-chunk-id');
const peekSectionTitle = document.getElementById('peek-section-title');
const peekPassageContent = document.getElementById('peek-passage-content');
const peekRerankReason = document.getElementById('peek-rerank-reason');

const HISTORY_STORAGE_KEY = 'sentinelrag_query_history';

// -------------------------------------------------------------
// Initialization & Input Handlers
// -------------------------------------------------------------

// Auto-expand textarea
queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
});

// Submit on Enter (without Shift)
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (queryInput.value.trim()) {
      queryForm.dispatchEvent(new Event('submit'));
    }
  }
});

// Check Server Health on Load
async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      serverStatusDot.style.background = '#10b981';
      serverStatusDot.classList.add('pulse');
      serverStatusText.textContent = `Online (${data.ollama_model})`;
    } else {
      throw new Error('Server returned non-200');
    }
  } catch (err) {
    serverStatusDot.style.background = '#f59e0b';
    serverStatusDot.classList.remove('pulse');
    serverStatusText.textContent = 'Engine Offline (Localhost:8000)';
  }
}

// -------------------------------------------------------------
// Pipeline Step Highlights
// -------------------------------------------------------------

const pipelineStepsMap = {
  'guardrail': stepGuardrail,
  'retrieve': stepRetrieve,
  'rerank': stepRerank,
  'generate': stepGenerate,
  'critic': stepCritic
};

function activateStep(stepName, message) {
  const steps = [stepGuardrail, stepRetrieve, stepRerank, stepGenerate, stepCritic];
  const targetStep = pipelineStepsMap[stepName];
  if (!targetStep) return;

  const targetIdx = steps.indexOf(targetStep);
  steps.forEach((step, idx) => {
    if (idx < targetIdx) {
      step.classList.remove('active');
      step.classList.add('completed');
    } else if (idx === targetIdx) {
      step.classList.remove('completed');
      step.classList.add('active');
    } else {
      step.classList.remove('active', 'completed');
    }
  });

  if (message) {
    progressTitle.textContent = message;
  }
}

function resetPipelineSteps() {
  const steps = [stepGuardrail, stepRetrieve, stepRerank, stepGenerate, stepCritic];
  steps.forEach(s => s.classList.remove('active', 'completed'));
  stepGuardrail.classList.add('active');
  progressTitle.textContent = 'Validating prompt injection & safety gate...';
  streamingTtftTag.classList.add('hidden');
}

// -------------------------------------------------------------
// Feature 1: Server-Sent Events (SSE) Live Token Streaming
// -------------------------------------------------------------

async function submitQueryStreaming(queryText) {
  const query = queryText.trim();
  if (!query) return;

  // UI state: loading
  btnSubmit.disabled = true;
  responseContainer.classList.remove('hidden');
  progressSection.classList.remove('hidden');
  resetPipelineSteps();

  // Reset answer card
  answerContent.innerHTML = '<span class="cursor-blink"></span>';
  liveStreamBadge.classList.remove('hidden');
  liveStreamBadge.textContent = 'LIVE';
  liveStreamBadge.style.display = 'inline-flex';
  citationsList.innerHTML = '<span style="color: var(--text-muted); font-size: 13px;">Retrieving sources...</span>';

  // Reset critic display
  criticMeter.style.borderColor = '#3b82f6';
  criticMeter.style.boxShadow = 'none';
  criticDecisionBadge.className = 'decision-badge';
  criticDecisionBadge.textContent = 'EVALUATING';
  criticScoreNumber.textContent = '--';
  checkValGrounded.textContent = 'Pending';
  checkIconGrounded.textContent = '⏳';
  checkValHallucination.textContent = 'Pending';
  checkIconHallucination.textContent = '⏳';
  checkValRetries.textContent = '1 Cycle';
  criticReasoningText.textContent = 'Awaiting generation stream...';

  bannerBadge.className = 'banner-badge';
  bannerBadge.textContent = 'STREAMING ANSWER';
  bannerLatency.textContent = 'Connecting...';
  bannerAuditScore.textContent = 'Confidence: --';

  const startTime = performance.now();
  let firstTokenReceived = false;
  let fullAnswerText = '';
  currentCitations = [];

  try {
    const response = await fetch(`${API_BASE_URL}/rag/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: query,
        thread_id: threadId,
        chat_history: chatHistory
      })
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server stream failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep partial line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;

        const rawData = trimmed.replace(/^data:\s*/, '');
        if (!rawData) continue;

        try {
          const event = JSON.parse(rawData);
          handleStreamEvent(event, startTime, () => {
            if (!firstTokenReceived) {
              firstTokenReceived = true;
              const ttftMs = Math.round(performance.now() - startTime);
              streamingTtftTag.textContent = `TTFT: ${ttftMs}ms`;
              streamingTtftTag.classList.remove('hidden');
            }
          }, (token) => {
            fullAnswerText += token;
            renderLiveToken(fullAnswerText);
          });
        } catch (e) {
          console.warn('Error parsing SSE event chunk:', e, rawData);
        }
      }
    }

    const elapsedSeconds = ((performance.now() - startTime) / 1000).toFixed(1);
    finalizeStreamingResponse(query, fullAnswerText, elapsedSeconds);

  } catch (err) {
    console.error('Streaming failure:', err);
    renderErrorResponse(query, err.message);
  } finally {
    progressSection.classList.add('hidden');
    btnSubmit.disabled = false;
  }
}

function handleStreamEvent(event, startTime, onFirstToken, onToken) {
  if (event.type === 'status') {
    activateStep(event.step, event.message);
  } else if (event.type === 'sources') {
    currentCitations = event.data || [];
    renderCitations(currentCitations);
  } else if (event.type === 'token') {
    onFirstToken();
    onToken(event.token);
  } else if (event.type === 'critic') {
    renderCriticResults(event.data);
  } else if (event.type === 'done') {
    bannerLatency.textContent = `Stream Complete · Status: ${event.status}`;
  } else if (event.type === 'error') {
    throw new Error(event.message || 'Stream processing error');
  }
}

function renderLiveToken(text) {
  // Format markdown bold & paragraphs
  let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\n\n/g, '<p></p>');
  answerContent.innerHTML = formatted + '<span class="cursor-blink"></span>';
}

function finalizeStreamingResponse(query, answerText, elapsedSeconds) {
  // Remove cursor
  const blinker = answerContent.querySelector('.cursor-blink');
  if (blinker) blinker.remove();

  liveStreamBadge.classList.add('hidden');
  liveStreamBadge.style.display = 'none';
  bannerLatency.textContent = `Completed in ${elapsedSeconds}s · 1 Attempt`;

  // Update multi-turn history
  if (answerText) {
    chatHistory.push({ role: 'user', content: query });
    chatHistory.push({ role: 'assistant', content: answerText });
    updateDialogueUI();
  }

  saveHistory(query);
}

// -------------------------------------------------------------
// Feature 2: Interactive Citation Deep Peek
// -------------------------------------------------------------

function renderCitations(sources) {
  citationsList.innerHTML = '';
  if (!sources || sources.length === 0) {
    citationsList.innerHTML = '<span style="color: var(--text-muted); font-size: 13px;">No direct citations retrieved.</span>';
    return;
  }

  sources.forEach((src, idx) => {
    const tag = document.createElement('div');
    tag.className = 'citation-tag';
    const scorePct = Math.round((src.score || 0) * 100);
    tag.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span>${src.document}${src.section ? ` · ${src.section}` : ''}</span>
      <span class="score-pill">${scorePct}%</span>
    `;

    // Clicking ANY citation triggers Deep Peek Modal!
    tag.addEventListener('click', () => openCitationPeekModal(src, idx));
    citationsList.appendChild(tag);
  });
}

function openCitationPeekModal(source, index) {
  peekDocName.textContent = source.document || 'Verified Document';
  peekScoreBadge.textContent = `Relevance Score: ${source.score ?? 'N/A'}`;
  peekChunkId.textContent = source.chunk_id ? `Chunk #${source.chunk_id}` : `Citation #${index + 1}`;
  peekSectionTitle.textContent = source.section || 'General Policy Provision';
  peekPassageContent.textContent = source.text || 'No raw passage text stored for this chunk.';
  peekRerankReason.textContent = source.reason
    ? `Rerank Strategy: ${source.reason}`
    : 'Retrieved via Reciprocal Rank Fusion (Qdrant Vector + BM25 Lexical search).';

  citationModal.classList.remove('hidden');
}

function closeCitationPeekModal() {
  citationModal.classList.add('hidden');
}

btnClosePeek.addEventListener('click', closeCitationPeekModal);
btnDismissPeek.addEventListener('click', closeCitationPeekModal);
citationModal.addEventListener('click', (e) => {
  if (e.target === citationModal) closeCitationPeekModal();
});

btnCopyPeek.addEventListener('click', () => {
  const text = peekPassageContent.textContent;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    btnCopyPeek.textContent = 'Copied!';
    setTimeout(() => { btnCopyPeek.textContent = 'Copy Passage'; }, 2000);
  });
});

// -------------------------------------------------------------
// Feature 3: Drag & Drop PDF/DOCX Uploader with Live Indexing
// -------------------------------------------------------------

function openUploadModal() {
  uploadProgressCard.classList.add('hidden');
  uploadResultBanner.classList.add('hidden');
  uploadModal.classList.remove('hidden');
}

function closeUploadModal() {
  uploadModal.classList.add('hidden');
}

btnOpenUpload.addEventListener('click', openUploadModal);
btnCloseUpload.addEventListener('click', closeUploadModal);
btnDoneUpload.addEventListener('click', closeUploadModal);
uploadModal.addEventListener('click', (e) => {
  if (e.target === uploadModal) closeUploadModal();
});

// Dropzone Click & Drag Handlers
dropzone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files && files.length > 0) {
    handleFileUpload(files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (fileInput.files && fileInput.files.length > 0) {
    handleFileUpload(fileInput.files[0]);
  }
});

async function handleFileUpload(file) {
  const allowed = ['.pdf', '.docx', '.md', '.txt', '.markdown'];
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  
  if (!allowed.includes(ext)) {
    alert(`File type "${ext}" not supported. Please upload a PDF, DOCX, MD, or TXT file.`);
    return;
  }

  // Show progress card
  uploadProgressCard.classList.remove('hidden');
  uploadResultBanner.classList.add('hidden');
  uploadFileName.textContent = file.name;
  uploadFileIcon.textContent = ext.replace('.', '').toUpperCase();
  uploadStatusText.textContent = 'Uploading & parsing document...';
  uploadBarFill.style.width = '35%';

  const formData = new FormData();
  formData.append('file', file);

  try {
    uploadBarFill.style.width = '65%';
    uploadStatusText.textContent = 'Chunking & vectorizing into Qdrant store...';

    const res = await fetch(`${API_BASE_URL}/rag/upload`, {
      method: 'POST',
      body: formData
    });

    uploadBarFill.style.width = '100%';

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed with status ${res.status}`);
    }

    const data = await res.json();
    uploadStatusText.textContent = 'Indexing complete!';
    uploadResultBanner.className = 'upload-result-banner success';
    uploadResultBanner.innerHTML = `
      <strong>✓ Document successfully ingested!</strong><br>
      Added <b>${data.chunks_indexed} chunks</b> to Qdrant vector database (Collection now contains ${data.total_chunks_in_qdrant} chunks).
      You can query "${data.filename}" immediately!
    `;
    uploadResultBanner.classList.remove('hidden');

    // Trigger server health re-check
    checkServerHealth();

  } catch (err) {
    console.error('File upload error:', err);
    uploadStatusText.textContent = 'Failed';
    uploadBarFill.style.background = '#ef4444';
    uploadResultBanner.className = 'upload-result-banner error';
    uploadResultBanner.innerHTML = `<strong>✕ Ingestion Failed:</strong> ${err.message}`;
    uploadResultBanner.classList.remove('hidden');
  }
}

// -------------------------------------------------------------
// Feature 4: Conversational Multi-Turn Memory & Dialogue
// -------------------------------------------------------------

function resetThread() {
  threadId = 'thread_' + Math.random().toString(36).substring(2, 9);
  chatHistory = [];
  currentCitations = [];
  updateDialogueUI();
  responseContainer.classList.add('hidden');
  queryInput.value = '';
  queryInput.focus();
}

btnNewThread.addEventListener('click', resetThread);
btnClearThread.addEventListener('click', resetThread);
btnClearDialogue.addEventListener('click', resetThread);

function updateDialogueUI() {
  const turns = chatHistory.length / 2;
  historyTurnPill.textContent = `${turns} Turn${turns === 1 ? '' : 's'}`;

  if (turns > 0) {
    threadBanner.classList.remove('hidden');
    threadTurnCounter.textContent = `Turn ${turns + 1}`;
    emptyDialogueMsg.classList.add('hidden');
  } else {
    threadBanner.classList.add('hidden');
    emptyDialogueMsg.classList.remove('hidden');
    dialogueList.innerHTML = '<div class="empty-dialogue" id="empty-dialogue-msg">Ask questions above. Multi-turn follow-ups will maintain context and display here.</div>';
    return;
  }

  // Render turns
  dialogueList.innerHTML = '';
  for (let i = 0; i < chatHistory.length; i += 2) {
    const userMsg = chatHistory[i];
    const botMsg = chatHistory[i + 1];

    const turnDiv = document.createElement('div');
    turnDiv.className = 'dialogue-turn';
    const preview = botMsg.content.length > 220 ? botMsg.content.substring(0, 220) + '...' : botMsg.content;

    turnDiv.innerHTML = `
      <div class="turn-user-q">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <span>${userMsg.content}</span>
      </div>
      <div class="turn-bot-a">${preview}</div>
    `;
    dialogueList.appendChild(turnDiv);
  }
}

// -------------------------------------------------------------
// Critic Node Rendering & Helpers
// -------------------------------------------------------------

function renderCriticResults(criticData) {
  const isPass = criticData.decision === 'PASS';
  const confidencePct = Math.round((criticData.score || 0) * 100);

  bannerAuditScore.textContent = `Confidence: ${confidencePct}%`;
  criticScoreNumber.textContent = `${confidencePct}%`;

  if (isPass) {
    bannerBadge.className = 'banner-badge badge-grounded';
    bannerBadge.textContent = 'GROUNDED ANSWER';
    criticMeter.style.borderColor = '#10b981';
    criticMeter.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.3)';
    criticDecisionBadge.className = 'decision-badge decision-pass';
    criticDecisionBadge.textContent = 'PASS';
  } else {
    bannerBadge.className = 'banner-badge badge-warning';
    bannerBadge.textContent = 'DEFENDED (REFUSAL)';
    criticMeter.style.borderColor = '#f59e0b';
    criticMeter.style.boxShadow = '0 0 20px rgba(245, 158, 11, 0.3)';
    criticDecisionBadge.className = 'decision-badge decision-pass';
    criticDecisionBadge.textContent = 'DEFENDED';
  }

  checkValGrounded.textContent = criticData.grounded ? 'Verified' : 'Flagged';
  checkIconGrounded.textContent = criticData.grounded ? '✓' : '✗';
  checkValHallucination.textContent = criticData.hallucination ? 'Detected' : 'Clean';
  checkIconHallucination.textContent = criticData.hallucination ? '✗' : '✓';
  criticReasoningText.textContent = criticData.reason || 'Answer verified strictly against knowledge base context.';
}

function renderErrorResponse(query, errorMsg) {
  responseContainer.classList.remove('hidden');
  bannerBadge.className = 'banner-badge badge-danger';
  bannerBadge.textContent = 'EXECUTION ERROR';
  bannerLatency.textContent = 'Failed';
  bannerAuditScore.textContent = '0%';
  criticScoreNumber.textContent = '0%';
  criticMeter.style.borderColor = '#ef4444';
  criticDecisionBadge.className = 'decision-badge decision-fail';
  criticDecisionBadge.textContent = 'ERROR';

  answerContent.innerHTML = `
    <p style="color: #ef4444; font-weight: 600;">Unable to complete query:</p>
    <p>${errorMsg}</p>
    <p style="color: var(--text-muted); font-size: 13px; margin-top: 10px;">
      Ensure the FastAPI backend is running on port 8000 and Ollama is active.
    </p>
  `;
  citationsList.innerHTML = '';
  criticReasoningText.textContent = `Backend error: ${errorMsg}`;
}

// -------------------------------------------------------------
// Copy & Ingest Controls
// -------------------------------------------------------------

btnCopyAnswer.addEventListener('click', () => {
  const text = answerContent.innerText;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const originalText = btnCopyAnswer.innerHTML;
    btnCopyAnswer.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
      <span style="color: #10b981;">Copied!</span>
    `;
    setTimeout(() => { btnCopyAnswer.innerHTML = originalText; }, 2000);
  });
});

btnReindex.addEventListener('click', async () => {
  const originalHtml = btnReindex.innerHTML;
  btnReindex.disabled = true;
  btnReindex.innerHTML = '<span>Indexing...</span>';

  try {
    const res = await fetch(`${API_BASE_URL}/rag/ingest`, { method: 'POST' });
    const data = await res.json();
    alert(`Success: Indexed ${data.indexed_in_qdrant} chunk vectors into Qdrant!`);
  } catch (err) {
    alert(`Ingest failed: ${err.message}`);
  } finally {
    btnReindex.innerHTML = originalHtml;
    btnReindex.disabled = false;
  }
});

// Form Submit Handler
queryForm.addEventListener('submit', (e) => {
  e.preventDefault();
  submitQueryStreaming(queryInput.value);
});

// Preset Chips Handler
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const query = chip.getAttribute('data-query');
    queryInput.value = query;
    queryInput.dispatchEvent(new Event('input'));
    submitQueryStreaming(query);
  });
});

// -------------------------------------------------------------
// History Management
// -------------------------------------------------------------

function loadHistory() {
  historyTags.innerHTML = '';
  try {
    const history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
    if (history.length === 0) {
      historyTags.innerHTML = '<span style="color: var(--text-muted); font-size: 12px;">No recent searches yet.</span>';
      return;
    }

    history.slice(0, 8).forEach(q => {
      const chip = document.createElement('button');
      chip.className = 'history-chip';
      chip.textContent = q;
      chip.title = q;
      chip.addEventListener('click', () => {
        queryInput.value = q;
        queryInput.dispatchEvent(new Event('input'));
        submitQueryStreaming(q);
      });
      historyTags.appendChild(chip);
    });
  } catch (e) {
    console.error('Error reading history:', e);
  }
}

function saveHistory(query) {
  try {
    let history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
    history = history.filter(item => item.toLowerCase() !== query.toLowerCase());
    history.unshift(query);
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history.slice(0, 10)));
    loadHistory();
  } catch (e) {
    console.error('Error saving history:', e);
  }
}

btnClearHistory.addEventListener('click', () => {
  localStorage.removeItem(HISTORY_STORAGE_KEY);
  loadHistory();
});

// Initial boot
checkServerHealth();
loadHistory();
