const form = document.querySelector('#form');
const promptBox = document.querySelector('#prompt');
const messages = document.querySelector('#messages');
const chatPanel = document.querySelector('main');
const monitorPanel = document.querySelector('#system-monitor');
const monitorToggle = document.querySelector('#monitor-toggle');
const monitorContent = document.querySelector('#monitor-content');
const sendButton = document.querySelector('#send');
const clearButton = document.querySelector('#clear');
const editBanner = document.querySelector('#edit-banner');
const cancelEditButton = document.querySelector('#cancel-edit');
const status = document.querySelector('#status');
const profileName = document.querySelector('#profile');
const modelSelect = document.querySelector('#model-select');
const loadModelButton = document.querySelector('#load-model');
const maxTokensInput = document.querySelector('#maxTokens');
const unlimitedTokensToggle = document.querySelector('#unlimitedTokens');
const reasoningToggle = document.querySelector('#reasoning');
const reasoningEffort = document.querySelector('#reasoningEffort');
const history = [];
const chatLog = [];
let currentProfileId = document.body.dataset.profile || 'default';
const chatStorageKey = 'n3komata.chat.v1';
const legacyChatStorageKey = `n3komata.chat.v1:${currentProfileId}`;

function assistantHistoryMessage(content, reasoningContent = '') {
  const message = {role: 'assistant', content};
  if (typeof reasoningContent === 'string' && reasoningContent) {
    message.reasoning_content = reasoningContent;
  }
  return message;
}
let controller = null;
let editingChatIndex = null;
let editingDraft = null;
let editingMessageNode = null;
let pageLeaving = false;
let modelSwitching = false;
let modelBusy = false;
let modelProfilesLoaded = false;
let monitorRequestInFlight = false;
let inferenceRequestInFlight = false;
let previousInferenceSample = null;
let lastSlotProgressAt = 0;
const slotProgressStaleMs = 2000;
let streamRateStartedAt = null;
let streamRateChunkCount = 0;
let lastStreamChunkAt = 0;
let lastStreamChartAt = 0;
let chatScrollFrame = null;
let chatScrollTarget = null;
const chartHistory = {gfx: [], spi: [], gpuTemp: [], gpuPower: [], ram: [], gtt: [], cpu: [], disk: [], context: [], liveRate: []};
let lastContextChartAt = 0;

const formatBytes = (value) => {
  if (!Number.isFinite(value)) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit++; }
  return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
};
const setText = (id, value) => { document.getElementById(id).textContent = value; };

function setMonitorExpanded(expanded, persist = true) {
  monitorPanel.classList.toggle('collapsed', !expanded);
  monitorToggle.setAttribute('aria-expanded', String(expanded));
  monitorContent.setAttribute('aria-hidden', String(!expanded));
  monitorContent.inert = !expanded;
  if (persist) {
    try { localStorage.setItem('n3komata.monitor.collapsed', String(!expanded)); } catch (_) {}
  }
}

let monitorExpanded = true;
try { monitorExpanded = localStorage.getItem('n3komata.monitor.collapsed') !== 'true'; } catch (_) {}
setMonitorExpanded(monitorExpanded, false);
monitorToggle.addEventListener('click', () => setMonitorExpanded(monitorToggle.getAttribute('aria-expanded') !== 'true'));

const svgNamespace = 'http://www.w3.org/2000/svg';
const nvmeLinkCeilingBytesPerSecond = 1_000_000_000;

function svgNode(name, attributes = {}, textContent = null) {
  const node = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (textContent !== null) node.textContent = textContent;
  return node;
}

function formatChartValue(id, value) {
  if (['chart-gpu-gfx', 'chart-gpu-spi', 'chart-cpu', 'chart-ram', 'chart-gtt', 'chart-context'].includes(id)) {
    return `${value.toFixed(0)}%`;
  }
  if (id === 'chart-gpu-temp') return `${value.toFixed(0)}°C`;
  if (id === 'chart-gpu-power') return `${value.toFixed(0)}W`;
  if (id === 'chart-disk') return `${(100 * value / nvmeLinkCeilingBytesPerSecond).toFixed(1)}%`;
  if (id === 'chart-live-rate') return `${value.toFixed(1)} tk/s`;
  return value.toFixed(1);
}

function formatAxisValue(id, value) {
  if (id === 'chart-disk') {
    return `${(100 * value / nvmeLinkCeilingBytesPerSecond).toFixed(0)}%`;
  }
  return formatChartValue(id, value);
}

function drawLine(id, values, value, ceiling = null, floor = 0) {
  values.push(Number.isFinite(value) ? value : null);
  if (values.length > 60) values.shift();
  const chart = document.getElementById(id);
  if (!chart) return;
  const present = values.filter(Number.isFinite);
  const upper = ceiling ?? Math.max(1, ...present) * 1.05;
  const span = Math.max(Number.EPSILON, upper - floor);
  const latest = present.length ? present[present.length - 1] : null;
  const isTall = chart.classList.contains('tall-chart');
  const isPaired = chart.classList.contains('paired-chart') || chart.classList.contains('subplot-chart');
  const top = isTall ? 32 : (isPaired ? 16 : 8);
  const bottom = isTall ? 256 : (isPaired ? 126 : 64);
  const left = 56;
  const right = 218;
  const yFor = sample => bottom - Math.max(0, Math.min(1, (sample - floor) / span)) * (bottom - top);
  chart.replaceChildren();
  chart.title = Number.isFinite(latest)
    ? `Recent range ${formatChartValue(id, floor)}–${formatChartValue(id, upper)}; latest ${formatChartValue(id, latest)}`
    : 'Waiting for samples';
  chart.append(svgNode('title', {}, chart.title));
  chart.append(svgNode('rect', {x: left, y: top, width: right - left, height: bottom - top, rx: 2, class: 'chart-background'}));
  for (let tick = 0; tick <= 4; tick++) {
    const y = top + (bottom - top) * tick / 4;
    chart.append(svgNode('line', {x1: left, x2: right, y1: y, y2: y, class: tick === 0 || tick === 4 ? 'chart-grid-edge' : 'chart-grid-line'}));
  }
  chart.append(svgNode('line', {x1: left, x2: left, y1: top, y2: bottom, class: 'chart-y-axis'}));
  chart.append(svgNode('text', {x: 1, y: top + 3, class: 'chart-axis-label'}, formatAxisValue(id, upper)));
  chart.append(svgNode('text', {x: 1, y: bottom + 3, class: 'chart-axis-label'}, formatAxisValue(id, floor)));

  let path = '';
  let segmentOpen = false;
  let lastPoint = null;
  values.forEach((sample, index) => {
    if (!Number.isFinite(sample)) { segmentOpen = false; return; }
    const x = values.length < 2 ? right : left + index * (right - left) / (values.length - 1);
    const y = yFor(sample);
    path += `${segmentOpen ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)} `;
    segmentOpen = true;
    lastPoint = {x, y};
  });
  if (path) chart.append(svgNode('path', {d: path.trim(), class: 'chart-line'}));
  if (lastPoint) chart.append(svgNode('circle', {cx: lastPoint.x, cy: lastPoint.y, r: 3, class: 'chart-current'}));
}

function drawPercentLine(id, values, value) {
  drawLine(id, values, value, 100, 0);
  const chart = document.getElementById(id);
  if (chart) {
    const description = Number.isFinite(value)
      ? `Scale: 0% to 100%; latest ${value.toFixed(1)}%`
      : 'Scale: 0% to 100%';
    chart.title = description;
    chart.setAttribute('aria-description', description);
    const title = chart.querySelector('title');
    if (title) title.textContent = description;
  }
}

function recentNumericRange(values, nextValue, minimumSpan) {
  const samples = values.concat(Number.isFinite(nextValue) ? [nextValue] : []).filter(Number.isFinite);
  if (!samples.length) return {floor: 0, ceiling: minimumSpan};
  const low = Math.min(...samples);
  const high = Math.max(...samples);
  const span = Math.max(high - low, minimumSpan);
  const padding = Math.max(minimumSpan * 0.12, span * 0.12);
  const midpoint = (low + high) / 2;
  const floor = Math.max(0, Math.min(low - padding, midpoint - span / 2 - padding));
  const ceiling = Math.max(high + padding, floor + minimumSpan);
  return {floor, ceiling};
}

function drawAutoRangeLine(id, values, value, minimumSpan) {
  const scale = recentNumericRange(values, value, minimumSpan);
  drawLine(id, values, value, scale.ceiling, scale.floor);
  const chart = document.getElementById(id);
  if (chart) {
    const description = `Recent scale: ${formatChartValue(id, scale.floor)}–${formatChartValue(id, scale.ceiling)}`;
    chart.setAttribute('aria-description', description);
    const title = chart.querySelector('title');
    if (title) title.textContent = description;
  }
}

function updateLiveRateFromStream(phase) {
  const now = performance.now();
  lastStreamChunkAt = now;
  if (streamRateStartedAt === null) streamRateStartedAt = now;
  streamRateChunkCount++;
  const elapsed = (now - streamRateStartedAt) / 1000;
  if (elapsed < 0.5 || (lastSlotProgressAt > 0 && now - lastSlotProgressAt < slotProgressStaleMs)) return;
  const estimatedRate = streamRateChunkCount / elapsed;
  setText('live-rate', `≈ ${estimatedRate.toFixed(1)} tk/s`);
  if (lastSlotProgressAt === 0 || now - lastSlotProgressAt >= slotProgressStaleMs) {
    setText('live-phase', `${phase} · stream estimate`);
  }
  if (now - lastStreamChartAt >= 500) {
    drawAutoRangeLine('chart-live-rate', chartHistory.liveRate, estimatedRate, 2);
    lastStreamChartAt = now;
  }
}

async function refreshMonitor() {
  if (monitorRequestInFlight) return;
  monitorRequestInFlight = true;
  try {
    const response = await fetch('/api/monitor', {cache: 'no-store'});
    if (!response.ok) throw new Error('telemetry endpoint unavailable');
    const data = await response.json();
    const gpu = data.gpu || {};
    setText('gpu-gfx', Number.isFinite(gpu.gui_active_percent) ? `${gpu.gui_active_percent.toFixed(1)}%` : '—');
    setText('gpu-spi', Number.isFinite(gpu.spi_busy_percent) ? `${gpu.spi_busy_percent.toFixed(1)}%` : '—');
    const gpuDetail = gpu.error || `${gpu.card || 'GPU'} · ${gpu.sample_hz || '—'} samples/s`;
    setText('gpu-detail', gpuDetail);
    document.getElementById('gpu-detail').title = gpuDetail;
    setText('gpu-temp', Number.isFinite(gpu.temperature_c) ? `${gpu.temperature_c.toFixed(1)} °C` : '—');
    setText('gpu-temp-label', gpu.temperature_label || 'temperature sensor');
    setText('gpu-power', Number.isFinite(gpu.power_w) ? `${gpu.power_w.toFixed(1)} W` : '—');
    setText('gpu-power-label', gpu.power_label || 'power sensor');
    const memory = data.memory;
    setText('ram-used', memory ? `${formatBytes(memory.used_bytes)} / ${formatBytes(memory.total_bytes)}` : '—');
    setText('ram-available', memory ? formatBytes(memory.available_bytes) : '—');
    const gtt = data.gtt || {};
    setText('gtt-used', Number.isFinite(gtt.used_bytes) && Number.isFinite(gtt.total_bytes)
      ? `${formatBytes(gtt.used_bytes)} / ${formatBytes(gtt.total_bytes)}` : 'Unavailable');
    setText('cpu-load', Number.isFinite(data.cpu_percent) ? `${data.cpu_percent.toFixed(1)}%` : '—');
    setText('disk-read', Number.isFinite(data.nvme_read_bytes_per_second)
      ? `${formatBytes(data.nvme_read_bytes_per_second)}/s` : '—');
    setText('disk-devices', (data.nvme_devices || []).join(', ') || 'NVMe counters unavailable');
    setText('monitor-source', `Updated ${new Date(data.sample_time * 1000).toLocaleTimeString()}`);
    const ramPercent = memory?.total_bytes ? 100 * memory.used_bytes / memory.total_bytes : null;
    const gttPercent = Number.isFinite(gtt.used_bytes) && gtt.total_bytes
      ? 100 * gtt.used_bytes / gtt.total_bytes : null;
    setText('ram-percent-label', Number.isFinite(ramPercent) ? `used / total · ${ramPercent.toFixed(1)}% used` : 'used / total');
    setText('gtt-percent-label', Number.isFinite(gttPercent) ? `used / total · ${gttPercent.toFixed(1)}% used` : 'used / total');
    drawPercentLine('chart-gpu-gfx', chartHistory.gfx, gpu.gui_active_percent);
    drawPercentLine('chart-gpu-spi', chartHistory.spi, gpu.spi_busy_percent);
    drawAutoRangeLine('chart-gpu-temp', chartHistory.gpuTemp, gpu.temperature_c, 4);
    drawAutoRangeLine('chart-gpu-power', chartHistory.gpuPower, gpu.power_w, 10);
    drawPercentLine('chart-ram', chartHistory.ram, ramPercent);
    drawPercentLine('chart-gtt', chartHistory.gtt, gttPercent);
    drawPercentLine('chart-cpu', chartHistory.cpu, data.cpu_percent);
    drawLine('chart-disk', chartHistory.disk, data.nvme_read_bytes_per_second, nvmeLinkCeilingBytesPerSecond);
  } catch (_) {
    setText('monitor-source', 'Telemetry unavailable');
  } finally {
    monitorRequestInFlight = false;
  }
}

async function refreshLiveInference() {
  if (inferenceRequestInFlight) return;
  inferenceRequestInFlight = true;
  try {
    const response = await fetch('/api/inference', {cache: 'no-store'});
    if (!response.ok) throw new Error('slot telemetry unavailable');
    const data = await response.json();
    const slots = Array.isArray(data.slots) ? data.slots : [];
    const contextSlots = slots.flatMap(slot => {
      const capacity = typeof slot.n_ctx === 'number' ? slot.n_ctx : null;
      if (!Number.isFinite(capacity) || capacity <= 0) return [];
      const prompt = typeof slot.n_prompt_tokens === 'number' ? slot.n_prompt_tokens : 0;
      const processed = typeof slot.n_prompt_tokens_processed === 'number'
        ? slot.n_prompt_tokens_processed : null;
      const cached = typeof slot.n_prompt_tokens_cache === 'number' ? slot.n_prompt_tokens_cache : 0;
      const decoded = typeof slot.n_decoded === 'number' ? slot.n_decoded : 0;
      const used = slot.is_processing && processed !== null
        ? processed + cached + decoded
        : prompt;
      return [{slot, capacity, used: Math.max(0, Math.min(capacity, used))}];
    });
    const activeContextSlots = contextSlots.filter(item => item.slot.is_processing);
    const contextPool = activeContextSlots.length ? activeContextSlots : contextSlots;
    const fullestSlot = contextPool.reduce((best, item) => !best || item.used / item.capacity > best.used / best.capacity
      ? item : best, null);
    if (fullestSlot) {
      const percent = 100 * fullestSlot.used / fullestSlot.capacity;
      const state = fullestSlot.slot.is_processing ? 'active'
        : (fullestSlot.used > 0 ? 'retained in slot cache' : 'idle');
      setText('context-used', `${Math.round(fullestSlot.used).toLocaleString()} / ${Math.round(fullestSlot.capacity).toLocaleString()}`);
      setText('context-detail', `Slot ${Number(fullestSlot.slot.id) + 1} · ${percent.toFixed(1)}% · ${state}`);
      const promptTokens = Number(fullestSlot.slot.n_prompt_tokens);
      const cachedTokens = Number(fullestSlot.slot.n_prompt_tokens_cache);
      setText('prompt-cache', Number.isFinite(promptTokens) && Number.isFinite(cachedTokens)
        ? `Prompt cache: ${Math.max(0, cachedTokens).toLocaleString()} / ${Math.max(0, promptTokens).toLocaleString()} tokens`
        : 'Prompt cache: unavailable');
      const now = performance.now();
      if (now - lastContextChartAt >= 1000) {
        drawPercentLine('chart-context', chartHistory.context, percent);
        lastContextChartAt = now;
      }
    } else {
      setText('context-used', '— / —');
      setText('context-detail', data.switching ? 'Loading model…' : 'Context data unavailable');
      setText('prompt-cache', 'Prompt cache: unavailable');
      drawPercentLine('chart-context', chartHistory.context, null);
    }

    const slot = slots.find(item => item.is_processing);
    if (slot) {
      const now = performance.now();
      const rawNextToken = Array.isArray(slot.next_token) ? slot.next_token[0] : slot.next_token;
      const tokenCount = Number(slot.n_decoded ?? rawNextToken?.n_decoded);
      const prompting = tokenCount === 0 && Number(slot.n_prompt_tokens_processed) < Number(slot.n_prompt_tokens);
      if (prompting) {
        if (lastStreamChunkAt === 0 || now - lastStreamChunkAt >= slotProgressStaleMs) {
          setText('live-phase', 'Prefill · slot');
        }
      } else if (lastStreamChunkAt === 0 || now - lastStreamChunkAt >= slotProgressStaleMs) {
        setText('live-phase', 'Decode · slot');
      }
      if (Number.isFinite(tokenCount)) {
        if (prompting) {
          setText('live-rate', '— tk/s');
        } else if (previousInferenceSample && tokenCount > previousInferenceSample.tokens) {
          const elapsed = (now - previousInferenceSample.time) / 1000;
          if (elapsed > 0) {
            const rate = (tokenCount - previousInferenceSample.tokens) / elapsed;
            lastSlotProgressAt = now;
            setText('live-phase', 'Decode · slot');
            setText('live-rate', `${rate.toFixed(1)} tk/s`);
            drawAutoRangeLine('chart-live-rate', chartHistory.liveRate, rate, 2);
          }
        } else if (!previousInferenceSample || tokenCount < previousInferenceSample.tokens) {
          setText('live-rate', 'Measuring…');
        }
        previousInferenceSample = {tokens: tokenCount, time: now};
      }
    } else if (!controller) {
      previousInferenceSample = null;
      lastSlotProgressAt = 0;
      lastStreamChunkAt = 0;
      setText('live-rate', '— tk/s');
      setText('live-phase', 'Idle');
    }
  } catch (_) {
    setText('live-phase', 'Slot stats unavailable');
  } finally {
    inferenceRequestInFlight = false;
  }
}

function updateReasoningControls() {
  reasoningEffort.disabled = !reasoningToggle.checked;
}
updateReasoningControls();
reasoningToggle.addEventListener('change', updateReasoningControls);

function updateMaxTokenControls() {
  maxTokensInput.disabled = unlimitedTokensToggle.checked;
}
updateMaxTokenControls();
unlimitedTokensToggle.addEventListener('change', updateMaxTokenControls);

function updateModelControls() {
  const changed = modelSelect.value && modelSelect.value !== currentProfileId;
  loadModelButton.disabled = modelSwitching || modelBusy || Boolean(controller) || !changed;
  modelSelect.disabled = modelSwitching || modelBusy || Boolean(controller);
  sendButton.disabled = modelSwitching || (modelBusy && !controller);
  clearButton.disabled = modelSwitching || Boolean(controller);
  for (const button of messages.querySelectorAll('.message-edit')) {
    button.disabled = modelSwitching || modelBusy || Boolean(controller) || editingChatIndex !== null;
  }
}

function setSendButtonMode(running) {
  sendButton.innerHTML = running
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1"></rect></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M22 2 11 13"></path><path d="m22 2-7 20-4-9-9-4Z"></path></svg>';
  sendButton.setAttribute('aria-label', running ? 'Stop generation' : 'Send message');
  sendButton.title = running ? 'Stop generation' : 'Send message';
  sendButton.classList.toggle('stop-action', running);
}

async function refreshModels() {
  try {
    const response = await fetch('/api/models', {cache: 'no-store'});
    if (!response.ok) return;
    const data = await response.json();
    modelBusy = Boolean(data.busy);
    const profiles = Array.isArray(data.profiles) ? data.profiles : [];
    const signature = profiles.map(item => `${item.category}\u0000${item.id}\u0000${item.name}`).join('\u0001');
    if (!modelProfilesLoaded || modelSelect.dataset.signature !== signature) {
      modelSelect.replaceChildren();
      for (const [category, label] of [['release', 'Release'], ['experimental', 'Experimental'], ['local', 'Local GGUF']]) {
        const items = profiles.filter(item => (item.category || 'release') === category);
        if (!items.length) continue;
        const group = document.createElement('optgroup');
        group.label = label;
        for (const item of items) {
          const option = document.createElement('option');
          option.value = item.id;
          option.textContent = item.name;
          option.title = item.description || item.name;
          group.append(option);
        }
        modelSelect.append(group);
      }
      modelSelect.dataset.signature = signature;
      modelProfilesLoaded = true;
    }
    if (data.current !== currentProfileId) {
      currentProfileId = data.current;
      document.body.dataset.profile = currentProfileId;
      modelSelect.value = currentProfileId;
      const selected = profiles.find(item => item.id === currentProfileId);
      if (selected) profileName.textContent = selected.name;
    } else if (!modelSwitching && (!modelSelect.value || modelSelect.value === currentProfileId)) {
      modelSelect.value = currentProfileId;
    }
    if (data.switching && !modelSwitching) {
      modelSwitching = true;
      status.textContent = 'Loading model…';
    } else if (!data.switching && modelSwitching && !loadModelButton.dataset.localSwitch) {
      modelSwitching = false;
    }
    updateModelControls();
  } catch (_) {}
}

async function loadSelectedModel() {
  if (controller || modelSwitching || modelBusy || !modelSelect.value || modelSelect.value === currentProfileId) return;
  modelSwitching = true;
  loadModelButton.dataset.localSwitch = 'true';
  loadModelButton.textContent = 'Loading…';
  status.textContent = 'Loading model…';
  updateModelControls();
  try {
    const response = await fetch('/api/models/select', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({profile_id: modelSelect.value}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    currentProfileId = data.profile_id;
    document.body.dataset.profile = currentProfileId;
    profileName.textContent = data.name || currentProfileId;
    modelSelect.value = currentProfileId;
    status.textContent = 'Ready';
    status.classList.add('ready');
    previousInferenceSample = null;
    lastSlotProgressAt = 0;
    setText('live-rate', '— tk/s');
    setText('live-phase', 'Idle');
  } catch (error) {
    status.textContent = `Switch failed: ${error.message}`;
    status.classList.remove('ready');
    await refreshModels();
  } finally {
    modelSwitching = false;
    delete loadModelButton.dataset.localSwitch;
    loadModelButton.textContent = 'Load';
    updateModelControls();
  }
}
loadModelButton.addEventListener('click', loadSelectedModel);
modelSelect.addEventListener('change', updateModelControls);

function shouldFollowChat(target = messages.lastElementChild) {
  const overflowY = getComputedStyle(chatPanel).overflowY;
  if (overflowY !== 'visible') {
    return chatPanel.scrollHeight - chatPanel.scrollTop - chatPanel.clientHeight < 96;
  }
  if (!target) return true;
  const bounds = target.getBoundingClientRect();
  return bounds.bottom <= window.innerHeight + 96 && bounds.top < window.innerHeight;
}

function scrollChatTo(target, behavior = 'auto') {
  if (getComputedStyle(chatPanel).overflowY !== 'visible') {
    chatPanel.scrollTo({top: chatPanel.scrollHeight, behavior});
  } else if (target) {
    target.scrollIntoView({behavior, block: 'end'});
  }
}

function scheduleChatScroll(target) {
  chatScrollTarget = target;
  if (chatScrollFrame !== null) return;
  chatScrollFrame = requestAnimationFrame(() => {
    chatScrollFrame = null;
    const currentTarget = chatScrollTarget;
    chatScrollTarget = null;
    if (currentTarget?.isConnected) scrollChatTo(currentTarget);
  });
}

async function writeClipboardText(text) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (_) {}
  }

  const temporaryInput = document.createElement('textarea');
  temporaryInput.value = text;
  temporaryInput.setAttribute('readonly', '');
  temporaryInput.style.position = 'fixed';
  temporaryInput.style.left = '-9999px';
  temporaryInput.style.top = '0';
  document.body.append(temporaryInput);
  let copied = false;
  try {
    temporaryInput.select();
    copied = document.execCommand('copy');
  } finally {
    temporaryInput.remove();
  }
  if (!copied) throw new Error('Clipboard access is unavailable');
}

function addMessage(role, text = '', forceScroll = false) {
  const node = document.createElement('div');
  node.className = `message ${role}`;
  const entryIndex = messages.children.length;
  node.dataset.chatIndex = String(entryIndex);
  let footer = null;
  if (role === 'user') {
    footer = document.createElement('div');
    footer.className = 'message-footer';
    const editButton = document.createElement('button');
    editButton.className = 'message-edit';
    editButton.type = 'button';
    editButton.title = 'Edit message';
    editButton.setAttribute('aria-label', 'Edit message');
    const icon = svgNode('svg', {viewBox: '0 0 24 24', 'aria-hidden': 'true'});
    icon.append(svgNode('path', {d: 'm4 16.5-.8 4.3 4.3-.8L19 8.5 15.5 5 4 16.5Z'}));
    icon.append(svgNode('path', {d: 'm13.8 6.7 3.5 3.5'}));
    editButton.append(icon, document.createTextNode('Edit'));
    editButton.addEventListener('click', () => beginMessageEdit(entryIndex, node));
    footer.append(editButton);
  } else if (role === 'assistant') {
    let copyText = String(text || '');
    footer = document.createElement('div');
    footer.className = 'message-footer';
    const metadata = document.createElement('span');
    metadata.className = 'message-metadata';
    footer.append(metadata);
    const copyButton = document.createElement('button');
    copyButton.className = 'message-copy';
    copyButton.type = 'button';
    copyButton.title = 'Copy response';
    copyButton.setAttribute('aria-label', 'Copy response');
    copyButton.disabled = !copyText;
    const icon = svgNode('svg', {viewBox: '0 0 24 24', 'aria-hidden': 'true'});
    icon.append(
      svgNode('path', {class: 'copy-glyph', d: 'M8 8h12v12H8z'}),
      svgNode('path', {class: 'copy-glyph', d: 'M16 8V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h4'}),
      svgNode('path', {class: 'copy-check', d: 'm5 12 4 4L19 6'}),
    );
    copyButton.append(icon);
    copyButton.addEventListener('click', async () => {
      copyButton.disabled = true;
      try {
        await writeClipboardText(copyText);
        copyButton.dataset.copied = 'true';
        copyButton.title = 'Copied';
        copyButton.setAttribute('aria-label', 'Response copied');
      } catch (_) {
        copyButton.title = 'Copy failed';
        copyButton.setAttribute('aria-label', 'Copy failed');
      }
      window.setTimeout(() => {
        if (copyButton.isConnected) {
          delete copyButton.dataset.copied;
          copyButton.title = 'Copy response';
          copyButton.setAttribute('aria-label', 'Copy response');
          copyButton.disabled = !copyText;
        }
      }, 1400);
    });
    footer.append(copyButton);
    node.setCopyText = (value) => {
      copyText = String(value || '');
      copyButton.disabled = !copyText;
    };
    node.setMessageMeta = (tokenCount, durationMs) => {
      const parts = [];
      if (Number.isFinite(tokenCount)) parts.push(`${Math.max(0, Math.round(tokenCount))} tokens`);
      if (Number.isFinite(durationMs)) {
        const totalSeconds = Math.max(0, Math.floor(durationMs / 1000));
        parts.push(`${Math.floor(totalSeconds / 60)}m ${String(totalSeconds % 60).padStart(2, '0')}s`);
      }
      metadata.textContent = parts.join(' · ');
      metadata.hidden = parts.length === 0;
    };
    node.setMessageMeta(null, null);
  }
  const body = document.createElement('div');
  body.className = 'message-content';
  renderMarkdownInto(body, text);
  node.append(body);
  if (footer) node.append(footer);
  const follow = forceScroll || shouldFollowChat();
  messages.append(node);
  if (follow) scheduleChatScroll(node);
  return node;
}

function historyLengthBeforeChatEntry(entryIndex) {
  let historyIndex = 0;
  for (const entry of chatLog.slice(0, entryIndex)) {
    const historyEntry = history[historyIndex];
    if (historyEntry?.role === entry.role && historyEntry.content === entry.content) historyIndex++;
  }
  return historyIndex;
}

function finishMessageEdit(restoreDraft = false) {
  editingMessageNode?.classList.remove('editing-message');
  editingMessageNode = null;
  editingChatIndex = null;
  editBanner.hidden = true;
  if (restoreDraft && editingDraft !== null) promptBox.value = editingDraft;
  editingDraft = null;
  updateModelControls();
}

function beginMessageEdit(entryIndex, node) {
  if (controller || modelSwitching || modelBusy || editingChatIndex !== null) return;
  const entry = chatLog[entryIndex];
  if (entry?.role !== 'user') return;
  editingDraft = promptBox.value;
  editingChatIndex = entryIndex;
  editingMessageNode = node;
  node.classList.add('editing-message');
  editBanner.hidden = false;
  promptBox.value = entry.content;
  updateModelControls();
  promptBox.focus();
  promptBox.setSelectionRange(promptBox.value.length, promptBox.value.length);
}

cancelEditButton.addEventListener('click', () => {
  finishMessageEdit(true);
  promptBox.focus();
});

function appendInlineMarkdown(parent, text, depth = 0) {
  if (depth > 5) {
    parent.append(document.createTextNode(text));
    return;
  }
  const tokenPattern = /(`+)([^`]+?)\1|\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)(?:\s+["']([^"']*)["'])?\)|\*\*(\S(?:.*?\S)?)\*\*|(?<!\w)__(\S(?:.*?\S)?)__(?!\w)|~~(\S(?:.*?\S)?)~~|\*(\S(?:.*?\S)?)\*|(?<!\w)_(\S(?:.*?\S)?)_(?!\w)/g;
  let cursor = 0;
  for (const match of text.matchAll(tokenPattern)) {
    const start = match.index ?? cursor;
    if (start > cursor) parent.append(document.createTextNode(text.slice(cursor, start)));
    let element;
    if (match[1]) {
      element = document.createElement('code');
      element.textContent = match[2];
    } else if (match[3]) {
      let safeUrl = null;
      try {
        const parsed = new URL(match[4], window.location.href);
        if (['http:', 'https:', 'mailto:'].includes(parsed.protocol)) safeUrl = parsed.href;
      } catch (_) {}
      if (safeUrl) {
        element = document.createElement('a');
        element.href = safeUrl;
        element.target = '_blank';
        element.rel = 'noopener noreferrer';
        if (match[5]) element.title = match[5];
        appendInlineMarkdown(element, match[3], depth + 1);
      } else {
        element = document.createTextNode(match[0]);
      }
    } else {
      const content = match[6] ?? match[7] ?? match[8] ?? match[9] ?? match[10];
      const kind = match[6] || match[7] ? 'strong' : (match[8] ? 'del' : 'em');
      element = document.createElement(kind);
      appendInlineMarkdown(element, content, depth + 1);
    }
    parent.append(element);
    cursor = start + match[0].length;
  }
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

function markdownCells(line) {
  let value = line.trim();
  if (value.startsWith('|')) value = value.slice(1);
  if (value.endsWith('|')) value = value.slice(0, -1);
  return value.split('|').map(cell => cell.trim());
}

function isTableStart(lines, index) {
  if (index + 1 >= lines.length || !lines[index].includes('|')) return false;
  const cells = markdownCells(lines[index + 1]);
  return lines[index + 1].includes('|') && cells.length > 0
    && cells.every(cell => /^:?-{3,}:?$/.test(cell));
}

function isMarkdownBlockStart(lines, index) {
  const line = lines[index];
  return /^ {0,3}(#{1,6})\s+/.test(line)
    || /^ {0,3}(`{3,}|~{3,})/.test(line)
    || /^\s*>/.test(line)
    || /^\s*(?:[-+*]\s+|\d+[.)]\s+)/.test(line)
    || /^ {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$/.test(line)
    || isTableStart(lines, index);
}

function renderMarkdownInto(target, source, depth = 0) {
  target.replaceChildren();
  if (depth > 5) {
    target.textContent = source;
    return;
  }
  const lines = String(source || '').replace(/\r\n?/g, '\n').split('\n');
  let index = 0;
  while (index < lines.length) {
    if (!lines[index].trim()) { index++; continue; }

    const fence = lines[index].match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (fence) {
      const marker = fence[1];
      const language = fence[2].trim().split(/\s+/, 1)[0];
      const codeLines = [];
      index++;
      while (index < lines.length && !new RegExp(`^ {0,3}${marker[0]}{${marker.length},}\\s*$`).test(lines[index])) {
        codeLines.push(lines[index++]);
      }
      if (index < lines.length) index++;
      const block = document.createElement('div');
      block.className = 'code-block';
      const toolbar = document.createElement('div');
      toolbar.className = 'code-block-toolbar';
      const languageLabel = document.createElement('span');
      languageLabel.textContent = /^[\w+-]+$/.test(language) ? language : 'Code';
      const copyButton = document.createElement('button');
      copyButton.className = 'code-copy-button';
      copyButton.type = 'button';
      copyButton.textContent = 'Copy';
      copyButton.setAttribute('aria-label', 'Copy code block');
      copyButton.addEventListener('click', async () => {
        copyButton.disabled = true;
        try {
          await writeClipboardText(code.textContent || '');
          copyButton.textContent = 'Copied';
          copyButton.dataset.copied = 'true';
        } catch (_) {
          copyButton.textContent = 'Copy failed';
        }
        window.setTimeout(() => {
          if (!copyButton.isConnected) return;
          copyButton.textContent = 'Copy';
          copyButton.disabled = false;
          delete copyButton.dataset.copied;
        }, 1400);
      });
      toolbar.append(languageLabel, copyButton);
      const pre = document.createElement('pre');
      const code = document.createElement('code');
      if (/^[\w+-]+$/.test(language)) code.className = `language-${language}`;
      code.textContent = codeLines.join('\n');
      pre.append(code);
      block.append(toolbar, pre);
      target.append(block);
      continue;
    }

    const heading = lines[index].match(/^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const element = document.createElement(`h${heading[1].length}`);
      appendInlineMarkdown(element, heading[2]);
      target.append(element);
      index++;
      continue;
    }

    if (/^ {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$/.test(lines[index])) {
      target.append(document.createElement('hr'));
      index++;
      continue;
    }

    if (/^\s*>/.test(lines[index])) {
      const quoteLines = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) {
        quoteLines.push(lines[index++].replace(/^\s*>\s?/, ''));
      }
      const quote = document.createElement('blockquote');
      renderMarkdownInto(quote, quoteLines.join('\n'), depth + 1);
      target.append(quote);
      continue;
    }

    const listMatch = lines[index].match(/^\s*(?:([-+*])|(\d+)[.)])\s+(.*)$/);
    if (listMatch) {
      const ordered = Boolean(listMatch[2]);
      const list = document.createElement(ordered ? 'ol' : 'ul');
      if (ordered && Number(listMatch[2]) > 1) list.start = Number(listMatch[2]);
      while (index < lines.length) {
        const item = lines[index].match(/^\s*(?:([-+*])|(\d+)[.)])\s+(.*)$/);
        if (!item || Boolean(item[2]) !== ordered) break;
        const li = document.createElement('li');
        appendInlineMarkdown(li, item[3]);
        list.append(li);
        index++;
      }
      target.append(list);
      continue;
    }

    if (isTableStart(lines, index)) {
      const headers = markdownCells(lines[index]);
      const alignments = markdownCells(lines[index + 1]).map(cell =>
        cell.startsWith(':') && cell.endsWith(':') ? 'center'
          : (cell.endsWith(':') ? 'right' : (cell.startsWith(':') ? 'left' : '')));
      const table = document.createElement('table');
      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      headers.forEach((header, column) => {
        const th = document.createElement('th');
        if (alignments[column]) th.style.textAlign = alignments[column];
        appendInlineMarkdown(th, header);
        headerRow.append(th);
      });
      thead.append(headerRow);
      table.append(thead);
      const tbody = document.createElement('tbody');
      index += 2;
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
        const cells = markdownCells(lines[index++]);
        const row = document.createElement('tr');
        headers.forEach((_, column) => {
          const td = document.createElement('td');
          if (alignments[column]) td.style.textAlign = alignments[column];
          appendInlineMarkdown(td, cells[column] || '');
          row.append(td);
        });
        tbody.append(row);
      }
      table.append(tbody);
      target.append(table);
      continue;
    }

    const paragraphLines = [];
    while (index < lines.length && lines[index].trim()
      && !(paragraphLines.length > 0 && isMarkdownBlockStart(lines, index))) {
      paragraphLines.push(lines[index++]);
    }
    const paragraph = document.createElement('p');
    paragraphLines.forEach((line, lineIndex) => {
      const hardBreak = /(?: {2,}|\\)$/.test(line);
      appendInlineMarkdown(paragraph, line.replace(/(?: {2,}|\\)$/, '').trim(), depth);
      if (lineIndex < paragraphLines.length - 1) {
        if (hardBreak) paragraph.append(document.createElement('br'));
        else paragraph.append(document.createTextNode(' '));
      }
    });
    target.append(paragraph);
  }
}

function persistChatState(generationActive = Boolean(controller)) {
  try {
    localStorage.setItem(chatStorageKey, JSON.stringify({
      version: 1,
      history,
      entries: chatLog,
      generationActive,
      settings: {
        maxTokens: maxTokensInput.value,
        unlimited: unlimitedTokensToggle.checked,
        reasoning: reasoningToggle.checked,
        reasoningEffort: reasoningEffort.value,
      },
    }));
  } catch (_) {}
}

function renderChatEntry(entry) {
  const node = addMessage(entry.role, entry.content);
  node.setCopyText?.(entry.content);
  node.setMessageMeta?.(entry.generatedTokens, entry.durationMs);
  if (entry.status === 'error') node.classList.add('error');
  if (entry.reasoning) {
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    const pre = document.createElement('pre');
    summary.textContent = 'Reasoning';
    pre.textContent = entry.reasoning;
    details.append(summary, pre);
    node.prepend(details);
  }
  if (entry.status === 'interrupted' || entry.status === 'stopped') {
    const note = document.createElement('small');
    note.className = 'message-status';
    note.textContent = entry.status === 'interrupted'
      ? 'Generation interrupted by page reload. Partial output restored.'
      : 'Generation stopped. Partial output kept.';
    node.insertBefore(note, node.querySelector('.message-footer'));
  }
  return node;
}

function restoreChatState() {
  try {
    if (!localStorage.getItem(chatStorageKey)) {
      const legacy = localStorage.getItem(legacyChatStorageKey);
      if (legacy) localStorage.setItem(chatStorageKey, legacy);
    }
    const saved = JSON.parse(localStorage.getItem(chatStorageKey) || 'null');
    if (!saved || saved.version !== 1 || !Array.isArray(saved.history) || !Array.isArray(saved.entries)) return;
    if (saved.settings && typeof saved.settings === 'object') {
      if (typeof saved.settings.maxTokens === 'string') maxTokensInput.value = saved.settings.maxTokens;
      if (typeof saved.settings.unlimited === 'boolean') unlimitedTokensToggle.checked = saved.settings.unlimited;
      if (typeof saved.settings.reasoning === 'boolean') reasoningToggle.checked = saved.settings.reasoning;
      if (typeof saved.settings.reasoningEffort === 'string') reasoningEffort.value = saved.settings.reasoningEffort;
      updateMaxTokenControls();
      updateReasoningControls();
    }
    for (const message of saved.history) {
      if ((message?.role === 'user' || message?.role === 'assistant') && typeof message.content === 'string') {
        if (message.role === 'assistant') {
          history.push(assistantHistoryMessage(message.content, message.reasoning_content));
        } else {
          history.push({role: 'user', content: message.content});
        }
      }
    }
    // Older saved chats kept reasoning in the rendered entry but omitted it from
    // the request history. Restore it there so the server can reuse the same prefix.
    const savedReasoningByAnswer = new Map();
    for (const savedEntry of saved.entries) {
      if (savedEntry?.role !== 'assistant' || typeof savedEntry.content !== 'string'
          || typeof savedEntry.reasoning !== 'string' || !savedEntry.reasoning) continue;
      const queue = savedReasoningByAnswer.get(savedEntry.content) || [];
      queue.push(savedEntry.reasoning);
      savedReasoningByAnswer.set(savedEntry.content, queue);
    }
    for (const message of history) {
      if (message.role !== 'assistant' || message.reasoning_content) continue;
      const queue = savedReasoningByAnswer.get(message.content);
      if (queue?.length) message.reasoning_content = queue.shift();
    }
    let interrupted = false;
    for (const savedEntry of saved.entries) {
      if ((savedEntry?.role !== 'user' && savedEntry?.role !== 'assistant') || typeof savedEntry.content !== 'string') continue;
      const entry = {
        role: savedEntry.role,
        content: savedEntry.content,
        reasoning: typeof savedEntry.reasoning === 'string' ? savedEntry.reasoning : '',
        status: savedEntry.status === 'streaming' ? 'interrupted' : savedEntry.status,
      };
      if (entry.status === 'interrupted') {
        interrupted = true;
        if (entry.content && !history.some(message => message.role === 'assistant' && message.content === entry.content)) {
          history.push(assistantHistoryMessage(entry.content, entry.reasoning));
        }
      }
      chatLog.push(entry);
      renderChatEntry(entry);
    }
    if (interrupted || saved.generationActive) persistChatState(false);
  } catch (_) {}
}

window.addEventListener('pagehide', () => {
  pageLeaving = true;
  persistChatState(Boolean(controller));
});
window.addEventListener('beforeunload', () => {
  pageLeaving = true;
  persistChatState(Boolean(controller));
});
restoreChatState();

sendButton.addEventListener('click', (event) => {
  if (!controller) return;
  event.preventDefault();
  controller.abort();
});

async function checkHealth() {
  if (!status) return;
  if (modelSwitching) {
    status.textContent = 'Loading model…';
    status.classList.remove('ready');
    return;
  }
  try {
    const response = await fetch('/api/health', {cache: 'no-store'});
    status.textContent = response.ok ? 'Ready' : 'Engine error';
    status.classList.toggle('ready', response.ok);
  } catch (_) {
    status.textContent = 'Disconnected';
    status.classList.remove('ready');
  }
}
checkHealth();
setInterval(checkHealth, 5000);
refreshModels();
setInterval(refreshModels, 5000);
refreshMonitor();
setInterval(refreshMonitor, 1000);
refreshLiveInference();
setInterval(refreshLiveInference, 500);

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = promptBox.value.trim();
  if (!text || controller || modelSwitching || modelBusy) return;
  if (editingChatIndex !== null) {
    const historyCut = historyLengthBeforeChatEntry(editingChatIndex);
    history.splice(historyCut);
    chatLog.splice(editingChatIndex);
    while (messages.children.length > editingChatIndex) messages.lastElementChild.remove();
    finishMessageEdit(false);
  }
  promptBox.value = '';
  const userEntry = {role: 'user', content: text};
  history.push(userEntry);
  chatLog.push({...userEntry});
  addMessage('user', text, true);
  const assistantEntry = {role: 'assistant', content: '', reasoning: '', status: 'streaming'};
  chatLog.push(assistantEntry);
  const bubble = addMessage('assistant', '', true);
  bubble.setCopyText('');
  const bubbleBody = bubble.querySelector('.message-content');
  bubbleBody.classList.add('streaming');
  const answerText = document.createTextNode('');
  bubbleBody.append(answerText);
  persistChatState(true);
  let answer = '';
  let reasoning = '';
  let reasoningBox = null;
  let reasoningText = null;
  let firstReasoningAt = null;
  let firstAnswerAt = null;
  let responseTimings = null;
  let responseUsageTokens = null;
  const generationStartedAt = performance.now();
  let followChat = true;
  previousInferenceSample = null;
  lastSlotProgressAt = 0;
  streamRateStartedAt = null;
  streamRateChunkCount = 0;
  lastStreamChunkAt = 0;
  lastStreamChartAt = 0;
  controller = new AbortController();
  updateModelControls();
  setText('live-rate', '— tk/s');
  setText('live-phase', 'Prefill');
  refreshLiveInference();
  setSendButtonMode(true);
  maxTokensInput.disabled = true;
  unlimitedTokensToggle.disabled = true;
  reasoningToggle.disabled = true;
  reasoningEffort.disabled = true;
  try {
    const reasoningEnabled = reasoningToggle.checked;
    const selectedEffort = reasoningEffort.value;
    const requestBody = {
      messages: history,
      stream: true,
      max_tokens: unlimitedTokensToggle.checked ? -1 : Number(maxTokensInput.value),
      temperature: 0.7,
      chat_template_kwargs: {enable_thinking: reasoningEnabled},
    };
    if (reasoningEnabled && selectedEffort !== 'default') {
      requestBody.reasoning_effort = selectedEffort;
    }
    const response = await fetch('/api/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      followChat = shouldFollowChat(bubble);
      pending += decoder.decode(value, {stream: true}).replaceAll('\r\n', '\n');
      const events = pending.split('\n\n');
      pending = events.pop() || '';
      for (const eventText of events) {
        for (const line of eventText.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;
          let item;
          try { item = JSON.parse(payload); } catch (_) { continue; }
          if (item.timings && typeof item.timings === 'object') responseTimings = item.timings;
          if (item.usage && item.usage.completion_tokens !== undefined
            && Number.isFinite(Number(item.usage.completion_tokens))) {
            responseUsageTokens = Number(item.usage.completion_tokens);
          }
          const delta = item.choices?.[0]?.delta || {};
          const thought = delta.reasoning_content || delta.reasoning || '';
          if (thought) {
            if (firstReasoningAt === null) firstReasoningAt = performance.now();
            updateLiveRateFromStream('Reasoning');
            if (!reasoningBox) {
              reasoningBox = document.createElement('details');
              const summary = document.createElement('summary');
              summary.textContent = 'Reasoning';
              const pre = document.createElement('pre');
              reasoningText = document.createTextNode('');
              pre.append(reasoningText);
              reasoningBox.append(summary, pre);
              bubble.prepend(reasoningBox);
            }
            reasoning += thought;
            assistantEntry.reasoning = reasoning;
            reasoningText.appendData(thought);
          }
          const piece = delta.content;
          if (typeof piece === 'string') {
            if (piece) updateLiveRateFromStream('Answer');
            if (piece && firstAnswerAt === null) firstAnswerAt = performance.now();
            answer += piece;
            assistantEntry.content = answer;
            bubble.setCopyText(answer);
            answerText.appendData(piece);
          }
        }
      }
      if (followChat) scheduleChatScroll(bubble);
    }
    bubbleBody.classList.remove('streaming');
    renderMarkdownInto(bubbleBody, answer);
    if (followChat) scheduleChatScroll(bubble);
    if (responseTimings) {
      setText('prompt-rate', `${Number(responseTimings.prompt_per_second || 0).toFixed(1)} tok/s`);
      setText('generation-rate', `${Number(responseTimings.predicted_per_second || 0).toFixed(1)} tok/s`);
      setText('live-rate', `${Number(responseTimings.predicted_per_second || 0).toFixed(1)} tk/s`);
    } else {
      setText('prompt-rate', 'Unavailable');
      setText('generation-rate', 'Unavailable');
    }
    setText('live-phase', 'Complete');
    setText('reasoning-span', firstReasoningAt !== null && firstAnswerAt !== null
      ? `Reasoning span ≈ ${((firstAnswerAt - firstReasoningAt) / 1000).toFixed(1)} s`
      : (reasoningToggle.checked ? 'Reasoning span: unavailable' : 'Reasoning: off'));
    const reportedTokens = responseTimings?.predicted_n ?? responseUsageTokens;
    const generatedTokens = reportedTokens !== null && reportedTokens !== undefined
      && Number.isFinite(Number(reportedTokens)) ? Number(reportedTokens) : null;
    const durationMs = performance.now() - generationStartedAt;
    assistantEntry.generatedTokens = generatedTokens;
    assistantEntry.durationMs = durationMs;
    bubble.setMessageMeta(generatedTokens, durationMs);
    assistantEntry.status = 'complete';
    history.push(assistantHistoryMessage(answer, reasoning));
    persistChatState(false);
  } catch (error) {
    if (!pageLeaving) {
      if (error.name === 'AbortError') {
        assistantEntry.status = 'stopped';
        assistantEntry.content = answer;
        assistantEntry.reasoning = reasoning;
        bubble.setCopyText(answer);
        assistantEntry.durationMs = performance.now() - generationStartedAt;
        bubble.setMessageMeta(null, assistantEntry.durationMs);
        bubbleBody.classList.remove('streaming');
        renderMarkdownInto(bubbleBody, answer);
        if (followChat) scheduleChatScroll(bubble);
        if (answer) history.push(assistantHistoryMessage(answer, reasoning));
        const note = document.createElement('small');
        note.className = 'message-status';
        note.textContent = 'Generation stopped. Partial output kept.';
        bubble.insertBefore(note, bubble.querySelector('.message-footer'));
        persistChatState(false);
      } else {
        bubble.classList.add('error');
        bubble.textContent = `Request failed: ${error.message}`;
        assistantEntry.status = 'error';
        assistantEntry.content = bubble.textContent;
        assistantEntry.reasoning = reasoning;
        history.pop();
        persistChatState(false);
      }
    }
  } finally {
    controller = null;
    modelBusy = false;
    updateModelControls();
    refreshModels();
    previousInferenceSample = null;
    sendButton.disabled = false;
    setSendButtonMode(false);
    unlimitedTokensToggle.disabled = false;
    updateMaxTokenControls();
    reasoningToggle.disabled = false;
    updateReasoningControls();
    promptBox.focus();
  }
});

clearButton.addEventListener('click', () => {
  if (controller) return;
  if (editingChatIndex !== null) finishMessageEdit(true);
  history.length = 0;
  chatLog.length = 0;
  messages.replaceChildren();
  try { localStorage.removeItem(chatStorageKey); } catch (_) {}
});
promptBox.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
