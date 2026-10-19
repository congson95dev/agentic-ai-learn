const chat = document.querySelector('#chat'), welcome = document.querySelector('#welcome');
const composer = document.querySelector('#composer'), input = document.querySelector('#message'), files = document.querySelector('#files');
const conversationList = document.querySelector('#conversationList'), title = document.querySelector('#chatTitle');
const approval = document.querySelector('#approval'), approvalText = document.querySelector('#approvalText'), voiceButton = document.querySelector('#voice');
let conversationId = null, approvalInFlight = false;

function escapeHtml(value) { const div = document.createElement('div'); div.textContent = value; return div.innerHTML; }
function toolCallInfo(tool) { return typeof tool === 'string' ? {name: tool, source: 'Tool'} : tool; }
function renderAssistant(item, content, toolCalls = [], isThinking = false) { const status = isThinking ? '<span class="tool-note thinking">Thinking<span class="dots">...</span></span>' : ''; const tools = toolCalls.map(toolCallInfo).map(tool => `<span class="tool-note" aria-label="Using ${escapeHtml(tool.source || 'Tool')}: ${escapeHtml(tool.name)}"><span class="tool-source ${tool.source === 'MCP' ? 'mcp' : ''}">Using ${escapeHtml(tool.source || 'Tool')}:</span> ${escapeHtml(tool.name)}</span>`).join(''); item.innerHTML = status + tools + marked.parse(content || ''); item.scrollIntoView({block: 'end'}); }
function addMessage(role, content, toolCalls = [], isThinking = false) { document.querySelector('#welcome')?.remove(); const item = document.createElement('article'); item.className = `message ${role}`; if (role === 'assistant') renderAssistant(item, content, toolCalls, isThinking); else item.textContent = content; chat.append(item); item.scrollIntoView({block: 'end'}); return item; }
function setFiles() { document.querySelector('#filePills').innerHTML = [...files.files].map(file => `<span class="file-pill">${escapeHtml(file.name)}</span>`).join(''); }
function autoresize() { input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 180)}px`; }
async function request(url, options = {}) { const response = await fetch(url, options); if (!response.ok) throw new Error((await response.json().catch(() => ({detail: 'Request failed'}))).detail); return response.status === 204 ? null : response.json(); }
async function refreshConversations() { const data = await request('/api/conversations'); conversationList.innerHTML = ''; data.forEach(c => { const row = document.createElement('div'); row.className = `conversation-row ${c.id === conversationId ? 'active' : ''}`; const button = document.createElement('button'); button.className = 'conversation'; button.textContent = c.title; button.onclick = () => loadConversation(c.id, c.title); const remove = document.createElement('button'); remove.className = 'delete-chat'; remove.type = 'button'; remove.title = 'Delete chat'; remove.ariaLabel = `Delete ${c.title}`; remove.textContent = '×'; remove.onclick = async event => { event.stopPropagation(); if (!confirm(`Delete “${c.title}”?`)) return; await request(`/api/conversations/${c.id}`, {method: 'DELETE'}); if (conversationId === c.id) await startNewChat(); else await refreshConversations(); }; row.append(button, remove); conversationList.append(row); }); }
async function startNewChat() { conversationId = null; chat.innerHTML = '<div class="welcome" id="welcome"><h1>How can I help you today?</h1><p>Ask a question, attach a document, or speak your message.</p></div>'; title.textContent = 'New chat'; approval.classList.add('hidden'); await refreshConversations(); input.focus(); }
async function createConversation() { const c = await request('/api/conversations', {method: 'POST'}); conversationId = c.id; await refreshConversations(); }
async function loadConversation(id, name) { conversationId = id; title.textContent = name; chat.innerHTML = ''; const messages = await request(`/api/conversations/${id}/messages`); messages.forEach(message => addMessage(message.role, message.content, message.tool_calls)); await refreshConversations(); document.querySelector('.sidebar').classList.remove('open'); }

async function readSse(response, onEvent) {
  const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
  while (true) {
    const {done, value} = await reader.read(); if (done) break;
    buffer += decoder.decode(value, {stream: true}); const frames = buffer.split('\n\n'); buffer = frames.pop();
    for (const frame of frames) { const event = frame.match(/^event: (.+)$/m)?.[1]; const data = frame.match(/^data: (.+)$/m)?.[1]; if (event && data) onEvent(event, JSON.parse(data)); }
  }
}

composer.addEventListener('submit', async event => {
  event.preventDefault(); const text = input.value.trim(); if (!text && !files.files.length) return;
  if (!conversationId) await createConversation(); addMessage('user', text || `Uploaded: ${[...files.files].map(file => file.name).join(', ')}`);
  const data = new FormData(); data.append('message', text); [...files.files].forEach(file => data.append('files', file)); input.value = ''; files.value = ''; setFiles(); autoresize(); composer.querySelector('.send').disabled = true;
  const assistant = addMessage('assistant', '', [], true); let answer = '', toolCalls = [], isThinking = true;
  try {
    const response = await fetch(`/api/conversations/${conversationId}/messages`, {method: 'POST', body: data});
    if (!response.ok) throw new Error((await response.json().catch(() => ({detail: 'Request failed'}))).detail);
    await readSse(response, (type, payload) => {
      if (type === 'token') { answer += payload.text; renderAssistant(assistant, answer, toolCalls, isThinking); }
      if (type === 'tool') { toolCalls.push(payload); renderAssistant(assistant, answer, toolCalls, isThinking); }
      if (type === 'done') { isThinking = false; renderAssistant(assistant, answer, toolCalls, isThinking); title.textContent = payload.title; refreshConversations(); if (payload.interrupt) { approvalText.textContent = payload.interrupt; approval.classList.remove('hidden'); } }
      if (type === 'error') throw new Error(payload.detail);
    });
  } catch (error) { isThinking = false; renderAssistant(assistant, answer || `Sorry, ${error.message}`, toolCalls, isThinking); }
  finally { composer.querySelector('.send').disabled = false; input.focus(); }
});

input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); composer.requestSubmit(); } });
approval.addEventListener('click', async event => { const approved = event.target.dataset.approved; if (approved === undefined || approvalInFlight) return; approvalInFlight = true; approval.classList.add('hidden'); const data = new FormData(); data.append('approved', approved); try { const result = await request(`/api/conversations/${conversationId}/approval`, {method: 'POST', body: data}); addMessage('assistant', result.answer, result.tool_calls); if (result.interrupt) { approvalText.textContent = result.interrupt; approval.classList.remove('hidden'); } } catch (error) { addMessage('assistant', `Sorry, ${error.message}`); approval.classList.remove('hidden'); } finally { approvalInFlight = false; } });
document.querySelector('#newChat').onclick = startNewChat; document.querySelector('#menu').onclick = () => document.querySelector('.sidebar').classList.toggle('open'); files.onchange = setFiles; input.oninput = autoresize;

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (Recognition) { const recognition = new Recognition(); recognition.lang = navigator.language || 'en-US'; recognition.interimResults = false; voiceButton.onclick = () => { recognition.start(); voiceButton.classList.add('listening'); document.querySelector('#voiceHint').textContent = 'Listening…'; }; recognition.onresult = event => { input.value += (input.value ? ' ' : '') + event.results[0][0].transcript; autoresize(); }; recognition.onend = () => { voiceButton.classList.remove('listening'); document.querySelector('#voiceHint').textContent = 'Files are saved locally and indexed for this chat.'; }; recognition.onerror = () => { voiceButton.classList.remove('listening'); document.querySelector('#voiceHint').textContent = 'Voice input was unavailable. Please try again.'; }; } else { voiceButton.disabled = true; document.querySelector('#voiceHint').textContent = 'Voice input is not supported by this browser.'; }
(async () => { try { await startNewChat(); } catch (error) { addMessage('assistant', `Could not start chat: ${error.message}`); } })();
