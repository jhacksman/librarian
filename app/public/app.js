const $ = (selector, parent = document) => parent.querySelector(selector);
const main = $('#main');
const number = new Intl.NumberFormat();
const state = {
  status: null, route: '', epoch: 0, library: { q: '', format: '', author: '', status: '', sort: 'title', offset: 0, limit: 24, view: 'grid' },
  seenBooks: new Map(), bookCache: new Map(), scope: null, draft: '', turns: [], asking: false,
  contentSearch: { query: '', bookId: '', result: null, scopeQuery: '', scopeOffset: 0 }, contentSearchSequence: 0, citationOrigin: null, readerStates: {}, readerModes: {}, readerReturn: null, readerCapture: null,
  pageCleanup: null, jobs: [], jobDetails: new Map(), jobOffsets: new Map(), openJobs: new Set(), poll: null, statusPoll: null, deferredJobs: false, importRoot: '', toastTimer: null,
};


const VIEW_STATE_KEY = 'librarian-view-state-v1';
function persistViewState() {
  try {
    let turns = state.turns.slice(-5);
    let value = JSON.stringify({ library: state.library, scope: state.scope, draft: state.draft, readerStates: state.readerStates, readerModes: state.readerModes, readerReturn: state.readerReturn, turns });
    while (value.length > 240000 && turns.length > 1) { turns = turns.slice(1); value = JSON.stringify({ library: state.library, scope: state.scope, draft: state.draft, readerStates: state.readerStates, readerModes: state.readerModes, readerReturn: state.readerReturn, turns }); }
    if (value.length > 240000) value = JSON.stringify({ library: state.library, scope: state.scope, draft: state.draft, readerStates: state.readerStates, readerModes: state.readerModes, readerReturn: state.readerReturn, turns: turns.map(turn => ({ question: turn.question, scope: turn.scope, error: 'This response could not be restored after refresh. Ask again to retrieve its sources.' })) });
    sessionStorage.setItem(VIEW_STATE_KEY, value);
  } catch { /* Navigation still works when browser storage is unavailable. */ }
}
try {
  const saved = JSON.parse(sessionStorage.getItem(VIEW_STATE_KEY) || 'null');
  if (saved && typeof saved === 'object') {
    for (const key of ['q', 'format', 'author', 'status']) if (typeof saved.library?.[key] === 'string' && saved.library[key].length <= 500) state.library[key] = saved.library[key];
    if (['title', 'author', 'recent'].includes(saved.library?.sort)) state.library.sort = saved.library.sort;
    if (['grid', 'list'].includes(saved.library?.view)) state.library.view = saved.library.view;
    if (Number.isSafeInteger(saved.library?.offset) && saved.library.offset >= 0) state.library.offset = saved.library.offset;
    if (saved.scope && typeof saved.scope.id === 'string' && saved.scope.id.length <= 4096 && typeof saved.scope.title === 'string') state.scope = { id: saved.scope.id, title: saved.scope.title.slice(0, 500), ...(saved.scope.readerContext && typeof saved.scope.readerContext === 'object' ? { readerContext: saved.scope.readerContext, detail: String(saved.scope.detail || '').slice(0, 1000) } : {}) };
    if (saved.readerStates && typeof saved.readerStates === 'object' && !Array.isArray(saved.readerStates)) state.readerStates = saved.readerStates;
    if (saved.readerModes && typeof saved.readerModes === 'object' && !Array.isArray(saved.readerModes)) state.readerModes = saved.readerModes;
    if (typeof saved.readerReturn === 'string' && saved.readerReturn.startsWith('#read/')) state.readerReturn = saved.readerReturn;
    if (typeof saved.draft === 'string') state.draft = saved.draft.slice(0, 4000);
    if (Array.isArray(saved.turns)) state.turns = saved.turns.slice(-5).filter(turn => typeof turn?.question === 'string').map(turn => ({ ...turn, pending: false, ...(turn.pending ? { error: 'Refreshing interrupted this question. Ask again to retrieve its sources.' } : {}) }));
  }
} catch { /* Ignore malformed or unavailable saved view state. */ }

const paths = {
  library: ['M4 4h5v16H4z', 'M11 4h4v16h-4z', 'm17 5 3-1 4 15-3 1z'],
  import: ['M12 16V3', 'm7 8 5-5 5 5', 'M4 14v6h16v-6'],
  ask: ['m12 3 2.3 6.7L21 12l-6.7 2.3L12 21l-2.3-6.7L3 12l6.7-2.3Z'],
  search: ['M21 21l-5-5', 'M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0'],
  grid: ['M3 3h7v7H3z', 'M14 3h7v7h-7z', 'M3 14h7v7H3z', 'M14 14h7v7h-7z'],
  list: ['M8 5h13', 'M8 12h13', 'M8 19h13', 'M3 5h.01', 'M3 12h.01', 'M3 19h.01'],
  arrow: ['M4 12h16', 'm14 6 6 6-6 6'], back: ['M20 12H4', 'm10 6-6 6 6 6'],
  chevron: ['m9 5 7 7-7 7'], down: ['m6 9 6 6 6-6'],
  book: ['M12 5v15', 'M12 5C9 3 5 3 2 4v15c3-1 7-1 10 1 3-2 7-2 10-1V4c-3-1-7-1-10 1Z'],
  folder: ['M3 7V4h6l2 3h10v13H3Z'], check: ['m5 12 4 4L19 6'],
  info: ['M12 11v6', 'M12 7h.01', 'M22 12A10 10 0 1 1 2 12a10 10 0 0 1 20 0'],
  file: ['M14 2H4v20h16V8Z', 'M14 2v6h6', 'M8 13h8', 'M8 17h6'],
  external: ['M14 3h7v7', 'M21 3 10 14', 'M10 3H3v18h18v-7'],
  send: ['m3 3 19 9-19 9 4-9Z', 'M7 12h15'],
  refresh: ['M21 4v6h-6', 'M3 20v-6h6', 'M20 10a8 8 0 0 0-14-5L3 8', 'M4 14a8 8 0 0 0 14 5l3-3'],
  close: ['m6 6 12 12', 'M18 6 6 18'], edit: ['m14 5 5 5', 'm4 15 12-12 5 5-12 12-6 1Z'],
};

function icon(name, size = 18) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  for (const [key, value] of Object.entries({ viewBox: '0 0 24 24', width: size, height: size, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.55, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true' })) svg.setAttribute(key, value);
  for (const path of paths[name] || paths.book) {
    const item = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    item.setAttribute('d', path); svg.append(item);
  }
  return svg;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  const hooks = { 'library-search': 'library-search', 'filter-format': 'library-format-filter', 'filter-status': 'library-status-filter', 'filter-author': 'library-author-filter', 'library-sort': 'library-sort', 'ask-question': 'ask-question' };
  if (hooks[attrs.id]) node.setAttribute('data-testid', hooks[attrs.id]);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = String(value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === 'value') node.value = value;
    else if (key === 'checked' || key === 'disabled' || key === 'hidden' || key === 'open') node[key] = value;
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat(Infinity)) if (child !== null && child !== undefined && child !== false) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}
function mount(...children) { main.replaceChildren(...children.filter((child) => child !== null && child !== undefined && child !== false)); }

function button(label, options = {}) {
  const { symbol, variant = '', onClick, href, ...attrs } = options;
  return el(href ? 'a' : 'button', { class: `button ${variant}`, ...(href ? { href } : { type: 'button', onclick: onClick }), ...attrs }, symbol ? icon(symbol, 16) : null, el('span', { text: label }));
}
function label(text, control) { return el('label', { class: 'sr-only', for: control.id, text }); }
function count(value) { return number.format(Number(value) || 0); }
function human(value) { return String(value ?? '').replace(/[_-]/g, ' ').replace(/^./, (char) => char.toUpperCase()); }
function authors(book) { return (Array.isArray(book?.authors) ? book.authors.join(', ') : String(book?.authors || '')).trim() || 'Author not identified'; }
function formatNames(book) { return [...new Set((Array.isArray(book?.formats) ? book.formats : []).map((item) => String(typeof item === 'string' ? item : item.format || '').toUpperCase()).filter(Boolean))]; }
function stringValue(value) { return typeof value === 'string' ? value : value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value); }
function itemId(item) { return String(item?.id ?? item?.jobId ?? ''); }
function titleOf(book) { return String(book?.title || 'Untitled book'); }
function dateLabel(value) {
  if (!value) return '';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value) : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}
function safeLocalURL(value) {
  if (typeof value !== 'string' || !value) return null;
  try { const url = new URL(value, location.origin); return url.origin === location.origin && ['http:', 'https:'].includes(url.protocol) ? url.href : null; } catch { return null; }
}
function locatorText(locator) {
  if (!locator) return 'Indexed passage';
  if (typeof locator === 'string') return locator;
  if (locator.label) return String(locator.label);
  const page = locator.page ?? locator.pageNumber ?? locator.page_number ?? locator.page_start;
  const end = locator.pageEnd ?? locator.page_end;
  if (page != null) return `Page ${page}${end && end !== page ? `–${end}` : ''}`;
  return String(locator.member ?? locator.href ?? locator.chapter ?? locator.section ?? stringValue(locator));
}
function warningsList(value) { return Array.isArray(value) ? value.map((item) => typeof item === 'string' ? item : item.message || item.detail || stringValue(item)) : value ? [stringValue(value)] : []; }
function notice(message, variant = '') { return el('div', { class: `notice ${variant}` }, icon('info', 17), el('p', { text: message })); }
function maintenanceDisabled() { return state.status?.capabilities?.maintenance === false; }
function statusTag(value) {
  const status = String(value || 'Unknown');
  const lower = status.toLowerCase();
  const variant = /error|fail/.test(lower) ? 'error' : /partial|warn|pending|processing|running|paused|unknown|queued/.test(lower) ? 'attention' : '';
  return el('span', { class: `status-tag ${variant}`, text: human(status) });
}
function toast(message) {
  const node = $('#toast'); clearTimeout(state.toastTimer); node.textContent = message; node.hidden = false;
  state.toastTimer = setTimeout(() => { node.hidden = true; }, 6000);
}
async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, { credentials: 'same-origin', ...options, headers: { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers }, ...(options.body && typeof options.body !== 'string' ? { body: JSON.stringify(options.body) } : {}) }); }
  catch { throw new Error('Your Spark could not be reached. Check the connection and try again.'); }
  let payload;
  try { payload = await response.json(); } catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
  if (!response.ok) throw new Error(String(payload.message || payload.error?.message || payload.error || `Request failed (${response.status}).`));
  return payload;
}

function nav() {
  const active = ['imports', 'ask', 'search', 'organize'].find(key => location.hash.startsWith(`#${key}`)) || 'library';
  $('#navigation').replaceChildren(...[['library', 'Library', 'library'], ['search', 'Find passages', 'search'], ['organize', 'Organize', 'folder'], ['imports', 'Imports', 'import'], ['ask', 'Ask your library', 'ask']].map(([key, name, symbol]) => el('a', { class: `nav-item ${key === active ? 'active' : ''}`, href: `#${key}`, title: name, 'aria-label': name, 'aria-current': key === active ? 'page' : null }, icon(symbol, 19), el('span', { text: name }), key === 'library' && state.status ? el('span', { class: 'nav-count', text: count(state.status.totals?.books) }) : null)));
}
function breadcrumb(...parts) {
  $('#breadcrumb').replaceChildren(...parts.flatMap((part, index) => [index ? el('span', { 'aria-hidden': 'true', text: '/' }) : null, typeof part === 'string' ? el('span', { text: part }) : el('a', { href: part.href, text: part.text })]).filter(Boolean));
}
function heading(eyebrow, title, description, actions = []) {
  return el('div', { class: 'page-heading' }, el('div', {}, el('p', { class: 'eyebrow', text: eyebrow }), el('h1', { class: 'page-title', text: title }), description ? el('p', { class: 'page-description', text: description }) : null), actions.length ? el('div', { class: 'heading-actions' }, actions) : null);
}
function loading(message = 'Loading…') { return el('div', { class: 'loading-state', role: 'status' }, el('span', { class: 'spinner', 'aria-hidden': 'true' }), el('p', { text: message })); }
function empty(title, description, action = null, symbol = 'book') { return el('div', { class: 'empty-state' }, el('div', { class: 'empty-symbol' }, icon(symbol, 27)), el('h2', { text: title }), el('p', { text: description }), action); }
function errorView(error, retry) { return el('div', { class: 'error-state', role: 'alert' }, el('h2', { text: 'Couldn’t open this view' }), el('p', { text: error.message }), button('Try again', { symbol: 'refresh', onClick: retry })); }
async function refreshStatus() {
  try {
    state.status = await api('/api/status');
    $('#connection').classList.remove('offline');
    $('#connection').replaceChildren(el('span', { class: 'connection-dot' }), el('span', { text: 'Connected to your Spark' }));
    nav();
    renderEmbeddingProgress();
    renderLibraryStats();
  } catch {
    $('#connection').classList.add('offline');
    $('#connection').replaceChildren(el('span', { class: 'connection-dot' }), el('span', { text: 'Spark connection unavailable' }));
  }
}
function renderLibraryStats() {
  const region = $('#library-stats'); if (!region) return;
  const totals = state.status?.totals || {};
  const values = [[totals.books, 'source books'], [totals.files, 'source files'], [totals.sections, 'sections'],
    [`${count(totals.embeddedChunks)} / ${count(totals.chunks)}`, 'chunks with vectors']];
  region.replaceChildren(...values.filter(([value]) => value !== undefined).map(([value, name]) => el('div', { class: 'stat' }, el('strong', { text: typeof value === 'string' ? value : count(value) }), el('span', { text: name }))));
}
function renderEmbeddingProgress() {
  const region = $('#embedding-progress'); if (!region) return;
  const model = state.status?.models?.embedding || {};
  const task = state.status?.models?.state || {};
  const total = Number(model.totalChunks || 0), embedded = Number(model.embeddedChunks || 0), failed = Number(model.failedChunks || 0);
  const busy = task.active === true;
  const run = async (retryFailed) => {
    try { await api('/api/embeddings', { method: 'POST', body: { retryFailed } }); await refreshStatus(); }
    catch (error) { toast(error.message); }
  };
  const summary = `${count(embedded)} / ${count(total)} chunks have vectors${failed ? ` · ${count(failed)} need attention` : ''}.`;
  region.replaceChildren(el('div', { class: 'panel-heading' }, el('h2', { text: 'Local search index' }), statusTag(busy ? 'running' : task.state && task.state !== 'idle' ? task.state : model.status || 'unavailable')),
    el('p', { class: 'field-help', text: summary }),
    model.configured ? el('div', { class: 'detail-actions' },
      button(busy ? 'Indexing…' : embedded ? 'Resume indexing' : 'Build search index', { disabled: maintenanceDisabled() || busy || total <= embedded + failed, symbol: 'refresh', 'data-testid': 'embedding-resume', onClick: () => run(false) }),
      failed ? button('Retry failed chunks', { disabled: maintenanceDisabled() || busy, variant: 'small', 'data-testid': 'embedding-retry', onClick: () => run(true) }) : null) : el('p', { class: 'field-help', text: 'Configure an installed local embedding model on the Spark to build vectors. Text search is available for extracted books.' }),
    maintenanceDisabled() ? el('p', { class: 'field-help', text: 'Index changes are disabled for this session. Existing passages remain searchable.' }) : null,
    ...(task.error || task.result?.error ? [notice(stringValue(task.error || task.result.error), 'error')] : []),
    ...warningsList(task.result?.warnings).map(warning => notice(warning)));
}
function scheduleStatusRefresh(epoch) {
  clearTimeout(state.statusPoll);
  state.statusPoll = setTimeout(async () => {
    if (epoch !== state.epoch) return;
    await refreshStatus();
    if (epoch === state.epoch) scheduleStatusRefresh(epoch);
  }, 4000);
}
function rememberBooks(books) { for (const book of books) state.seenBooks.set(String(book.id), book); }
function cover(book) {
  const colors = ['#45584b', '#816047', '#586a77', '#aa674c', '#746d56', '#484d63', '#827270', '#677562'];
  let hash = 0; for (const char of String(book.id || book.title)) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  const fallback = el('div', { class: 'cover-fallback' }, el('span', { class: 'cover-kicker', text: 'Your personal library' }), el('span', { class: 'cover-title', text: titleOf(book) }), el('span', { class: 'cover-line' }), el('span', { class: 'cover-author', text: authors(book) }));
  fallback.style.setProperty('--cover-bg', colors[hash % colors.length]);
  const artwork = el('div', { class: 'book-art', 'aria-hidden': 'true' }, fallback);
  const source = safeLocalURL(book.coverUrl);
  if (source) {
    const image = el('img', { src: source, alt: '', loading: 'lazy', decoding: 'async' });
    image.addEventListener('error', () => image.replaceWith(fallback), { once: true });
    artwork.replaceChildren(image);
  }
  return artwork;
}
function bookCard(book) {
  return el('article', { class: 'book-card', 'data-testid': 'book-card', 'data-book-id': book.id }, el('a', { class: 'book-card-link', href: book.groupId ? `#group/${encodeURIComponent(book.groupId)}` : `#book/${encodeURIComponent(book.id)}` }, cover(book), el('div', { class: 'book-copy' }, el('h2', { text: titleOf(book) }), el('p', { class: 'book-author', text: authors(book) }))), el('div', { class: 'book-card-meta' }, formatNames(book).map((format) => el('span', { class: 'format-tag', text: format })), statusTag(book.status), (book.metadataQuality?.needsReview || book.members?.some(member => member.metadataQuality?.needsReview)) ? el('span', {class: 'format-tag', text: 'Metadata needs review'}) : null));
}
function selectControl(id, name, entries, value, onChange) {
  const control = el('select', { id, class: 'filter-select', onchange: (event) => onChange(event.target.value) }, entries.map(([key, text]) => el('option', { value: key, text })));
  control.value = value;
  return [label(name, control), control];
}
function facetOptions(name) {
  const raw = state.status?.facets?.[name] || [];
  return (Array.isArray(raw) ? raw : Object.keys(raw)).map((item) => typeof item === 'string' ? item : item.value ?? item.name ?? item.author ?? item.status).filter(Boolean);
}

async function libraryPage(epoch) {
  await refreshStatus(); if (epoch !== state.epoch) return;
  breadcrumb('Library');
  const filters = state.library;
  const region = el('div', { id: 'catalog-results', 'aria-live': 'polite' }, loading('Finding your books…'));
  const summary = el('span', { class: 'filter-summary' });
  const queryInput = el('input', { id: 'library-search', type: 'search', value: filters.q, placeholder: 'Search your books…', autocomplete: 'off' });
  const searchForm = el('form', { class: 'search-field', role: 'search' }, icon('search', 17), label('Search books by title or author', queryInput), queryInput);
  let debounce;
  const changeFilter = (key, value) => { filters[key] = value; filters.offset = 0; persistViewState(); updateCatalog(); };
  queryInput.addEventListener('input', () => { filters.q = queryInput.value.trim(); filters.offset = 0; persistViewState(); clearTimeout(debounce); debounce = setTimeout(() => { if (region.isConnected && epoch === state.epoch) updateCatalog(); }, 250); });
  state.pageCleanup = () => clearTimeout(debounce);
  searchForm.addEventListener('submit', (event) => { event.preventDefault(); clearTimeout(debounce); changeFilter('q', queryInput.value.trim()); });
  const authorOptions = [...new Set([...facetOptions('authors'), ...[...state.seenBooks.values()].flatMap((book) => Array.isArray(book.authors) ? book.authors : book.authors ? [book.authors] : []), ...(filters.author ? [filters.author] : [])])].sort((a, b) => a.localeCompare(b));
  const statuses = [...new Set([...facetOptions('statuses'), ...[...state.seenBooks.values()].map((book) => book.status).filter(Boolean), ...(filters.status ? [filters.status] : [])])];
  const views = el('div', { class: 'view-switch', role: 'group', 'aria-label': 'Library layout' }, ...['grid', 'list'].map((view) => el('button', { type: 'button', class: `icon-button ${filters.view === view ? 'active' : ''}`, 'aria-label': `${human(view)} view`, 'aria-pressed': filters.view === view ? 'true' : 'false', title: `${human(view)} view`, onclick: (event) => { filters.view = view; persistViewState(); for (const item of views.children) { const active = item === event.currentTarget; item.classList.toggle('active', active); item.setAttribute('aria-pressed', String(active)); } const list = $('.book-grid, .book-list', region); if (list) list.className = `book-${view}`; } }, icon(view, 16))));
  const totals = state.status?.totals || {};
  const stats = [
    [totals.books, 'books'], [totals.sourceFiles ?? totals.files ?? totals.sources, 'source files'],
    [totals.sections, 'sections'],
  ].filter(([value]) => value !== undefined).map(([value, name]) => el('div', { class: 'stat' }, el('strong', { text: count(value) }), el('span', { text: name })));
  const embedding = state.status?.models?.embedding || {};
  const vectorCount = totals.vectorChunks ?? totals.embeddedChunks ?? totals.vectors ?? embedding.embeddedChunks ?? embedding.vectorChunks ?? embedding.coverage?.embedded;
  if (vectorCount !== undefined) stats.push(el('div', { class: 'stat' }, el('strong', { text: `${count(vectorCount)}${totals.chunks != null ? ` / ${count(totals.chunks)}` : ''}` }), el('span', { text: 'chunks with vectors' })));
  mount(heading('Your collection', 'A world on your bookshelf.', 'All your books, a little closer. Pick up a thought, find a passage, or start somewhere new.', [button('Import books', { href: '#imports', symbol: 'import', variant: 'primary' })]), el('div', { class: 'stats-row', id: 'library-stats' }, stats),
    el('div', { class: 'library-tools' }, searchForm, selectControl('library-sort', 'Sort books', [['title', 'Title · A to Z'], ['author', 'Author · A to Z'], ['recent', 'Recently added']], filters.sort, (value) => changeFilter('sort', value)), views),
    el('div', { class: 'filter-row' }, el('span', { class: 'filter-label', text: 'Browse by' }), selectControl('filter-format', 'Book format', [['', 'All formats'], ...['pdf', 'epub', 'mobi', 'prc'].map(format => [format, format.toUpperCase()])], filters.format, (value) => changeFilter('format', value)), selectControl('filter-author', 'Author', [['', 'All authors'], ...authorOptions.map((name) => [name, name])], filters.author, (value) => changeFilter('author', value)), selectControl('filter-status', 'Indexing status', [['', 'All statuses'], ...statuses.map((value) => [value, human(value)])], filters.status, (value) => changeFilter('status', value)), summary), region);
  let request = 0;
  async function updateCatalog() {
    const current = ++request;
    region.setAttribute('aria-busy', 'true');
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(filters)) if (key !== 'view' && value !== '') params.set(key, value);
    try {
      const data = await api(`/api/catalog?${params}`);
      if (epoch !== state.epoch || current !== request) return;
      const books = (data.items || []).map(card => {
        const member = card.members.find(item => card.matchingBookIds.includes(item.id)) || card.members[0];
        return { ...member, title: card.title, formats: Object.keys(card.formats), status: member.sectionCount ? 'completed' : 'queued', groupId: card.type === 'group' ? card.id : null };
      });
      const total = Number(data.total) || 0; rememberBooks(books);
      summary.textContent = `${count(total)} ${total === 1 ? 'book' : 'books'}${filters.q || filters.format || filters.author || filters.status ? ' found' : ' in your collection'}`;
      if (!books.length) {
        const filtered = Boolean(filters.q || filters.format || filters.author || filters.status);
        region.replaceChildren(empty(filtered ? 'No books on this shelf yet.' : 'Your next chapter starts here.', filtered ? 'Try a different search or clear your filters to see more of your collection.' : 'Import a folder of PDF and EPUB files from your Spark to begin building your library.', filtered ? button('Clear filters', { onClick: () => { Object.assign(filters, { q: '', format: '', author: '', status: '', offset: 0 }); persistViewState(); renderRoute(); } }) : button('Import your first books', { href: '#imports', symbol: 'import', variant: 'primary' })));
      } else {
        const previous = button('Previous', { symbol: 'back', variant: 'small', disabled: filters.offset <= 0, onClick: () => { filters.offset = Math.max(0, filters.offset - filters.limit); persistViewState(); updateCatalog(); } });
        const next = button('Next', { symbol: 'arrow', variant: 'small', disabled: filters.offset + books.length >= total, onClick: () => { filters.offset += filters.limit; persistViewState(); updateCatalog(); } });
        region.replaceChildren(el('div', { class: `book-${filters.view}` }, books.map(bookCard)), el('div', { class: 'pagination' }, el('span', { class: 'pagination-info', text: `Showing ${count(filters.offset + 1)}–${count(filters.offset + books.length)} of ${count(total)} books` }), el('div', { class: 'pagination-actions' }, previous, next)));
      }
      for (const [id, list, field] of [['filter-author', books.flatMap((book) => Array.isArray(book.authors) ? book.authors : book.authors ? [book.authors] : []), 'author'], ['filter-status', books.map((book) => book.status).filter(Boolean), 'status']]) {
        const select = $(`#${id}`); if (!select) continue;
        for (const value of [...new Set(list)]) if (![...select.options].some((option) => option.value === value)) select.append(el('option', { value, text: field === 'status' ? human(value) : value }));
      }
    } catch (error) { if (epoch === state.epoch && current === request) region.replaceChildren(errorView(error, updateCatalog)); }
    finally { if (epoch === state.epoch && current === request) region.setAttribute('aria-busy', 'false'); }
  }
  await updateCatalog();
  renderLibraryStats(); scheduleStatusRefresh(epoch);
}

async function getBook(id, fresh = false) {
  if (!fresh && state.bookCache.has(String(id))) return state.bookCache.get(String(id));
  const book = await api(`/api/books/${encodeURIComponent(id)}`);
  state.bookCache.set(String(id), book); state.seenBooks.set(String(id), book); return book;
}
function askAbout(book) { state.scope = { id: String(book.id), title: titleOf(book) }; persistViewState(); location.hash = '#ask'; }
function sourceLink(format, labelText = 'Open file') {
  const url = safeLocalURL(format.sourceUrl);
  return url ? button(labelText, { href: url, target: '_blank', rel: 'noopener', symbol: 'external', variant: 'small', 'aria-label': `${labelText} (${String(format.format || 'source').toUpperCase()}, opens in a new tab)` }) : el('span', { class: 'file-note', text: 'Source unavailable' });
}
async function bookPage(id, epoch) {
  const book = await getBook(id, true); if (epoch !== state.epoch) return;
  breadcrumb({ text: 'Library', href: '#library' }, titleOf(book));
  const toc = Array.isArray(book.toc) ? book.toc : [];
  const progress = book.readingProgress?.sectionId;
  const start = progress || book.sections?.[0]?.id || toc[0]?.id;
  const detail = el('div', { class: 'detail-info' }, el('p', { class: 'eyebrow', text: 'In your collection' }), el('h1', { class: 'page-title', text: titleOf(book) }), el('p', { class: 'detail-author', text: authors(book) }), el('div', { class: 'book-card-meta' }, formatNames(book).map((name) => el('span', { class: 'format-tag', text: name })), statusTag(book.status)), el('div', { class: 'detail-actions' }, start ? button(progress ? 'Continue reading' : 'Start reading', { href: `#read/${encodeURIComponent(start)}`, symbol: 'book', variant: 'primary' }) : null, button('Ask this book', { symbol: 'ask', onClick: () => askAbout(book) }), button('Edit details', { symbol: 'edit', onClick: () => editBook(book, editor) }), button('Reindex source', { symbol: 'refresh', variant: 'small', disabled: maintenanceDisabled(), title: maintenanceDisabled() ? 'Reindexing is disabled for this session.' : null, onClick: async () => {
    try { await api('/api/reindex', { method: 'POST', body: { bookIds: [book.id] } }); state.bookCache.clear(); toast('Reindexing the verified local source.'); location.hash = '#imports'; }
    catch (error) { toast(error.message); }
  } })));
  const metadata = book.metadata || {};
  const creators = Array.isArray(metadata.creators) ? metadata.creators.map(credit => `${credit.name} (${credit.role})`).join('; ') : '';
  const authorBasis = metadata.sourceMetadata?.recovery?.fields?.authors?.roleBasis;
  const fields = [['Author', authors(book)], ['Author credit', authorBasis], ['Creator credits', creators],
    ['Edition', metadata.edition || 'Edition not identified'], ['Publisher', metadata.publisher ?? book.publisher], ['Language', metadata.language ?? book.language], ['Published', metadata.published ?? metadata.publicationDate ?? metadata.date], ['Tags', Array.isArray(metadata.tags) ? metadata.tags.join(', ') : metadata.tags], ['Sections / chunks', `${count(book.sectionCount ?? toc.length)} sections · ${count(book.chunkCount)} chunks`], ['Book ID', book.id]].filter(([, value]) => value !== undefined && value !== null && value !== '');
  const editor = el('div', { id: 'metadata-editor' });
  const sources = (book.formats || []).filter((format) => typeof format === 'object');
  mount(button('Back to library', { href: '#library', symbol: 'back', variant: 'quiet small' }), el('div', { class: 'detail-hero', 'data-testid': 'book-detail', 'data-book-id': book.id }, cover(book), detail), editor,
    maintenanceDisabled() ? notice('Reindexing is disabled for this session. Reading, search and book details remain available.') : null,
    ...warningsList(book.warnings).map((warning) => notice(warning)),
    metadata.description || book.description ? el('p', { class: 'page-description detail-description', text: metadata.description || book.description }) : null,
    el('div', { class: 'detail-layout' }, el('section', { class: 'panel', 'aria-label': 'Table of contents' }, el('div', { class: 'panel-heading' }, el('h2', { text: 'Between the covers' }), el('span', { text: `${count(toc.length)} sections` })), toc.length ? el('ol', { class: 'toc', 'data-testid': 'book-toc' }, toc.map((section, index) => el('li', {}, el(section.unresolvedFragment ? 'span' : 'a', { href: tocURL(section), title: section.unresolvedFragment ? 'This fragment has no extracted-text location. Open its section from the reader.' : null, 'data-section-id': section.id }, el('span', { class: 'toc-number', text: String(index + 1).padStart(2, '0') }), el('span', { class: 'toc-title', text: section.title || `Section ${index + 1}` }), el('span', { class: 'toc-location', text: locatorText(section.locator) }), icon('chevron', 13))))) : empty('No readable sections yet', 'Check the import’s file dispositions for extraction or indexing details.')),
      el('aside', { class: 'detail-aside' }, el('section', { class: 'panel' }, el('div', { class: 'panel-heading' }, el('h2', { text: 'Book details' })), el('dl', { class: 'metadata' }, fields.map(([name, value]) => el('div', {}, el('dt', { text: name }), el('dd', { class: name === 'Book ID' ? 'mono' : '', text: value }))))), el('section', { class: 'panel' }, el('div', { class: 'panel-heading' }, el('h2', { text: 'Original files' }), el('span', { text: `${sources.length} sources` })), el('div', { class: 'source-files' }, sources.length ? sources.map((format) => el('div', { class: 'source-file' }, icon('file', 18), el('div', {}, el('strong', { text: String(format.format || 'File').toUpperCase() }), el('p', { class: 'file-note', text: human(format.status || 'Source file') })), sourceLink(format))) : el('p', { class: 'field-help', text: 'No source files available.' }))))));
}
function editBook(book, region) {
  if (region.childElementCount) { region.replaceChildren(); return; }
  const metadata = book.metadata || {};
  const form = el('form', { class: 'panel metadata-edit' });
  const inputs = {};
  for (const [key, name, value] of [['title', 'Title', titleOf(book)], ['authors', 'Authors (one per line)', Array.isArray(book.authors) ? book.authors.join('\n') : book.authors || ''], ['publisher', 'Publisher', metadata.publisher ?? book.publisher ?? ''], ['language', 'Language', metadata.language ?? book.language ?? ''], ['tags', 'Tags (comma separated)', Array.isArray(metadata.tags) ? metadata.tags.join(', ') : metadata.tags || ''], ['description', 'Description', metadata.description ?? book.description ?? '']]) {
    const input = el(key === 'description' || key === 'authors' ? 'textarea' : 'input', { id: `edit-${key}`, class: 'field-input', value, ...(key === 'title' ? { required: true, maxlength: 1000 } : {}), ...(key === 'description' ? { rows: 4 } : {}), ...(key === 'authors' ? { rows: 2 } : {}) });
    inputs[key] = input;
    form.append(el('div', { class: key === 'description' ? 'edit-wide' : '' }, el('label', { class: 'field-label', for: input.id, text: name }), input));
  }
  const errorRegion = el('p', { class: 'inline-error edit-wide', role: 'alert' });
  const save = button('Save details', { type: 'submit', variant: 'primary', symbol: 'check' });
  form.append(el('div', { class: 'edit-wide detail-actions' }, save, button('Cancel', { onClick: () => region.replaceChildren() })), errorRegion);
  form.addEventListener('submit', async (event) => {
    event.preventDefault(); errorRegion.textContent = ''; save.disabled = true;
    const payload = Object.fromEntries(Object.entries(inputs).map(([key, input]) => [key, input.value.trim()]));
    payload.authors = payload.authors.split('\n').map((value) => value.trim()).filter(Boolean);
    payload.tags = payload.tags.split(',').map((value) => value.trim()).filter(Boolean);
    if (!payload.title) { errorRegion.textContent = 'Enter a title for this book.'; save.disabled = false; return; }
    try { await api(`/api/books/${encodeURIComponent(book.id)}`, { method: 'PUT', body: payload }); state.bookCache.delete(String(book.id)); toast('Book details saved.'); if (region.isConnected) renderRoute(); }
    catch (error) { errorRegion.textContent = error.message; save.disabled = false; }
  });
  region.replaceChildren(form); inputs.title.focus();
}

function tocURL(item) {
  const locator = item.locator?.derived || item.locator || {};
  const original = ['pdf', 'epub'].includes(item.locator?.format);
  if (item.unresolvedFragment && (!original || !locator.heading)) return null;
  return readerURL(item.id, original ? 'original' : 'extracted', { start: item.charStart, end: item.charEnd, fragment: locator.heading, focus: 1 });
}

function sectionText(text, selection) {
  const body = el('div', { class: 'section-text' });
  const start = Number(selection.get('start')); const end = Number(selection.get('end'));
  if (selection.has('start') && selection.has('end') && Number.isSafeInteger(start) && Number.isSafeInteger(end) && start >= 0 && end > start && end <= text.length) {
    body.append(document.createTextNode(text.slice(0, start)), el('mark', { class: 'citation-highlight', id: 'cited-passage', tabindex: '-1', text: text.slice(start, end) }), document.createTextNode(text.slice(end)));
  } else body.textContent = text || 'This section contains no extracted text.';
  return body;
}
function readerLocator(section) { return section.locator?.derived || section.locator || {}; }
function readerURL(sectionId, mode, values = {}) {
  const params = new URLSearchParams({ view: mode });
  for (const key of ['start', 'end', 'fragment', 'focus']) if (values[key] !== undefined && values[key] !== null && values[key] !== '') params.set(key, values[key]);
  return `#read/${encodeURIComponent(sectionId)}?${params}`;
}
function sourceDescription(book, section) {
  const locator = readerLocator(section);
  const edition = [book.metadata?.edition, book.publisher, book.metadata?.publicationDate].filter(Boolean).join(' · ');
  const file = book.formats?.find(file => typeof file === 'object' && file.format === section.locator?.format);
  return [edition || 'Selected source edition', file?.name || `Source ${book.id}`, locator.page != null ? `Physical page ${locator.page}` : `EPUB section: ${section.title || locator.member || 'Current section'}`, locator.member].filter(Boolean).join(' · ');
}
async function readerPage(id, epoch, selection = new URLSearchParams()) {
  const section = await api(`/api/sections/${encodeURIComponent(id)}`);
  const book = await getBook(section.bookId); if (epoch !== state.epoch) return;
  const locator = readerLocator(section), supported = ['pdf', 'epub'].includes(section.locator?.format);
  const saved = state.readerStates[id] || { mode: supported ? 'original' : 'extracted', original: {}, extracted: {} };
  state.readerStates[id] = saved;
  let mode = ['original', 'extracted'].includes(selection.get('view')) ? selection.get('view') : saved.mode;
  if (!supported) mode = 'extracted';
  saved.mode = mode;
  const range = { start: selection.get('start'), end: selection.get('end'), fragment: selection.get('fragment') };
  const hasRange = selection.has('start') && selection.has('end');
  const selectedRange = JSON.stringify(range);
  const freshSelection = (hasRange || !!range.fragment) && (selection.get('focus') === '1' || saved.selectedRange !== selectedRange);
  if (freshSelection) { saved.original = {}; saved.extracted = {}; }
  saved.selectedRange = selectedRange;
  let original = null, disposed = false, viewSequence = 0;
  breadcrumb({ text: 'Library', href: '#library' }, { text: titleOf(book), href: `#book/${encodeURIComponent(book.id)}` }, section.title || 'Reader');
  const body = el('div', { class: 'reader-view-body', 'data-testid': 'reader-view-body' });
  const source = el('p', { class: 'reader-scope-description', 'data-testid': 'reader-source-identity', text: sourceDescription(book, section) });
  const capture = () => {
    if (mode === 'original' && original) saved.original = original.capturePosition();
    if (mode === 'extracted') saved.extracted = { scrollY: window.scrollY, scrollX: window.scrollX };
    saved.mode = mode; state.readerModes[book.id] = mode;
    state.readerReturn = readerURL(id, mode, range);
    persistViewState();
  };
  state.readerCapture = capture;
  state.pageCleanup = () => { capture(); disposed = true; original?.destroy(); if (state.readerCapture === capture) state.readerCapture = null; };
  const switches = el('div', { class: 'reader-view-switch', role: 'group', 'aria-label': 'Reader view' });
  const originalButton = button(section.locator?.format === 'pdf' ? 'Original PDF' : 'Original EPUB', { 'data-testid': 'reader-original', disabled: !supported, onClick: () => changeMode('original') });
  const extractedButton = button('Extracted text', { 'data-testid': 'reader-extracted', onClick: () => changeMode('extracted') });
  switches.append(originalButton, extractedButton);
  const reading = el('article', { class: 'reading-sheet', 'data-testid': 'reader-content', 'data-book-id': section.bookId, 'data-section-id': section.id, 'data-physical-page': locator.page, 'data-member': locator.member }, el('p', { class: 'eyebrow', text: titleOf(book) }), el('h1', { text: section.title || 'Untitled section' }), source, switches, body,
    el('div', { class: 'reader-pagination' }, section.previousId ? button('Previous section', { href: readerURL(section.previousId, mode), symbol: 'back', 'data-testid': 'reader-previous' }) : el('span', { class: 'field-help', text: 'Beginning of book' }), section.nextId ? button('Next section', { href: readerURL(section.nextId, mode), symbol: 'arrow', 'data-testid': 'reader-next' }) : el('span', { class: 'field-help', text: 'End of book' })));
  const toc = Array.isArray(book.toc) ? book.toc : [];
  const tocDetails = el('details', { class: 'reader-toc', 'data-testid': 'reader-contents', open: window.matchMedia('(min-width:851px)').matches }, el('summary', { text: 'Contents' }), el('nav', { 'aria-label': 'Book sections' }, toc.map((item, index) => {
    const target = item.locator?.derived || item.locator || {};
    const url = item.unresolvedFragment && !target.heading ? null : readerURL(item.id, mode, { start: item.charStart, end: item.charEnd, fragment: target.heading, focus: 1 });
    return el(url ? 'a' : 'span', { href: url, class: item.id === section.id ? 'active' : '', text: `${String(index + 1).padStart(2, '0')}  ${item.title || `Section ${index + 1}`}` });
  })));
  const askCurrent = () => {
    capture();
    const position = mode === 'original' ? saved.original : {};
    const fragment = position?.fragment || range.fragment;
    state.scope = { id: String(book.id), title: titleOf(book), detail: sourceDescription(book, section), readerContext: {
      bookId: String(book.id), sectionId: section.id, scope: 'section',
      ...(locator.page != null ? { page: locator.page } : {}), ...(locator.member ? { member: locator.member } : {}),
      ...(fragment ? { fragment } : {}), ...(hasRange ? { charStart: Number(range.start), charEnd: Number(range.end) } : {}),
    } };
    persistViewState(); location.hash = '#ask';
  };
  mount(el('div', { class: 'reader-heading' }, el('div', { class: 'reader-context' }, button(titleOf(book), { href: `#book/${encodeURIComponent(book.id)}`, symbol: 'back', variant: 'quiet small', 'data-testid': 'reader-back' })), el('div', { class: 'reader-actions' }, state.citationOrigin === '#search' && state.contentSearch.result ? button('Back to search', { href: '#search', symbol: 'back', variant: 'small', 'data-testid': 'reader-return-search' }) : state.turns.length || state.scope?.readerContext ? button('Back to conversation', { href: '#ask', symbol: 'back', variant: 'small', 'data-testid': 'reader-return-ask' }) : null,
    button(locator.page != null ? 'Ask about this page' : 'Ask about this section', { symbol: 'ask', 'data-testid': 'reader-ask-context', onClick: askCurrent }), button('Ask this book', { variant: 'small', onClick: () => { capture(); askAbout(book); } }))), warningsList(book.warnings).length ? el('details', { class: 'reader-source-notes', 'data-testid': 'reader-source-notes' }, el('summary', { text: `Source notes (${warningsList(book.warnings).length})` }), ...warningsList(book.warnings).map(warning => notice(warning))) : null, el('div', { class: 'reader-layout' }, tocDetails, reading));
  async function showView() {
    const sequence = ++viewSequence;
    reading.setAttribute('data-view-ready', 'false');
    originalButton.setAttribute('aria-pressed', String(mode === 'original')); extractedButton.setAttribute('aria-pressed', String(mode === 'extracted'));
    reading.setAttribute('data-reader-mode', mode);
    for (const [testId, sectionId] of [['reader-previous', section.previousId], ['reader-next', section.nextId]]) if (sectionId) reading.querySelector(`[data-testid="${testId}"]`)?.setAttribute('href', readerURL(sectionId, mode));
    for (const link of tocDetails.querySelectorAll('a')) { const url = new URL(link.href); const [route, query = ''] = url.hash.split('?'); const params = new URLSearchParams(query); params.set('view', mode); link.href = `${route}?${params}`; }
    original?.destroy(); original = null;
    if (mode === 'extracted') {
      body.replaceChildren(sectionText(String(section.text || ''), selection));
      if (hasRange && !Number.isFinite(saved.extracted?.scrollY)) { const mark = $('#cited-passage'); mark?.scrollIntoView({ block: 'center' }); mark?.focus({ preventScroll: true }); }
      else window.scrollTo(saved.extracted?.scrollX || 0, saved.extracted?.scrollY || 0);
      reading.setAttribute('data-view-ready', 'true');
      return;
    }
    body.replaceChildren(loading('Opening the original source…'));
    try {
      const { createOriginalView } = await import('/reader-views.mjs');
      const view = await createOriginalView({ section, book, position: { ...saved.original, page: locator.page, member: locator.member, ...(range.fragment ? { fragment: range.fragment } : {}), ...(hasRange ? { charStart: Number(range.start), charEnd: Number(range.end) } : {}) },
        onPosition: position => { saved.original = position; }, onNavigate: target => {
          const match = book.sections?.find(item => { const value = readerLocator(item); return target.sectionId ? item.id === target.sectionId : target.page != null ? value.page === target.page : value.member === target.member; });
          if (match) { capture(); location.hash = readerURL(match.id, 'original', { fragment: target.fragment, focus: 1 }); }
        } });
      if (disposed || epoch !== state.epoch || sequence !== viewSequence) { view.destroy(); return; }
      original = view; body.replaceChildren(view.element); await view.ready;
      if (!disposed && sequence === viewSequence) { reading.setAttribute('data-view-ready', 'true'); capture(); }
    } catch (error) { if (!disposed && sequence === viewSequence) { reading.setAttribute('data-view-ready', 'error'); body.replaceChildren(notice(`The original view could not be opened: ${error.message}. You can read the extracted text.`, 'error')); } }
  }
  function changeMode(next) {
    if (next === mode) return;
    capture(); mode = next; saved.mode = next;
    history.replaceState(null, '', readerURL(id, mode, range)); state.route = location.hash;
    persistViewState(); void showView();
  }
  await showView();
  try { await api(`/api/books/${encodeURIComponent(book.id)}/progress`, { method: 'POST', body: { sectionId: section.id } }); book.readingProgress = { sectionId: section.id }; }
  catch { if (epoch === state.epoch) reading.append(el('p', { class: 'field-help', role: 'status', text: 'Your place could not be saved. You can still continue reading.' })); }
}

function rootPaths() { return (state.status?.importRoots || []).map((value) => typeof value === 'string' ? value : value.path || value.root).filter(Boolean); }
function jobCounts(job) {
  const counts = job.counts || job.progress || {};
  const total = Number(job.totalFiles ?? job.total ?? counts.totalFiles ?? counts.total ?? job.files?.length ?? 0);
  const done = Number(job.processedFiles ?? job.processed ?? counts.processedFiles ?? counts.processed ?? counts.completed ?? counts.done ?? 0);
  return { total: Number.isFinite(total) ? total : 0, done: Number.isFinite(done) ? done : 0 };
}
function jobStatus(job) { return job.state || job.status || 'unknown'; }
function activeJob(job) { return /running|queued|pending|processing|scanning|importing/i.test(jobStatus(job)); }
async function loadJob(id) {
  const response = await api(`/api/imports/${encodeURIComponent(id)}?${new URLSearchParams({ limit: '100', offset: String(state.jobOffsets.get(id) || 0) })}`);
  const detail = response.job ? { ...response.job, files: response.files || response.job.files } : response;
  state.jobDetails.set(id, detail); return detail;
}
function renderJob(job, region) {
  const id = itemId(job); const detail = state.jobDetails.get(id); const data = detail ? { ...detail, ...job, files: detail.files } : job;
  const { total, done } = jobCounts(data);
  const path = data.root || data.path || data.sourceRoot || `Import ${id}`;
  const files = Array.isArray(data.files) ? data.files : [];
  const progress = el('progress', { max: Math.max(total, 1), 'aria-label': 'Files processed', 'data-testid': 'import-progress', ...(total ? { value: Math.min(done, total) } : activeJob(data) ? {} : { value: 0 }) });
  const details = el('details', { class: 'file-dispositions', open: state.openJobs.has(id) }, el('summary', { text: `File dispositions${files.length ? ` (${count(files.length)})` : ''}` }));
  const fileRegion = el('div'); details.append(fileRegion);
  function showFiles() {
    if (!files.length) { fileRegion.replaceChildren(el('p', { class: 'field-help', text: activeJob(data) ? 'Files will appear as this folder is scanned.' : 'No file dispositions returned for this import.' })); return; }
    const offset = state.jobOffsets.get(id) || 0;
    const page = async (nextOffset) => {
      state.jobOffsets.set(id, nextOffset);
      try { await loadJob(id); if (region.isConnected) region.replaceWith(renderJob(job, region)); }
      catch (error) { toast(error.message); }
    };
    fileRegion.replaceChildren(el('div', { class: 'table-scroll' }, el('table', { class: 'file-table' }, el('thead', {}, el('tr', {}, ['File', 'Disposition', 'Details'].map((text) => el('th', { scope: 'col', text })))), el('tbody', {}, files.map((file) => el('tr', {}, el('td', { text: file.relative_path || file.path || file.name || file.sourcePath || file.filename || file.id || 'Unnamed file' }), el('td', {}, statusTag(file.disposition || file.status)), el('td', {}, el('span', { text: stringValue(file.message || file.error || file.reason || file.detail || '') }), file.bookId || file.book_id ? el('div', {}, el('a', { href: `#book/${encodeURIComponent(file.bookId || file.book_id)}`, text: 'Open book' })) : null)))))));
    fileRegion.append(el('div', { class: 'pagination' }, el('span', { class: 'pagination-info', text: `Files ${count(offset + 1)}–${count(offset + files.length)}${total ? ` of ${count(total)}` : ''}` }), el('div', { class: 'pagination-actions' }, button('Previous files', { variant: 'small', disabled: offset === 0, onClick: () => page(Math.max(0, offset - 100)) }), button('Next files', { variant: 'small', disabled: total ? offset + files.length >= total : files.length < 100, onClick: () => page(offset + 100) }))));
  }
  details.addEventListener('toggle', async () => {
    if (details.open) {
      state.openJobs.add(id);
      if (!detail) {
        fileRegion.replaceChildren(loading('Loading file dispositions…'));
        try { await loadJob(id); if (region.isConnected) region.replaceWith(renderJob(job, region)); }
        catch (error) { fileRegion.replaceChildren(notice(error.message, 'error')); }
      }
    } else state.openJobs.delete(id);
  });
  if (detail || files.length) showFiles();
  const resume = button('Resume import', { symbol: 'refresh', variant: 'small', disabled: maintenanceDisabled(), 'data-testid': 'import-resume', onClick: async () => {
    resume.disabled = true;
    try { await api(`/api/imports/${encodeURIComponent(id)}/resume`, { method: 'POST', body: {} }); state.jobDetails.delete(id); toast('Import resumed.'); await refreshImports(); }
    catch (error) { toast(error.message); resume.disabled = maintenanceDisabled(); }
  } });
  const pause = button('Pause import', { variant: 'small', disabled: maintenanceDisabled(), onClick: async () => {
    pause.disabled = true;
    try { await api(`/api/imports/${encodeURIComponent(id)}/pause`, { method: 'POST', body: {} }); toast('Pause requested. The current file will stop safely.'); await refreshImports(); }
    catch (error) { toast(error.message); pause.disabled = maintenanceDisabled(); }
  } });
  const card = el('article', { class: 'panel import-job', 'data-job-id': id, 'data-testid': 'import-job' }, el('div', { class: 'job-heading' }, el('div', { class: 'job-icon' }, icon('folder', 19)), el('div', { class: 'job-title' }, el('h3', { text: path }), el('p', { text: [dateLabel(data.createdAt || data.startedAt || data.started_at), id ? `Import ${id}` : ''].filter(Boolean).join(' · ') })), statusTag(jobStatus(data))), el('div', { class: 'job-progress' }, progress, el('span', { text: total ? `${count(done)} / ${count(total)} source files` : activeJob(data) ? 'Scanning source files…' : 'No source files counted' })), el('div', { class: 'job-footer' }, el('span', { text: data.message || (activeJob(data) ? 'You can leave this page. Importing continues on your Spark.' : 'Review each file below for the complete import result.') }), data.canPause ? pause : data.canResume ? resume : data.externalOwner ? el('span', { class: 'field-help', text: 'Running in another import process.' }) : null), data.recoveryNotice ? notice(data.recoveryNotice) : null, ...warningsList(data.warnings).map((warning) => notice(warning)), data.error ? notice(stringValue(data.error), 'error') : null, details);
  region = card;
  return card;
}
async function refreshImports() {
  const epoch = state.epoch;
  const region = $('#import-jobs'); if (!region) return;
  try {
    const data = await api('/api/imports');
    await refreshStatus();
    if (epoch !== state.epoch || !region.isConnected) return;
    state.jobs = Array.isArray(data.items) ? data.items : [];
    for (const job of state.jobs) if (state.openJobs.has(itemId(job))) {
      try { await loadJob(itemId(job)); } catch { /* Keep the last successfully returned disposition list. */ }
    }
    if (epoch !== state.epoch || !region.isConnected) return;
    // Preserve focus on a job action while polling. Its next refresh happens after focus leaves.
    state.deferredJobs = region.contains(document.activeElement);
    if (!state.deferredJobs) paintImportJobs(region);
    $('#import-job-count').textContent = `${count(state.jobs.length)} ${state.jobs.length === 1 ? 'import' : 'imports'}`;
    clearTimeout(state.poll);
    if (state.jobs.some(activeJob)) state.poll = setTimeout(refreshImports, 3000);
  } catch (error) { if (epoch === state.epoch && region.isConnected) region.replaceChildren(errorView(error, refreshImports)); }
}
function paintImportJobs(region) {
  region.replaceChildren(...(state.jobs.length ? state.jobs.map((job) => renderJob(job, null)) : [empty('A fresh start.', 'Your import history will appear here, with a disposition for every source file.', null, 'folder')]));
  state.deferredJobs = false;
}
async function importsPage(epoch) {
  await refreshStatus(); if (epoch !== state.epoch) return;
  breadcrumb('Imports');
  const roots = rootPaths();
  const input = el('input', { id: 'import-root', class: 'field-input', type: 'text', disabled: maintenanceDisabled(), value: state.importRoot || (roots.length === 1 ? roots[0] : ''), placeholder: roots[0] || '/path/to/your/books', required: true, list: 'import-roots', autocomplete: 'off', spellcheck: 'false' });
  input.addEventListener('input', () => { state.importRoot = input.value; });
  const start = button('Import folder', { type: 'submit', symbol: 'import', variant: 'primary', disabled: maintenanceDisabled() });
  const error = el('p', { class: 'inline-error', role: 'alert' });
  const form = el('form', { class: 'panel import-form' }, el('label', { class: 'field-label', for: input.id, text: 'Folder on your Spark' }), el('div', { class: 'input-action' }, input, start), el('datalist', { id: 'import-roots' }, roots.map((path) => el('option', { value: path }))), el('p', { class: 'field-help', text: 'Stage your PDF and EPUB files in an allowed folder on the Spark, then enter its path. This is a server folder, not a folder on this browser’s device.' }), roots.length ? el('p', { class: 'field-help', text: `Allowed roots: ${roots.join(' · ')}` }) : el('p', { class: 'field-help', text: 'The server validates the folder against its configured import roots.' }), error);
  form.addEventListener('submit', async (event) => {
    event.preventDefault(); error.textContent = ''; if (maintenanceDisabled()) return; const root = input.value.trim(); if (!root) return;
    start.disabled = true;
    try { const result = await api('/api/imports', { method: 'POST', body: { root } }); state.importRoot = root; const id = itemId(result.job || result); if (id) state.openJobs.add(id); toast('Import started. Your Spark will keep track of each file.'); if (epoch === state.epoch) await refreshImports(); refreshStatus(); }
    catch (failure) { error.textContent = failure.message; }
    finally { start.disabled = maintenanceDisabled(); }
  });
  mount(maintenanceDisabled() ? heading('Your library', 'Import history', 'Imports, reindexing and index changes are disabled for this session. You can keep reading, searching and organizing books.') : heading('Make room for new ideas', 'Bring your books along.', 'Import a folder once. Pick up where you left off, and see exactly what happened to every file.'), el('div', { class: 'import-intro' }, form, el('aside', { class: 'import-side-note' }, icon('book', 24), el('h2', { text: 'A shelf, not a duplicate pile.' }), el('p', { text: 'PDF and EPUB sources stay traceable. File dispositions show what was added, skipped, or needs attention.' }))), el('section', { class: 'panel', id: 'embedding-progress', 'aria-label': 'Local search index' }), el('div', { class: 'section-heading' }, el('h2', { text: 'Import history' }), el('p', { id: 'import-job-count' })), el('div', { id: 'import-jobs', class: 'import-jobs' }, loading('Loading import history…')));
  const jobs = $('#import-jobs');
  jobs.addEventListener('focusout', () => queueMicrotask(() => {
    if (jobs.isConnected && state.deferredJobs && !jobs.contains(document.activeElement)) paintImportJobs(jobs);
  }));
  renderEmbeddingProgress(); scheduleStatusRefresh(epoch);
  await refreshImports();
}

function modelDescription() {
  const models = state.status?.models;
  if (!models) return 'Model status unavailable';
  const model = models.chat || models.generation || models.llm || (Array.isArray(models) ? models.find((item) => /chat|generation|llm/.test(item.role || item.type || '')) : null);
  if (!model) return 'Local model status unavailable';
  if (typeof model === 'string') return `${model} · configured`;
  const name = model.name || model.model || model.id || 'Local model';
  if (model.reachable === true || model.available === true || model.status === 'ready') return `${name} · reachable`;
  if (model.configured === false) return 'No chat model configured';
  if (model.reachable === false || model.available === false) return `${name} · unavailable`;
  return `${name} · configured`;
}
function openCitation(hit) {
  const sectionId = hit.sectionId ?? hit.section_id;
  if (!sectionId) return;
  state.readerCapture?.();
  state.citationOrigin = location.hash;
  const locator = hit.locator?.derived || hit.locator || {};
  const saved = state.readerStates[sectionId];
  const mode = state.readerModes[hit.bookId] || saved?.mode || 'extracted';
  location.hash = readerURL(sectionId, mode, { start: locator.charStart ?? hit.locator?.charStart, end: locator.charEnd ?? hit.locator?.charEnd, fragment: Number.isSafeInteger(locator.charStart ?? hit.locator?.charStart) ? undefined : locator.fragment, focus: 1 });
}
function citationButton(hit, index) {
  const sectionId = hit.sectionId ?? hit.section_id;
  return el('button', { type: 'button', class: 'citation', 'data-testid': 'citation-link', 'data-section-id': sectionId, 'data-book-id': hit.bookId, disabled: !sectionId, 'aria-label': sectionId ? `Read source ${index + 1}: ${hit.title || 'Book'}, ${locatorText(hit.locator)}` : `Source ${index + 1}: reader section unavailable`, onclick: () => openCitation(hit) }, el('span', { class: 'citation-number', text: hit.citation ?? index + 1 }), el('span', { class: 'citation-body' }, el('span', { class: 'citation-title', text: hit.title || 'Source passage' }), el('span', { class: 'citation-locator', text: `${locatorText(hit.locator)}${hit.bookId ? ` · Book ${hit.bookId}` : ''}` }), hit.text ? el('span', { class: 'citation-excerpt', text: hit.text }) : null), icon('arrow', 15));
}
function answerWithCitations(answer, citations) {
  const body = el('div', { class: 'answer-text', 'data-testid': 'ask-answer' });
  let start = 0;
  for (const match of String(answer).matchAll(/\[(\d+)\]/g)) {
    body.append(document.createTextNode(String(answer).slice(start, match.index)));
    const hit = citations.find((item, index) => Number(item.citation ?? index + 1) === Number(match[1]));
    if (hit) body.append(el('button', { type: 'button', class: 'inline-citation', 'data-testid': 'inline-citation', 'aria-label': `Read supporting source ${match[1]}`, text: match[0], onclick: () => openCitation(hit) }));
    else body.append(document.createTextNode(match[0]));
    start = match.index + match[0].length;
  }
  body.append(document.createTextNode(String(answer).slice(start))); return body;
}

function quotedCitation(hit, index) {
  const link = citationButton(hit, index);
  link.classList.add('quoted-citation');
  link.setAttribute('data-testid', 'excerpt-link');
  if (hit.omittedBefore || hit.omittedAfter) {
    link.querySelector('.citation-body').append(el('span', { class: 'excerpt-boundary',
      text: hit.omittedBefore && hit.omittedAfter ? 'Text continues before and after this excerpt.'
        : hit.omittedBefore ? 'Earlier text is outside this excerpt.' : 'Text continues after this excerpt.' }));
  }
  return link;
}
function renderTurn(turn) {
  const question = el('div', { class: 'chat-question' }, el('span', { class: 'chat-question-scope', text: turn.scope ? `Asked in ${turn.scope.title}${turn.scope.readerContext ? ` · ${turn.scope.detail || 'Current section'}` : ''}` : 'Asked across your library' }), turn.question);
  let content;
  if (turn.pending) content = el('div', { role: 'status' }, loading('Reading the relevant passages…'));
  else if (turn.error) content = notice(turn.error, 'error');
  else {
    const response = turn.response || {};
    const excerptSources = response.responseKind === 'source_excerpts' && Array.isArray(response.extractive?.passages);
    const excerpts = excerptSources && response.status === 'excerpts';
    const noEvidence = response.answerStatus === 'no_answer' || ['no_evidence', 'insufficient_evidence'].includes(response.responseKind);
    const unavailable = response.responseKind === 'model_unavailable' || response.status === 'model_unavailable' || response.status === 'local_ai_unavailable' || response.answerStatus === 'unavailable';
    const supporting = Array.isArray(response.extractive?.passages) ? response.extractive.passages : [];
    const generated = response.responseKind === 'generated' || response.responseKind === 'generated_answer' || response.responseKind === 'answer' || (!response.abstained && !!response.answer);
    const citations = excerptSources || unavailable ? supporting : Array.isArray(response.citations) && response.citations.length ? response.citations : noEvidence ? supporting : [];
    const answer = excerpts ? 'These are source excerpts. No generated answer was requested for this response.'
      : unavailable ? (response.answer || 'The local answer model is unavailable. A generated answer could not be produced. The retrieved excerpts below are available to inspect.')
      : noEvidence || response.abstained ? (response.answer || 'I couldn’t find enough evidence in the selected sources to answer this question.')
      : response.answer || 'The local model returned no answer. Inspect the source passages or try another question.';
    const timing = typeof response.timing === 'number' ? `${response.timing.toFixed(1)} ms` : response.timing?.totalMs != null ? `${Math.round(response.timing.totalMs)} ms` : '';
    const title = excerpts ? 'Source excerpts only' : unavailable ? 'Local answer unavailable' : noEvidence || response.abstained ? 'More evidence needed' : 'Answer from your library';
    content = el('div', { class: 'chat-response', 'data-response-kind': response.responseKind, 'data-generated-answer': String(generated && !unavailable && !noEvidence && !excerpts && !response.abstained), 'data-answer-status': response.answerStatus || (unavailable ? 'unavailable' : response.abstained ? 'no_answer' : 'answered') },
      el('div', { class: 'response-heading' }, icon(excerpts ? 'book' : 'ask', 17), el('span', { text: title }),
        el('span', { class: 'response-mode', text: [response.mode ? human(response.mode) : '', typeof response.model === 'string' ? response.model : response.model?.name, timing].filter(Boolean).join(' · ') })),
      answerWithCitations(answer, generated ? citations : []),
      citations.length ? el('p', { class: 'citation-label', text: `${citations.length} ${citations.length === 1 ? 'source passage' : 'source passages'} · open to read in context` }) : null,
      askBookGroups(response, citations, excerptSources || unavailable),
      ...warningsList(response.warnings).map((warning) => el('p', { class: 'response-warning', text: warning })),
      excerptSources ? el('p', { class: 'response-warning', text: response.extractive.limitations })
        : response.limitations ? el('p', { class: 'response-warning', text: response.limitations }) : null);
  }
  return el('article', { class: 'chat-turn', 'data-testid': turn.pending ? 'ask-pending' : 'ask-result' }, question, content);
}
function askBookGroups(response, citations, excerpts) {
  const books = new Map((response.books || []).map(book => [String(book.id), book]));
  const groups = new Map();
  for (const hit of [...(response.hits || []), ...citations]) {
    if (!hit.bookId) continue;
    const id = String(hit.bookId);
    const book = books.get(id) || state.seenBooks.get(id) || { id, title: hit.title || 'Untitled book', authors: hit.authors || [], coverUrl: `/api/books/${encodeURIComponent(id)}/thumbnail` };
    const identity = book.identity?.id || id;
    if (!groups.has(identity)) groups.set(identity, { book, members: new Map() });
    const group = groups.get(identity); group.members.set(id, book);
    if ((group.book.cover?.kind || group.book.thumbnail?.kind) === 'fallback'
        && (book.cover?.kind || book.thumbnail?.kind) && (book.cover?.kind || book.thumbnail?.kind) !== 'fallback') group.book = book;
  }
  if (!groups.size) return null;
  return el('div', { class: 'ask-book-groups' }, [...groups.values()].map(group => {
    const book = group.book;
    const formats = [...new Set([...group.members.values()].flatMap(formatNames))];
    const passages = citations.flatMap((hit, index) => group.members.has(String(hit.bookId)) ? [[hit, index]] : []);
    return el('section', { class: 'ask-book-group', 'data-testid': 'ask-source-book', 'data-book-id': book.id,
      'data-source-book-ids': [...group.members.keys()].join(' ') },
      el('a', { class: 'ask-book-identity', href: `#book/${encodeURIComponent(book.id)}` },
        el('div', { class: 'ask-source-cover' }, cover(book)),
        el('div', { class: 'ask-source-copy' }, el('h3', { text: titleOf(book) }),
          el('p', { text: authors(book) }), el('span', { class: 'format-tag', text: formats.join(' / ') || 'Source book' }),
          (book.cover?.kind || book.thumbnail?.kind) === 'fallback' ? el('span', { class: 'ask-cover-note', text: 'No cover available' }) : null)),
      passages.length ? el('div', { class: 'citations' }, passages.map(([hit, index]) => (excerpts ? quotedCitation : citationButton)(hit, index))) : null);
  }));
}
function askPage() {
  breadcrumb('Ask your library');
  const conversation = el('div', { class: 'chat-messages', id: 'chat-messages', 'aria-live': 'polite', 'aria-relevant': 'additions text' });
  const textarea = el('textarea', { id: 'ask-question', placeholder: 'What would you like to understand?', rows: 3, required: true, maxlength: 4000, value: state.draft });
  textarea.addEventListener('input', () => { state.draft = textarea.value; persistViewState(); });
  const send = button(state.asking ? 'Reading…' : 'Ask library', { type: 'submit', variant: 'primary', symbol: 'send', disabled: state.asking, 'data-testid': 'ask-submit' });
  const picker = el('div', { class: 'scope-picker', hidden: true });
  const scopeTitle = el('div', { class: 'scope-title', 'data-testid': 'ask-scope-title' });
  const scopeDetail = el('p', { class: 'scope-detail', 'data-testid': 'ask-scope-detail' });
  const updateScope = () => { scopeTitle.textContent = state.scope ? state.scope.title : 'All books in your library'; scopeDetail.textContent = state.scope?.readerContext ? state.scope.detail || 'Current section of this source edition' : state.scope ? 'Whole book · selected source edition' : 'Search and answer across every source book'; };
  updateScope();
  const change = button('Change', { variant: 'small', 'aria-expanded': 'false', onClick: () => {
    picker.hidden = !picker.hidden; change.setAttribute('aria-expanded', String(!picker.hidden));
    if (picker.hidden) return;
    const search = el('input', { type: 'search', id: 'scope-search', placeholder: 'Find a book by title or author…', autocomplete: 'off' });
    const results = el('div', { class: 'scope-results', 'aria-live': 'polite' });
    const choose = (book) => { state.scope = book ? { id: String(book.id), title: titleOf(book) } : null; persistViewState(); askPage(); $('#ask-question')?.focus(); };
    picker.replaceChildren(button('Search all books', { symbol: 'library', variant: 'small', onClick: () => choose(null) }), el('div', { class: 'search-field' }, icon('search', 16), label('Find a book to ask', search), search), results);
    let timer; let request = 0;
    const find = async () => {
      const current = ++request;
      try {
        const response = await api(`/api/books?${new URLSearchParams({ q: search.value.trim(), limit: '20', offset: '0', sort: 'title' })}`);
        if (current !== request || !results.isConnected) return;
        const items = response.items || []; rememberBooks(items);
        results.replaceChildren(...(items.length ? items.map((book) => el('button', { type: 'button', class: 'scope-option', onclick: () => choose(book) }, el('strong', { text: titleOf(book) }), el('span', { text: `${authors(book)} · ${formatNames(book).join(' / ')} · ID ${book.id}` }))) : [el('p', { class: 'field-help', text: 'No matching books. Try another title or author.' })]));
      } catch (error) { if (current === request && results.isConnected) results.replaceChildren(notice(error.message, 'error')); }
    };
    search.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(find, 250); });
    find(); search.focus();
  } });
  const suggestions = ['What is the central argument?', 'Explain a concept with examples.', 'Where do these authors disagree?'];
  const welcome = el('div', { class: 'chat-empty' }, el('div', { class: 'ask-symbol' }, icon('ask', 29)), el('h2', { text: 'Follow your curiosity.' }), el('p', { text: 'Ask a question and your local model will answer from retrieved passages, with citations you can open. Use Find passages to look up text directly.' }), el('div', { class: 'question-examples' }, suggestions.map((question) => el('button', { type: 'button', text: question, onclick: () => { textarea.value = question; state.draft = question; persistViewState(); textarea.focus(); } }))));
  if (state.turns.length) conversation.replaceChildren(...state.turns.map(renderTurn)); else conversation.append(welcome);
  const form = el('form', { class: 'composer' }, label('Your question', textarea), textarea, el('div', { class: 'composer-footer' }, el('span', { class: 'composer-help', text: '⌘ / Ctrl + Enter to ask' }), send));
  const submit = async () => {
    const question = textarea.value.trim(); if (!question || state.asking) return;
    const turn = { question, scope: state.scope ? { ...state.scope } : null, pending: true };
    state.turns.push(turn); state.asking = true; state.draft = ''; textarea.value = ''; send.disabled = true; persistViewState();
    conversation.replaceChildren(...state.turns.map(renderTurn));
    try { turn.response = await api('/api/ask', { method: 'POST', body: { question, responseMode: 'auto', ...(turn.scope ? { bookId: turn.scope.id, ...(turn.scope.readerContext ? { readerContext: turn.scope.readerContext } : {}) } : {}) } }); }
    catch (error) { turn.error = error.message; }
    finally { turn.pending = false; state.asking = false; persistViewState(); if (location.hash === '#ask' && state.route === '#ask') { askPage(); const target = $('#ask-question'); target?.focus({ preventScroll: true }); } }
  };
  form.addEventListener('submit', (event) => { event.preventDefault(); submit(); });
  textarea.addEventListener('keydown', (event) => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); submit(); } });
  mount(el('div', { class: 'ask-layout' }, el('div', { class: 'ask-header' }, el('div', {}, el('p', { class: 'eyebrow', text: 'A conversation with your collection' }), el('h1', { class: 'page-title', text: 'Ask your library.' })), el('div', { class: 'model-pill' }, icon('ask', 12), el('span', { text: modelDescription() }))), el('div', { class: 'scope-bar' }, icon('book', 19), el('div', { class: 'scope-info' }, el('p', { class: 'scope-label', text: 'LOOKING IN' }), scopeTitle, scopeDetail), change), state.scope ? el('div', { class: 'scope-actions' }, state.scope.readerContext ? button('Ask the whole book', { variant: 'small', 'data-testid': 'ask-whole-book', onClick: () => { state.scope = { id: state.scope.id, title: state.scope.title }; persistViewState(); askPage(); } }) : null, button('Ask the whole library', { variant: 'small', 'data-testid': 'ask-clear-scope', onClick: () => { state.scope = null; persistViewState(); askPage(); } }), state.readerReturn ? button('Return to reading', { href: state.readerReturn, variant: 'small', symbol: 'book', 'data-testid': 'ask-return-reader' }) : null) : state.readerReturn ? button('Return to reading', { href: state.readerReturn, variant: 'small', 'data-testid': 'ask-return-reader' }) : null, picker, conversation, form, el('p', { class: 'ask-footnote', text: 'Answers can miss context or make mistakes. Follow the citations and check the original passages. Each question searches the selected scope independently.' })));
}

function contentSearchPage(epoch) {
  breadcrumb('Find passages');
  const query = el('input', { id: 'content-query', type: 'search', class: 'field-input', required: true, maxlength: 4000, value: state.contentSearch.query, placeholder: 'A phrase, concept, command, or line of code…', 'data-testid': 'content-query' });
  const scopeQuery = el('input', { id: 'content-scope-query', type: 'search', class: 'field-input', maxlength: 1000, value: state.contentSearch.scopeQuery, placeholder: 'Find a source book by title or author…', autocomplete: 'off' });
  const scope = el('select', { id: 'content-scope', class: 'filter-select' });
  const scopeNotice = el('div', { class: 'field-help', 'aria-live': 'polite' });
  const pageSize = 20;
  let scopeSequence = 0, scopeTimer;
  const option = book => el('option', { value: book.id, title: `Source ${book.id}`, text: `${titleOf(book)} · ${authors(book)} · ${formatNames(book).join(' / ')} · ${String(book.id).slice(0, 12)}` });
  const updateOptions = books => {
    const selected = state.contentSearch.bookId;
    const retained = selected && !books.some(book => String(book.id) === selected) ? state.seenBooks.get(selected) : null;
    scope.replaceChildren(el('option', { value: '', text: 'All books' }), ...(retained ? [option(retained)] : []), ...books.map(option));
    // Retain an explicit scope while catalog requests are pending or filtered.
    if (selected && ![...scope.options].some(item => item.value === selected)) scope.append(el('option', { value: selected, text: `Selected source · ID ${selected}` }));
    scope.value = selected;
  };
  updateOptions([]);
  const previous = button('Previous books', { variant: 'small', disabled: true, onClick: () => {
    state.contentSearch.scopeOffset = Math.max(0, state.contentSearch.scopeOffset - pageSize); loadScopes();
  } });
  const next = button('Next books', { variant: 'small', disabled: true, onClick: () => {
    state.contentSearch.scopeOffset += pageSize; loadScopes();
  } });
  const loadScopes = async () => {
    if (epoch !== state.epoch) return;
    clearTimeout(scopeTimer);
    const sequence = ++scopeSequence;
    previous.disabled = true; next.disabled = true;
    scopeNotice.textContent = 'Finding source books…';
    try {
      const response = await api(`/api/books?${new URLSearchParams({ q: state.contentSearch.scopeQuery.trim(), limit: String(pageSize), offset: String(state.contentSearch.scopeOffset), sort: 'title' })}`);
      if (epoch !== state.epoch || sequence !== scopeSequence) return;
      const books = response.items || []; rememberBooks(books); updateOptions(books);
      const offset = state.contentSearch.scopeOffset;
      scopeNotice.textContent = response.total ? `${count(offset + 1)}–${count(offset + books.length)} of ${count(response.total)} matching source books` : 'No matching source books. Try another title or author.';
      previous.disabled = offset === 0; next.disabled = offset + books.length >= response.total;
    } catch (error) {
      if (epoch !== state.epoch || sequence !== scopeSequence) return;
      scopeNotice.replaceChildren(notice(error.message, 'error'));
    }
  };
  scopeQuery.addEventListener('input', () => {
    state.contentSearch.scopeQuery = scopeQuery.value; state.contentSearch.scopeOffset = 0;
    ++scopeSequence; clearTimeout(scopeTimer); previous.disabled = true; next.disabled = true;
    scopeTimer = setTimeout(loadScopes, 250);
  });
  scopeQuery.addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); loadScopes(); } });
  const results = el('div', { class: 'citations', 'aria-live': 'polite', 'data-testid': 'content-results' });
  query.addEventListener('input', () => { state.contentSearch.query = query.value; });
  scope.addEventListener('change', () => { state.contentSearch.bookId = scope.value; });
  const showResult = result => {
    const hits = result.hits || [];
    results.replaceChildren(el('p', { class: 'field-help', text: `${count(hits.length)} passages · ${human(result.mode)} search` }),
      ...warningsList(result.warnings).map(warning => notice(warning)),
      ...(hits.length ? hits.map(citationButton) : [empty('No matching passages', 'Try a shorter phrase or a different word.')]));
  };
  if (state.contentSearch.result) showResult(state.contentSearch.result);
  const submit = button('Find passages', { type: 'submit', variant: 'primary', symbol: 'search', 'data-testid': 'content-submit' });
  const form = el('form', { class: 'panel content-search-form' }, el('label', { class: 'field-label', for: query.id, text: 'Search inside your books' }), el('div', { class: 'input-action' }, query, submit),
    el('label', { class: 'field-label', for: scopeQuery.id, text: 'Choose a source book, or search all books' }), scopeQuery, label('Search scope', scope), scope,
    el('div', { class: 'pagination' }, scopeNotice, previous, next));
  form.addEventListener('submit', async event => {
    event.preventDefault(); submit.disabled = true; results.replaceChildren(loading('Searching the text…'));
    const submitted = { query: query.value.trim(), bookId: scope.value };
    const sequence = ++state.contentSearchSequence;
    try {
      const result = await api('/api/search', { method: 'POST', body: { query: submitted.query, ...(submitted.bookId ? { bookId: submitted.bookId } : {}), limit: 20 } });
      if (sequence !== state.contentSearchSequence) return;
      state.contentSearch = { ...state.contentSearch, ...submitted, result };
      if (epoch !== state.epoch) return;
      showResult(result);
    } catch (error) { if (epoch === state.epoch) results.replaceChildren(notice(error.message, 'error')); }
    finally { submit.disabled = false; }
  });
  mount(heading('Follow a thought', 'Find it in the text.', 'Search the contents and open each passage in its original context.'), form, results);
  loadScopes();
}

async function groupPage(id, epoch) {
  const group = await api(`/api/catalog/groups/${encodeURIComponent(id)}`); if (epoch !== state.epoch) return;
  breadcrumb({ text: 'Library', href: '#library' }, group.title);
  mount(heading('Linked formats and editions', group.title, 'Choose a source book to read or ask. Each source keeps its own citations and reading place.'),
    el('div', { class: 'book-grid' }, group.members.map(book => el('div', {}, bookCard(book), button('Unlink from this group', { variant: 'small', onClick: async () => {
      try { await api('/api/catalog/unlink', { method: 'POST', body: { bookId: book.id } }); toast('Source book unlinked.'); await groupPage(id, epoch); }
      catch (error) { toast(error.message); }
    } })))), button('Review other formats', { href: '#organize', symbol: 'folder' }));
}

async function organizePage(epoch) {
  breadcrumb('Organize');
  const region = el('div', { class: 'import-jobs' }, loading('Finding related formats…'));
  const selected = new Map();
  const query = el('input', { id: 'group-search', type: 'search', class: 'field-input', placeholder: 'Find source books by title or author…' });
  const title = el('input', { id: 'group-title', class: 'field-input', placeholder: 'Title for the linked group', maxlength: 500 });
  const choices = el('div', { class: 'scope-results' });
  const selection = el('div', { class: 'detail-actions' });
  const link = button('Link selected books', { variant: 'primary', disabled: true, onClick: async () => {
    try { const result = await api('/api/catalog/groups', { method: 'POST', body: { bookIds: [...selected.keys()], ...(title.value.trim() ? { title: title.value.trim() } : {}) } }); state.bookCache.clear(); location.hash = `#group/${encodeURIComponent(result.group.id)}`; }
    catch (error) { toast(error.message); }
  } });
  const showSelection = () => {
    link.disabled = selected.size < 2;
    selection.replaceChildren(...[...selected.values()].map(book => button(`Remove ${titleOf(book)}`, { variant: 'small', onClick: () => { selected.delete(book.id); showSelection(); } })));
  };
  let searchSequence = 0, timer;
  const search = async () => {
    const sequence = ++searchSequence;
    try {
      const data = await api(`/api/books?${new URLSearchParams({ q: query.value.trim(), limit: '30' })}`);
      if (epoch !== state.epoch || sequence !== searchSequence) return;
      choices.replaceChildren(...data.items.map(book => el('button', { type: 'button', class: 'scope-option', onclick: () => { selected.set(book.id, book); showSelection(); } },
        el('strong', { text: titleOf(book) }), el('span', { text: `${authors(book)} · ${formatNames(book).join(' / ')} · ${book.id.slice(0, 12)}` }))));
    } catch (error) { if (epoch === state.epoch) choices.replaceChildren(notice(error.message, 'error')); }
  };
  query.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(search, 250); });
  mount(heading('Keep your shelves together', 'Formats, thoughtfully linked.', 'Review related files or select books yourself. Linking changes the catalog grouping; source text and citations stay attached to their original book.'),
    el('section', { class: 'panel' }, el('h2', { text: 'Create a group' }), label('Find source books to link', query), query, choices, selection, label('Group title', title), title, link),
    el('div', { class: 'section-heading' }, el('h2', { text: 'Suggested relationships' })), region);
  let offset = 0;
  const loadSuggestions = async () => {
    try {
      const data = await api(`/api/catalog/suggestions?limit=15&offset=${offset}`); if (epoch !== state.epoch) return;
      region.replaceChildren(...data.items.map(item => el('article', { class: 'panel' }, el('h3', { text: item.title }),
        el('p', { class: 'field-help', text: item.signal === 'filename_only' ? 'Matching folder and filename only.' : 'Related title, author, or identifier metadata.' }),
        el('ul', {}, item.members.map(member => el('li', {}, el('a', { href: `#book/${encodeURIComponent(member.id)}`, text: `${member.title} · ${member.files.map(file => file.format.toUpperCase()).join(' / ')}` })))),
        ...item.cautions.map(caution => el('p', { class: 'field-help', text: caution })),
        item.additionalBookIds.length ? notice('Some books already belong to larger groups. Select every member above to combine those groups.') : button('Link these source books', { variant: 'small', onClick: async () => {
          try { await api('/api/catalog/groups', { method: 'POST', body: { bookIds: item.bookIds, title: item.title } }); toast('Source books linked.'); await loadSuggestions(); }
          catch (error) { toast(error.message); }
        } }))),
        data.total ? el('div', { class: 'pagination' }, el('span', { text: `${offset + 1}–${offset + data.items.length} of ${data.total} suggestions` }),
          button('Previous suggestions', { disabled: offset === 0, onClick: () => { offset = Math.max(0, offset - 15); loadSuggestions(); } }),
          button('Next suggestions', { disabled: offset + data.items.length >= data.total, onClick: () => { offset += 15; loadSuggestions(); } })) : empty('No unreviewed suggestions', 'You can still select books above to create a group.'));
    } catch (error) { if (epoch === state.epoch) region.replaceChildren(errorView(error, loadSuggestions)); }
  };
  await Promise.all([search(), loadSuggestions()]);
}

async function renderRoute() {
  const hash = location.hash || '#library';
  if (hash === '#main') { main.focus(); return; }
  state.pageCleanup?.(); state.pageCleanup = null; persistViewState();
  const epoch = ++state.epoch; state.route = hash; clearTimeout(state.poll); clearTimeout(state.statusPoll); nav();
  main.setAttribute('aria-busy', 'true'); main.replaceChildren(loading());
  try {
    if (hash === '#library' || hash === '#') await libraryPage(epoch);
    else if (hash === '#imports') await importsPage(epoch);
    else if (hash === '#ask') askPage();
    else if (hash === '#search') contentSearchPage(epoch);
    else if (hash === '#organize') await organizePage(epoch);
    else if (hash.startsWith('#group/')) await groupPage(decodeURIComponent(hash.slice(7)), epoch);
    else if (hash.startsWith('#book/')) await bookPage(decodeURIComponent(hash.slice(6)), epoch);
    else if (hash.startsWith('#read/')) { const [id, query = ''] = hash.slice(6).split('?'); await readerPage(decodeURIComponent(id), epoch, new URLSearchParams(query)); }
    else { breadcrumb('Library'); main.replaceChildren(empty('This shelf doesn’t exist.', 'Head back to your library to find your next read.', button('Go to library', { href: '#library' }))); }
  } catch (error) { if (epoch === state.epoch) main.replaceChildren(errorView(error, renderRoute)); }
  finally { if (epoch === state.epoch) { main.setAttribute('aria-busy', 'false'); document.title = `${hash === '#ask' ? 'Ask your library' : hash === '#imports' ? 'Imports' : hash.startsWith('#read/') ? 'Reader' : 'Your library'} — Librarian`; } }
}
window.addEventListener('hashchange', () => { renderRoute(); if (location.hash !== '#main') { window.scrollTo({ top: 0 }); main.focus({ preventScroll: true }); } });
document.querySelector('.skip-link').addEventListener('click', (event) => { event.preventDefault(); main.focus(); main.scrollIntoView({ block: 'start' }); });
await refreshStatus();
await renderRoute();
