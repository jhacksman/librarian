import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createApp } from '../src/server.mjs';
import { readConfig } from '../src/config.mjs';
import { LibraryStore, digest } from '../src/store.mjs';
import { processIdentity } from '../src/process-lock.mjs';

async function fixture(t, { beforeStart, configure, retrieval, env = {} } = {}) {
  const root = await mkdtemp(join(tmpdir(), 'librarian-api-'));
  const config = readConfig({ LIBRARIAN_DATA_DIR: join(root, 'library'), LIBRARIAN_PORT: '0', ...env });
  configure?.(config);
  if (beforeStart) {
    const initial = new LibraryStore(config.dbPath, { dataDir: config.dataDir });
    try { await beforeStart(initial); } finally { initial.close(); }
  }
  const app = await createApp({ config, retrieval });
  await new Promise(done => app.server.listen(0, '127.0.0.1', done));
  config.origin = `http://127.0.0.1:${app.server.address().port}`;
  t.after(async () => { await app.close(); await rm(root, { recursive: true, force: true }); });
  const request = (path, body, method = 'POST', headers = {}) => fetch(config.origin + path, {
    method: body === undefined ? 'GET' : method,
    headers: { origin: config.origin, ...(body === undefined ? {} : { 'content-type': 'application/json' }), ...headers },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  return { ...app, root, request };
}

function seed(store, id, title, author, format = 'epub') {
  store.db.prepare('INSERT INTO books(id,title,authors_json,created_at,updated_at) VALUES(?,?,?,?,?)')
    .run(id, title, JSON.stringify([author]), '2026-10-04', '2026-10-04');
  store.db.prepare('INSERT INTO sections(id,book_id,ordinal,title,text,locator_json) VALUES(?,?,?,?,?,?)')
    .run(`${id}:section`, id, 0, 'Chapter one', 'Continuity 🦊 keeps the next sentence with its source.', JSON.stringify({ format, member: 'chapter.xhtml', offsetBasis: 'utf16' }));
  store.db.prepare('INSERT INTO files(id,path,source_path,relative_path,size,format,status,book_id,updated_at,group_key) VALUES(?,?,?,?,?,?,?,?,?,?)')
    .run(`${id}:file`, '/not-managed/source', `/original/${id}`, `${id}.${format}`, 10, format, 'completed', id, '2026-10-04', id);
}

function interruptedJob(store, { dispositions = [], state = 'running', inventoryComplete = true, interrupted = false } = {}) {
  const job = store.createJob('/unavailable/finalization-originals', { inventoryComplete, interrupted, completed: false });
  for (const [ordinal, status] of dispositions.entries()) {
    const file = store.registerFile(job.id, { sourcePath: `/unavailable/finalization-originals/${job.id}-${ordinal}.epub`,
      relativePath: `${ordinal}.epub`, size: 12, format: 'epub', groupKey: job.id,
      ...(status === 'inventory_conflict' ? { error: 'Immutable source identity conflict' } : {}) }, ordinal);
    if (status === 'queued' || status === 'inventory_conflict') continue;
    const attempt = store.beginFile(job.id, file.id);
    if (status === 'running') continue;
    if (status === 'completed') {
      // Persist actual book/chunk/citation rows as a completed file transaction,
      // then leave the job itself running, as in the crash finalization window.
      store.db.prepare('UPDATE files SET staged=1,sha256=? WHERE id=?').run(digest(`${job.id}:${ordinal}`), file.id);
      const bookId = store.commitBook(job.id, file.id, attempt.receiptId, {
        title: 'Durable completion evidence', authors: ['Synthetic author'],
        sections: [{ ordinal: 0, title: 'Evidence', text: 'The saved 🧭 passage must survive status reconciliation.',
          locator: { format: 'epub', member: 'evidence.xhtml' } }],
      });
      const section = store.db.prepare('SELECT id FROM sections WHERE book_id=?').get(bookId);
      store.db.prepare('INSERT INTO reading_progress(book_id,section_id,updated_at) VALUES(?,?,?)')
        .run(bookId, section.id, '2026-10-04');
    } else store.finishFile(job.id, file.id, attempt.receiptId, { status,
      error: status === 'failed' ? 'Synthetic extraction failed' : null });
  }
  store.refreshJob(job.id, { state, current: { phase: 'finalizing' } });
  return job.id;
}

function durableRows(store) {
  return Object.fromEntries(['files', 'books', 'sections', 'chunks', 'reading_progress', 'import_job_files', 'import_receipts']
    .map(table => [table, store.db.prepare(`SELECT * FROM ${table} ORDER BY rowid`).all()]));
}

function maintenanceRows(store) {
  return { ...durableRows(store),
    import_jobs: store.db.prepare('SELECT * FROM import_jobs ORDER BY rowid').all(),
    import_lock: store.db.prepare('SELECT * FROM import_lock ORDER BY rowid').all() };
}

test('default server starts with explicitly unavailable local AI and a live catalog', async t => {
  const app = await fixture(t);
  const status = await (await app.request('/api/status')).json();
  assert.equal(status.models.embedding.configured, false);
  assert.equal(status.models.chat.configured, false);
  assert.deepEqual(status.capabilities, { maintenance: true });
  assert.equal(status.totals.files, 0);
  const home = await app.request('/');
  assert.equal(home.status, 200);
  assert.match(home.headers.get('content-security-policy'), /script-src 'self'/);
});

test('custom configuration without a maintenance flag retains enabled import recovery', async t => {
  let id;
  const app = await fixture(t, {
    configure(config) { delete config.maintenanceEnabled; },
    beforeStart(store) { id = interruptedJob(store, { dispositions: ['queued'] }); },
  });
  assert.equal((await (await app.request('/api/status')).json()).capabilities.maintenance, true);
  const job = (await (await app.request('/api/imports')).json()).items.find(item => item.id === id);
  assert.equal(job.state, 'paused');
  assert.equal(job.canResume, true);
});

test('disabled maintenance preserves interrupted jobs and stale locks during startup and import reads', async t => {
  let saved, ids;
  const app = await fixture(t, { env: { LIBRARIAN_MAINTENANCE: 'disabled', LIBRARIAN_IMPORT_ROOTS: ':' },
    beforeStart(store) {
      ids = [
        interruptedJob(store, { dispositions: ['completed'] }),
        interruptedJob(store, { dispositions: ['running', 'queued'] }),
        interruptedJob(store, { dispositions: ['failed'], state: 'paused' }),
      ];
      store.db.prepare('INSERT INTO import_lock(id,owner,pid,job_id,started_at,process_identity) VALUES(1,?,?,?,?,?)')
        .run('stale-importer', 2147483647, ids[1], '2026-10-04', 'absent-process');
      saved = maintenanceRows(store);
    },
  });
  assert.deepEqual(maintenanceRows(app.store), saved, 'Startup must not recover or finalize interrupted imports');
  const status = await (await app.request('/api/status')).json();
  assert.deepEqual(status.capabilities, { maintenance: false });
  assert.deepEqual(status.importRoots, []);
  for (let read = 0; read < 2; read++) {
    const listed = await (await app.request('/api/imports')).json();
    for (const id of ids) {
      const job = listed.items.find(item => item.id === id);
      assert.equal(job.canPause, false); assert.equal(job.canResume, false);
      const response = await app.request(`/api/imports/${id}`);
      assert.equal(response.status, 200);
      const detail = await response.json();
      assert.equal(detail.canPause, false); assert.equal(detail.canResume, false);
    }
    assert.deepEqual(maintenanceRows(app.store), saved, 'Status/list/detail must preserve jobs, receipts, sources and stale ownership');
  }
});

test('disabled maintenance rejects same-origin mutations before body or target validation', async t => {
  let id, embeddingCalls = 0;
  const app = await fixture(t, { env: { LIBRARIAN_MAINTENANCE: 'disabled', LIBRARIAN_EMBEDDING_MODEL: 'synthetic-embedding' },
    retrieval: { async embedPending() { embeddingCalls++; throw new Error('Maintenance must not reach the embedding worker'); } },
    beforeStart(store) { id = interruptedJob(store, { dispositions: ['completed', 'queued'], state: 'paused' }); },
  });
  const importRoot = app.config.importRoots[0];
  await mkdir(importRoot, { recursive: true });
  const bookId = app.store.db.prepare('SELECT id FROM books LIMIT 1').get().id;
  const saved = maintenanceRows(app.store);
  const mutations = [
    ['/api/imports', { root: importRoot }],
    [`/api/imports/${id}/resume`, {}],
    [`/api/imports/${id}/pause`, {}],
    ['/api/reindex', { bookIds: [bookId] }],
    ['/api/embeddings', { retryFailed: true }],
    [`/api/imports/${id}`, {}, 'DELETE'],
    ['/api/imports/future-operation', {}, 'PATCH'],
  ];
  for (const [path, body, method = 'POST'] of mutations) {
    const response = await app.request(path, body, method);
    assert.equal(response.status, 403, path);
    assert.match((await response.json()).error, /maintenance is disabled/i);
    assert.deepEqual(maintenanceRows(app.store), saved, path);
  }
  for (const path of ['/api/imports', '/api/imports/missing/resume', '/api/imports/missing/pause', '/api/reindex', '/api/embeddings']) {
    const response = await fetch(app.config.origin + path, {
      method: 'POST', headers: { origin: app.config.origin, 'content-type': 'text/plain' }, body: 'not JSON',
    });
    assert.equal(response.status, 403, `${path} must be denied before JSON or target validation`);
    assert.match((await response.json()).error, /maintenance is disabled/i);
  }
  assert.equal(embeddingCalls, 0);
  assert.deepEqual(maintenanceRows(app.store), saved);
});

test('disabled maintenance keeps reading, source search, progress, metadata and catalog grouping available', async t => {
  const app = await fixture(t, { env: { LIBRARIAN_MAINTENANCE: 'disabled' } });
  interruptedJob(app.store, { state: 'completed', dispositions: ['completed'] });
  const { id: bookId } = app.store.db.prepare('SELECT id FROM books LIMIT 1').get();
  const section = app.store.db.prepare('SELECT * FROM sections WHERE book_id=?').get(bookId);
  seed(app.store, 'other-edition', 'Other edition', 'Synthetic author', 'pdf');
  const managed = join(app.config.dataDir, 'sources'); await mkdir(managed, { recursive: true });
  const copy = join(managed, 'synthetic.pdf'); await writeFile(copy, 'synthetic source bytes');
  const { id: fileId } = app.store.db.prepare('SELECT id FROM files WHERE book_id=?').get(bookId);
  app.store.db.prepare("UPDATE files SET path=?,format='pdf' WHERE id=?").run(copy, fileId);

  for (const path of ['/api/books', '/api/catalog', `/api/books/${encodeURIComponent(bookId)}`, `/api/sections/${encodeURIComponent(section.id)}`]) {
    assert.equal((await app.request(path)).status, 200, path);
  }
  const searchResponse = await app.request('/api/search', { query: 'saved passage', bookId });
  assert.equal(searchResponse.status, 200);
  assert.ok((await searchResponse.json()).hits.some(hit => hit.bookId === bookId));
  const askResponse = await app.request('/api/ask', { question: 'saved passage', bookId, responseMode: 'excerpts' });
  assert.equal(askResponse.status, 200);
  const ask = await askResponse.json();
  assert.equal(ask.responseKind, 'source_excerpts'); assert.equal(ask.metrics.chatMs, 0);
  assert.ok(ask.extractive.passages.some(passage => passage.bookId === bookId));
  assert.equal((await app.request(`/api/books/${encodeURIComponent(bookId)}/progress`, { sectionId: section.id })).status, 200);
  assert.equal((await app.request(`/api/books/${encodeURIComponent(bookId)}`, { title: 'Reader edit', tags: ['review'] }, 'PUT')).status, 200);
  const detail = await (await app.request(`/api/books/${encodeURIComponent(bookId)}`)).json();
  assert.equal(detail.title, 'Reader edit'); assert.deepEqual(detail.metadata.tags, ['review']);
  assert.equal(detail.readingProgress.sectionId, section.id);

  const linked = await app.request('/api/catalog/groups', { bookIds: [bookId, 'other-edition'], title: 'Reader group' });
  assert.equal(linked.status, 200);
  const { group } = await linked.json();
  assert.equal((await app.request(`/api/catalog/groups/${encodeURIComponent(group.id)}`)).status, 200);
  assert.equal((await (await app.request('/api/catalog')).json()).total, 1);
  assert.equal((await app.request('/api/catalog/unlink', { bookId: 'other-edition' })).status, 200);
  assert.equal((await (await app.request('/api/catalog')).json()).total, 2);
  const source = await app.request(`/api/files/${encodeURIComponent(fileId)}/source`);
  assert.equal(source.status, 200); assert.equal(await source.text(), 'synthetic source bytes');
  assert.equal(app.store.db.prepare('SELECT text FROM sections WHERE id=?').get(section.id).text, section.text);
});

test('HTTP ask exposes exact ungenerated excerpts and validates the response mode', async t => {
  const app = await fixture(t);
  interruptedJob(app.store, { state: 'completed', dispositions: ['completed'] });
  const book = app.store.db.prepare('SELECT id FROM books LIMIT 1').get();
  const response = await app.request('/api/ask', { question: 'saved passage', bookId: book.id, responseMode: 'excerpts' });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.responseKind, 'source_excerpts');
  assert.equal(result.status, 'excerpts');
  assert.equal(result.abstained, true);
  assert.equal(result.metrics.chatMs, 0);
  const quote = result.extractive.passages[0];
  const section = app.store.db.prepare('SELECT * FROM sections WHERE id=?').get(quote.sectionId);
  assert.equal(quote.bookId, book.id);
  assert.equal(quote.text, section.text.slice(quote.locator.charStart, quote.locator.charEnd));
  assert.deepEqual(result.books.map(source => source.id), [book.id]);
  assert.equal(result.books[0].title, 'Durable completion evidence');
  assert.equal(result.books[0].thumbnail.kind, 'fallback');
  assert.equal(result.books[0].coverUrl, `/api/books/${book.id}/thumbnail`);
  assert.equal((await app.request(result.books[0].coverUrl)).status, 200);
  assert.equal((await app.request('/api/ask', { question: 'saved passage', responseMode: 'invalid' })).status, 400);
  assert.equal((await app.request('/api/ask', { question: 'saved passage', responseMode: null })).status, 400);
});

test('import file views suppress stale completed reasons without rewriting stored history', async t => {
  const app = await fixture(t);
  const id = interruptedJob(app.store, { state: 'completed_with_errors',
    dispositions: ['completed', 'unsupported', 'auxiliary', 'failed', 'duplicate'] });
  const oldReason = 'Source retained; MOBI/PRC text extraction is not supported';
  for (const file of app.store.listFiles({ jobId: id })) {
    const reason = file.status === 'auxiliary' ? 'Torrent retained as auxiliary metadata; no download performed' : oldReason;
    // Reproduce persisted rows from the older importer, including the already
    // completed extraction. Read-only projection must handle them without reindex.
    const metadata = JSON.stringify({ ...file.metadata, reason });
    app.store.db.prepare('UPDATE files SET metadata_json=? WHERE id=?').run(metadata, file.id);
    app.store.db.prepare('UPDATE import_receipts SET metadata_json=? WHERE job_id=? AND file_id=?').run(metadata, id, file.id);
    // Historical job disposition can differ from the current file. The reason
    // projection follows the displayed job status, not the joined file status.
    if (file.status === 'unsupported') app.store.db.prepare("UPDATE files SET status='completed' WHERE id=?").run(file.id);
    if (file.status === 'duplicate') app.store.db.prepare("UPDATE files SET status='unsupported' WHERE id=?").run(file.id);
  }
  const before = durableRows(app.store);
  const response = await app.request(`/api/imports/${id}`);
  assert.equal(response.status, 200);
  const { files } = await response.json();
  assert.equal(files.length, 5);
  for (const file of files) {
    assert.equal(file.reason, ['unsupported', 'auxiliary'].includes(file.status) ? file.metadata.reason : null);
    assert.ok(file.metadata.reason, 'Raw historical metadata remains available');
  }
  const completed = files.find(file => file.status === 'completed');
  assert.equal(completed.metadata.extraction.searchable, true);
  assert.ok(completed.bookId);
  assert.deepEqual(durableRows(app.store), before, 'Reading current dispositions must not mutate receipts, corpus or source metadata');
});

test('catalog filters apply before pagination, reader links and edits persist', async t => {
  const app = await fixture(t);
  seed(app.store, 'a', 'Alphabet', 'Author A');
  seed(app.store, 'b', 'Zoology', 'Author B', 'pdf');
  const list = await (await app.request('/api/books?author=Author%20B&format=pdf&limit=1')).json();
  assert.equal(list.total, 1); assert.equal(list.items[0].id, 'b');
  assert.equal((await app.request('/api/books/b/progress', { sectionId: 'a:section' })).status, 400);
  assert.equal((await app.request('/api/books/b/progress', { sectionId: 'b:section' })).status, 200);
  assert.equal((await app.request('/api/books/b', { title: 'Revised', tags: ['local'], authors: ['Editor'] }, 'PUT')).status, 200);
  const detail = await (await app.request('/api/books/b')).json();
  assert.equal(detail.readingProgress.sectionId, 'b:section');
  assert.equal(detail.title, 'Revised'); assert.deepEqual(detail.metadata.tags, ['local']);
  const section = await (await app.request('/api/sections/b%3Asection')).json();
  assert.equal(section.bookId, 'b'); assert.match(section.text, /🦊/);
});

test('writes reject other origins and source links cannot serve original NAS paths', async t => {
  const app = await fixture(t); seed(app.store, 'a', 'Book', 'Writer');
  const rejected = await app.request('/api/books/a', { title: 'Changed' }, 'PUT', { origin: 'https://example.invalid' });
  assert.equal(rejected.status, 403);
  const original = join(app.root, 'original.pdf'); await writeFile(original, '0123456789');
  app.store.db.prepare('UPDATE files SET path=? WHERE id=?').run(original, 'a:file');
  assert.equal((await app.request('/api/files/a%3Afile/source')).status, 409);
  const managed = join(app.config.dataDir, 'sources'); await mkdir(managed);
  const copy = join(managed, 'copy.pdf'); await writeFile(copy, '0123456789');
  app.store.db.prepare("UPDATE files SET path=?,format='pdf' WHERE id=?").run(copy, 'a:file');
  const response = await app.request('/api/files/a%3Afile/source', undefined, 'GET', { range: 'bytes=2-5' });
  assert.equal(response.status, 206); assert.equal(await response.text(), '2345');
  assert.equal(response.headers.get('content-range'), 'bytes 2-5/10');
});

test('restart exposes complete interrupted inventories for resume and never acknowledges an ineffective pause', async t => {
  let readyId, incompleteId;
  const app = await fixture(t, { beforeStart(store) {
    readyId = store.createJob('/unavailable/originals', { inventoryComplete: true }).id;
    store.registerFile(readyId, { sourcePath: '/unavailable/originals/sample.epub', relativePath: 'sample.epub', size: 0, format: 'epub', groupKey: 'restart-fixture' }, 0);
    incompleteId = store.createJob('/unavailable/incomplete', { inventoryComplete: false }).id;
    store.refreshJob(readyId, { state: 'running', current: { fileId: 'previous-worker' } });
    store.refreshJob(incompleteId, { state: 'running' });
    store.db.prepare('INSERT INTO import_lock(id,owner,pid,job_id,started_at,process_identity) VALUES(1,?,?,?,?,?)')
      .run('stale-pid-identity', process.pid, readyId, new Date().toISOString(), 'different-boot:0');
  } });
  // A reused PID is verifiable on Spark/Linux; other hosts retain uncertain locks.
  if (!processIdentity()) return t.skip('Linux process identity required');
  const { items } = await (await app.request('/api/imports')).json();
  const ready = items.find(job => job.id === readyId), incomplete = items.find(job => job.id === incompleteId);
  assert.equal(ready.state, 'paused'); assert.equal(ready.canResume, true); assert.equal(ready.canPause, false);
  assert.equal(ready.current, null); assert.match(ready.recoveryNotice, /Resume/);
  assert.equal(incomplete.state, 'failed'); assert.equal(incomplete.canResume, false);
  assert.equal((await app.request(`/api/imports/${readyId}/pause`, {})).status, 409);
  assert.equal((await app.request(`/api/imports/${incompleteId}/resume`, {})).status, 409);
  assert.equal((await app.request('/api/embeddings', {})).status, 409);
});

test('startup respects a live external importer and exposes no false browser pause or resume', async t => {
  let id;
  const app = await fixture(t, { beforeStart(store) {
    id = interruptedJob(store, { dispositions: ['completed'] });
    store.db.prepare('INSERT INTO import_lock(id,owner,pid,job_id,started_at,process_identity) VALUES(1,?,?,?,?,?)')
      .run('live-test-importer', process.pid, id, new Date().toISOString(), processIdentity());
  } });
  const job = (await (await app.request('/api/imports')).json()).items.find(item => item.id === id);
  assert.equal(job.state, 'running'); assert.equal(job.externalOwner, true);
  assert.equal(job.canPause, false); assert.equal(job.canResume, false);
  assert.equal((await app.request(`/api/imports/${id}/pause`, {})).status, 409);
  assert.equal((await app.request(`/api/imports/${id}/resume`, {})).status, 409);
  assert.equal(app.store.db.prepare('SELECT owner FROM import_lock WHERE id=1').get().owner, 'live-test-importer');
});

test('restart finalizes terminal-only inventories from durable dispositions without new source work or receipts', async t => {
  const cases = [
    { name: 'successful and retained dispositions', dispositions: ['completed', 'duplicate', 'unsupported', 'auxiliary'], expected: 'completed' },
    { name: 'retryable terminal failure', dispositions: ['completed', 'failed'], expected: 'completed_with_errors', canResume: true },
    { name: 'nonretryable inventory conflict', dispositions: ['inventory_conflict'], expected: 'completed_with_errors' },
    { name: 'complete empty inventory', dispositions: [], expected: 'completed' },
    { name: 'previous restart already paused terminal success', dispositions: ['completed'], state: 'paused', interrupted: true, expected: 'completed' },
    { name: 'previous restart already paused terminal conflict', dispositions: ['inventory_conflict'], state: 'paused', interrupted: true, expected: 'completed_with_errors' },
  ];
  for (const scenario of cases) await t.test(scenario.name, async t => {
    let id, saved;
    const app = await fixture(t, { beforeStart(store) {
      id = interruptedJob(store, scenario);
      saved = durableRows(store);
    } });
    const first = (await (await app.request('/api/imports')).json()).items.find(item => item.id === id);
    assert.equal(first.state, scenario.expected);
    assert.equal(first.summary.completed, true);
    assert.equal(first.current, null);
    assert.equal(first.totalFiles, scenario.dispositions.length);
    assert.equal(first.processedFiles, scenario.dispositions.length);
    assert.equal(first.canPause, false);
    assert.equal(first.canResume, scenario.canResume === true);
    assert.match(first.recoveryNotice, /finalized from the saved results/);
    assert.deepEqual(durableRows(app.store), saved);
    // Listing again is idempotent: it must not append attempts or redo completion.
    const second = (await (await app.request('/api/imports')).json()).items.find(item => item.id === id);
    assert.deepEqual(second, first);
    assert.deepEqual(durableRows(app.store), saved);
  });
});

test('restart does not finalize unfinished, deliberately paused or incomplete inventories', async t => {
  const cases = [
    { name: 'queued work remains', dispositions: ['completed', 'queued'], expected: 'paused', canResume: true },
    { name: 'running file remains', dispositions: ['completed', 'running'], expected: 'paused', canResume: true },
    { name: 'deliberate pause with unfinished work', dispositions: ['completed', 'queued'], state: 'paused', expected: 'paused', canResume: true, unchanged: true },
    { name: 'earlier interrupted pause with unfinished work', dispositions: ['running'], state: 'paused', interrupted: true, expected: 'paused', canResume: true, unchanged: true },
    { name: 'incomplete empty inventory', dispositions: [], inventoryComplete: false, expected: 'failed' },
    { name: 'incomplete inventory with terminal files', dispositions: ['completed'], inventoryComplete: false, expected: 'failed' },
  ];
  for (const scenario of cases) await t.test(scenario.name, async t => {
    let id, saved, before;
    const app = await fixture(t, { beforeStart(store) {
      id = interruptedJob(store, scenario);
      saved = durableRows(store); before = store.getJob(id);
    } });
    const job = (await (await app.request('/api/imports')).json()).items.find(item => item.id === id);
    assert.equal(job.state, scenario.expected);
    assert.equal(job.summary.completed, false);
    assert.equal(job.canResume, scenario.canResume === true);
    assert.deepEqual(durableRows(app.store), saved);
    if (scenario.unchanged) assert.deepEqual(app.store.getJob(id), before);
  });
});

test('catalog grouping exposes explicit source choices and never becomes a retrieval scope', async t => {
  const app = await fixture(t);
  seed(app.store, 'epub-source', 'A linked title', 'Writer', 'epub');
  seed(app.store, 'pdf-source', 'A linked title', 'Writer', 'pdf');
  const linkedResponse = await app.request('/api/catalog/groups', { bookIds: ['epub-source', 'pdf-source'], title: 'One catalog entry' });
  assert.equal(linkedResponse.status, 200);
  const { group } = await linkedResponse.json();
  const catalog = await (await app.request('/api/catalog?format=pdf')).json();
  assert.equal(catalog.total, 1); assert.equal(catalog.items[0].type, 'group');
  assert.deepEqual(catalog.items[0].matchingBookIds, ['pdf-source']);
  assert.equal(catalog.items[0].sourceBookCount, 2);
  const detail = await (await app.request(`/api/catalog/groups/${group.id}`)).json();
  assert.deepEqual(detail.members.map(book => book.id).sort(), ['epub-source', 'pdf-source']);
  assert.equal((await app.request('/api/ask', { question: 'What does this say?', bookId: group.id })).status, 400);
  assert.equal((await app.request('/api/catalog/unlink', { bookId: 'pdf-source' })).status, 200);
  assert.equal((await (await app.request('/api/catalog')).json()).total, 2);
  assert.equal(app.store.db.prepare('SELECT count(*) n FROM sections').get().n, 2);
});

test('EPUB table-of-contents entries preserve distinct fragment offsets within one source section', async t => {
  const app = await fixture(t); seed(app.store, 'anchors', 'Anchors', 'Writer');
  const text = 'Early section\n\nLater 🦊 section';
  const later = text.indexOf('Later');
  const locator = { format: 'epub', member: 'chapter.xhtml', anchors: { early: { charStart: 0, charEnd: 13 }, later: { charStart: later, charEnd: text.length } } };
  app.store.db.prepare('UPDATE sections SET text=?,locator_json=? WHERE book_id=?').run(text, JSON.stringify(locator), 'anchors');
  app.store.db.prepare('UPDATE books SET metadata_json=? WHERE id=?').run(JSON.stringify({ toc: [
    { title: 'Early', locator: { format: 'epub', member: 'chapter.xhtml', heading: 'early' } },
    { title: 'Later', locator: { format: 'epub', member: 'chapter.xhtml', heading: 'later' } },
    { title: 'Missing', locator: { format: 'epub', member: 'chapter.xhtml', heading: 'missing' } },
  ] }), 'anchors');
  const { toc } = await (await app.request('/api/books/anchors')).json();
  assert.equal(toc[0].id, toc[1].id); assert.notEqual(toc[0].charStart, toc[1].charStart);
  assert.equal(text.slice(toc[1].charStart, toc[1].charEnd), 'Later 🦊 section');
  assert.equal(toc[2].unresolvedFragment, true); assert.equal(toc[2].charStart, undefined);
});


test('Ask derives exact reader source scope before generation and Search stays lookup', async t => {
  const asks = [], searches = [];
  const app = await fixture(t, { retrieval: {
    async ask(request) { asks.push(request); return { responseKind: 'generated', answer: 'The current source establishes continuity [1].', abstained: false, citations: [], hits: [] }; },
    async search(request) { searches.push(request); return { hits: [] }; },
  } });
  seed(app.store, 'reader-source', 'Reader source', 'Fixture author', 'pdf');
  app.store.db.prepare('UPDATE sections SET locator_json=? WHERE id=?').run(JSON.stringify({ format: 'pdf', page: 17 }), 'reader-source:section');
  seed(app.store, 'other-source', 'Different edition', 'Fixture author');
  const response = await app.request('/api/ask', { question: 'What does this page explain?', readerContext: { bookId: 'reader-source', sectionId: 'reader-source:section', scope: 'section' } });
  assert.equal(response.status, 200);
  assert.equal((await response.json()).responseKind, 'generated');
  assert.deepEqual(asks[0], { question: 'What does this page explain?', bookId: 'reader-source', responseMode: 'auto', readerContext: { bookId: 'reader-source', sectionId: 'reader-source:section', scope: 'section', page: 17 } });
  assert.equal((await app.request('/api/ask', { question: 'Explain the whole book', bookId: 'reader-source' })).status, 200);
  assert.equal(asks[1].bookId, 'reader-source'); assert.equal(asks[1].readerContext, undefined);
  assert.equal((await app.request('/api/ask', { question: 'Explain the whole library' })).status, 200);
  assert.equal(asks[2].bookId, undefined); assert.equal(asks[2].readerContext, undefined);
  assert.equal((await app.request('/api/search', { query: 'continuity', bookId: 'reader-source' })).status, 200);
  assert.equal(searches.length, 1); assert.equal(asks.length, 3);
  assert.equal(searches[0].bookId, 'reader-source'); assert.equal(searches[0].readerContext, undefined);
});

test('Ask rejects spoofed reader page, edition, section, fragment and offsets before inference', async t => {
  let calls = 0;
  const app = await fixture(t, { retrieval: { async ask() { calls++; return { hits: [] }; } } });
  seed(app.store, 'reader-source', 'Reader source', 'Fixture author');
  seed(app.store, 'other-source', 'Different edition', 'Fixture author');
  const context = { bookId: 'reader-source', sectionId: 'reader-source:section', scope: 'section' };
  const invalid = [null, [], 'reader-source', {}, { ...context, bookId: 'missing' },
    { ...context, sectionId: 'other-source:section' }, { ...context, scope: 'library' },
    { ...context, page: 2 }, { ...context, member: '../other.xhtml' }, { ...context, fragment: 'invented' },
    { ...context, text: 'Spoofed passage' }, { ...context, fileId: 'other-source:file' },
    { ...context, charStart: 0 }, { ...context, charStart: -1, charEnd: 4 },
    { ...context, charStart: 0, charEnd: 999999 }, { ...context, charStart: 0.5, charEnd: 4 }];
  for (const readerContext of invalid) assert.equal((await app.request('/api/ask', { question: 'Explain this', readerContext })).status, 400, JSON.stringify(readerContext));
  assert.equal((await app.request('/api/ask', { question: 'Explain this', bookId: 'other-source', readerContext: context })).status, 400);
  assert.equal((await app.request('/api/search', { query: 'continuity', readerContext: context })).status, 400);
  assert.equal(calls, 0, 'Invalid scope cannot broaden to another source or trigger inference');
});

test('Ask accepts exact EPUB anchors and UTF-16 citation spans from the stored source', async t => {
  const calls = [];
  const app = await fixture(t, { retrieval: { async ask(request) { calls.push(request); return { hits: [] }; } } });
  seed(app.store, 'reader-source', 'Reader source', 'Fixture author');
  app.store.db.prepare('UPDATE sections SET locator_json=? WHERE id=?').run(JSON.stringify({ format: 'epub', member: 'chapter.xhtml', anchors: { heading: { charStart: 0, charEnd: 13 } } }), 'reader-source:section');
  const readerContext = { bookId: 'reader-source', sectionId: 'reader-source:section', scope: 'section', member: 'chapter.xhtml', fragment: 'heading', charStart: 0, charEnd: 13 };
  assert.equal((await app.request('/api/ask', { question: 'Explain this section', readerContext })).status, 200);
  assert.deepEqual(calls[0].readerContext, readerContext);
});
