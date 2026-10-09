/**
 * Spark-only synthetic maintenance-guard browser regression. Source-only until
 * the coordinator runs it with prepared Node 24 ARM64 and /opt/playwright.
 * LIBRARIAN_MAINTENANCE_BROWSER_OUTPUT names a fresh /tmp directory whose parent
 * already exists. No installation, corpus access, model calls or shared state.
 * One original two-section EPUB is normally imported before createApp starts.
 * Completed import history is the browser fixture; interrupted/resumable jobs
 * and skipped recovery are covered separately by the server tests.
 * The coordinator must impose a 120-second owned-process boundary as well as
 * this driver's 90-second work budget and bounded cleanup. No quality claim.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {lstat, mkdir, mkdtemp, readFile, realpath, rename, rm, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {basename, dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {performance} from 'node:perf_hooks';
import {expectedSection, syntheticEpub} from './recovery-fixtures.mjs';

const PLAYWRIGHT = '/opt/playwright/node_modules/playwright';
const QUERY = 'saffron compass beacon';
const BOOK = {filename: 'maintenance-fixture.epub', title: 'Synthetic Maintenance Handbook', sections: [
  {title: 'A searchable signal', paragraph: 'The saffron compass beacon glows beside the synthetic reading desk. This original fixture contains no private book text.'},
  {title: 'A saved reading place', paragraph: 'The reader places a violet bookmark at the second section. Maintenance restrictions leave this reading position available.'},
]};
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const short = (value) => String(value ?? '').slice(0, 500);
const within = (parent, child) => {
  const part = relative(parent, child);
  return part === '' || (!isAbsolute(part) && part !== '..' && !part.startsWith(`..${sep}`));
};
async function outsideGit(directory) {
  for (let current = directory; ; current = dirname(current)) {
    try { await lstat(join(current, '.git')); throw new Error('Evidence must be outside every Git checkout'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (dirname(current) === current) return;
  }
}
async function bounded(promise, milliseconds, message) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(message)), milliseconds); })]); }
  finally { clearTimeout(timer); }
}

async function main() {
  const started = performance.now();
  assert.equal(process.platform, 'linux', 'Execute only in the approved Spark Linux runtime');
  assert.equal(process.arch, 'arm64', 'Use the prepared Spark ARM64 runtime');
  assert.ok(/^v24\./.test(process.version), 'Use Node 24');
  assert.equal(process.env.PLAYWRIGHT_MODULE || PLAYWRIGHT, PLAYWRIGHT);
  const requested = process.env.LIBRARIAN_MAINTENANCE_BROWSER_OUTPUT;
  assert.ok(requested && isAbsolute(requested), 'Set LIBRARIAN_MAINTENANCE_BROWSER_OUTPUT to a fresh absolute /tmp path');
  const requestedPath = resolve(requested), temporaryRoot = await realpath('/tmp');
  assert.ok(within('/tmp', requestedPath) && requestedPath !== '/tmp');
  const parent = await realpath(dirname(requestedPath));
  assert.ok(within(temporaryRoot, parent), 'Output parent must resolve beneath /tmp');
  const output = join(parent, basename(requestedPath));
  await outsideGit(output); await mkdir(output); // Existing output is never merged or deleted.
  const report = {schemaVersion: 1, synthetic: true, completed: false, passed: false,
    startedAt: new Date().toISOString(), scope: 'Real createApp/LibraryStore maintenance-disabled desktop/mobile regression',
    limits: {totalBudgetMs: 120000, workBudgetMs: 90000, importBudgetMs: 20000, parserBytes: 256 * 1024,
      screenshotCount: 4, screenshotBytes: 5 * 1024 * 1024, reportBytes: 256 * 1024,
      snapshotBytes: 128 * 1024, snapshotCount: 2, outputBytes: 21 * 1024 * 1024,
      browserRequestsPerProfile: 80, directRequests: 20, requestBodyBytes: 4096,
      responseBodyBytes: 128 * 1024, logBytes: 32768, logEntries: 128},
    limitations: ['One normally completed import; interrupted/resumable ownership cases belong to the server tests.',
      'Lexical search and synthetic reading only; no semantic, model, production-corpus or answer-quality acceptance.',
      'Coordinator owns the outer 120-second process-tree bound; this driver closes only its own resources.'],
    environment: {platform: process.platform, architecture: process.arch, node: process.version},
    sourceHashes: {}, steps: [], profiles: [], screenshots: [], artifacts: [], requests: [],
    progressWrites: [], logs: [], logBytes: 0, modelTransportAttempts: 0, cleanup: {}};
  const deadline = started + report.limits.workBudgetMs;
  const remaining = (maximum = 10000) => {
    const value = Math.min(maximum, Math.floor(deadline - performance.now()));
    assert.ok(value > 0, 'Maintenance browser exhausted its work budget'); return value;
  };
  const cleanupRemaining = (maximum) => {
    const value = Math.min(maximum, Math.floor(started + report.limits.totalBudgetMs - performance.now()));
    assert.ok(value > 0, 'Maintenance browser exhausted its cleanup budget'); return value;
  };
  const abort = new AbortController();
  let workspace, store, app, browser, context, browserServer, browserProcess, origin, baseline, imported;
  let sections, chunk, bookId, expired = false;
  const workTimer = setTimeout(() => {
    expired = true; abort.abort();
    // Closing owned pages interrupts pending browser work. Launch itself uses
    // Playwright's native timeout, never a race that can orphan a late launch.
    void context?.close().catch(() => {});
    app?.server.closeAllConnections();
  }, Math.max(1, deadline - performance.now()));
  const checkpoint = async () => {
    report.updatedAt = new Date().toISOString();
    const bytes = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
    assert.ok(bytes.length <= report.limits.reportBytes, 'Report byte limit');
    await writeFile(join(output, 'report.json.tmp'), bytes);
    await rename(join(output, 'report.json.tmp'), join(output, 'report.json'));
  };
  const step = async (name, action) => {
    remaining(); const row = {name, passed: false}; report.steps.push(row); const start = performance.now();
    try { await action(row); remaining(); row.passed = true; }
    finally { row.milliseconds = Math.round(performance.now() - start); await checkpoint(); }
  };
  const retain = async (filename, value) => {
    assert.ok(report.artifacts.length < report.limits.snapshotCount);
    const bytes = Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
    assert.ok(bytes.length <= report.limits.snapshotBytes, 'Snapshot byte limit');
    await writeFile(join(output, filename), bytes, {flag: 'wx'});
    report.artifacts.push({filename, bytes: bytes.length, sha256: sha256(bytes)});
  };
  // Cap retained browser messages and JavaScript stdout/stderr together. The
  // coordinator separately caps native process logs at its job boundary.
  const log = (kind, value) => {
    const source = String(value ?? ''), bytes = Buffer.byteLength(source);
    if (report.logBytes + bytes > report.limits.logBytes || report.logs.length >= report.limits.logEntries) {
      report.logLimitExceeded = true; report.logsTruncated = true; abort.abort(); return;
    }
    report.logBytes += bytes; report.logs.push({kind, text: short(source), sourceBytes: bytes});
  };
  const savedWrites = [process.stdout, process.stderr].map((stream) => ({stream, write: stream.write}));
  for (const {stream} of savedWrites) stream.write = function (value, encoding, callback) {
    log(stream === process.stdout ? 'stdout' : 'stderr', Buffer.isBuffer(value) ? value.toString('utf8') : value);
    const done = typeof encoding === 'function' ? encoding : callback;
    if (typeof done === 'function') queueMicrotask(done); return true;
  };
  const content = (extended = false) => {
    const order = {books: 'id', files: 'id', sections: 'id', chunks: 'id', chunks_fts: 'chunk_id',
      import_jobs: 'id', import_job_files: 'job_id,file_id', import_receipts: 'id', import_lock: 'id',
      ...(extended ? {embedding_failures: 'chunk_id,model_identity', catalog_groups: 'id',
        group_members: 'book_id', catalog_group_events: 'id'} : {})};
    return Object.fromEntries(Object.entries(order).map(([table, sort]) => [table,
      store.db.prepare(`SELECT * FROM ${table} ORDER BY ${sort}`).all()]));
  };
  const progress = () => store.db.prepare('SELECT * FROM reading_progress ORDER BY book_id').all();
  const request = async (path, body) => {
    assert.ok(report.requests.length < report.limits.directRequests);
    const raw = body === undefined ? undefined : JSON.stringify(body);
    assert.ok(Buffer.byteLength(raw || '') <= report.limits.requestBodyBytes);
    const response = await fetch(origin + path, {method: raw === undefined ? 'GET' : 'POST',
      headers: {origin, ...(raw === undefined ? {} : {'content-type': 'application/json'})}, body: raw,
      redirect: 'error', signal: AbortSignal.any([abort.signal, AbortSignal.timeout(remaining(5000))])});
    let length = 0; const parts = [];
    for await (const part of response.body) {
      length += part.length; assert.ok(length <= report.limits.responseBodyBytes, 'HTTP response byte limit'); parts.push(part);
    }
    const value = JSON.parse(Buffer.concat(parts).toString('utf8'));
    report.requests.push({path, method: raw === undefined ? 'GET' : 'POST', status: response.status,
      requestBytes: Buffer.byteLength(raw || ''), responseBytes: length});
    return {status: response.status, value};
  };
  const screenshot = async (page, profile, suffix) => {
    assert.ok(report.screenshots.length < report.limits.screenshotCount);
    const bytes = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining(5000)});
    assert.ok(bytes.length <= report.limits.screenshotBytes);
    const filename = `${profile.name}-${suffix}.png`;
    await writeFile(join(output, filename), bytes, {flag: 'wx'});
    report.screenshots.push({filename, bytes: bytes.length, sha256: sha256(bytes), width: profile.width, height: profile.height});
  };
  try {
    await checkpoint();
    for (const path of ['scripts/maintenance-browser.mjs', 'scripts/recovery-fixtures.mjs', 'package.json',
      'src/config.mjs', 'src/store.mjs', 'src/importer.mjs', 'src/process-lock.mjs', 'src/import-lifecycle.mjs',
      'src/extract.mjs', 'src/extract-worker.mjs', 'src/convert.mjs', 'src/retrieval.mjs', 'src/catalog.mjs',
      'src/server.mjs', 'public/app.js', 'public/index.html', 'public/styles.css']) {
      report.sourceHashes[path] = sha256(await readFile(new URL(`../${path}`, import.meta.url)));
    }
    const [{LibraryStore}, {importLibrary}, {createApp}, {readConfig}] = await Promise.all([
      import('../src/store.mjs'), import('../src/importer.mjs'), import('../src/server.mjs'), import('../src/config.mjs'),
    ]);
    workspace = await realpath(await mkdtemp(join(temporaryRoot, 'librarian-maintenance-browser-')));
    const originals = join(workspace, 'originals'); await mkdir(originals);
    const config = readConfig({LIBRARIAN_DATA_DIR: join(workspace, 'state'), LIBRARIAN_HOST: '127.0.0.1',
      LIBRARIAN_PORT: '0', LIBRARIAN_IMPORT_ROOTS: originals, LIBRARIAN_MAINTENANCE: 'disabled'});
    assert.equal(config.maintenanceEnabled, false);
    assert.equal(config.retrieval.chatModel, null); assert.equal(config.retrieval.embeddingModel, null);
    config.retrieval.fetch = async () => { report.modelTransportAttempts++; throw new Error('Model transport is forbidden in this synthetic driver'); };
    store = new LibraryStore(config.dbPath, {dataDir: config.dataDir});
    await step('normal-prestart-import-of-one-owned-epub', async (row) => {
      const bytes = syntheticEpub(BOOK); assert.ok(bytes.length <= 16384);
      bookId = sha256(bytes); report.fixture = {...BOOK, sha256: bookId, bytes: bytes.length};
      await writeFile(join(originals, BOOK.filename), bytes, {flag: 'wx', mode: 0o400});
      const job = await importLibrary(store, originals, {signal: AbortSignal.any([abort.signal,
        AbortSignal.timeout(remaining(report.limits.importBudgetMs))]), timeoutMs: 15000,
        maxWorkerOutputBytes: report.limits.parserBytes});
      assert.equal(job.state, 'completed'); assert.equal(job.total, 1); assert.equal(job.processed, 1);
      assert.equal(job.summary.statuses.completed, 1); assert.equal(job.summary.statuses.failed, 0);
      report.importJobId = job.id;
      imported = content(); assert.equal(imported.books.length, 1); assert.equal(imported.files.length, 1);
      sections = store.db.prepare('SELECT * FROM sections ORDER BY ordinal').all();
      assert.equal(sections.length, 2); assert.equal(imported.chunks.length, 2); assert.equal(imported.chunks_fts.length, 2);
      for (const [index, section] of sections.entries()) assert.equal(section.text, expectedSection(BOOK.sections[index]));
      chunk = store.db.prepare('SELECT * FROM chunks WHERE section_id=?').get(sections[0].id);
      assert.equal(chunk.text, sections[0].text); assert.deepEqual(progress(), []);
      assert.equal(imported.files[0].book_id, bookId); assert.equal(imported.files[0].staged, 1);
      assert.equal(imported.files[0].status, 'completed'); assert.equal(imported.import_jobs.length, 1);
      assert.equal(imported.import_lock.length, 0);
      for (const item of imported.chunks) {
        for (const field of ['embedding_json', 'embedding_model', 'embedding_dimension', 'embedding_norm']) assert.equal(item[field], null);
      }
      assert.equal(sha256(await readFile(imported.files[0].path)), bookId);
      row.books = 1; row.sections = 2; row.chunks = 2; row.jobId = job.id;
    });
    app = await createApp({config, store});
    assert.deepEqual(content(), imported, 'App startup must preserve imported content and history');
    baseline = content(true);
    for (const table of ['embedding_failures', 'catalog_groups', 'group_members', 'catalog_group_events']) assert.deepEqual(baseline[table], []);
    await retain('before-browser.json', {content: baseline, readingProgress: progress(), originalSha256: bookId});
    await new Promise((done, reject) => { app.server.once('error', reject); app.server.listen(0, '127.0.0.1', done); });
    origin = `http://127.0.0.1:${app.server.address().port}`; config.origin = origin; report.origin = origin;
    await step('real-http-status-and-maintenance-denials', async (row) => {
      const status = await request('/api/status'); assert.equal(status.status, 200);
      assert.equal(status.value.capabilities.maintenance, false);
      assert.equal(status.value.models.chat.configured, false); assert.equal(status.value.models.embedding.configured, false);
      assert.equal(status.value.totals.books, 1); assert.equal(status.value.totals.chunks, 2);
      const imports = await request('/api/imports'); assert.equal(imports.status, 200); assert.equal(imports.value.items.length, 1);
      for (const job of imports.value.items) { assert.equal(job.canResume, false); assert.equal(job.canPause, false); }
      for (const [path, body] of [['/api/imports', {root: originals}], ['/api/reindex', {bookIds: [bookId]}],
        [`/api/imports/${report.importJobId}/resume`, {}], [`/api/imports/${report.importJobId}/pause`, {}],
        ['/api/embeddings', {retryFailed: true}]]) {
        const denied = await request(path, body); assert.equal(denied.status, 403, path);
        assert.match(denied.value.error, /maintenance is disabled/i);
      }
      assert.deepEqual(content(true), baseline); assert.deepEqual(progress(), []);
      row.maintenance = false; row.deniedMutations = 5; row.immutableContentPreserved = true;
    });
    // Browser launch is deliberately after seeding and HTTP checks. Native
    // timeouts retain Playwright's launch cleanup; no uncancelled launch race.
    const require = createRequire(import.meta.url), {chromium} = require(PLAYWRIGHT);
    report.environment.playwright = require(`${PLAYWRIGHT}/package.json`).version;
    browserServer = await chromium.launchServer({host: '127.0.0.1', headless: true,
      chromiumSandbox: true, timeout: remaining(15000)});
    browserProcess = browserServer.process(); report.environment.ownedBrowserPid = browserProcess.pid;
    assert.equal(new URL(browserServer.wsEndpoint()).hostname, '127.0.0.1');
    browser = await chromium.connect(browserServer.wsEndpoint(), {timeout: remaining(5000)});
    report.environment.chromium = browser.version();
    for (const profile of [{name: 'desktop', width: 1365, height: 900}, {name: 'mobile', width: 390, height: 844}]) {
      await step(`${profile.name}-guard-search-reader-progress`, async (row) => {
        const result = {...profile, requests: [], errors: [], unexpectedRequests: []}; report.profiles.push(result);
        context = await browser.newContext({viewport: {width: profile.width, height: profile.height},
          isMobile: profile.name === 'mobile', hasTouch: profile.name === 'mobile', deviceScaleFactor: 1,
          serviceWorkers: 'block', acceptDownloads: false, locale: 'en-US', timezoneId: 'UTC', reducedMotion: 'reduce'});
        const page = await context.newPage(); page.setDefaultTimeout(remaining(5000)); page.setDefaultNavigationTimeout(remaining(7000));
        page.on('pageerror', (error) => { if (result.errors.length < 20) result.errors.push(short(error.message)); });
        page.on('console', (message) => { if (['error', 'warning'].includes(message.type())) log(`${profile.name}-${message.type()}`, message.text()); });
        await context.route('**/*', async (route) => {
          const request = route.request(), url = new URL(request.url()), method = request.method();
          const bodyBytes = request.postDataBuffer()?.length || 0;
          const readable = ['/', '/index.html', '/app.js', '/styles.css', '/favicon.ico', '/api/status', '/api/books', '/api/catalog', '/api/imports'].includes(url.pathname)
            || url.pathname === `/api/books/${bookId}` || url.pathname === `/api/books/${bookId}/thumbnail` || url.pathname === `/api/imports/${report.importJobId}`
            || sections.some((section) => url.pathname === `/api/sections/${encodeURIComponent(section.id)}`);
          const allowed = url.origin === origin && ((method === 'GET' && readable)
            || (method === 'POST' && ['/api/search', `/api/books/${bookId}/progress`].includes(url.pathname)));
          if (!allowed || bodyBytes > report.limits.requestBodyBytes || result.requests.length >= report.limits.browserRequestsPerProfile || expired) {
            if (result.unexpectedRequests.length < 20) result.unexpectedRequests.push({method, path: short(url.pathname), bodyBytes});
            return route.abort('blockedbyclient');
          }
          result.requests.push({method, path: url.pathname, query: short(url.search), bodyBytes}); return route.continue();
        });
        const goto = async (hash) => {
          const response = await page.goto(`${origin}/${hash}`, {waitUntil: 'domcontentloaded', timeout: remaining(7000)});
          if (response) assert.equal(response.status(), 200);
          await page.locator('#main[aria-busy="false"]').waitFor({state: 'visible', timeout: remaining()});
        };
        await goto(`#book/${bookId}`);
        const reindex = page.getByRole('button', {name: 'Reindex source', exact: true});
        await reindex.waitFor({state: 'visible'}); assert.equal(await reindex.isDisabled(), true);
        assert.ok(await page.getByText('Reindexing is disabled for this session. Reading, search and book details remain available.', {exact: true}).isVisible());
        result.reindexVisibleDisabled = true;
        await goto('#imports');
        const root = page.locator('#import-root'), submit = page.getByRole('button', {name: 'Import folder', exact: true});
        await root.waitFor({state: 'visible'}); assert.equal(await root.isDisabled(), true);
        assert.ok(await submit.isVisible()); assert.equal(await submit.isDisabled(), true);
        await page.locator(`[data-testid="import-job"][data-job-id="${report.importJobId}"]`).waitFor({state: 'visible'});
        assert.equal(await page.getByRole('button', {name: /^(Resume|Pause) import$/}).count(), 0);
        assert.ok(await page.getByText('Imports, reindexing and index changes are disabled for this session. You can keep reading, searching and organizing books.', {exact: true}).isVisible());
        result.importsDisabled = true; result.resumePauseActions = 0; result.maintenanceNoticeVisible = true;
        await screenshot(page, profile, 'maintenance-disabled');
        await goto('#search');
        await page.locator(`#content-scope option[value="${bookId}"]`).waitFor({state: 'attached'});
        await page.locator('#content-scope').selectOption(bookId); await page.locator('#content-query').fill(QUERY);
        const searched = page.waitForResponse((response) => response.url() === `${origin}/api/search`
          && response.request().method() === 'POST', {timeout: remaining()});
        const [response] = await Promise.all([searched,
          Promise.resolve().then(() => page.locator('[data-testid="content-submit"]').click())]);
        assert.equal(response.status(), 200);
        assert.equal(response.request().postDataJSON().bookId, bookId);
        assert.equal(response.request().postDataJSON().query, QUERY);
        const raw = await response.body(); assert.ok(raw.length <= report.limits.responseBodyBytes);
        const found = JSON.parse(raw.toString('utf8'));
        assert.equal(found.mode, 'lexical'); assert.equal(found.hits.length, 1);
        const hit = found.hits[0]; assert.equal(hit.bookId, bookId); assert.equal(hit.sectionId, sections[0].id);
        assert.equal(hit.chunkId, chunk.id); assert.equal(hit.locator.sourceSha256, bookId);
        assert.equal(hit.locator.offsetBasis, 'section_utf16_code_units');
        assert.ok(Number.isSafeInteger(hit.locator.charStart) && Number.isSafeInteger(hit.locator.charEnd));
        assert.ok(hit.locator.charStart >= 0 && hit.locator.charEnd > hit.locator.charStart && hit.locator.charEnd <= sections[0].text.length);
        assert.equal(hit.text, sections[0].text.slice(hit.locator.charStart, hit.locator.charEnd));
        const read = async (section, action) => {
          const saved = page.waitForResponse((item) => item.url() === `${origin}/api/books/${bookId}/progress`
            && item.request().method() === 'POST' && item.request().postDataJSON()?.sectionId === section.id, {timeout: remaining()});
          const [response] = await Promise.all([saved, Promise.resolve().then(action)]);
          assert.equal(response.status(), 200);
          assert.deepEqual(await response.json(), {ok: true});
          await page.locator(`[data-testid="reader-content"][data-section-id="${section.id}"]`).waitFor({state: 'visible'});
          assert.equal(await page.locator('.section-text').textContent(), section.text);
          const current = progress(); assert.equal(current.length, 1);
          assert.equal(current[0].book_id, bookId); assert.equal(current[0].section_id, section.id);
          assert.ok(Number.isFinite(Date.parse(current[0].updated_at)));
          report.progressWrites.push({profile: profile.name, sectionId: section.id, httpStatus: response.status(), row: current[0]});
        };
        await read(sections[0], () => page.locator('[data-testid="content-results"] [data-testid="citation-link"]').click());
        assert.equal(await page.locator('.citation-highlight').textContent(), hit.text);
        await read(sections[1], () => page.locator('[data-testid="reader-next"]').click());
        await screenshot(page, profile, 'reader-progress');
        const detail = await request(`/api/books/${bookId}`); assert.equal(detail.status, 200);
        assert.equal(detail.value.readingProgress.sectionId, sections[1].id);
        assert.deepEqual(content(true), baseline, 'Only reading_progress may change during browser use');
        assert.deepEqual(result.errors, []); assert.deepEqual(result.unexpectedRequests, []);
        result.search = {mode: found.mode, hits: found.hits.length, chunkId: hit.chunkId, textSha256: sha256(hit.text)};
        result.readerUsable = true; result.savedSectionId = sections[1].id; result.contentPreserved = true;
        row.searchMode = found.mode; row.progressWrites = 2;
        await bounded(context.close(), remaining(5000), 'Owned browser context did not close'); context = null;
      });
    }
    await step('after-content-and-progress-accounting', async (row) => {
      const after = content(true); assert.deepEqual(after, baseline);
      assert.deepEqual(store.db.prepare('PRAGMA integrity_check').all().map((item) => item.integrity_check), ['ok']);
      assert.deepEqual(store.db.prepare('PRAGMA foreign_key_check').all(), []);
      assert.equal(sha256(await readFile(join(originals, BOOK.filename))), bookId);
      assert.equal(sha256(await readFile(imported.files[0].path)), bookId);
      assert.equal(report.modelTransportAttempts, 0); assert.ok(!report.logLimitExceeded);
      assert.equal(report.progressWrites.length, 4); assert.equal(report.screenshots.length, 4);
      const writes = report.profiles.flatMap((profile) => profile.requests).filter((item) => item.method === 'POST');
      assert.equal(writes.filter((item) => item.path.endsWith('/progress')).length, 4);
      assert.equal(writes.filter((item) => item.path === '/api/search').length, 2);
      assert.equal(writes.length, 6, 'Browser must not send maintenance writes');
      const current = progress(); assert.equal(current.length, 1); assert.equal(current[0].section_id, sections[1].id);
      report.accounting = {beforeContentSha256: sha256(JSON.stringify(baseline)), afterContentSha256: sha256(JSON.stringify(after)),
        books: 1, files: 1, sections: 2, chunks: 2, ftsRows: 2, progressRowsBefore: 0, progressRowsAfter: 1,
        acknowledgedProgressWrites: 4, sourceBytesPreserved: true, integrity: 'ok', foreignKeyViolations: 0};
      await retain('after-browser.json', {content: after, readingProgress: current,
        acknowledgedProgressWrites: report.progressWrites, originalSha256: bookId, accounting: report.accounting});
      row.immutableContentPreserved = true; row.progressOnly = true;
    });
    report.completed = true; report.passed = true;
  } catch (error) {
    report.error = {name: error.name, message: short(error.message)}; process.exitCode = 1;
  } finally {
    clearTimeout(workTimer); abort.abort();
    try {
      if (browserServer) await bounded(browserServer.close(), cleanupRemaining(8000), 'Owned Chromium server close timed out');
      report.cleanup.browserClosed = !browserProcess || browserProcess.exitCode !== null || browserProcess.signalCode !== null;
      assert.ok(report.cleanup.browserClosed, 'Owned Chromium parent has not exited');
    } catch (error) {
      report.cleanup.browserError = short(error.message);
      try {
        if (browserServer) await bounded(browserServer.kill(), cleanupRemaining(3000), 'Owned Chromium kill timed out');
        report.cleanup.browserForcedClosed = !browserProcess || browserProcess.exitCode !== null || browserProcess.signalCode !== null;
      } catch (failure) { report.cleanup.browserKillError = short(failure.message); }
    }
    try {
      if (app) { app.server.closeAllConnections(); await bounded(app.close(), cleanupRemaining(5000), 'Owned HTTP cleanup timed out'); }
      report.cleanup.serverClosed = !app || !app.server.listening;
    } catch (error) { report.cleanup.serverError = short(error.message); }
    if (report.cleanup.serverClosed) {
      try { store?.close(); report.cleanup.storeClosed = true; }
      catch (error) { report.cleanup.storeError = short(error.message); }
    }
    if (workspace && report.cleanup.serverClosed && report.cleanup.storeClosed && report.cleanup.browserClosed) {
      try { await bounded(rm(workspace, {recursive: true, force: true}), cleanupRemaining(3000), 'Owned workspace removal timed out'); report.cleanup.workspaceRemoved = true; }
      catch (error) { report.cleanup.workspaceError = short(error.message); }
    }
    if (expired || report.logLimitExceeded || Object.keys(report.cleanup).some((key) => key.endsWith('Error'))) {
      report.passed = false; process.exitCode = 1;
    }
    report.finishedAt = new Date().toISOString(); report.elapsedMs = Math.round(performance.now() - started);
    if (report.elapsedMs >= report.limits.totalBudgetMs) { report.passed = false; report.totalBudgetExceeded = true; process.exitCode = 1; }
    report.retainedPayloadBytes = [...report.artifacts, ...report.screenshots].reduce((sum, item) => sum + item.bytes, 0);
    try {
      assert.ok(report.retainedPayloadBytes + 2 * report.limits.reportBytes <= report.limits.outputBytes);
      await checkpoint();
      if (performance.now() - started >= report.limits.totalBudgetMs && !report.totalBudgetExceeded) {
        report.passed = false; report.totalBudgetExceeded = true; process.exitCode = 1;
        report.elapsedMs = Math.round(performance.now() - started);
        await checkpoint(); // Never accept a final evidence write that exceeded the total budget.
      }
    }
    finally { for (const {stream, write} of savedWrites) stream.write = write; }
  }
}

await main().catch((error) => { process.stderr.write(`Maintenance browser driver: ${short(error.message)}\n`); process.exitCode = 1; });
