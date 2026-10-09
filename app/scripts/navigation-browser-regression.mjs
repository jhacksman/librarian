/**
 * Source-only regression; run only through the Spark coordinator. Coordinator runs only on prepared Spark Linux ARM64 Node24.
 * Uses existing /opt/playwright; never install or run locally. No live service.
 * LIBRARIAN_NAVIGATION_APP_ROOT: absolute path of immutable staged app/ source.
 * LIBRARIAN_NAVIGATION_OUTPUT: fresh absolute /tmp directory; parent must exist.
 * Same frozen cases for original and candidate, no expected-failure allowance.
 * Coordinator must impose a 180-second process-tree bound and capped native logs.
 * This is regression evidence, not the independent black-box acceptance verdict.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {lstat, mkdir, mkdtemp, readFile, realpath, rename, rm, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {basename, dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {pathToFileURL} from 'node:url';
import {performance} from 'node:perf_hooks';

const PLAYWRIGHT = '/opt/playwright/node_modules/playwright';
// v2 permits only the owned books' bounded image endpoints. Navigation assertions are unchanged.
const QUESTION = 'How is the saffron compass beacon stored?';
const DRAFT = 'Retained synthetic navigation question';
const EXPECTED = {q: 'Navigation fixture', format: 'epub', author: 'Recovery Fixture',
  status: 'completed', sort: 'recent', view: 'list', offset: 24};
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const short = value => String(value ?? '').slice(0, 700);
const within = (parent, child) => {
  const part = relative(parent, child);
  return part === '' || (!isAbsolute(part) && part !== '..' && !part.startsWith(`..${sep}`));
};
async function bounded(promise, milliseconds, message) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(message)), milliseconds); })]); }
  finally { clearTimeout(timer); }
}
async function outsideGit(path) {
  for (let current = path; ; current = dirname(current)) {
    try { await lstat(join(current, '.git')); throw new Error('Evidence must be outside Git'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (dirname(current) === current) return;
  }
}

async function main() {
  assert.equal(process.platform, 'linux'); assert.equal(process.arch, 'arm64'); assert.match(process.version, /^v24\./);
  assert.equal(process.env.PLAYWRIGHT_MODULE || PLAYWRIGHT, PLAYWRIGHT);
  const rootInput = process.env.LIBRARIAN_NAVIGATION_APP_ROOT;
  const outInput = process.env.LIBRARIAN_NAVIGATION_OUTPUT;
  assert.ok(rootInput && isAbsolute(rootInput), 'Set source app root');
  assert.ok(outInput && isAbsolute(outInput), 'Set fresh /tmp output');
  const root = await realpath(rootInput), tmp = await realpath('/tmp');
  const requested = resolve(outInput); assert.ok(within('/tmp', requested) && requested !== '/tmp');
  const parent = await realpath(dirname(requested)); assert.ok(within(tmp, parent));
  const output = join(parent, basename(requested)); await outsideGit(output); await mkdir(output);
  const started = performance.now(), deadline = started + 150000;
  const remaining = (max = 7000) => {
    const value = Math.min(max, Math.floor(deadline - performance.now())); assert.ok(value > 0, 'Work deadline exhausted'); return value;
  };
  const report = {schemaVersion: 1, synthetic: true, completed: false, passed: false,
    acceptanceVerdict: false, sourceRoot: root, startedAt: new Date().toISOString(), sourceHashes: {},
    environment: {platform: process.platform, arch: process.arch, node: process.version},
    limits: {workMs: 150000, outerMs: 180000, browserRequestsPerProfile: 1000,
      responseTasks: 240, requestBodyBytes: 4096, responseBodyBytes: 131072,
      reportBytes: 786432, screenshots: 4, screenshotBytes: 5242880},
    limitations: ['Synthetic catalog and lexical Ask only; no model, private corpus, live LAN or answer-quality claim.',
      'Source cause of reported empty catalog is not presumed; unchanged original may pass navigation cases.',
      'Network failures in the owned fixture distinguish HTTP health from visible empty DOM; no session service is contacted.',
      'Separate fresh black-box critic owns acceptance; this driver only supplies implementation regression evidence.'],
    fixtures: [], steps: [], profiles: [], screenshots: [], cleanup: {}, modelTransportAttempts: 0};
  let workspace, store, app, browserServer, browser, context, expired = false;
  const abort = new AbortController();
  const timer = setTimeout(() => { expired = true; abort.abort(); void context?.close().catch(() => {}); app?.server.closeAllConnections(); }, 150000);
  const checkpoint = async () => {
    const bytes = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
    assert.ok(bytes.length <= report.limits.reportBytes, 'Report byte cap');
    await writeFile(join(output, 'report.json.tmp'), bytes); await rename(join(output, 'report.json.tmp'), join(output, 'report.json'));
  };
  const step = async (name, action, fatal = false) => {
    remaining(); const row = {name, passed: false}, start = performance.now(); report.steps.push(row);
    try { await action(row); remaining(); row.passed = true; }
    catch (error) { row.error = short(error.stack || error.message); if (fatal) throw error; }
    finally { row.ms = Math.round(performance.now() - start); await checkpoint(); }
    return row.passed;
  };
  try {
    await checkpoint();
    for (const path of ['public/app.js', 'public/index.html', 'public/styles.css', 'src/server.mjs', 'src/store.mjs',
      'src/importer.mjs', 'src/catalog.mjs', 'src/retrieval.mjs', 'src/config.mjs', 'scripts/recovery-fixtures.mjs']) {
      report.sourceHashes[path] = sha256(await readFile(join(root, path)));
    }
    report.sourceHashes.driver = sha256(await readFile(new URL(import.meta.url)));
    const moduleAt = path => import(pathToFileURL(join(root, path)).href);
    const [{LibraryStore}, {importLibrary}, {createApp}, {readConfig}, {syntheticEpub}] = await Promise.all([
      moduleAt('src/store.mjs'), moduleAt('src/importer.mjs'), moduleAt('src/server.mjs'), moduleAt('src/config.mjs'), moduleAt('scripts/recovery-fixtures.mjs')]);
    workspace = await realpath(await mkdtemp(join(tmp, 'librarian-navigation-')));
    const originals = join(workspace, 'originals'); await mkdir(originals);
    const config = readConfig({LIBRARIAN_DATA_DIR: join(workspace, 'state'), LIBRARIAN_HOST: '127.0.0.1',
      LIBRARIAN_PORT: '0', LIBRARIAN_IMPORT_ROOTS: originals, LIBRARIAN_MAINTENANCE: 'disabled'});
    assert.equal(config.retrieval.chatModel, null); assert.equal(config.retrieval.embeddingModel, null);
    config.retrieval.fetch = async () => { report.modelTransportAttempts++; throw new Error('No model transport allowed'); };
    store = new LibraryStore(config.dbPath, {dataDir: config.dataDir});
    await step('owned-real-catalog-import', async row => {
      for (let index = 0; index < 26; index++) {
        const book = {filename: `navigation-${String(index).padStart(2, '0')}.epub`, title: `Navigation fixture ${String(index).padStart(2, '0')}`,
          sections: [{title: 'Navigation evidence', paragraph: `The saffron compass beacon is stored in a violet cedar box. Synthetic source ${index} has unique marker navigation-${index}.`}]};
        const bytes = syntheticEpub(book); assert.ok(bytes.length < 16384);
        await writeFile(join(originals, book.filename), bytes, {flag: 'wx', mode: 0o400});
        report.fixtures.push({id: sha256(bytes), title: book.title, filename: book.filename, bytes: bytes.length});
      }
      const job = await importLibrary(store, originals, {signal: AbortSignal.any([abort.signal, AbortSignal.timeout(remaining(30000))]), timeoutMs: 15000, maxWorkerOutputBytes: 262144});
      assert.equal(job.state, 'completed'); assert.equal(job.processed, 26); assert.equal(job.summary.statuses.completed, 26);
      row.jobId = job.id;
    }, true);
    const dbFingerprint = () => sha256(JSON.stringify(Object.fromEntries(['books', 'files', 'sections', 'chunks', 'chunks_fts', 'reading_progress']
      .map(table => [table, store.db.prepare(`SELECT * FROM ${table} ORDER BY rowid`).all()]))));
    const baselineDb = dbFingerprint(); report.baselineDatabase = baselineDb;
    app = await createApp({config, store});
    await new Promise((done, reject) => { app.server.once('error', reject); app.server.listen(0, '127.0.0.1', done); });
    const origin = `http://127.0.0.1:${app.server.address().port}`; config.origin = origin; report.origin = origin;
    const require = createRequire(import.meta.url), {chromium} = require(PLAYWRIGHT);
    report.environment.playwright = require(`${PLAYWRIGHT}/package.json`).version;
    browserServer = await chromium.launchServer({host: '127.0.0.1', headless: true, chromiumSandbox: true, timeout: remaining(20000)});
    report.environment.ownedBrowserPid = browserServer.process().pid;
    assert.equal(new URL(browserServer.wsEndpoint()).hostname, '127.0.0.1');
    browser = await chromium.connect(browserServer.wsEndpoint(), {timeout: remaining()}); report.environment.chromium = browser.version();
    for (const profile of [{name: 'desktop', width: 1365, height: 900}, {name: 'mobile', width: 390, height: 844}]) {
      const result = {...profile, requests: [], responses: [], console: [], pageErrors: [], requestFailures: [], unexpectedRequests: [], states: []};
      report.profiles.push(result);
      context = await browser.newContext({viewport: {width: profile.width, height: profile.height}, isMobile: profile.name === 'mobile',
        hasTouch: profile.name === 'mobile', deviceScaleFactor: 1, serviceWorkers: 'block', acceptDownloads: false,
        locale: 'en-US', timezoneId: 'UTC', reducedMotion: 'reduce'});
      const page = await context.newPage(); page.setDefaultTimeout(5000); page.setDefaultNavigationTimeout(7000);
      let nextAsk = 'real', heldAsk = null, askHit = null, activeCase = '', selected = null, expectedIds = [];
      const responseTasks = [];
      const requestCases = new WeakMap();
      page.on('pageerror', error => { if (result.pageErrors.length < 30) result.pageErrors.push({case: activeCase, text: short(error.message)}); });
      page.on('console', msg => { if (result.console.length < 60 && ['error', 'warning'].includes(msg.type())) result.console.push({case: activeCase, type: msg.type(), text: short(msg.text())}); });
      page.on('requestfailed', req => { if (result.requestFailures.length < 30) result.requestFailures.push({case: requestCases.get(req) || activeCase, path: new URL(req.url()).pathname, error: req.failure()?.errorText}); });
      page.on('response', response => {
        if (responseTasks.length >= report.limits.responseTasks) return;
        const path = new URL(response.url()).pathname;
        if (!['/api/catalog', '/api/status', '/api/ask'].includes(path)) return;
        const task = (async () => {
          const entry = {case: requestCases.get(response.request()) || activeCase, path, status: response.status(), query: new URL(response.url()).search}; result.responses.push(entry);
          try {
            const bytes = await bounded(response.body(), remaining(), 'Response evidence timeout');
            assert.ok(bytes.length <= report.limits.responseBodyBytes); entry.bytes = bytes.length; entry.sha256 = sha256(bytes);
            const body = JSON.parse(bytes.toString('utf8'));
            if (path === '/api/catalog') Object.assign(entry, {total: body.total, cards: body.items?.length,
              sourceIds: body.items?.flatMap(card => card.members?.map(book => book.id) || [])});
            if (path === '/api/status') entry.books = body.totals?.books;
            if (path === '/api/ask') Object.assign(entry, {mode: body.mode, hits: body.hits?.length, error: body.error});
          } catch (error) { entry.evidenceError = short(error.message); }
        })(); responseTasks.push(task);
      });
      await context.route('**/*', async route => {
        const req = route.request(), url = new URL(req.url()), method = req.method(), bodyBytes = req.postDataBuffer()?.length || 0;
        requestCases.set(req, activeCase);
        const knownGet = ['/', '/index.html', '/app.js', '/styles.css', '/favicon.ico', '/api/status', '/api/catalog', '/api/books'].includes(url.pathname)
          || report.fixtures.some(book => [ `/api/books/${book.id}`, `/api/books/${book.id}/thumbnail`, `/api/books/${book.id}/cover` ].includes(url.pathname));
        const allowed = url.origin === origin && ((method === 'GET' && knownGet) || (method === 'POST' && url.pathname === '/api/ask'));
        if (!allowed || expired || bodyBytes > report.limits.requestBodyBytes || result.requests.length >= report.limits.browserRequestsPerProfile) {
          if (result.unexpectedRequests.length < 30) result.unexpectedRequests.push({method, path: url.pathname, origin: url.origin}); return route.abort('blockedbyclient');
        }
        const entry = {case: activeCase, method, path: url.pathname, query: url.search, bodyBytes};
        if (method === 'POST') entry.body = req.postDataJSON(); result.requests.push(entry);
        // Browser decoration is outside the navigation product oracle.
        if (url.pathname === '/favicon.ico') return route.fulfill({status: 204, body: ''});
        if (url.pathname === '/api/ask') {
          const mode = nextAsk; nextAsk = 'real'; entry.transport = mode;
          if (mode === 'pending') { assert.equal(heldAsk, null); heldAsk = route; askHit?.(); return; }
          if (mode === 'failure') return route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({error: 'Synthetic Ask temporarily unavailable'})});
        }
        return route.continue();
      });
      const ready = async hash => {
        await page.waitForFunction(expected => location.hash === expected && document.querySelector('#main')?.getAttribute('aria-busy') === 'false', hash, {timeout: remaining()});
        if (hash === '#library') await page.locator('#catalog-results[aria-busy="false"]').waitFor({timeout: remaining()});
      };
      const nav = async hash => { await page.locator(`#navigation a[href="${hash}"]`).click(); await ready(hash); };
      const snapshot = async name => {
        if (result.states.length >= 32) return;
        const state = await page.evaluate(() => ({hash: location.hash, title: document.title, mainBusy: document.querySelector('#main')?.getAttribute('aria-busy'),
          catalogBusy: document.querySelector('#catalog-results')?.getAttribute('aria-busy'),
          ids: [...document.querySelectorAll('[data-testid="book-card"]')].map(node => node.dataset.bookId),
          values: Object.fromEntries(['library-search', 'filter-format', 'filter-author', 'filter-status', 'library-sort'].map(id => [id, document.getElementById(id)?.value ?? null])),
          layout: document.querySelector('.view-switch [aria-pressed="true"]')?.getAttribute('aria-label'),
          pagination: document.querySelector('.pagination-info')?.textContent ?? null,
          scope: document.querySelector('.scope-title')?.textContent ?? null, draft: document.querySelector('#ask-question')?.value ?? null,
          pending: document.querySelectorAll('[data-testid="ask-pending"]').length,
          turns: [...document.querySelectorAll('.chat-turn')].map(node => node.textContent.slice(0, 1500)),
          mainText: document.querySelector('#main')?.textContent.slice(0, 6000),
          storage: Object.fromEntries(Object.entries(sessionStorage).map(([key, value]) => [key, value.slice(0, 2000)]))}));
        result.states.push({name, ...state}); return state;
      };
      const assertLibrary = async (retained = false) => {
        await ready('#library'); const state = await snapshot(activeCase);
        assert.ok(state.ids.length > 0, 'Library must visibly contain cards after navigation');
        assert.equal(state.ids.length, retained ? 2 : 24);
        if (retained) {
          assert.deepEqual(state.ids, expectedIds, 'Exact source IDs/order retained');
          assert.deepEqual(state.values, {'library-search': EXPECTED.q, 'filter-format': EXPECTED.format,
            'filter-author': EXPECTED.author, 'filter-status': EXPECTED.status, 'library-sort': EXPECTED.sort});
          assert.equal(state.layout, 'List view'); assert.match(state.pagination, /25–26 of 26/);
        }
      };
      const run = async (name, action) => {
        activeCase = name;
        return step(`${profile.name}-${name}`, async row => {
          try { await action(row); }
          finally { try { row.state = await snapshot(`${name}-final`); } catch (error) { row.snapshotError = short(error.message); } }
        });
      };
      const submit = async mode => {
        nextAsk = mode; const hit = mode === 'pending' ? new Promise(done => { askHit = done; }) : null;
        const response = mode === 'pending' ? null : page.waitForResponse(res => new URL(res.url()).pathname === '/api/ask', {timeout: remaining()});
        // Keep a click failure from leaving a rejected observer unhandled;
        // awaiting the original promise still rejects the case honestly.
        response?.catch(() => {});
        await page.locator('#ask-question').fill(QUESTION); await page.getByTestId('ask-submit').click();
        if (hit) { await bounded(hit, remaining(), 'Pending Ask was not intercepted'); askHit = null; await page.getByTestId('ask-pending').last().waitFor(); }
        else { const res = await response; assert.equal(res.status(), mode === 'failure' ? 503 : 200);
          await page.waitForFunction(() => !document.querySelector('[data-testid="ask-pending"]') && !document.querySelector('[data-testid="ask-submit"]')?.disabled, null, {timeout: remaining()}); return res.json(); }
      };
      await run('repeated-exact-library-ask-submit-library', async () => {
        const response = await page.goto(`${origin}/#library`, {waitUntil: 'domcontentloaded', timeout: remaining()}); assert.equal(response.status(), 200);
        for (let repeat = 0; repeat < 3; repeat++) { await assertLibrary(); await nav('#ask'); await submit('real'); await nav('#library'); await assertLibrary(); }
      });
      await run('ask-shows-each-hit-book-and-preserves-citation-targets', async row => {
        await nav('#ask'); const body = await submit('real');
        const excerpts = body.responseKind === 'source_excerpts';
        const citations = excerpts ? body.extractive?.passages || [] : body.citations || [];
        const ids = [...new Set([...(body.hits || []), ...citations].map(hit => hit.bookId).filter(Boolean))].sort();
        assert.ok(ids.length > 1, 'Synthetic global Ask must retrieve multiple source books');
        const turn = page.getByTestId('ask-result').last(), groups = turn.getByTestId('ask-source-book');
        assert.deepEqual((await groups.evaluateAll(nodes => nodes.flatMap(node => node.dataset.sourceBookIds.split(' ')))).sort(), ids);
        assert.equal(await groups.count(), ids.length, 'One identity per distinct synthetic source book');
        for (const id of ids) {
          const group = groups.filter({has: page.locator(`a[href="#book/${id}"]`)});
          assert.equal(await group.locator('h3').textContent(), report.fixtures.find(book => book.id === id).title);
          assert.match(await group.textContent(), /No cover available/);
          const image = group.locator('img'); await image.waitFor();
          await bounded(image.evaluate(image => image.decode()), remaining(), 'Book thumbnail did not decode');
          assert.equal(await image.evaluate(image => image.naturalWidth > 0 && image.getBoundingClientRect().width <= 60), true);
        }
        const links = turn.getByTestId(excerpts ? 'excerpt-link' : 'citation-link');
        assert.deepEqual((await links.evaluateAll(nodes => nodes.map(node => [node.dataset.bookId, node.dataset.sectionId, node.querySelector('.citation-number')?.textContent]))).sort(),
          citations.map((hit, index) => [String(hit.bookId), String(hit.sectionId), String(hit.citation ?? index + 1)]).sort());
        row.sourceIds = ids; row.citationTargets = citations.map(hit => ({bookId: hit.bookId, sectionId: hit.sectionId, locator: hit.locator}));
        const bytes = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining()});
        assert.ok(bytes.length <= report.limits.screenshotBytes); const filename = `${profile.name}-ask-identities.png`;
        await writeFile(join(output, filename), bytes, {flag: 'wx'}); report.screenshots.push({filename, bytes: bytes.length, sha256: sha256(bytes)});
        await nav('#library'); await assertLibrary();
      });
      await run('input-immediately-before-leaving-library-is-retained', async () => {
        await nav('#library');
        // Dispatch a normal input event and activate the ordinary navigation
        // link in the same task, strictly before the 250 ms search debounce.
        await page.evaluate(() => {
          const input = document.querySelector('#library-search'); input.value = 'Navigation fixture 0';
          input.dispatchEvent(new Event('input', {bubbles: true})); document.querySelector('#navigation a[href="#ask"]').click();
        });
        await ready('#ask'); await nav('#library');
        const state = await snapshot(activeCase);
        assert.equal(state.values['library-search'], 'Navigation fixture 0'); assert.equal(state.ids.length, 10);
      });
      const setup = await run('retained-query-filter-sort-view-pagination-book', async () => {
        await nav('#library'); await page.locator('#library-search').fill(EXPECTED.q); await page.locator('#library-search').press('Enter');
        await ready('#library'); await page.locator('#filter-format').selectOption(EXPECTED.format); await ready('#library');
        await page.locator('#filter-author').selectOption(EXPECTED.author); await ready('#library');
        await page.locator('#filter-status').selectOption(EXPECTED.status); await ready('#library');
        await page.locator('#library-sort').selectOption(EXPECTED.sort); await ready('#library');
        await page.getByRole('button', {name: 'List view', exact: true}).click(); await page.getByRole('button', {name: 'Next', exact: true}).click(); await ready('#library');
        expectedIds = await page.getByTestId('book-card').evaluateAll(nodes => nodes.map(node => node.dataset.bookId)); assert.equal(expectedIds.length, 2);
        selected = report.fixtures.find(book => book.id === expectedIds[0]); assert.ok(selected);
        await assertLibrary(true); await page.locator(`[data-book-id="${selected.id}"] .book-card-link`).click(); await ready(`#book/${selected.id}`);
        assert.equal(await page.getByTestId('book-detail').getAttribute('data-book-id'), selected.id);
        await page.getByRole('button', {name: 'Ask this book', exact: true}).click(); await ready('#ask');
        assert.equal(await page.locator('.scope-title').textContent(), selected.title); await submit('real');
        assert.equal(result.requests.filter(req => req.path === '/api/ask').at(-1).body.bookId, selected.id);
        await nav('#library'); await assertLibrary(true);
      });
      if (setup) {
        await run('browser-back-forward-preserves-scope-and-draft', async () => {
          await page.goBack(); await ready('#ask'); assert.equal(await page.locator('.scope-title').textContent(), selected.title);
          await page.locator('#ask-question').fill(DRAFT); await page.goForward(); await assertLibrary(true);
          await page.goBack(); await ready('#ask'); assert.equal(await page.locator('#ask-question').inputValue(), DRAFT);
          await nav('#library'); await assertLibrary(true);
        });
        await run('pending-ask-completes-away-from-library', async () => {
          await nav('#ask'); await submit('pending'); await nav('#library'); await assertLibrary(true);
          const route = heldAsk; heldAsk = null;
          const completed = page.waitForResponse(res => new URL(res.url()).pathname === '/api/ask', {timeout: remaining()});
          completed.catch(() => {});
          await route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({mode: 'excerpt', abstained: true,
            answer: 'Synthetic held Ask completed.', hits: [], warnings: []})});
          assert.equal((await completed).status(), 200);
          // Cross browser-task boundary; completion may settle after response event.
          await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
          await assertLibrary(true); await nav('#ask'); assert.equal(await page.getByTestId('ask-pending').count(), 0);
          assert.equal(await page.getByTestId('ask-submit').isEnabled(), true); await nav('#library'); await assertLibrary(true);
        });
        await run('failed-ask-leaves-catalog-and-selected-source-usable', async () => {
          await nav('#ask'); await submit('failure'); assert.match(await page.getByTestId('ask-result').last().textContent(), /Synthetic Ask temporarily unavailable/);
          assert.equal(await page.locator('.scope-title').textContent(), selected.title); await nav('#library'); await assertLibrary(true);
        });
        await run('refresh-library-retains-controls-layout-page', async () => {
          await page.reload({waitUntil: 'domcontentloaded', timeout: remaining()}); await assertLibrary(true);
        });
        await run('refresh-ask-retains-selected-source-draft-and-turn', async () => {
          await nav('#ask'); await page.locator('#ask-question').fill(DRAFT);
          const priorTurns = await page.locator('.chat-turn').count(); assert.ok(priorTurns > 0);
          await page.reload({waitUntil: 'domcontentloaded', timeout: remaining()}); await ready('#ask');
          assert.equal(await page.locator('.scope-title').textContent(), selected.title); assert.equal(await page.locator('#ask-question').inputValue(), DRAFT);
          assert.ok(await page.locator('.chat-turn').count() > 0); await nav('#library'); await assertLibrary(true);
        });
        await run('refresh-pending-ask-recovers-as-explicit-interruption', async () => {
          await nav('#ask'); await submit('pending'); await page.reload({waitUntil: 'domcontentloaded', timeout: remaining()});
          if (heldAsk) { await heldAsk.abort('aborted').catch(() => {}); heldAsk = null; }
          await ready('#ask'); assert.equal(await page.getByTestId('ask-pending').count(), 0);
          assert.equal(await page.getByTestId('ask-submit').isEnabled(), true);
          assert.match(await page.getByTestId('ask-result').last().textContent(), /refresh.*interrupt|interrupt.*refresh/i);
          assert.equal(await page.locator('.scope-title').textContent(), selected.title); await nav('#library'); await assertLibrary(true);
        });
      } else {
        // Never make a failed setup appear to have passed dependent coverage.
        for (const name of ['browser-back-forward-preserves-scope-and-draft', 'pending-ask-completes-away-from-library',
          'failed-ask-leaves-catalog-and-selected-source-usable', 'refresh-library-retains-controls-layout-page',
          'refresh-ask-retains-selected-source-draft-and-turn', 'refresh-pending-ask-recovers-as-explicit-interruption']) {
          report.steps.push({name: `${profile.name}-${name}`, passed: false, blocked: 'Retained-state setup failed; case not executed'});
        }
      }
      if (heldAsk) { await heldAsk.abort('aborted').catch(() => {}); heldAsk = null; }
      await run('catalog-health-console-network-proof', async row => {
        const healthy = await fetch(`${origin}/api/catalog?limit=24&offset=0&sort=title`, {signal: AbortSignal.timeout(remaining())});
        const bytes = Buffer.from(await healthy.arrayBuffer()); assert.ok(bytes.length <= report.limits.responseBodyBytes);
        const catalog = JSON.parse(bytes.toString('utf8')); row.directCatalog = {status: healthy.status, total: catalog.total, cards: catalog.items?.length, sha256: sha256(bytes)};
        assert.equal(healthy.status, 200); assert.equal(catalog.total, 26); assert.equal(catalog.items.length, 24);
        assert.deepEqual(result.unexpectedRequests, []); assert.deepEqual(result.pageErrors, []);
        row.unexpectedConsole = result.console.filter(entry => !(entry.case === 'failed-ask-leaves-catalog-and-selected-source-usable'
          && entry.type === 'error' && /503|Failed to load resource/i.test(entry.text)));
        row.unexpectedNetworkFailures = result.requestFailures.filter(entry => !(entry.case === 'refresh-pending-ask-recovers-as-explicit-interruption'
          && entry.path === '/api/ask' && /ABORTED|aborted/i.test(entry.error || '')));
        assert.deepEqual(row.unexpectedConsole, []); assert.deepEqual(row.unexpectedNetworkFailures, []);
        assert.ok(result.requests.length < report.limits.browserRequestsPerProfile);
      });
      await Promise.allSettled(responseTasks);
      if (report.screenshots.length < report.limits.screenshots) {
        const bytes = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining()});
        assert.ok(bytes.length <= report.limits.screenshotBytes); const filename = `${profile.name}-final.png`;
        await writeFile(join(output, filename), bytes, {flag: 'wx'}); report.screenshots.push({filename, bytes: bytes.length, sha256: sha256(bytes)});
      }
      await context.close(); context = null; await checkpoint();
    }
    await step('immutable-catalog-and-no-model-transport', async row => {
      row.database = dbFingerprint(); assert.equal(row.database, baselineDb); assert.equal(report.modelTransportAttempts, 0);
    });
    report.completed = true; report.passed = report.steps.every(step => step.passed);
    report.emptyCatalogObservations = report.profiles.flatMap(profile => profile.states.filter(state => state.hash === '#library' && state.catalogBusy === 'false' && state.ids.length === 0).map(state => ({profile: profile.name, name: state.name, mainText: state.mainText})));
    report.interpretation = report.emptyCatalogObservations.length
      ? 'Visible empty catalog observed: correlate state, filter controls and matching catalog response before attributing cause.'
      : 'No empty catalog observed in captured states; source cause of reported live symptom remains unproved.';
    if (!report.passed) process.exitCode = 1;
  } catch (error) { report.error = short(error.stack || error.message); process.exitCode = 1; }
  finally {
    clearTimeout(timer); abort.abort();
    try { if (context) await bounded(context.close(), 3000, 'Context cleanup timeout'); report.cleanup.contextClosed = true; }
    catch (error) { report.cleanup.contextError = short(error.message); }
    try { if (browser) await bounded(browser.close(), 5000, 'Browser connection cleanup timeout');
      if (browserServer) await bounded(browserServer.close(), 5000, 'Owned browser process cleanup timeout'); report.cleanup.browserClosed = true; }
    catch (error) { report.cleanup.browserError = short(error.message); try { await browserServer?.kill(); } catch {} }
    try { if (app?.server.listening) { app.server.closeAllConnections(); await bounded(new Promise(done => app.server.close(done)), 3000, 'Server cleanup timeout'); }
      report.cleanup.serverClosed = !app?.server.listening; }
    catch (error) { report.cleanup.serverError = short(error.message); }
    try { store?.close(); report.cleanup.storeClosed = true; }
    catch (error) { report.cleanup.storeError = short(error.message); }
    if (workspace && report.cleanup.serverClosed && report.cleanup.browserClosed && report.cleanup.storeClosed) {
      await rm(workspace, {recursive: true, force: true}).then(() => { report.cleanup.workspaceRemoved = true; }).catch(error => { report.cleanup.workspaceError = short(error.message); });
    }
    if (Object.keys(report.cleanup).some(key => key.endsWith('Error'))) { report.passed = false; process.exitCode = 1; }
    report.finishedAt = new Date().toISOString(); await checkpoint();
  }
}
await main().catch(error => { process.stderr.write(`Navigation browser draft: ${short(error.message)}\n`); process.exitCode = 1; });
