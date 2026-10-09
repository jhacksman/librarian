/**
 * Spark-only real HTTP/browser regression for fresh passage-search scope.
 * LIBRARIAN_SEARCH_SCOPE_OUTPUT must name a fresh directory beneath /tmp.
 * Uses the prepared Node 24 ARM64 and /opt/playwright runtime; installs nothing.
 * Owns 29 original synthetic EPUBs, their actual imports, a temporary database,
 * one loopback createApp server and fresh desktop/mobile browser contexts.
 * No mocked API, setContent, model calls, private corpus or quality claims.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {lstat, mkdir, mkdtemp, readFile, realpath, rename, rm, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {basename, dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {performance} from 'node:perf_hooks';
import {expectedSection, syntheticEpub} from './recovery-fixtures.mjs';

const PLAYWRIGHT = '/opt/playwright/node_modules/playwright';
const FILTER = 'Zulu shared handbook';
const QUERY = 'saffron compass beacon';
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const short = (value) => String(value ?? '').slice(0, 500);
const within = (parent, child) => {
  const part = relative(parent, child);
  return part === '' || (!isAbsolute(part) && part !== '..' && !part.startsWith(`..${sep}`));
};

async function outsideGit(directory) {
  for (let current = directory; ; current = dirname(current)) {
    try { await lstat(join(current, '.git')); throw new Error('Search-scope output must be outside every Git checkout'); }
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
  assert.equal(process.platform, 'linux', 'Execute only in the approved Spark Linux runtime');
  assert.equal(process.arch, 'arm64', 'Use the prepared Spark ARM64 runtime');
  assert.ok(/^v24\./.test(process.version), 'Use Node 24');
  assert.equal(process.env.PLAYWRIGHT_MODULE || PLAYWRIGHT, PLAYWRIGHT);
  const requested = process.env.LIBRARIAN_SEARCH_SCOPE_OUTPUT;
  assert.ok(requested && isAbsolute(requested), 'LIBRARIAN_SEARCH_SCOPE_OUTPUT must be an absolute fresh /tmp path');
  const requestedPath = resolve(requested);
  const temporaryRoot = await realpath('/tmp');
  assert.ok(within('/tmp', requestedPath) && requestedPath !== '/tmp');
  await outsideGit(requestedPath);
  await mkdir(dirname(requestedPath), {recursive: true});
  const output = join(await realpath(dirname(requestedPath)), basename(requestedPath));
  assert.ok(within(temporaryRoot, output) && output !== temporaryRoot);
  await outsideGit(output);
  await mkdir(output); // Refuse stale evidence from a previous run.
  const report = {schemaVersion: 1, completed: false, passed: false, startedAt: new Date().toISOString(),
    scope: 'Fresh passage-search scope and exact citation round trip against real LibraryStore/createApp HTTP',
    fixtures: [], sourceHashes: {}, steps: [], profiles: [], screenshots: [], artifacts: [], cleanup: {},
    environment: {platform: process.platform, architecture: process.arch, node: process.version},
    models: 'No chat or embedding model is configured; assertions exercise real lexical retrieval only',
    limits: {totalBudgetMs: 180000, importBudgetMs: 90000, reportBytes: 256 * 1024,
      artifactBytes: 128 * 1024, screenshotBytes: 5 * 1024 * 1024, browserRequests: 120},
  };
  const deadline = performance.now() + report.limits.totalBudgetMs;
  const remaining = (maximum = 15000) => {
    const value = Math.min(maximum, Math.floor(deadline - performance.now()));
    assert.ok(value > 0, 'Search-scope driver exhausted its time budget'); return value;
  };
  const checkpoint = async () => {
    report.updatedAt = new Date().toISOString();
    const raw = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
    assert.ok(raw.length <= report.limits.reportBytes);
    await writeFile(join(output, 'report.json.tmp'), raw);
    await rename(join(output, 'report.json.tmp'), join(output, 'report.json'));
  };
  const step = async (name, action) => {
    remaining(); const row = {name, passed: false}; report.steps.push(row); const started = performance.now();
    try { await action(row); row.passed = true; }
    finally { row.milliseconds = Math.round(performance.now() - started); await checkpoint(); }
  };
  const retain = async (filename, value) => {
    const raw = Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
    assert.ok(raw.length <= report.limits.artifactBytes);
    await writeFile(join(output, filename), raw, {flag: 'wx'});
    report.artifacts.push({filename, bytes: raw.length, sha256: sha256(raw)});
  };
  let workspace; let store; let app; let browser; let context;
  let origin; let target; let twin; let targetSection; let targetChunk;
  await checkpoint();
  try {
    for (const name of ['search-scope-browser.mjs', 'recovery-fixtures.mjs']) {
      report.sourceHashes[`scripts/${name}`] = sha256(await readFile(new URL(name, import.meta.url)));
    }
    for (const name of ['config.mjs', 'store.mjs', 'importer.mjs', 'process-lock.mjs', 'import-lifecycle.mjs',
      'extract.mjs', 'extract-worker.mjs', 'convert.mjs', 'retrieval.mjs', 'catalog.mjs', 'server.mjs']) {
      report.sourceHashes[`src/${name}`] = sha256(await readFile(new URL(`../src/${name}`, import.meta.url)));
    }
    for (const name of ['app.js', 'index.html', 'styles.css']) {
      report.sourceHashes[`public/${name}`] = sha256(await readFile(new URL(`../public/${name}`, import.meta.url)));
    }
    const [{LibraryStore}, {importLibrary}, {createApp}, {readConfig}] = await Promise.all([
      import('../src/store.mjs'), import('../src/importer.mjs'), import('../src/server.mjs'), import('../src/config.mjs'),
    ]);
    workspace = await realpath(await mkdtemp(join(tmpdir(), 'librarian-search-scope-')));
    const originals = join(workspace, 'originals'); await mkdir(originals);
    const config = readConfig({LIBRARIAN_DATA_DIR: join(workspace, 'state'), LIBRARIAN_PORT: '0', LIBRARIAN_IMPORT_ROOTS: originals});
    assert.equal(config.retrieval.chatModel, null); assert.equal(config.retrieval.embeddingModel, null);
    store = new LibraryStore(config.dbPath, {dataDir: config.dataDir});
    await step('seed-real-epubs-with-off-page-duplicate-titles', async (row) => {
      for (let index = 0; index < 29; index += 1) {
        const number = String(index + 1).padStart(2, '0');
        const book = {filename: `scope-${number}.epub`, title: index < 27 ? `Shelf source ${number}` : FILTER,
          sections: [{title: 'Scope evidence', paragraph: `The saffron compass beacon belongs to synthetic source ${number}. Its retained proof is scope-marker-${number}.`}]};
        const bytes = syntheticEpub(book);
        await writeFile(join(originals, book.filename), bytes, {flag: 'wx', mode: 0o400});
        const fixture = {...book, sha256: sha256(bytes), bytes: bytes.length};
        report.fixtures.push(fixture);
        if (index === 27) twin = fixture;
        if (index === 28) target = fixture;
      }
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), remaining(report.limits.importBudgetMs));
      let job;
      try { job = await importLibrary(store, originals, {signal: controller.signal, timeoutMs: 15000}); }
      finally { clearTimeout(timer); }
      assert.equal(job.state, 'completed'); assert.equal(job.total, 29); assert.equal(job.processed, 29);
      assert.equal(job.summary.statuses.completed, 29); assert.equal(job.summary.statuses.failed, 0);
      const files = store.listFiles({limit: 100});
      assert.equal(files.length, 29);
      for (const fixture of report.fixtures) {
        const file = files.find((item) => item.relative_path === fixture.filename); assert.ok(file);
        assert.equal(file.sha256, fixture.sha256); assert.equal(file.book_id, fixture.sha256);
        assert.equal(file.staged, 1); assert.equal(sha256(await readFile(file.path)), fixture.sha256);
      }
      assert.notEqual(target.sha256, twin.sha256);
      assert.equal(store.db.prepare('SELECT count(*) n FROM books').get().n, 29);
      assert.equal(store.db.prepare('SELECT count(*) n FROM sections').get().n, 29);
      assert.equal(store.db.prepare('SELECT count(*) n FROM chunks').get().n, 29);
      assert.equal(store.db.prepare('SELECT count(*) n FROM chunks_fts').get().n, 29);
      targetSection = store.db.prepare('SELECT * FROM sections WHERE book_id=?').get(target.sha256);
      targetChunk = store.db.prepare('SELECT * FROM chunks WHERE book_id=?').get(target.sha256);
      assert.equal(targetSection.text, expectedSection(target.sections[0]));
      assert.equal(targetChunk.text, targetSection.text);
      row.jobId = job.id; row.books = 29; row.targetSourceId = target.sha256; row.duplicateTitleSourceId = twin.sha256;
    });
    app = await createApp({config, store});
    await new Promise((done, reject) => { app.server.once('error', reject); app.server.listen(0, '127.0.0.1', done); });
    origin = `http://127.0.0.1:${app.server.address().port}`; config.origin = origin; report.origin = origin;
    await step('verify-target-is-not-in-first-catalog-page', async (row) => {
      const response = await fetch(`${origin}/api/catalog?sort=title&limit=24&offset=0`, {signal: AbortSignal.timeout(remaining(5000))});
      assert.equal(response.status, 200); const catalog = await response.json();
      assert.equal(catalog.total, 29); assert.equal(catalog.items.length, 24);
      const ids = catalog.items.flatMap((card) => card.members.map((member) => member.id));
      assert.ok(!ids.includes(target.sha256)); assert.ok(!ids.includes(twin.sha256));
      row.total = catalog.total; row.firstPageBooks = ids.length; row.targetAbsent = true;
      await retain('catalog-first-page.json', {total: catalog.total, sourceIds: ids});
    });
    const require = createRequire(import.meta.url); const {chromium} = require(PLAYWRIGHT);
    report.environment.playwright = require(`${PLAYWRIGHT}/package.json`).version;
    browser = await chromium.launch({headless: true, chromiumSandbox: true, timeout: remaining(20000)});
    report.environment.chromium = browser.version();
    for (const profile of [{name: 'desktop', width: 1365, height: 900}, {name: 'mobile', width: 390, height: 844}]) {
      await step(`${profile.name}-fresh-scope-and-citation-round-trip`, async (row) => {
        const result = {...profile, browserErrors: [], unexpectedRequests: [], requests: []}; report.profiles.push(result);
        context = await browser.newContext({viewport: {width: profile.width, height: profile.height},
          isMobile: profile.name === 'mobile', hasTouch: profile.name === 'mobile', deviceScaleFactor: 1,
          serviceWorkers: 'block', javaScriptEnabled: true, acceptDownloads: false,
          locale: 'en-US', timezoneId: 'UTC', reducedMotion: 'reduce'});
        const page = await context.newPage(); page.setDefaultTimeout(10000); page.setDefaultNavigationTimeout(15000);
        page.on('pageerror', (error) => { if (result.browserErrors.length < 20) result.browserErrors.push(short(error.message)); });
        await context.route('**/*', async (route) => {
          const request = route.request(); const url = new URL(request.url());
          const allowed = url.origin === origin && (['GET', 'HEAD'].includes(request.method()) || (request.method() === 'POST'
            && (['/api/search', '/api/ask'].includes(url.pathname) || /^\/api\/books\/[^/]+\/progress$/.test(url.pathname))));
          if (!allowed || result.requests.length >= report.limits.browserRequests) {
            if (result.unexpectedRequests.length < 20) result.unexpectedRequests.push({method: request.method(), pathname: url.pathname, origin: url.origin});
            return route.abort('blockedbyclient');
          }
          result.requests.push({method: request.method(), pathname: url.pathname, query: url.search});
          return route.continue();
        });
        const response = await page.goto(`${origin}/#search`, {waitUntil: 'domcontentloaded'}); assert.equal(response.status(), 200);
        await page.locator('#main[aria-busy="false"]').waitFor({state: 'visible'});
        assert.equal(await page.locator('#content-scope').inputValue(), '');
        const isFilteredPickerResponse = (item) => {
          const url = new URL(item.url());
          return url.origin === origin && url.pathname === '/api/books' && url.searchParams.get('q') === FILTER
            && item.request().method() === 'GET';
        };
        const pickerResponse = page.waitForResponse(isFilteredPickerResponse, {timeout: remaining()});
        await page.locator('#content-scope-query').fill(FILTER);
        const catalogResponse = await pickerResponse; assert.equal(catalogResponse.status(), 200);
        const choices = await catalogResponse.json();
        assert.equal(choices.total, 2);
        assert.deepEqual(choices.items.map((book) => book.id).sort(), [target.sha256, twin.sha256].sort());
        await page.locator(`#content-scope option[value="${target.sha256}"]`).waitFor({state: 'attached'});
        await page.locator(`#content-scope option[value="${twin.sha256}"]`).waitFor({state: 'attached'});
        const labels = await page.locator('#content-scope option').evaluateAll((options) => options.filter((option) => option.value).map((option) => option.textContent));
        assert.equal(new Set(labels).size, labels.length, 'Duplicate titles need distinguishable source labels');
        assert.ok(result.requests.every((request) => request.pathname !== '/api/catalog'
          && !/^\/api\/books\/[^/]+/.test(request.pathname)), 'Scope selection must not depend on prior catalog/detail visits');
        await page.locator('#content-scope').selectOption(target.sha256);
        await page.locator('#content-query').fill(QUERY);
        const searched = page.waitForResponse((item) => item.url() === `${origin}/api/search` && item.request().method() === 'POST', {timeout: remaining()});
        await page.locator('[data-testid="content-submit"]').click();
        const searchResponse = await searched; assert.equal(searchResponse.status(), 200);
        assert.equal(searchResponse.request().postDataJSON().bookId, target.sha256);
        assert.equal(searchResponse.request().postDataJSON().query, QUERY);
        const found = await searchResponse.json();
        assert.equal(found.mode, 'lexical'); assert.equal(found.hits.length, 1);
        for (const hit of found.hits) {
          assert.equal(hit.bookId, target.sha256); assert.equal(hit.sectionId, targetSection.id);
          assert.equal(hit.chunkId, targetChunk.id); assert.equal(hit.locator.sourceSha256, target.sha256);
          assert.equal(hit.locator.offsetBasis, 'section_utf16_code_units');
          assert.ok(Number.isSafeInteger(hit.locator.charStart) && Number.isSafeInteger(hit.locator.charEnd));
          assert.ok(hit.locator.charStart >= 0 && hit.locator.charEnd > hit.locator.charStart && hit.locator.charEnd <= targetSection.text.length);
          assert.equal(targetSection.text.slice(hit.locator.charStart, hit.locator.charEnd), hit.text);
          assert.ok(hit.text.includes('scope-marker-29')); assert.ok(!hit.text.includes('scope-marker-28'));
        }
        await retain(`${profile.name}-search-response.json`, found);
        const hit = found.hits[0];
        await page.locator('[data-testid="content-results"] [data-testid="citation-link"]').click();
        await page.locator(`[data-testid="reader-content"][data-section-id="${targetSection.id}"]`).waitFor({state: 'visible'});
        assert.equal(await page.locator('.section-text').textContent(), targetSection.text);
        assert.equal(await page.locator('.citation-highlight').textContent(), hit.text);
        const route = new URL(page.url()); const [reader, rangeText] = route.hash.split('?');
        assert.equal(reader, `#read/${encodeURIComponent(targetSection.id)}`);
        const range = new URLSearchParams(rangeText);
        assert.ok(range.has('start') && range.has('end'));
        assert.equal(Number(range.get('start')), hit.locator.charStart); assert.equal(Number(range.get('end')), hit.locator.charEnd);
        const returnedPickerResponse = page.waitForResponse(isFilteredPickerResponse, {timeout: remaining()});
        await page.getByRole('link', {name: 'Back to search', exact: true}).click();
        assert.equal((await returnedPickerResponse).status(), 200);
        // This second option is absent from the provisional retained selection.
        // Its appearance proves the asynchronous picker reload has rendered.
        await page.locator(`#content-scope option[value="${twin.sha256}"]`).waitFor({state: 'attached'});
        await page.locator('#content-scope').waitFor({state: 'visible'});
        await page.waitForFunction((id) => document.querySelector('#content-scope')?.value === id, target.sha256, {timeout: remaining()});
        assert.equal(new URL(page.url()).hash, '#search');
        assert.equal(await page.locator('#content-query').inputValue(), QUERY);
        assert.equal(await page.locator('#content-scope-query').inputValue(), FILTER);
        assert.equal(await page.locator('[data-testid="content-results"] [data-testid="citation-link"]').count(), 1);
        assert.equal(await page.locator('[data-testid="content-results"] [data-testid="citation-link"]').getAttribute('data-book-id'), target.sha256);
        const png = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining(10000)});
        assert.ok(png.length <= report.limits.screenshotBytes);
        const filename = `${profile.name}-scope-preserved.png`; await writeFile(join(output, filename), png, {flag: 'wx'});
        report.screenshots.push({filename, bytes: png.length, sha256: sha256(png), width: profile.width, height: profile.height});
        const pickerRequests = result.requests.filter((request) => request.pathname === '/api/books');
        assert.ok(pickerRequests.length >= 3, 'Initial, filtered, and return picker requests must reach the real catalog');
        for (const request of pickerRequests) {
          const parameters = new URLSearchParams(request.query);
          assert.equal(parameters.get('limit'), '20', 'Source picker must use bounded catalog pages');
          assert.equal(parameters.get('offset'), '0'); assert.equal(parameters.get('sort'), 'title');
        }

        // The same real source is useful from Ask while generation is disabled.
        // Quotes must stay visible and preserve the reader's exact highlight.
        await page.goto(`${origin}/#book/${target.sha256}`, {waitUntil: 'domcontentloaded'});
        await page.getByRole('button', {name: 'Ask this book', exact: true}).click();
        await page.locator('#ask-question').fill(QUERY);
        const asked = page.waitForResponse((item) => item.url() === `${origin}/api/ask`
          && item.request().method() === 'POST', {timeout: remaining()});
        await page.locator('[data-testid="ask-submit"]').click();
        const askResponse = await asked; assert.equal(askResponse.status(), 200);
        const answer = await askResponse.json();
        assert.equal(askResponse.request().postDataJSON().bookId, target.sha256);
        assert.equal(answer.responseKind, 'source_excerpts'); assert.equal(answer.abstained, true);
        assert.equal(answer.status, 'local_ai_unavailable'); assert.equal(answer.metrics.chatMs, 0);
        assert.equal(answer.extractive.passages.length, 1);
        const quote = answer.extractive.passages[0];
        assert.equal(quote.bookId, target.sha256); assert.equal(quote.sectionId, targetSection.id);
        assert.equal(quote.text, targetSection.text.slice(quote.locator.charStart, quote.locator.charEnd));
        await page.locator('[data-response-kind="source_excerpts"]').waitFor({state: 'visible'});
        assert.match(await page.locator('.response-heading').textContent(), /Local answer unavailable/);
        assert.equal(await page.locator('[data-response-kind="source_excerpts"]').getAttribute('data-generated-answer'), 'false');
        assert.equal(await page.locator('[data-response-kind="source_excerpts"]').getAttribute('data-answer-status'), 'unavailable');
        assert.equal(await page.locator('.answer-text').textContent(), answer.answer);
        const quoteLink = page.locator('[data-testid="excerpt-link"]');
        assert.equal(await quoteLink.locator('.citation-excerpt').textContent(), quote.text);
        assert.equal(await quoteLink.locator('.citation-excerpt').evaluate(el => getComputedStyle(el).whiteSpace), 'pre-wrap');
        assert.equal(await quoteLink.locator('.citation-excerpt').evaluate(el => getComputedStyle(el).display), 'block');
        await retain(`${profile.name}-excerpts-response.json`, answer);
        const excerptPng = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining(10000)});
        assert.ok(excerptPng.length <= report.limits.screenshotBytes);
        const excerptFilename = `${profile.name}-cited-excerpts.png`;
        await writeFile(join(output, excerptFilename), excerptPng, {flag: 'wx'});
        report.screenshots.push({filename: excerptFilename, bytes: excerptPng.length, sha256: sha256(excerptPng), width: profile.width, height: profile.height});
        await quoteLink.click();
        await page.locator(`[data-testid="reader-content"][data-section-id="${targetSection.id}"]`).waitFor({state: 'visible'});
        assert.equal(await page.locator('.citation-highlight').textContent(), quote.text);
        await page.getByRole('link', {name: 'Back to conversation', exact: true}).click();
        await page.locator('[data-testid="excerpt-link"]').waitFor({state: 'visible'});
        assert.equal(await page.locator('[data-testid="excerpt-link"] .citation-excerpt').textContent(), quote.text);
        result.excerptRoundTripPreserved = true;
        assert.deepEqual(result.browserErrors, []); assert.deepEqual(result.unexpectedRequests, []);
        result.targetSourceId = target.sha256; result.hitCount = found.hits.length; result.roundTripPreserved = true;
        row.targetSourceId = target.sha256; row.hitCount = found.hits.length;
        await bounded(context.close(), 10000, 'Browser context did not close'); context = null;
      });
    }
    assert.deepEqual(store.db.prepare('PRAGMA integrity_check').all().map((row) => row.integrity_check), ['ok']);
    report.completed = true; report.passed = true;
  } catch (error) {
    report.error = {name: error.name, message: short(error.message)}; process.exitCode = 1;
  } finally {
    try { if (browser) await bounded(browser.close(), 10000, 'Browser cleanup exceeded its deadline'); report.cleanup.browserClosed = true; }
    catch (error) { report.cleanup.browserError = short(error.message); }
    try { if (app) await bounded(app.close(), 10000, 'Owned HTTP server cleanup exceeded its deadline'); report.cleanup.serverClosed = true; }
    catch (error) { report.cleanup.serverError = short(error.message); }
    if (report.cleanup.serverClosed) {
      try { if (store) store.close(); report.cleanup.storeClosed = true; }
      catch (error) { report.cleanup.storeError = short(error.message); }
    }
    if (workspace && report.cleanup.serverClosed && report.cleanup.storeClosed) {
      await rm(workspace, {recursive: true, force: true}).then(() => { report.cleanup.workspaceRemoved = true; })
        .catch((error) => { report.cleanup.workspaceError = short(error.message); });
    }
    if (Object.keys(report.cleanup).some((key) => key.endsWith('Error'))) { report.passed = false; process.exitCode = 1; }
    report.finishedAt = new Date().toISOString(); await checkpoint();
  }
}

await main().catch((error) => { process.stderr.write(`Search-scope driver: ${short(error.message)}\n`); process.exitCode = 1; });
