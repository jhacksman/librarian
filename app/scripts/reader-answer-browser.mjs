/**
 * Run ONLY as a finite job through the Spark coordinator on Linux ARM64 Node24.
 * Uses prepared /opt/playwright. Never installs packages, models or services.
 * LIBRARIAN_READER_APP_ROOT: immutable absolute staged app/ source directory.
 * LIBRARIAN_READER_OUTPUT: fresh absolute /tmp evidence directory.
 * LIBRARIAN_READER_MODEL_ENDPOINT: existing authorized loopback model origin.
 * LIBRARIAN_READER_MODEL: already installed model name, provider ollama/openai.
 * Coordinator supplies process-tree timeout 420 seconds; work bound 360 seconds.
 * Synthetic evidence only; source hashes, real HTTP/model metadata, desktop/mobile.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {lstat, mkdir, mkdtemp, readFile, realpath, rename, rm, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {pathToFileURL} from 'node:url';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const short = value => String(value ?? '').slice(0, 1200);
const within = (parent, child) => { const rel = relative(parent, child); return !isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${sep}`); };
async function outsideGit(path) {
  for (let current = path; ; current = dirname(current)) {
    try { await lstat(join(current, '.git')); throw new Error('Evidence must be outside Git'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (dirname(current) === current) return;
  }
}
// Inline citations are buttons. Derive their expected destination from the
// actual response before clicking; never accept a URL sampled after the click.
function firstCitationRoute(answer, mode) {
  const marker = /\[(\d+)\]/.exec(answer.answer); assert.ok(marker);
  const hit = answer.citations.find((item,index) => Number(item.citation ?? index+1) === Number(marker[1]));
  assert.ok(hit?.sectionId);
  const locator = hit.locator?.derived || hit.locator || {};
  const start = locator.charStart ?? hit.locator?.charStart, end = locator.charEnd ?? hit.locator?.charEnd;
  const params = new URLSearchParams({view: mode});
  for (const [key,value] of [['start',start],['end',end],['fragment',Number.isSafeInteger(start)?undefined:locator.fragment],['focus',1]])
    if (value !== undefined && value !== null && value !== '') params.set(key,value);
  return `#read/${encodeURIComponent(hit.sectionId)}?${params}`;
}
// Readiness belongs to the requested route and native source, not stale chrome.
async function waitForReaderRoute(page, remaining, hash, mode) {
  const expectedHash = hash || new URL(page.url()).hash;
  await page.waitForFunction(expected => {
    if (location.hash !== expected.hash || document.querySelector('#main')?.getAttribute('aria-busy') !== 'false') return false;
    const [route, query = ''] = expected.hash.split('?');
    if (!route.startsWith('#read/')) return true;
    const id = decodeURIComponent(route.slice(6)), reader = document.querySelector('[data-testid="reader-content"]');
    const wantedMode = expected.mode || new URLSearchParams(query).get('view');
    if (reader?.getAttribute('data-section-id') !== id || reader.getAttribute('data-view-ready') !== 'true'
      || (wantedMode && reader.getAttribute('data-reader-mode') !== wantedMode)) return false;
    if (reader.getAttribute('data-reader-mode') === 'extracted') return true;
    const physicalPage = reader.getAttribute('data-physical-page');
    if (physicalPage) {
      const canvas = reader.querySelector('[data-testid="original-pdf-canvas"]');
      return canvas?.width > 0 && canvas.height > 0
        && canvas.getAttribute('aria-label')?.startsWith(`Original PDF physical page ${physicalPage} of `)
        && reader.querySelector('[data-testid="original-pdf-viewport"]')?.getAttribute('aria-busy') === 'false';
    }
    const frame = reader.querySelector('[data-testid="original-epub-frame"]');
    const path = `/api/sections/${encodeURIComponent(id)}/original/document`;
    return !!frame?.contentDocument?.body && new URL(frame.src).pathname === path
      && new URL(frame.contentDocument.URL).pathname === path;
  }, {hash: expectedHash, mode}, {timeout: remaining(45000)});
}
function syntheticPdf() {
  const texts = ['Fixture cover. The amber lantern manual.',
    'The amber lantern burns for exactly 42 minutes. The keeper records the interval in the copper ledger.',
    'Store the amber lantern beside the eastern window. A blue cloth protects its polished handle.'];
  const objects = ['<< /Type /Catalog /Pages 2 0 R >>', '<< /Type /Pages /Kids [3 0 R 5 0 R 7 0 R] /Count 3 >>'];
  for (const [index, text] of texts.entries()) {
    objects.push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 9 0 R >> >> /Contents ${4 + index * 2} 0 R >>`);
    const stream = `BT /F1 11 Tf 48 710 Td (${text}) Tj ET\n`;
    objects.push(`<< /Length ${Buffer.byteLength(stream)} >>\nstream\n${stream}endstream`);
  }
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
  let pdf = '%PDF-1.4\n', offsets = [0];
  for (const [index, object] of objects.entries()) { offsets.push(Buffer.byteLength(pdf)); pdf += `${index + 1} 0 obj\n${object}\nendobj\n`; }
  const xref = Buffer.byteLength(pdf);
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${offsets.slice(1).map(value => `${String(value).padStart(10, '0')} 00000 n \n`).join('')}trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(pdf);
}

async function main() {
  assert.equal(process.platform, 'linux'); assert.equal(process.arch, 'arm64'); assert.match(process.version, /^v24\./);
  const sourceInput = process.env.LIBRARIAN_READER_APP_ROOT, outputInput = process.env.LIBRARIAN_READER_OUTPUT;
  assert.ok(sourceInput && isAbsolute(sourceInput)); assert.ok(outputInput && isAbsolute(outputInput));
  assert.ok(within('/tmp', resolve(outputInput)) && resolve(outputInput) !== '/tmp');
  const root = await realpath(sourceInput), parent = await realpath(dirname(outputInput)); assert.ok(within(await realpath('/tmp'), parent));
  const output = join(parent, outputInput.split('/').at(-1)); await outsideGit(output); await mkdir(output);
  const endpoint = new URL(process.env.LIBRARIAN_READER_MODEL_ENDPOINT || '');
  assert.ok(['http:', 'https:'].includes(endpoint.protocol) && ['127.0.0.1', '[::1]'].includes(endpoint.hostname));
  assert.equal(endpoint.username + endpoint.password + endpoint.search + endpoint.hash, ''); assert.ok(endpoint.pathname === '/' || ((process.env.LIBRARIAN_READER_MODEL_PROVIDER || 'ollama') === 'openai' && endpoint.pathname === '/v1'));
  const model = process.env.LIBRARIAN_READER_MODEL; assert.ok(typeof model === 'string' && model.length > 0 && model.length <= 200);
  const provider = process.env.LIBRARIAN_READER_MODEL_PROVIDER || 'ollama'; assert.ok(['ollama', 'openai'].includes(provider));
  const report = {schemaVersion: 1, synthetic: true, acceptanceVerdict: false, passed: false, completed: false,
    sourceRoot: root, startedAt: new Date().toISOString(), sourceHashes: {}, model: {endpoint: endpoint.href.replace(/\/$/, ''), name: model, provider, revision: process.env.LIBRARIAN_READER_MODEL_REVISION || ''},
    limits: {workMs: 360000, outerMs: 420000, modelCalls: 12, requestsPerProfile: 450, requestBytes: 8192, responseBytes: 262144, reportBytes: 1048576, screenshots: 4},
    limitations: ['Owned synthetic sources only; local model metadata and exact quote verification do not constitute independent semantic acceptance.', 'No private corpus, LAN service, external API, new credentials, downloads or installations.'],
    fixtures: [], steps: [], profiles: [], modelCalls: [], screenshots: [], cleanup: {}};
  let workspace, store, browser, browserServer, context, activeApp, expired = false; const apps = [];
  const deadline = Date.now() + report.limits.workMs, controller = new AbortController();
  const remaining = (max = 15000) => { const value = Math.min(max, deadline - Date.now()); assert.ok(value > 0, 'Work deadline exceeded'); return value; };
  const timer = setTimeout(() => { expired = true; controller.abort(); void context?.close().catch(() => {}); for (const app of apps) app.server.closeAllConnections(); }, report.limits.workMs);
  const checkpoint = async () => { const bytes = Buffer.from(JSON.stringify(report, null, 2)); assert.ok(bytes.length <= report.limits.reportBytes); await writeFile(join(output, 'report.json.tmp'), bytes); await rename(join(output, 'report.json.tmp'), join(output, 'report.json')); };
  const step = async (name, action) => { const entry = {name, passed: false}; report.steps.push(entry); const start = Date.now(); try { await action(entry); entry.passed = true; } catch (error) { entry.error = short(error.stack || error.message); } finally { entry.ms = Date.now() - start; await checkpoint(); } return entry.passed; };
  try {
    for (const path of ['public/app.js', 'public/styles.css', 'public/reader-views.mjs', 'public/reader-views.css', 'src/server.mjs', 'src/original-reader.mjs', 'src/retrieval.mjs', 'src/answer-context.mjs', 'src/config.mjs', 'scripts/recovery-fixtures.mjs']) report.sourceHashes[path] = sha(await readFile(join(root, path)));
    report.sourceHashes.driver = sha(await readFile(new URL(import.meta.url)));
    const at = path => import(pathToFileURL(join(root, path)).href);
    const [{LibraryStore}, {importLibrary}, {createApp}, {readConfig}, {syntheticEpub}] = await Promise.all([at('src/store.mjs'), at('src/importer.mjs'), at('src/server.mjs'), at('src/config.mjs'), at('scripts/recovery-fixtures.mjs')]);
    workspace = await mkdtemp('/tmp/librarian-reader-answer-'); const originals = join(workspace, 'originals'); await mkdir(originals);
    const epubBytes = syntheticEpub({filename: 'lantern.epub', title: 'Amber Lantern EPUB', sections: [
      {title: 'Lantern timing', paragraph: `The amber lantern burns for exactly 42 minutes. The keeper records the interval in the copper ledger. ${'The keeper checks the copper ledger before each use. '.repeat(90)}`},
      {title: 'Lantern storage', paragraph: 'Store the amber lantern beside the eastern window. A blue cloth protects its polished handle.'}]});
    const pdfBytes = syntheticPdf(); await writeFile(join(originals, 'lantern.epub'), epubBytes, {flag: 'wx', mode: 0o400}); await writeFile(join(originals, 'lantern.pdf'), pdfBytes, {flag: 'wx', mode: 0o400});
    report.fixtures = [{id: sha(epubBytes), format: 'epub', bytes: epubBytes.length}, {id: sha(pdfBytes), format: 'pdf', bytes: pdfBytes.length}];
    const makeConfig = chat => readConfig({LIBRARIAN_DATA_DIR: join(workspace, 'state'), LIBRARIAN_HOST: '127.0.0.1', LIBRARIAN_PORT: '0', LIBRARIAN_IMPORT_ROOTS: originals,
      LIBRARIAN_MAINTENANCE: 'disabled', LIBRARIAN_MODEL_ENDPOINT: endpoint.href.replace(/\/$/, ''), LIBRARIAN_CHAT_REVISION: process.env.LIBRARIAN_READER_MODEL_REVISION || '', LIBRARIAN_MODEL_PROVIDER: provider, LIBRARIAN_CHAT_MODEL: chat || '', LIBRARIAN_MODEL_TIMEOUT_MS: '35000', LIBRARIAN_CONTEXT_TOKENS: process.env.LIBRARIAN_READER_CONTEXT_TOKENS || '8192'});
    const config = makeConfig(model); store = new LibraryStore(config.dbPath, {dataDir: config.dataDir});
    const job = await importLibrary(store, originals, {signal: AbortSignal.any([controller.signal, AbortSignal.timeout(remaining(60000))]), timeoutMs: 20000, maxWorkerOutputBytes: 262144});
    assert.equal(job.state, 'completed'); assert.equal(job.processed, 2);
    const pdfBook = sha(pdfBytes), epubBook = sha(epubBytes);
    const pdfSection = store.db.prepare("SELECT * FROM sections WHERE book_id=? AND json_extract(locator_json,'$.page')=2").get(pdfBook);
    const epubSection = store.db.prepare('SELECT * FROM sections WHERE book_id=? ORDER BY ordinal LIMIT 1').get(epubBook); assert.ok(pdfSection && epubSection);
    report.fixtures[0].sectionId = epubSection.id; report.fixtures[1].sectionId = pdfSection.id;
    config.retrieval.fetch = async (input, options = {}) => {
      assert.ok(report.modelCalls.length < report.limits.modelCalls, 'Model call cap'); const url = new URL(input); assert.equal(url.origin, endpoint.origin);
      const row = {path: url.pathname, method: options.method || 'GET', requestedAt: new Date().toISOString()}; report.modelCalls.push(row);
      const start = Date.now(); const response = await fetch(input, {...options, signal: AbortSignal.any([controller.signal, ...(options.signal ? [options.signal] : [])])}); row.httpStatus = response.status; row.ms = Date.now() - start; return response;
    };
    const startApp = async settings => { const app = await createApp({config: settings, store}); apps.push(app); await new Promise((done, reject) => { app.server.once('error', reject); app.server.listen(0, '127.0.0.1', done); }); settings.origin = `http://127.0.0.1:${app.server.address().port}`; return app; };
    activeApp = await startApp(config); const missingApp = await startApp(makeConfig(null));
    const require = createRequire(import.meta.url), {chromium} = require('/opt/playwright/node_modules/playwright'); report.environment = {platform: process.platform, arch: process.arch, node: process.version, playwright: require('/opt/playwright/node_modules/playwright/package.json').version};
    browserServer = await chromium.launchServer({host: '127.0.0.1', headless: true, chromiumSandbox: true, timeout: remaining(20000)}); assert.equal(new URL(browserServer.wsEndpoint()).hostname, '127.0.0.1'); report.environment.browserPid = browserServer.process().pid;
    browser = await chromium.connect(browserServer.wsEndpoint(), {timeout: remaining()}); report.environment.chromium = browser.version();
    for (const profile of [{name: 'desktop', width: 1365, height: 900}, {name: 'mobile', width: 390, height: 844}]) {
      const evidence = {...profile, asks: [], errors: [], unexpectedRequests: [], requests: 0}; report.profiles.push(evidence);
      context = await browser.newContext({viewport: {width: profile.width, height: profile.height}, isMobile: profile.name === 'mobile', hasTouch: profile.name === 'mobile', serviceWorkers: 'block', acceptDownloads: false, locale: 'en-US', timezoneId: 'UTC', reducedMotion: 'reduce'});
      const origins = new Set(apps.map(app => app.config.origin));
      await context.route('**/*', async route => {
        const req = route.request(), url = new URL(req.url()); evidence.requests++;
        const owned = origins.has(url.origin), path = url.pathname;
        const known = ['/', '/index.html', '/app.js', '/styles.css', '/reader-views.mjs', '/reader-views.css', '/favicon.ico', '/api/status', '/api/catalog', '/api/books', '/api/ask', '/api/search'].includes(path)
          || path.startsWith('/api/reader-assets/') || report.fixtures.some(item => path.startsWith(`/api/books/${item.id}`) || path.startsWith(`/api/sections/${encodeURIComponent(item.sectionId)}`))
          || /^\/api\/sections\/[^/]+(?:\/original(?:\/document|\/resource|\/pdf)?)?$/.test(path);
        if (!owned || !known || expired || evidence.requests > report.limits.requestsPerProfile || (req.postDataBuffer()?.length || 0) > report.limits.requestBytes) { if (evidence.unexpectedRequests.length < 30) evidence.unexpectedRequests.push({origin: url.origin, path, method: req.method()}); return route.abort('blockedbyclient'); }
        if (path === '/favicon.ico') return route.fulfill({status: 204, body: ''});
        if (path === '/api/ask' && req.method() === 'POST') evidence.asks.push({origin: url.origin, request: req.postDataJSON()});
        return route.continue();
      });
      const page = await context.newPage(); page.setDefaultTimeout(remaining(18000)); page.setDefaultNavigationTimeout(remaining(20000));
      page.on('pageerror', error => { if (evidence.errors.length < 20) evidence.errors.push(short(error.message)); });
      const ready = (hash, mode) => waitForReaderRoute(page, remaining, hash, mode);
      const open = async (hash, app = activeApp) => { const response = await page.goto(`${app.config.origin}/${hash}`, {waitUntil: 'domcontentloaded', timeout: remaining(20000)}); assert.equal(response.status(), 200); await ready(); };
      const clickReady = async selector => { const target = page.getByTestId(selector); const hash = await target.getAttribute('href') || (selector === 'reader-ask-context' || selector === 'reader-return-ask' ? '#ask' : null); await target.click(); if (hash?.startsWith('#')) await page.waitForFunction(expected => location.hash === expected, hash, {timeout: remaining()}); await ready(hash, selector === 'reader-original' ? 'original' : selector === 'reader-extracted' ? 'extracted' : undefined); };
      const submit = async question => {
        const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/ask', {timeout: remaining(45000)}); responsePromise.catch(() => {});
        await page.getByTestId('ask-question').fill(question); await page.getByTestId('ask-submit').click(); const response = await responsePromise; assert.equal(response.status(), 200);
        const bytes = await response.body(); assert.ok(bytes.length <= report.limits.responseBytes); const result = JSON.parse(bytes);
        evidence.asks.at(-1).response = {sha256: sha(bytes), bytes: bytes.length, status: result.status, answerStatus: result.answerStatus, responseKind: result.responseKind, answer: result.answer, inference: result.inference, readerContext: result.readerContext, citations: result.citations?.map(hit => ({bookId: hit.bookId, sectionId: hit.sectionId, locator: hit.locator, text: hit.text})), support: result.support};
        await page.waitForFunction(() => !document.querySelector('[data-testid="ask-pending"]'), null, {timeout: remaining()}); return result;
      };
      const verifyCitations = result => { assert.ok(result.citations?.length); for (const hit of result.citations) { const section = store.db.prepare('SELECT book_id,text FROM sections WHERE id=?').get(hit.sectionId); assert.equal(section.book_id, hit.bookId); assert.equal(section.text.slice(hit.locator.charStart, hit.locator.charEnd), hit.text); } };
      await step(`${profile.name}-pdf-original-scope-generated-answer-citation-return`, async row => {
        await open(`#read/${encodeURIComponent(pdfSection.id)}`); const reader = page.getByTestId('reader-content'); assert.equal(await reader.getAttribute('data-reader-mode'), 'original');
        const canvas = page.getByTestId('original-pdf-canvas'); await canvas.waitFor(); assert.equal(await canvas.evaluate(node => node.width > 0 && node.height > 0), true);
        assert.match(await canvas.getAttribute('aria-label'), /physical page 2 of 3/); assert.match(await page.getByTestId('reader-source-identity').textContent(), /Physical page 2/);
        await clickReady('reader-ask-context'); assert.match(await page.getByTestId('ask-scope-detail').textContent(), /Physical page 2/);
        const answer = await submit('How long does the amber lantern burn according to this page?'); const request = evidence.asks.at(-1).request;
        assert.equal(request.responseMode, 'auto'); assert.deepEqual(request.readerContext, {bookId: pdfBook, sectionId: pdfSection.id, scope: 'section', page: 2});
        assert.equal(answer.inference.requested, true); assert.equal(answer.inference.completed, true); assert.equal(answer.responseKind, 'generated'); assert.equal(answer.abstained, false); assert.match(answer.answer, /42/); verifyCitations(answer);
        assert.equal(await page.getByTestId('ask-result').last().locator('.chat-response').getAttribute('data-generated-answer'), 'true');
        const inline = page.getByTestId('ask-result').last().getByTestId('inline-citation').first(); const citationHash = firstCitationRoute(answer, 'original'); await inline.click(); await ready(citationHash, 'original'); assert.equal(await page.getByTestId('reader-content').getAttribute('data-reader-mode'), 'original');
        await clickReady('reader-extracted'); const mark = page.locator('#cited-passage'); await mark.waitFor(); assert.ok((await mark.textContent()).length > 0);
        row.citationText = await mark.textContent(); assert.ok(answer.citations.some(hit => hit.text === row.citationText));
        await clickReady('reader-return-ask'); await clickReady('ask-return-reader'); assert.equal(await page.getByTestId('reader-content').getAttribute('data-reader-mode'), 'extracted');
      });
      await step(`${profile.name}-epub-reflow-scroll-toggle-back-scope-release`, async row => {
        await open(`#read/${encodeURIComponent(epubSection.id)}`); const frame = page.getByTestId('original-epub-frame'); await frame.waitFor();
        assert.equal(await page.getByTestId('reader-content').getAttribute('data-physical-page'), null);
        assert.match(await page.getByTestId('reader-source-identity').textContent(), /EPUB section/);
        const scroll = await frame.evaluate(node => { const doc = node.contentDocument; doc.scrollingElement.scrollTop = 250; doc.dispatchEvent(new Event('scroll')); return doc.scrollingElement.scrollTop; }); assert.ok(scroll > 0);
        await clickReady('reader-extracted'); await clickReady('reader-original'); await page.getByTestId('original-epub-frame').waitFor();
        await page.waitForFunction(expected => document.querySelector('[data-testid="original-epub-frame"]')?.contentDocument?.scrollingElement.scrollTop === expected, scroll, {timeout: remaining()});
        await clickReady('reader-ask-context'); assert.match(await page.getByTestId('ask-scope-detail').textContent(), /EPUB section/);
        const answer = await submit('How long does the amber lantern burn in this section?'); assert.equal(answer.inference.completed, true); assert.equal(answer.abstained, false); assert.match(answer.answer, /42/); verifyCitations(answer);
        assert.equal(evidence.asks.at(-1).request.readerContext.member, JSON.parse(epubSection.locator_json).member); assert.equal(evidence.asks.at(-1).request.readerContext.page, undefined);
        await clickReady('ask-whole-book'); assert.match(await page.getByTestId('ask-scope-detail').textContent(), /Whole book/);
        await clickReady('ask-clear-scope'); assert.equal(await page.getByTestId('ask-scope-title').textContent(), 'All books in your library');
        const empty = await submit('What does the nonexistent source book "Qzxv Unindexed Navigation Manual 908172" say about ion engines?'); assert.equal(empty.abstained, true); assert.equal(empty.answerStatus, 'no_answer');
        assert.equal(evidence.asks.at(-1).request.bookId, undefined); assert.equal(evidence.asks.at(-1).request.readerContext, undefined);
        assert.equal(await page.getByTestId('ask-result').last().locator('.chat-response').getAttribute('data-generated-answer'), 'false');
        row.originalScroll = scroll;
      });
      await step(`${profile.name}-missing-model-honest-visible-state`, async () => {
        await open(`#read/${encodeURIComponent(pdfSection.id)}`, missingApp); await clickReady('reader-ask-context');
        const answer = await submit('How long does the amber lantern burn?'); assert.equal(answer.answerStatus, 'unavailable'); assert.equal(answer.inference.completed, false);
        assert.match(await page.getByTestId('ask-result').last().textContent(), /Local answer unavailable/); assert.equal(await page.getByTestId('ask-result').last().locator('.chat-response').getAttribute('data-generated-answer'), 'false');
        assert.ok(await page.getByTestId('ask-result').last().getByTestId('excerpt-link').count());
      });
      await step(`${profile.name}-negative-scopes-before-inference`, async () => {
        const count = report.modelCalls.length;
        for (const readerContext of [{bookId: pdfBook, sectionId: epubSection.id, scope: 'section'}, {bookId: pdfBook, sectionId: pdfSection.id, scope: 'section', page: 999}, {bookId: epubBook, sectionId: epubSection.id, scope: 'section', member: '../spoof.xhtml'}]) {
          const response = await context.request.post(`${activeApp.config.origin}/api/ask`, {headers: {Origin: activeApp.config.origin}, data: {question: 'Explain this', readerContext}, timeout: remaining()}); assert.equal(response.status(), 400);
        }
        assert.equal(report.modelCalls.length, count);
      });
      await step(`${profile.name}-rendering-evidence`, async () => {
        for (const [name, section] of [['pdf', pdfSection], ['epub', epubSection]]) {
          await open(`#read/${encodeURIComponent(section.id)}?view=original`); await page.getByTestId(name === 'pdf' ? 'original-pdf-canvas' : 'original-epub-frame').waitFor();
          const bytes = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining()}); assert.ok(bytes.length <= 5242880); const filename = `${profile.name}-${name}.png`; await writeFile(join(output, filename), bytes, {flag: 'wx'}); report.screenshots.push({filename, bytes: bytes.length, sha256: sha(bytes)});
        }
        assert.deepEqual(evidence.unexpectedRequests, []); assert.deepEqual(evidence.errors, []);
      });
      await context.close(); context = null;
    }
    report.completed = true; report.passed = report.steps.every(row => row.passed) && report.modelCalls.some(row => row.httpStatus === 200);
  } catch (error) { report.error = short(error.stack || error.message); }
  finally {
    clearTimeout(timer); controller.abort();
    try { await context?.close(); await browser?.close(); await browserServer?.close(); report.cleanup.browserClosed = true; } catch (error) { report.cleanup.browserError = short(error.message); }
    for (const app of apps) { app.server.closeAllConnections(); try { await app.close(); } catch (error) { report.cleanup.serverError = short(error.message); } }
    try { store?.close(); if (workspace) await rm(workspace, {recursive: true, force: true}); report.cleanup.workspaceRemoved = true; } catch (error) { report.cleanup.workspaceError = short(error.message); }
    await checkpoint(); process.exitCode = report.completed && report.passed ? 0 : 1;
  }
  process.stdout.write(JSON.stringify({passed: report.passed, completed: report.completed, output, steps: report.steps.map(row => ({name: row.name, passed: row.passed, error: row.error}))}) + '\n');
}
// Existing-service mode consumes only coordinator-owned isolated-clone origin.
// No source import, SQLite access, model/service launch or fixture creation.
// CASES is a frozen JSON object {cases:[{id,bookId,sectionId?,page?,member?,
// charStart?,charEnd?,fragment?,question,expected?:"generated"|"no_answer"}]}.
// If sectionId is absent, page/member is resolved from the server's own catalog.
async function existingServiceProof() {
  assert.equal(process.platform, 'linux'); assert.equal(process.arch, 'arm64'); assert.match(process.version, /^v24\./);
  const origin = new URL(process.env.LIBRARIAN_READER_ORIGIN || '');
  assert.ok(origin.protocol === 'http:' && ['127.0.0.1', '[::1]'].includes(origin.hostname));
  assert.ok(origin.port && !['3474', '3475'].includes(origin.port), 'Use fresh coordinator-owned clone port');
  assert.equal(origin.username + origin.password + origin.search + origin.hash, ''); assert.equal(origin.pathname, '/');
  const root = await realpath(process.env.LIBRARIAN_READER_APP_ROOT || '');
  const casesPath = process.env.LIBRARIAN_READER_CASES; assert.ok(casesPath && isAbsolute(casesPath));
  const casesBytes = await readFile(casesPath); assert.ok(casesBytes.length <= 32768);
  const cases = JSON.parse(casesBytes).cases; assert.ok(Array.isArray(cases) && cases.length >= 1 && cases.length <= 4);
  for (const item of cases) { assert.match(item.bookId, /^[a-f0-9]{64}$/); assert.ok(typeof item.question === 'string' && item.question.length <= 4000); assert.ok(item.sectionId || Number.isInteger(item.page) || typeof item.member === 'string'); }
  const outputInput = process.env.LIBRARIAN_READER_OUTPUT; assert.ok(outputInput && isAbsolute(outputInput) && within('/tmp', resolve(outputInput)) && resolve(outputInput) !== '/tmp');
  const parent = await realpath(dirname(outputInput)); assert.ok(within(await realpath('/tmp'), parent));
  const output = join(parent, outputInput.split('/').at(-1)); await outsideGit(output); await mkdir(output);
  const report = {schemaVersion: 1, mode: 'existing_service', synthetic: false, acceptanceVerdict: false, passed: false, completed: false,
    origin: origin.origin, sourceRoot: root, casesHash: sha(casesBytes), sourceHashes: {}, startedAt: new Date().toISOString(),
    limits: {workMs: 360000, outerMs: 420000, requestsPerProfile: 500, requestBytes: 8192, responseBytes: 1048576, reportBytes: 4194304},
    limitations: ['Browser transport, source identity, exact citations and position evidence only. Independent critic owns semantic quality acceptance.', 'Existing service and model are owned and bounded by the coordinator; this driver launches only an owned browser. No import, metadata mutation, model install or private source upload.'],
    steps: [], profiles: [], screenshots: [], cleanup: {}};
  let browserServer, browser, context, expired = false; const deadline = Date.now() + report.limits.workMs;
  const remaining = (max = 15000) => { const value = Math.min(max, deadline - Date.now()); assert.ok(value > 0, 'Work deadline exceeded'); return value; };
  const timer = setTimeout(() => { expired = true; void context?.close().catch(() => {}); }, report.limits.workMs);
  const checkpoint = async () => { const bytes = Buffer.from(JSON.stringify(report, null, 2)); assert.ok(bytes.length <= report.limits.reportBytes); await writeFile(join(output, 'report.json.tmp'), bytes); await rename(join(output, 'report.json.tmp'), join(output, 'report.json')); };
  const step = async (name, action) => { const row = {name, passed: false}; report.steps.push(row); const start = Date.now(); try { await action(row); row.passed = true; } catch (error) { row.error = short(error.stack || error.message); } finally { row.ms = Date.now() - start; await checkpoint(); } return row.passed; };
  const sourceBooks = new Map(), sourceSections = new Map();
  try {
    for (const path of ['public/app.js', 'public/styles.css', 'public/reader-views.mjs', 'public/reader-views.css', 'src/server.mjs', 'src/original-reader.mjs', 'src/retrieval.mjs', 'src/answer-context.mjs', 'src/config.mjs']) report.sourceHashes[path] = sha(await readFile(join(root, path)));
    report.sourceHashes.driver = sha(await readFile(new URL(import.meta.url)));
    const require = createRequire(import.meta.url), {chromium} = require('/opt/playwright/node_modules/playwright');
    report.environment = {platform: process.platform, arch: process.arch, node: process.version, playwright: require('/opt/playwright/node_modules/playwright/package.json').version};
    browserServer = await chromium.launchServer({host: '127.0.0.1', headless: true, chromiumSandbox: true, timeout: remaining(20000)}); assert.equal(new URL(browserServer.wsEndpoint()).hostname, '127.0.0.1'); report.environment.browserPid = browserServer.process().pid;
    browser = await chromium.connect(browserServer.wsEndpoint(), {timeout: remaining()}); report.environment.chromium = browser.version();
    for (const profile of [{name: 'desktop', width: 1365, height: 900}, {name: 'mobile', width: 390, height: 844}]) {
      const evidence = {...profile, requests: 0, asks: [], errors: [], unexpectedRequests: []}; report.profiles.push(evidence);
      context = await browser.newContext({viewport: {width: profile.width, height: profile.height}, isMobile: profile.name === 'mobile', hasTouch: profile.name === 'mobile', serviceWorkers: 'block', acceptDownloads: false, reducedMotion: 'reduce', timezoneId: 'UTC', locale: 'en-US'});
      await context.route('**/*', async route => {
        const req = route.request(), url = new URL(req.url()); evidence.requests++;
        const bookMatch = /^\/api\/books\/([^/]+)(?:\/(thumbnail|cover|progress))?$/.exec(url.pathname);
        const sectionMatch = /^\/api\/sections\/([^/]+)(?:\/original(?:\/(?:pdf|document|resource))?)?$/.exec(url.pathname);
        const knownGet = ['/', '/index.html', '/app.js', '/styles.css', '/reader-views.mjs', '/reader-views.css', '/favicon.ico', '/api/status'].includes(url.pathname)
          || url.pathname.startsWith('/api/reader-assets/') || (bookMatch && sourceBooks.has(decodeURIComponent(bookMatch[1]))) || (sectionMatch && sourceSections.has(decodeURIComponent(sectionMatch[1])));
        const knownPost = url.pathname === '/api/ask' || (bookMatch && bookMatch[2] === 'progress' && sourceBooks.has(decodeURIComponent(bookMatch[1])));
        const allowed = url.origin === origin.origin && ((req.method() === 'GET' && knownGet) || (req.method() === 'POST' && knownPost));
        if (!allowed || expired || evidence.requests > report.limits.requestsPerProfile || (req.postDataBuffer()?.length || 0) > report.limits.requestBytes) { if (evidence.unexpectedRequests.length < 20) evidence.unexpectedRequests.push({origin: url.origin, method: req.method(), path: url.pathname}); return route.abort('blockedbyclient'); }
        if (url.pathname === '/favicon.ico') return route.fulfill({status: 204, body: ''});
        if (url.pathname === '/api/ask') evidence.asks.push({request: req.postDataJSON()}); return route.continue();
      });
      const getJSON = async path => { const response = await context.request.get(`${origin.origin}${path}`, {timeout: remaining(20000)}); assert.equal(response.status(), 200); const bytes = await response.body(); assert.ok(bytes.length <= report.limits.responseBytes); return JSON.parse(bytes); };
      for (const item of cases) {
        if (!sourceBooks.has(item.bookId)) { const book = await getJSON(`/api/books/${encodeURIComponent(item.bookId)}`); assert.equal(book.id, item.bookId); sourceBooks.set(book.id, book); for (const section of book.sections) sourceSections.set(section.id, {bookId: book.id, ...section}); }
        if (!item.sectionId) { const section = sourceBooks.get(item.bookId).sections.find(section => item.page != null ? section.locator.page === item.page : section.locator.member === item.member); assert.ok(section, 'Coordinator case source locator exists'); item.sectionId = section.id; }
        assert.equal(sourceSections.get(item.sectionId)?.bookId, item.bookId);
      }
      const page = await context.newPage(); page.setDefaultTimeout(remaining(20000)); page.setDefaultNavigationTimeout(remaining(20000)); page.on('pageerror', error => { if (evidence.errors.length < 20) evidence.errors.push(short(error.message)); });
      const ready = (hash, mode) => waitForReaderRoute(page, remaining, hash, mode);
      const click = async testId => { const target = page.getByTestId(testId); const hash = await target.getAttribute('href') || (testId === 'reader-ask-context' || testId === 'reader-return-ask' ? '#ask' : null); await target.click(); if (hash?.startsWith('#')) await page.waitForFunction(expected => location.hash === expected, hash, {timeout: remaining()}); await ready(hash, testId === 'reader-original' ? 'original' : testId === 'reader-extracted' ? 'extracted' : undefined); };
      for (const [index, item] of cases.entries()) await step(`${profile.name}-${item.id || index}-actual-reader-ask-citation`, async row => {
        const source = await getJSON(`/api/sections/${encodeURIComponent(item.sectionId)}`); assert.equal(source.bookId, item.bookId); if (item.page != null) assert.equal(source.locator.page, item.page); if (item.member) assert.equal(source.locator.member, item.member);
        const params = new URLSearchParams({view: 'original'}); if (item.charStart != null) { params.set('start', item.charStart); params.set('end', item.charEnd); params.set('focus', 1); } if (item.fragment) params.set('fragment', item.fragment);
        await page.goto(`${origin.origin}/#read/${encodeURIComponent(item.sectionId)}?${params}`, {waitUntil: 'domcontentloaded', timeout: remaining(20000)}); await ready();
        const isPdf = source.locator.format === 'pdf'; await page.getByTestId(isPdf ? 'original-pdf-canvas' : 'original-epub-frame').waitFor(); assert.equal(await page.getByTestId('reader-content').getAttribute('data-reader-mode'), 'original');
        row.source = {bookId: source.bookId, sectionId: source.id, locator: source.locator, textSha256: sha(Buffer.from(source.text)), textUtf16Units: source.text.length};
        row.original = isPdf ? await page.getByTestId('original-pdf-canvas').evaluate(canvas => ({width: canvas.width, height: canvas.height, visibleWidth: canvas.getBoundingClientRect().width, ariaLabel: canvas.getAttribute('aria-label')}))
          : await page.getByTestId('original-epub-frame').evaluate(frame => { const doc = frame.contentDocument; const style = frame.contentWindow.getComputedStyle(doc.body); return {memberURL: frame.src, sandbox: frame.getAttribute('sandbox'), documentHeight: doc.scrollingElement.scrollHeight, viewportHeight: doc.scrollingElement.clientHeight, stylesheets: [...doc.querySelectorAll('link[rel="stylesheet"]')].map(node => node.href), bodyStyle: {fontFamily: style.fontFamily, fontSize: style.fontSize, color: style.color}, images: [...doc.images].slice(0, 20).map(image => ({src: image.src, complete: image.complete, width: image.naturalWidth, height: image.naturalHeight}))}; });
        if (isPdf) assert.ok(row.original.width > 0 && row.original.height > 0);
        const originalPng = `${profile.name}-case${index}-original.png`, originalBytes = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining()}); assert.ok(originalBytes.length <= 5242880); await writeFile(join(output, originalPng), originalBytes, {flag: 'wx'}); report.screenshots.push({filename: originalPng, view: 'original', bytes: originalBytes.length, sha256: sha(originalBytes)});

        const getScroll = () => page.getByTestId('original-epub-frame').evaluate(frame => frame.contentDocument.scrollingElement.scrollTop);
        if (!isPdf) {
          const frame = page.getByTestId('original-epub-frame'); row.scroll = await frame.evaluate(frame => { const doc = frame.contentDocument; doc.scrollingElement.scrollTop += 240; doc.dispatchEvent(new Event('scroll')); return doc.scrollingElement.scrollTop; });
          await click('reader-extracted'); await click('reader-original'); await page.getByTestId('original-epub-frame').waitFor(); await page.waitForFunction(expected => document.querySelector('[data-testid="original-epub-frame"]')?.contentDocument?.scrollingElement.scrollTop === expected, row.scroll, {timeout: remaining()}); assert.equal(await getScroll(), row.scroll);
        }
        await click('reader-ask-context'); assert.match(await page.getByTestId('ask-scope-detail').textContent(), isPdf ? new RegExp(`Physical page ${item.page ?? source.locator.page}`) : /EPUB section/);
        const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/ask', {timeout: remaining(45000)}); responsePromise.catch(() => {});
        await page.getByTestId('ask-question').fill(item.question); await page.getByTestId('ask-submit').click(); const response = await responsePromise; assert.equal(response.status(), 200);
        const bytes = await response.body(); assert.ok(bytes.length <= report.limits.responseBytes); const answer = JSON.parse(bytes), request = evidence.asks.at(-1).request;
        assert.equal(request.responseMode, 'auto'); assert.equal(request.bookId, item.bookId); assert.equal(request.readerContext.sectionId, item.sectionId); assert.equal(request.readerContext.scope, 'section');
        if (item.charStart != null) { assert.equal(request.readerContext.charStart, item.charStart); assert.equal(request.readerContext.charEnd, item.charEnd); }
        const filename = `${profile.name}-case${index}-ask.json`; await writeFile(join(output, filename), bytes, {flag: 'wx'}); row.response = {filename, bytes: bytes.length, sha256: sha(bytes), status: answer.status, answerStatus: answer.answerStatus, inference: answer.inference};
        await page.waitForFunction(() => !document.querySelector('[data-testid="ask-pending"]'), null, {timeout: remaining()});
        if (item.expected === 'no_answer') { assert.equal(answer.abstained, true); assert.equal(answer.answerStatus, 'no_answer'); }
        else { assert.equal(answer.inference.requested, true); assert.equal(answer.inference.completed, true); assert.equal(answer.responseKind, 'generated'); assert.equal(answer.abstained, false); assert.ok(answer.answer?.trim().length > 40); }
        row.citations = [];
        for (const hit of answer.citations || []) { assert.equal(hit.bookId, item.bookId); const section = await getJSON(`/api/sections/${encodeURIComponent(hit.sectionId)}`); assert.equal(section.bookId, hit.bookId); assert.equal(section.text.slice(hit.locator.charStart, hit.locator.charEnd), hit.text); row.citations.push({sectionId: hit.sectionId, locator: hit.locator, textSha256: sha(Buffer.from(hit.text))}); }
        if (!answer.abstained) {
          assert.ok(row.citations.length); const inline = page.getByTestId('ask-result').last().getByTestId('inline-citation').first(); const citationHash = firstCitationRoute(answer, 'original'); await inline.click(); await ready(citationHash, 'original');
          assert.equal(await page.getByTestId('reader-content').getAttribute('data-book-id'), item.bookId); assert.equal(await page.getByTestId('reader-content').getAttribute('data-reader-mode'), 'original');
          await click('reader-extracted'); const text = await page.locator('#cited-passage').textContent(); assert.ok(answer.citations.some(hit => hit.text === text)); row.roundtrip = {citationTextSha256: sha(Buffer.from(text)), readerSectionId: await page.getByTestId('reader-content').getAttribute('data-section-id')};
          await click('reader-return-ask'); await click('ask-return-reader'); assert.equal(await page.getByTestId('reader-content').getAttribute('data-reader-mode'), 'extracted');
        }
        const image = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining()}); assert.ok(image.length <= 5242880); const png = `${profile.name}-case${index}.png`; await writeFile(join(output, png), image, {flag: 'wx'}); report.screenshots.push({filename: png, bytes: image.length, sha256: sha(image)});
        await page.locator('#navigation a[href="#ask"]').click(); await ready('#ask'); await click('ask-whole-book'); assert.match(await page.getByTestId('ask-scope-detail').textContent(), /Whole book/); await click('ask-clear-scope'); assert.equal(await page.getByTestId('ask-scope-title').textContent(), 'All books in your library');
      });
      await step(`${profile.name}-browser-health`, async () => { assert.deepEqual(evidence.errors, []); assert.deepEqual(evidence.unexpectedRequests, []); });
      await context.close(); context = null;
    }
    report.completed = true; report.passed = report.steps.every(row => row.passed);
  } catch (error) { report.error = short(error.stack || error.message); }
  finally { clearTimeout(timer); try { await context?.close(); await browser?.close(); await browserServer?.close(); report.cleanup.browserClosed = true; } catch (error) { report.cleanup.browserError = short(error.message); } await checkpoint(); process.exitCode = report.completed && report.passed ? 0 : 1; }
  process.stdout.write(JSON.stringify({passed: report.passed, completed: report.completed, output, steps: report.steps.map(row => ({name: row.name, passed: row.passed, error: row.error}))}) + '\n');
}
if (process.env.LIBRARIAN_READER_MODE === 'existing_service') await existingServiceProof(); else { assert.ok(!process.env.LIBRARIAN_READER_MODE || process.env.LIBRARIAN_READER_MODE === 'synthetic'); await main(); }
