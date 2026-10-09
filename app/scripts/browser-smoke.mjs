/**
 * Spark-only live HTTP smoke; run against an already staged, running application.
 * Required: LIBRARIAN_DEMO_ORIGIN, LIBRARIAN_DEMO_OUTPUT (fresh, outside Git).
 * Optional: LIBRARIAN_DEMO_BOOK_ID, LIBRARIAN_DEMO_QUESTION,
 *   LIBRARIAN_DEMO_DATASET=synthetic|existing-catalog|unspecified,
 *   LIBRARIAN_DEMO_REQUIRE_CITATION=1 (recommended for a known synthetic question).
 * Uses the prepared /opt/playwright/node_modules/playwright; installs nothing.
 * Reads/import-progress browsing only; reading-position saves are intentional.
 * No imports, metadata edits, embedding jobs, raw HTML, HAR, traces or videos.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, realpath, rename, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const DRIVER = fileURLToPath(import.meta.url);
const MODULE = '/opt/playwright/node_modules/playwright';
const UI_TIMEOUT = 15_000;
const ASK_TIMEOUT = 180_000;
const MAX_EVENTS = 100;
const MAX_PNG = 5 * 1024 * 1024;
const MAX_REPORT = 512 * 1024;
const LOCATORS = {
  mainReady: '#main[aria-busy="false"]',
  catalogReady: '#catalog-results[aria-busy="false"]',
  cards: '.book-card-link', search: '#library-search',
  format: '#filter-format', author: '#filter-author', status: '#filter-status',
  sort: '#library-sort', toc: '.toc a[href^="#read/"]',
  reader: '.section-text', readerBack: '.reader-context a[href^="#book/"]',
  question: '#ask-question', submit: '.composer button[type="submit"]',
  result: '.chat-turn .chat-response', citations: '.chat-turn:last-child .citation',
  contentResults: '[data-testid="content-results"]',
  imports: '#import-jobs', jobs: '#import-jobs .import-job',
};
const sha = (value) => createHash('sha256').update(value).digest('hex');
const short = (value) => String(value ?? '').slice(0, 500);
const within = (parent, child) => {
  const part = relative(parent, child);
  return part === '' || (!part.startsWith(`..${sep}`) && part !== '..' && !isAbsolute(part));
};

async function outsideGit(path) {
  for (let dir = path; ; dir = dirname(dir)) {
    try {
      await lstat(join(dir, '.git'));
      throw new Error('LIBRARIAN_DEMO_OUTPUT must be outside every Git checkout');
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (dirname(dir) === dir) return;
  }
}

function safeURL(value) {
  try {
    const url = new URL(value);
    // Keep routes and query names for diagnosis, never questions, tokens or filters.
    return `${url.origin}${url.pathname}${url.search ? `?keys=${[...new Set(url.searchParams.keys())].join(',')}` : ''}${url.hash.split('?')[0]}`;
  } catch { return '[invalid URL]'; }
}

async function main() {
  assert.equal(process.platform, 'linux', 'Run this driver on Spark, not native macOS');
  assert.equal(process.arch, 'arm64', 'Use the prepared Spark ARM64 browser runtime');
  assert.ok(process.env.LIBRARIAN_DEMO_ORIGIN, 'LIBRARIAN_DEMO_ORIGIN is required');
  const parsed = new URL(process.env.LIBRARIAN_DEMO_ORIGIN);
  assert.ok(['http:', 'https:'].includes(parsed.protocol));
  assert.ok(!parsed.username && !parsed.password && !parsed.search && !parsed.hash && parsed.pathname === '/',
    'LIBRARIAN_DEMO_ORIGIN must be a single HTTP(S) origin without credentials');
  const origin = parsed.origin;
  const requestedOutput = process.env.LIBRARIAN_DEMO_OUTPUT;
  assert.ok(requestedOutput && isAbsolute(requestedOutput), 'LIBRARIAN_DEMO_OUTPUT must be an absolute path');
  const requested = resolve(requestedOutput);
  await outsideGit(requested);
  await mkdir(dirname(requested), { recursive: true });
  const parent = await realpath(dirname(requested));
  const output = join(parent, requested.slice(dirname(requested).length + 1));
  await outsideGit(output);
  assert.ok(!within(resolve(dirname(DRIVER), '..'), output), 'Output must not be inside app source');
  await mkdir(output); // Refuse existing output: stale success/screenshots cannot survive.
  const dataset = process.env.LIBRARIAN_DEMO_DATASET || 'unspecified';
  assert.ok(['synthetic', 'existing-catalog', 'unspecified'].includes(dataset), 'Invalid dataset label');
  const question = process.env.LIBRARIAN_DEMO_QUESTION || 'What is the central argument?';
  assert.ok(question.trim() && question.length <= 4_000, 'Question must contain 1–4000 characters');
  const requestedBook = process.env.LIBRARIAN_DEMO_BOOK_ID || null;
  assert.ok(!requestedBook || (requestedBook.length <= 256 && !/[\u0000-\u001f]/u.test(requestedBook)), 'Invalid book ID');
  const requireCitation = process.env.LIBRARIAN_DEMO_REQUIRE_CITATION === '1';
  const report = {
    schema_version: 1, completed: false, passed: false, started_at: new Date().toISOString(),
    scope: 'Actual desktop/mobile journeys against a running Librarian HTTP application; no setContent',
    origin, driver_sha256: sha(await readFile(DRIVER)), locators: LOCATORS,
    dataset: { declared: dataset, provenance: 'Operator label; this driver does not certify corpus provenance or answer quality', real_book_validation_claimed: false },
    question: { sha256: sha(question), characters: question.length, explicitly_supplied: Boolean(process.env.LIBRARIAN_DEMO_QUESTION) },
    requested_book_id: requestedBook, require_citation: requireCitation,
    privacy: 'Viewport screenshots only; no raw DOM, book/answer text, request bodies, response bodies, trace, HAR or video retained',
    allowed_writes: ['POST /api/ask', 'POST /api/search', 'POST /api/books/:id/progress'],
    environment: { platform: process.platform, arch: process.arch, node: process.version },
    journeys: [], console_errors: [], console_warnings: [], page_errors: [],
    http_failures: [], request_failures: [], guarded_requests: [], diagnostics_dropped: 0,
    api_timings: [], timings_are_a_benchmark: false, browser_closed: false,
  };
  const checkpoint = async () => {
    report.updated_at = new Date().toISOString();
    const raw = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
    assert.ok(raw.length <= MAX_REPORT, 'Report size limit exceeded');
    await writeFile(join(output, 'report.json.tmp'), raw);
    await rename(join(output, 'report.json.tmp'), join(output, 'report.json'));
  };
  const event = (key, value) => {
    if (report[key].length < MAX_EVENTS) report[key].push(value);
    else report.diagnostics_dropped += 1;
  };
  let browser;
  await checkpoint();
  try {
    assert.equal(process.env.PLAYWRIGHT_MODULE || MODULE, MODULE, 'Use the existing prepared Playwright module');
    const require = createRequire(import.meta.url);
    const { chromium } = require(MODULE);
    report.environment.playwright = require(`${MODULE}/package.json`).version;
    browser = await chromium.launch({ headless: true, chromiumSandbox: true, timeout: 20_000 });
    report.environment.chromium = browser.version();

    for (const [name, viewport, mobile] of [['desktop', { width: 1365, height: 900 }, false], ['mobile', { width: 390, height: 844 }, true]]) {
      const journey = { name, viewport, mobile, completed: false, steps: [], screenshots: [], unavailable: [] };
      report.journeys.push(journey);
      const context = await browser.newContext({ viewport, deviceScaleFactor: 1, isMobile: mobile,
        hasTouch: mobile, javaScriptEnabled: true, serviceWorkers: 'block', acceptDownloads: false,
        locale: 'en-US', timezoneId: 'UTC', colorScheme: 'light', reducedMotion: 'reduce' });
      const page = await context.newPage();
      page.setDefaultTimeout(UI_TIMEOUT);
      page.setDefaultNavigationTimeout(30_000);
      const starts = new WeakMap();
      const guarded = new WeakSet();
      let stage = 'open';
      page.on('console', (message) => {
        if (message.type() === 'error' || message.type() === 'warning') {
          event(message.type() === 'error' ? 'console_errors' : 'console_warnings', { journey: name, stage, text: short(message.text()) });
        }
      });
      page.on('pageerror', (error) => event('page_errors', { journey: name, stage, name: error.name, message: short(error.message) }));
      page.on('request', (request) => starts.set(request, performance.now()));
      page.on('response', (response) => {
        const request = response.request();
        const record = { journey: name, stage, method: request.method(), url: safeURL(response.url()), status: response.status(),
          milliseconds_to_headers: Math.round((performance.now() - (starts.get(request) ?? performance.now())) * 100) / 100 };
        if (new URL(response.url()).pathname.startsWith('/api/')) event('api_timings', record);
        if (response.status() >= 400) event('http_failures', record);
      });
      page.on('requestfailed', (request) => {
        if (!guarded.has(request)) event('request_failures', { journey: name, stage, method: request.method(), url: safeURL(request.url()), error: short(request.failure()?.errorText) });
      });
      await context.route('**/*', async (route) => {
        const request = route.request(); const url = new URL(request.url()); const method = request.method();
        const allowed = url.origin === origin && (['GET', 'HEAD'].includes(method)
          || (method === 'POST' && (['/api/ask', '/api/search'].includes(url.pathname) || /^\/api\/books\/[^/]+\/progress$/.test(url.pathname))));
        if (allowed) return route.continue();
        guarded.add(request);
        event('guarded_requests', { journey: name, stage, method, url: safeURL(url.href), reason: 'Outside origin or outside explicitly allowed read/ask/reading-progress workflow' });
        return route.abort('blockedbyclient');
      });
      const ready = () => page.locator(LOCATORS.mainReady).waitFor({ state: 'visible' });
      const readerReady = async (hash) => {
        const base = hash.split('?')[0];
        const sectionId = decodeURIComponent(base.slice('#read/'.length));
        await page.waitForFunction((expected) => document.querySelector('#main')?.getAttribute('aria-busy') === 'false'
          && document.querySelector('[data-testid="reader-content"]')?.getAttribute('data-section-id') === expected,
        sectionId, { timeout: UI_TIMEOUT });
        await page.locator(LOCATORS.reader).waitFor({ state: 'visible' });
        assert.equal(new URL(page.url()).hash.split('?')[0], base);
      };
      const shot = async (label) => {
        const filename = `${name}-${label}.png`;
        const raw = await page.screenshot({ type: 'png', fullPage: false, animations: 'disabled', timeout: UI_TIMEOUT });
        assert.ok(raw.length <= MAX_PNG, 'Viewport screenshot exceeds 5 MiB');
        await writeFile(join(output, filename), raw, { flag: 'wx' });
        journey.screenshots.push({ filename, bytes: raw.length, sha256: sha(raw), width: raw.readUInt32BE(16), height: raw.readUInt32BE(20), full_page: false, route: safeURL(page.url()) });
      };
      const step = async (label, locator, action) => {
        stage = label; const start = performance.now();
        const row = { name: label, locator, passed: false }; journey.steps.push(row);
        try { await action(row); row.passed = true; }
        finally { row.milliseconds = Math.round((performance.now() - start) * 100) / 100; await checkpoint(); }
      };
      const catalogChange = async (field, value, action) => {
        const pending = page.waitForResponse((response) => {
          const url = new URL(response.url());
          return response.request().method() === 'GET' && url.origin === origin && url.pathname === '/api/catalog' && (url.searchParams.get(field) || '') === value;
        });
        const [response] = await Promise.all([pending, action()]);
        assert.equal(response.status(), 200, 'Catalog request failed');
        await response.finished();
        await page.locator(LOCATORS.catalogReady).waitFor({ state: 'visible' });
      };
      const getBook = async (id) => {
        const url = `${origin}/api/books/${encodeURIComponent(id)}`;
        const start = performance.now();
        let response;
        try { response = await context.request.get(url, { timeout: UI_TIMEOUT }); }
        catch (error) {
          event('request_failures', { journey: name, stage, method: 'GET', url: safeURL(url), error: short(error.message), channel: 'metadata observation' });
          throw error;
        }
        const observation = { journey: name, stage, method: 'GET', url: safeURL(url), status: response.status(),
          milliseconds: Math.round((performance.now() - start) * 100) / 100, channel: 'metadata observation' };
        event('api_timings', observation);
        if (response.status() >= 400) event('http_failures', observation);
        assert.equal(response.status(), 200, 'Reading book identity/progress failed');
        return response.json(); // Keep metadata in memory only; never persist this body.
      };
      let bookId;
      let selectedBookTitle;
      let initialSection;
      try {
        await step('library', '#navigation a[href="#library"], #catalog-results', async (row) => {
          const response = await page.goto(`${origin}/#library`, { waitUntil: 'domcontentloaded' });
          assert.equal(response?.status(), 200, 'Application document did not load');
          await ready(); await page.locator(LOCATORS.catalogReady).waitFor({ state: 'visible' });
          row.visible_books = await page.locator(LOCATORS.cards).count();
          assert.ok(row.visible_books > 0, 'Stage a readable synthetic catalog or provide an existing catalog before running');
          await shot('01-library');
        });
        await step('browse-filters', `${LOCATORS.search}, ${LOCATORS.format}, ${LOCATORS.status}, ${LOCATORS.sort}`, async (row) => {
          const title = (await page.locator(`${LOCATORS.cards} h2`).first().innerText()).slice(0, 120);
          await catalogChange('q', title.trim(), async () => { await page.locator(LOCATORS.search).fill(title); await page.locator(LOCATORS.search).press('Enter'); });
          assert.ok(await page.locator(LOCATORS.cards).count() > 0, 'Searching the visible title lost every result');
          await catalogChange('q', '', async () => { await page.locator(LOCATORS.search).fill(''); await page.locator(LOCATORS.search).press('Enter'); });
          row.filters = [];
          for (const [selector, field] of [[LOCATORS.format, 'format'], [LOCATORS.author, 'author'], [LOCATORS.status, 'status']]) {
            const values = await page.locator(`${selector} option`).evaluateAll((options) => options.map((item) => item.value).filter(Boolean));
            if (!values.length) { journey.unavailable.push(`${field} filter has no populated options`); continue; }
            await catalogChange(field, values[0], () => page.locator(selector).selectOption(values[0]));
            row.filters.push({ field, value_sha256: sha(values[0]), result_count: await page.locator(LOCATORS.cards).count() });
            await catalogChange(field, '', () => page.locator(selector).selectOption(''));
          }
          await catalogChange('sort', 'recent', () => page.locator(LOCATORS.sort).selectOption('recent'));
          await catalogChange('sort', 'title', () => page.locator(LOCATORS.sort).selectOption('title'));
          await shot('02-filtered-library');
        });
        await step('book-detail-and-toc', `${LOCATORS.cards}, ${LOCATORS.toc}`, async (row) => {
          const links = page.locator(LOCATORS.cards);
          const candidates = requestedBook ? [requestedBook] : await links.evaluateAll((items) => items.slice(0, 10).map((item) => item.closest('[data-book-id]')?.dataset.bookId).filter(Boolean));
          let book;
          for (const id of candidates) { const candidate = await getBook(id); if (Array.isArray(candidate.toc) && candidate.toc.some(item => !item.unresolvedFragment)) { bookId = id; book = candidate; break; } }
          assert.ok(book, 'No readable book among candidates; supply LIBRARIAN_DEMO_BOOK_ID for a staged book');
          selectedBookTitle = book.title;
          journey.book_id = bookId; journey.reading_progress_before = book.readingProgress || null;
          const href = `#book/${encodeURIComponent(bookId)}`;
          const matching = page.locator(LOCATORS.cards);
          let clicked = false;
          for (let i = 0; i < await matching.count(); i += 1) {
            if (await matching.nth(i).getAttribute('href') === href) { await matching.nth(i).click(); clicked = true; break; }
          }
          if (!clicked) await page.goto(`${origin}/${href}`, { waitUntil: 'domcontentloaded' });
          row.selection = clicked ? 'Visible catalog link clicked' : 'Explicit requested book route opened';
          await ready(); await page.locator(LOCATORS.toc).first().waitFor({ state: 'visible' });
          row.sections = await page.locator(LOCATORS.toc).count();
          assert.equal(row.sections, book.toc.filter(item => !item.unresolvedFragment).length, 'Clickable TOC differs from resolved book destinations');
          row.unresolved_toc_destinations = book.toc.filter(item => item.unresolvedFragment).length;
          journey.toc_sections = row.sections;
          initialSection = await page.locator(LOCATORS.toc).first().getAttribute('href');
          await shot('03-book-detail');
        });
        await step('reader-next-previous-return', `${LOCATORS.toc}, .reader-pagination a, ${LOCATORS.readerBack}`, async (row) => {
          await page.locator(LOCATORS.toc).first().click(); await readerReady(initialSection);
          row.section_text_characters = await page.locator(LOCATORS.reader).evaluate((node) => node.textContent.length);
          await shot('04-reader');
          const next = page.getByRole('link', { name: 'Next section', exact: true });
          if (await next.count()) {
            const nextHash = await next.getAttribute('href');
            assert.notEqual(nextHash, initialSection);
            await next.click(); await readerReady(nextHash);
            await page.getByRole('link', { name: 'Previous section', exact: true }).click(); await readerReady(initialSection);
            row.next_previous = 'passed';
          } else {
            row.next_previous = 'unavailable'; journey.unavailable.push('Selected TOC target has no next physical section');
          }
          await page.locator(LOCATORS.readerBack).click(); await ready();
          await page.locator(LOCATORS.toc).first().waitFor({ state: 'visible' });
          assert.equal(new URL(page.url()).hash, `#book/${encodeURIComponent(bookId)}`);
        });
        await step('ask-and-citation-jump', `button[name="Ask this book"], ${LOCATORS.question}, ${LOCATORS.submit}, ${LOCATORS.citations}`, async (row) => {
          await page.getByRole('button', { name: 'Ask this book', exact: true }).click(); await ready();
          await page.locator(LOCATORS.question).fill(question);
          const waiting = page.waitForResponse((response) => response.url() === `${origin}/api/ask` && response.request().method() === 'POST', { timeout: ASK_TIMEOUT });
          const [response] = await Promise.all([waiting, page.locator(LOCATORS.submit).click()]);
          assert.equal(response.status(), 200, 'Ask request failed');
          const answer = await response.json();
          assert.equal(typeof answer.abstained, 'boolean', 'Ask must explicitly report abstention');
          assert.ok(Array.isArray(answer.citations), 'Ask must return a citation list');
          assert.ok(answer.citations.every((citation) => citation.bookId === bookId), 'Scoped ask returned a different book');
          await page.locator(LOCATORS.result).last().waitFor({ state: 'visible', timeout: ASK_TIMEOUT });
          row.abstained = answer.abstained; row.mode = short(answer.mode); row.citations = answer.citations.length;
          row.answer_characters = typeof answer.answer === 'string' ? answer.answer.length : 0;
          row.answer_sha256 = typeof answer.answer === 'string' ? sha(answer.answer) : null;
          await page.locator(LOCATORS.result).last().scrollIntoViewIfNeeded(); await shot('05-ask');
          const citations = page.locator(`${LOCATORS.citations}:not(:disabled)`);
          if (await citations.count()) {
            const first = answer.citations.find((citation) => citation.sectionId ?? citation.section_id);
            assert.ok(first, 'Enabled citation has no source section in response');
            await citations.first().click();
            await readerReady(`#read/${encodeURIComponent(first.sectionId ?? first.section_id)}`);
            if (Number.isSafeInteger(first.locator?.charStart) && Number.isSafeInteger(first.locator?.charEnd)) {
              const range = new URLSearchParams(new URL(page.url()).hash.split('?')[1] || '');
              assert.equal(range.get('start'), String(first.locator.charStart), 'Citation start offset differs');
              assert.equal(range.get('end'), String(first.locator.charEnd), 'Citation end offset differs');
              const mark = page.locator('#cited-passage');
              await mark.waitFor({ state: 'visible' });
              assert.equal(await mark.textContent(), first.text, 'Highlighted passage differs from cited excerpt');
              row.citation_range = { start: first.locator.charStart, end: first.locator.charEnd, exact_excerpt_match: true };
            }
            row.citation_jump = 'passed'; await shot('06-citation-reader');
          } else {
            row.citation_jump = 'unavailable'; journey.unavailable.push('Ask returned no enabled reader citation');
            assert.ok(!requireCitation, 'This synthetic smoke requires a usable source citation');
          }
        });
        await step('content-search-and-return', `#content-query, ${LOCATORS.contentResults}`, async (row) => {
          await page.locator('#navigation a[href="#search"]').click(); await ready();
          await page.locator('#content-query').fill(question);
          await page.locator('#content-scope-query').fill(selectedBookTitle);
          await page.locator('#content-scope-query').press('Enter');
          await page.locator('#content-scope').selectOption(bookId);
          const pending = page.waitForResponse(response => response.url() === `${origin}/api/search` && response.request().method() === 'POST');
          const [response] = await Promise.all([pending, page.locator('[data-testid="content-submit"]').click()]);
          assert.equal(response.status(), 200); const result = await response.json();
          assert.ok(result.hits.every(hit => hit.bookId === bookId), 'Content search escaped its selected source');
          row.hits = result.hits.length;
          if (result.hits.length) {
            const first = result.hits[0];
            await page.locator(`${LOCATORS.contentResults} [data-testid="citation-link"]`).first().click();
            await readerReady(`#read/${encodeURIComponent(first.sectionId)}`);
            assert.equal(await page.locator('#cited-passage').textContent(), first.text);
            await page.getByRole('link', { name: 'Back to search', exact: true }).click(); await ready();
            assert.equal(await page.locator('#content-query').inputValue(), question);
            assert.equal(await page.locator('#content-scope-query').inputValue(), selectedBookTitle);
            assert.equal(await page.locator('#content-scope').inputValue(), bookId);
            assert.equal(await page.locator(`${LOCATORS.contentResults} [data-testid="citation-link"]`).count(), result.hits.length);
            row.restored_after_reader = true;
          } else journey.unavailable.push('This question returned no direct content-search hits');
          await shot('07-content-search');
        });
        await step('import-progress-read-only', '#navigation a[href="#imports"], #import-jobs', async (row) => {
          const pending = page.waitForResponse((response) => response.url() === `${origin}/api/imports` && response.request().method() === 'GET');
          const [response] = await Promise.all([pending, page.locator('#navigation a[href="#imports"]').click()]);
          assert.equal(response.status(), 200, 'Import history request failed');
          await response.finished(); await ready();
          await page.locator(LOCATORS.imports).waitFor({ state: 'visible' });
          row.existing_jobs = await page.locator(LOCATORS.jobs).count();
          row.progress = await page.locator(`${LOCATORS.jobs} progress`).evaluateAll((items) => items.slice(0, 20).map((item) => ({ value: item.hasAttribute('value') ? item.value : null, max: item.max })));
          if (!row.existing_jobs) journey.unavailable.push('No existing import jobs; empty import history inspected');
          await shot('07-import-progress');
        });
        const after = await getBook(bookId);
        journey.reading_progress_after = after.readingProgress || null;
        journey.completed = true;
      } catch (error) {
        journey.error = { stage, name: error.name, message: short(error.message) };
        try { await shot('failure'); } catch (captureError) { journey.failure_screenshot_error = short(captureError.message); }
      } finally {
        await context.close(); journey.context_closed = true; await checkpoint();
      }
    }
    report.completed = report.journeys.length === 2 && report.journeys.every((journey) => journey.completed);
    report.passed = report.completed && ['console_errors', 'page_errors', 'http_failures', 'request_failures', 'guarded_requests'].every((key) => report[key].length === 0);
  } catch (error) {
    report.error = { name: error.name, message: short(error.message) };
  } finally {
    if (browser) {
      try { await browser.close(); report.browser_closed = true; }
      catch (error) { report.cleanup_error = short(error.message); report.passed = false; }
    }
    await checkpoint();
  }
  console.log(JSON.stringify({ report: join(output, 'report.json'), completed: report.completed, passed: report.passed, dataset, journeys: report.journeys.length }));
  if (!report.passed) process.exitCode = 1;
}

main().catch((error) => { console.error(`${error.name}: ${short(error.message)}`); process.exitCode = 1; });
