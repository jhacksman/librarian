// Spark coordinator only: finite read-only browser proof against an isolated candidate.
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {join} from 'node:path';
import {parseFrozenGold, FROZEN_GOLD_SHA256} from '../src/frozen-gold-recovery.mjs';

assert.equal(process.platform, 'linux'); assert.equal(process.arch, 'arm64'); assert.match(process.version, /^v24\./);
const [baseInput, goldPath, applicationPath, output, ...extra] = process.argv.slice(2);
assert.ok(baseInput && goldPath && applicationPath && output && !extra.length,
  'Usage: node scripts/frozen-gold-browser.mjs ISOLATED_LOOPBACK_URL GOLD.json APPLICATION.json FRESH_OUTPUT');
const base = new URL(baseInput);
assert.equal(base.protocol, 'http:'); assert.equal(base.hostname, '127.0.0.1'); assert.equal(base.pathname, '/');
assert.ok(base.port && base.port !== '3475', 'Use the isolated candidate listener');
const gold = parseFrozenGold(await readFile(goldPath)), application = JSON.parse(await readFile(applicationPath, 'utf8'));
assert.equal(application.schema, 'librarian-frozen-gold-application/v1'); assert.equal(application.failed, 0);
assert.equal(application.completed, 18); assert.equal(gold.records.length, 18);
await mkdir(output, {mode: 0o700});
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const report = {schema: 'librarian-frozen-gold-browser-proof/v1', goldSha256: FROZEN_GOLD_SHA256,
  independentAcceptanceVerdict: null, passed: false, actual: [], screenshots: [], errors: [], forbiddenRequests: [],
  limitation: 'Author-supplied finite API/DOM regression proof; independent critic owns gold comparison and visual acceptance.'};
let screenshotBytes = 0;
const saveScreenshot = async (png, filename) => {
  screenshotBytes += png.length;
  assert.ok(png.length <= 5 * 1024 ** 2 && screenshotBytes <= 48 * 1024 ** 2, 'Screenshot byte cap');
  await writeFile(join(output, filename), png, {flag: 'wx'});
  report.screenshots.push({filename, bytes: png.length, sha256: hash(png)});
};
const {chromium} = createRequire(import.meta.url)('/opt/playwright/node_modules/playwright');
const browser = await chromium.launch({headless: true, chromiumSandbox: true, timeout: 20000});
let context;
try {
  context = await browser.newContext({viewport: {width: 1440, height: 1100}, serviceWorkers: 'block'});
  await context.route('**/*', route => {
    const request = route.request(), url = new URL(request.url());
    if (url.origin !== base.origin || request.method() !== 'GET') {
      report.forbiddenRequests.push({method: request.method(), url: request.url()}); return route.abort();
    }
    return route.continue();
  });
  const page = await context.newPage(); page.setDefaultTimeout(7000);
  page.on('pageerror', error => report.errors.push(String(error.message).slice(0, 1000)));
  const status = await page.request.get(new URL('/api/status', base).href, {maxRedirects: 0});
  assert.equal(status.status(), 200); assert.equal((await status.json()).capabilities.maintenance, false);
  for (const record of gold.records) {
    const id = record.source_sha256, before = application.before.find(book => book.id === id);
    const applied = application.after.find(book => book.id === id);
    assert.ok(before && applied); assert.deepEqual(applied.cover, before.cover);
    const response = await page.request.get(new URL(`/api/books/${id}`, base).href, {maxRedirects: 0});
    assert.equal(response.status(), 200); const book = await response.json();
    const receipt = book.metadata?.sourceMetadata?.recovery;
    assert.equal(receipt.goldSha256, FROZEN_GOLD_SHA256); assert.equal(receipt.sourceSha256, id);
    const actual = {title: book.title, authors: book.authors, creators: book.metadata?.creators || null,
      edition: book.metadata?.edition || null};
    for (const name of ['title', 'authors', 'creators', 'edition']) {
      const expected = record[name].value === null || receipt.preservedFields.includes(name)
        ? before[name] : record[name].value;
      assert.deepEqual(actual[name], expected, `${id}: ${name}`);
    }
    await page.goto(`${base.href}#library`, {waitUntil: 'networkidle'});
    const search = page.locator('#library-search'); await search.waitFor();
    // Library filters belong to the visible search control, not hash routing.
    // Wait for this submitted title's catalog response before checking its card.
    const filtered = page.waitForResponse(response => {
      const url = new URL(response.url());
      return url.origin === base.origin && url.pathname === '/api/catalog'
        && url.searchParams.get('q') === book.title && response.request().method() === 'GET';
    });
    filtered.catch(() => {});
    await search.fill(book.title); await search.press('Enter');
    const catalogResponse = await filtered; assert.equal(catalogResponse.status(), 200);
    await page.locator('#catalog-results[aria-busy="false"]').waitFor();
    const card = page.locator(`[data-testid="book-card"][data-book-id="${id}"]`);
    await card.waitFor(); assert.equal((await card.locator('h2').innerText()).trim(), book.title);
    assert.equal((await card.locator('.book-author').innerText()).trim(), book.authors.length ? book.authors.join(', ') : 'Author not identified');
    const cover = card.locator('img');
    assert.equal(await cover.count(), 1, 'The retained cover must load, not silently become a placeholder');
    await card.scrollIntoViewIfNeeded();
    await page.waitForFunction(sourceId => {
      const img = document.querySelector(`[data-testid="book-card"][data-book-id="${sourceId}"] img`);
      return img?.complete && img.naturalWidth > 0;
    }, id);
    await page.goto(`${base.href}#book/${id}`, {waitUntil: 'networkidle'});
    const detail = page.locator(`[data-testid="book-detail"][data-book-id="${id}"]`); await detail.waitFor();
    assert.equal((await detail.locator('h1').innerText()).trim(), book.title);
    const fields = await page.locator('dl.metadata > div').evaluateAll(rows => Object.fromEntries(rows.map(row =>
      [row.querySelector('dt').textContent.trim(), row.querySelector('dd').textContent.trim()])));
    assert.equal(fields.Author, book.authors.length ? book.authors.join(', ') : 'Author not identified');
    assert.equal(fields.Edition, actual.edition || 'Edition not identified');
    if (actual.creators) assert.equal(fields['Creator credits'], actual.creators.map(credit => `${credit.name} (${credit.role})`).join('; '));
    else assert.equal(fields['Creator credits'], undefined);
    if (receipt.fields.authors?.roleBasis) assert.equal(fields['Author credit'], receipt.fields.authors.roleBasis);
    await saveScreenshot(await page.screenshot({type: 'png', animations: 'disabled', fullPage: false}), `${id}-desktop-detail.png`);
    await saveScreenshot(await page.locator('dl.metadata').screenshot({type: 'png', animations: 'disabled'}), `${id}-desktop-fields.png`);
    report.actual.push({id, api: actual, libraryCardVisible: true, coverLoaded: true, detailFields: fields,
      libraryFilter: {route: '#library', query: book.title, catalogStatus: catalogResponse.status()},
      preservedFields: receipt.preservedFields, unresolvedFields: receipt.unresolvedFields});
  }
  await page.setViewportSize({width: 390, height: 844});
  for (const index of [0, 3, 7, 17]) {
    const id = gold.records[index].source_sha256;
    await page.goto(`${base.href}#book/${id}`, {waitUntil: 'networkidle'});
    await page.locator(`[data-testid="book-detail"][data-book-id="${id}"]`).waitFor();
    await saveScreenshot(await page.locator('dl.metadata').screenshot({type: 'png', animations: 'disabled'}), `${id}-mobile-fields.png`);
  }
  assert.equal(report.forbiddenRequests.length, 0); assert.equal(report.errors.length, 0);
  report.passed = true;
} catch (error) {report.errors.push(String(error.stack).slice(0, 4000)); process.exitCode = 1;}
finally {
  if (context) await context.close(); await browser.close();
  await writeFile(join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
}
