/**
 * Spark-only synthetic durability/UI regression. Required:
 *   LIBRARIAN_RECOVERY_OUTPUT=/fresh/absolute/path/outside/git
 * Run with the prepared Node 24 ARM64 + Playwright runtime; installs nothing.
 * This owns four loopback fixture-server processes, one mock model endpoint,
 * two synthetic EPUBs and one fresh temporary database. Only owned fixture or
 * browser processes receive termination signals. Initial slow extraction uses a documented IPC
 * fixture, while restarted Resume runs the actual HTTP importer/parser worker.
 * The model is a deterministic workflow mock: no real-model/quality claim.
 */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {createServer} from 'node:http';
import {lstat, mkdir, mkdtemp, readFile, readdir, realpath, rename, rm, unlink, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {dirname, isAbsolute, join, resolve} from 'node:path';
import {performance} from 'node:perf_hooks';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {BOOKS, expectedSection, syntheticEpub} from './recovery-fixtures.mjs';

const DRIVER = fileURLToPath(import.meta.url);
const FIXTURE = fileURLToPath(new URL('./recovery-fixture-server.mjs', import.meta.url));
const PLAYWRIGHT = '/opt/playwright/node_modules/playwright';
const QUESTION = 'How many minutes does the amber lantern burn?';
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const short = (value) => String(value ?? '').slice(0, 500);

async function outsideGit(directory) {
  for (let current = directory; ; current = dirname(current)) {
    try { await lstat(join(current, '.git')); throw new Error('Recovery output must be outside every Git checkout'); }
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
  assert.equal(process.platform, 'linux', 'Run only in the approved Spark Linux runtime');
  assert.equal(process.arch, 'arm64', 'Use the prepared Spark ARM64 runtime');
  assert.ok(/^v24\./.test(process.version), 'Use Node 24');
  assert.equal(process.env.PLAYWRIGHT_MODULE || PLAYWRIGHT, PLAYWRIGHT);
  const requested = process.env.LIBRARIAN_RECOVERY_OUTPUT;
  assert.ok(requested && isAbsolute(requested), 'LIBRARIAN_RECOVERY_OUTPUT must be an absolute fresh path');
  await outsideGit(resolve(requested));
  await mkdir(dirname(resolve(requested)), {recursive: true});
  const output = join(await realpath(dirname(resolve(requested))), resolve(requested).slice(dirname(resolve(requested)).length + 1));
  await outsideGit(output);
  await mkdir(output); // Never merge a fresh run with stale evidence.
  const report = {schemaVersion: 1, completed: false, passed: false, startedAt: new Date().toISOString(),
    scope: 'Synthetic real SIGKILL/restart and live browser Resume against createApp and LibraryStore',
    initialImport: 'IPC test fixture invokes actual inventory/runImport/extractBook, holding the second extraction after staging; resumed import uses ordinary HTTP/default parser worker',
    model: 'Owned loopback deterministic Ollama-protocol mock; recovery workflow only, not real model or answer-quality acceptance',
    environment: {platform: process.platform, architecture: process.arch, node: process.version},
    sourceHashes: {}, fixtures: [], steps: [], servers: [], screenshots: [], snapshots: [],
    modelRequests: [], browserErrors: [], unexpectedRequests: [], hardKills: [], cleanup: {},
    limits: {totalBudgetMs: 180000, reportBytes: 512 * 1024, snapshotBytes: 256 * 1024,
      screenshotBytes: 5 * 1024 * 1024, modelRequests: 24, modelBodyBytes: 65536},
  };
  const deadline = performance.now() + report.limits.totalBudgetMs;
  const remaining = (maximum = 20000) => {
    const value = Math.min(maximum, Math.floor(deadline - performance.now()));
    assert.ok(value > 0, 'Recovery driver exhausted its total budget'); return value;
  };
  const checkpoint = async () => {
    report.updatedAt = new Date().toISOString();
    const raw = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
    assert.ok(raw.length <= report.limits.reportBytes, 'Recovery report exceeded its bound');
    await writeFile(join(output, 'report.json.tmp'), raw);
    await rename(join(output, 'report.json.tmp'), join(output, 'report.json'));
  };
  const step = async (name, action) => {
    remaining(); const row = {name, passed: false}; report.steps.push(row); const start = performance.now();
    try { await action(row); row.passed = true; }
    finally { row.milliseconds = Math.round(performance.now() - start); await checkpoint(); }
  };
  const poll = async (label, read, predicate, maximum = 20000) => {
    const stop = performance.now() + remaining(maximum);
    for (let attempt = 0; attempt < 400 && performance.now() < stop; attempt += 1) {
      const value = await read(); if (predicate(value)) return value;
      await delay(75);
    }
    throw new Error(`Timed out waiting for ${label}`);
  };
  let workspace; let browser; let context; let page; let modelServer; let current; let smoke;
  const owned = []; const modelSockets = new Set();
  let origin = ''; let modelPhase = 'idle'; let heldModelRequest = false;
  let fixtureFailure;
  const ownedChildren = async (server) => {
    try {
      const text = await readFile(`/proc/${server.pid}/task/${server.pid}/children`, 'utf8');
      const ids = text.trim() ? text.trim().split(/\s+/).map(Number) : [];
      assert.ok(ids.length <= 8 && ids.every(Number.isSafeInteger), 'Unexpected fixture child count');
      return await Promise.all(ids.map(async (pid) => {
        try { const stat = await readFile(`/proc/${pid}/stat`, 'utf8'); return {pid, identity: stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19]}; }
        catch (error) { if (error.code === 'ENOENT') return null; throw error; }
      })).then((rows) => rows.filter(Boolean));
    } catch (error) { if (error.code === 'ENOENT') return []; throw error; }
  };
  const awaitParserWatchdog = async (children) => {
    // The production extractor checks its parent once per second. Do not remove
    // the temporary workspace while an observed child still owns its files.
    await delay(1250);
    for (let attempt = 0; attempt < 20; attempt += 1) {
      const alive = await Promise.all(children.map(async ({pid, identity}) => {
        try { const stat = await readFile(`/proc/${pid}/stat`, 'utf8'); const fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
          return fields[19] === identity && fields[0] !== 'Z'; }
        catch (error) { if (error.code === 'ENOENT') return false; throw error; }
      }));
      if (!alive.some(Boolean)) return;
      await delay(100);
    }
    throw new Error('Observed owned parser child survived its parent watchdog');
  };
  await checkpoint();
  try {
    for (const relative of ['recovery-browser.mjs', 'recovery-fixture-server.mjs', 'recovery-fixtures.mjs', 'browser-smoke.mjs']) {
      report.sourceHashes[relative] = sha(await readFile(new URL(relative, import.meta.url)));
    }
    for (const relative of ['server.mjs', 'config.mjs', 'store.mjs', 'importer.mjs', 'import-lifecycle.mjs', 'process-lock.mjs',
      'catalog.mjs', 'retrieval.mjs', 'extract.mjs', 'extract-worker.mjs', 'convert.mjs']) {
      report.sourceHashes[`src/${relative}`] = sha(await readFile(new URL(`../src/${relative}`, import.meta.url)));
    }
    for (const relative of ['app.js', 'index.html', 'styles.css']) {
      report.sourceHashes[`public/${relative}`] = sha(await readFile(new URL(`../public/${relative}`, import.meta.url)));
    }
    const {DatabaseSync} = await import('node:sqlite');
    workspace = await realpath(await mkdtemp(join(tmpdir(), 'librarian-recovery-')));
    await mkdir(join(workspace, 'originals'));
    for (const book of BOOKS) {
      const bytes = syntheticEpub(book);
      await writeFile(join(workspace, 'originals', book.filename), bytes, {flag: 'wx', mode: 0o400});
      report.fixtures.push({filename: book.filename, title: book.title, sha256: sha(bytes), bytes: bytes.length, sections: 2});
    }
    const snapshot = () => {
      const db = new DatabaseSync(join(workspace, 'state', 'library.sqlite'), {readOnly: true, timeout: 5000});
      try {
        db.exec('BEGIN');
        const value = {
          jobs: db.prepare('SELECT id,state,total,processed,summary_json FROM import_jobs ORDER BY id').all(),
          files: db.prepare('SELECT id,relative_path,path,source_path,sha256,size,status,book_id,staged,attempts,error FROM files ORDER BY relative_path').all(),
          receipts: db.prepare('SELECT id,job_id,file_id,attempt,status,error,started_at,finished_at FROM import_receipts ORDER BY id').all(),
          sections: db.prepare('SELECT id,book_id,ordinal,title,text,locator_json FROM sections ORDER BY book_id,ordinal').all(),
          chunks: db.prepare('SELECT id,book_id,section_id,ordinal,text,locator_json,embedding_json,embedding_model,embedding_dimension,embedding_norm FROM chunks ORDER BY id').all(),
          fts: db.prepare('SELECT chunk_id,book_id,text FROM chunks_fts ORDER BY chunk_id').all(),
          lexicalLantern: db.prepare("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'lantern' ORDER BY chunk_id").all(),
          integrity: db.prepare('PRAGMA integrity_check').all(),
        };
        db.exec('COMMIT'); return value;
      } finally { db.close(); }
    };
    const retain = async (label, value) => {
      const raw = Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
      assert.ok(raw.length <= report.limits.snapshotBytes);
      const filename = `${label}.json`; await writeFile(join(output, filename), raw, {flag: 'wx'});
      report.snapshots.push({filename, bytes: raw.length, sha256: sha(raw)});
    };
    const request = async (route, body) => {
      const response = await fetch(origin + route, {method: body === undefined ? 'GET' : 'POST',
        headers: {origin, ...(body === undefined ? {} : {'content-type': 'application/json'})},
        ...(body === undefined ? {} : {body: JSON.stringify(body)}), signal: AbortSignal.timeout(remaining(5000))});
      assert.ok(response.ok, `${route}: HTTP ${response.status}`); return response.json();
    };
    modelServer = createServer(async (req, res) => {
      try {
        assert.equal(req.method, 'POST');
        assert.ok(['/api/embed', '/api/chat'].includes(req.url));
        let size = 0; const pieces = [];
        for await (const part of req) { size += part.length; assert.ok(size <= report.limits.modelBodyBytes); pieces.push(part); }
        const body = JSON.parse(Buffer.concat(pieces).toString('utf8'));
        assert.ok(report.modelRequests.length < report.limits.modelRequests);
        const record = {index: report.modelRequests.length, phase: modelPhase, route: req.url, fulfilled: false};
        report.modelRequests.push(record);
        const send = (value) => { record.fulfilled = true; res.writeHead(200, {'content-type': 'application/json'}); res.end(JSON.stringify(value)); };
        if (req.url === '/api/embed') {
          assert.equal(body.model, 'synthetic-recovery-embedding'); assert.equal(body.truncate, false);
          assert.ok(Array.isArray(body.input) && body.input.length === 1 && typeof body.input[0] === 'string');
          const input = body.input[0]; record.kind = input.startsWith('document: ') ? 'document' : 'query';
          assert.ok(input.startsWith(`${record.kind}: `)); record.inputSha256 = sha(input);
          if (modelPhase === 'embedding-initial' && record.kind === 'document'
              && report.modelRequests.filter((item) => item.phase === modelPhase && item.kind === 'document').length === 2) {
            record.held = true; heldModelRequest = true;
            const timer = setTimeout(() => res.destroy(), 45000);
            res.once('close', () => { clearTimeout(timer); record.closed = true; });
            return;
          }
          send({embeddings: [[3, 4]]});
        } else {
          assert.equal(body.model, 'synthetic-recovery-chat');
          const input = JSON.parse(body.messages.find((item) => item.role === 'user').content);
          assert.equal(input.question, QUESTION);
          const evidence = input.passages.find((item) => item.text.includes(BOOKS[0].sections[0].paragraph));
          assert.ok(evidence, 'Mock answer requires the actual recovered source passage');
          record.referencedChunk = evidence.chunkId;
          send({done: true, done_reason: 'stop', message: {content: JSON.stringify({abstain: false,
            answer: `The amber lantern burns for exactly 42 minutes. [${evidence.citation}]`})}});
        }
      } catch (error) {
        fixtureFailure = error; res.writeHead(500, {'content-type': 'application/json'}); res.end('{"error":"Recovery mock rejected request"}');
      }
    });
    modelServer.on('connection', (socket) => { modelSockets.add(socket); socket.once('close', () => modelSockets.delete(socket)); });
    await new Promise((resolveListen, reject) => { modelServer.once('error', reject); modelServer.listen(0, '127.0.0.1', resolveListen); });
    const modelOrigin = `http://127.0.0.1:${modelServer.address().port}`;
    const startServer = async (phase) => {
      assert.ok(!current || current.closed, 'Previous owned server must exit before restart');
      const row = {phase, pid: null, closed: false, messages: [], stderr: ''};
      const child = spawn(process.execPath, ['--disallow-code-generation-from-strings', FIXTURE, workspace, phase, modelOrigin],
        {stdio: ['ignore', 'pipe', 'pipe', 'ipc'], env: {PATH: '/usr/local/bin:/usr/bin:/bin', HOME: workspace, LANG: 'C.UTF-8'}});
      row.pid = child.pid; row.child = child; row.exit = new Promise((done) => child.once('close', (code, signal) => {
        row.closed = true; row.code = code; row.signal = signal; done({code, signal});
      }));
      child.on('error', (error) => { row.error = short(error.message); });
      child.on('message', (message) => {
        if (row.messages.length >= 10) { row.error = 'Fixture IPC exceeded its bound'; child.kill('SIGKILL'); return; }
        row.messages.push(message);
      });
      let logBytes = 0;
      const log = (part) => {
        logBytes += part.length;
        if (logBytes > 32768) { row.error = 'Fixture logs exceeded their bound'; child.kill('SIGKILL'); return; }
        row.stderr += part.toString('utf8');
      };
      child.stdout.on('data', log); child.stderr.on('data', log);
      owned.push(row); current = row;
      const ready = await poll('fixture server readiness', () => {
        assert.ok(!row.closed && !row.error, row.error || row.stderr || 'Fixture exited before ready');
        const failure = row.messages.find((item) => item.type === 'failure'); assert.ok(!failure, failure?.message);
        return row.messages.find((item) => item.type === 'ready');
      }, Boolean);
      assert.equal(ready.pid, child.pid); origin = ready.origin;
      const parsed = new URL(origin); assert.equal(parsed.hostname, '127.0.0.1');
      report.servers.push({phase, pid: row.pid, origin}); return row;
    };
    const stopServer = async (hard = false) => {
      const server = current; assert.ok(server && !server.closed);
      // Stop browser polling before changing the owned server's origin/lifetime.
      // Background work is server-owned and remains running until the signal.
      if (page) await page.goto('about:blank', {timeout: remaining(5000)});
      const children = await ownedChildren(server);
      server.observedChildPids = children.map((child) => child.pid);
      assert.equal(server.child.kill(hard ? 'SIGKILL' : 'SIGTERM'), true);
      const result = await bounded(server.exit, 7000, 'Owned fixture server did not exit');
      if (hard) {
        assert.equal(result.signal, 'SIGKILL');
        try { await awaitParserWatchdog(children); }
        catch (error) { server.cleanupError = short(error.message); throw error; }
        report.hardKills.push({phase: server.phase, pid: server.pid, signal: result.signal,
          observedChildPids: children.map((child) => child.pid), parserWatchdogAwaited: true});
      }
      else { assert.equal(result.code, 0, server.stderr); }
    };
    const require = createRequire(import.meta.url); const {chromium} = require(PLAYWRIGHT);
    report.environment.playwright = require(`${PLAYWRIGHT}/package.json`).version;
    browser = await chromium.launch({headless: true, chromiumSandbox: true, timeout: remaining()});
    report.environment.chromium = browser.version();
    context = await browser.newContext({viewport: {width: 1365, height: 900}, serviceWorkers: 'block',
      javaScriptEnabled: true, acceptDownloads: false, locale: 'en-US', timezoneId: 'UTC', reducedMotion: 'reduce'});
    page = await context.newPage(); page.setDefaultTimeout(10000); page.setDefaultNavigationTimeout(15000);
    page.on('pageerror', (error) => { if (report.browserErrors.length < 20) report.browserErrors.push(short(error.message)); });
    await context.route('**/*', async (route) => {
      const req = route.request(); const url = new URL(req.url());
      const allowed = url.origin === origin && (['GET', 'HEAD'].includes(req.method()) || (req.method() === 'POST'
        && (/^\/api\/imports\/[^/]+\/resume$/.test(url.pathname) || ['/api/embeddings', '/api/ask', '/api/search'].includes(url.pathname)
          || /^\/api\/books\/[^/]+\/progress$/.test(url.pathname))));
      if (allowed) return route.continue();
      if (report.unexpectedRequests.length < 20) report.unexpectedRequests.push({method: req.method(), origin: url.origin, pathname: url.pathname});
      return route.abort('blockedbyclient');
    });
    const importsPage = async () => {
      const target = `${origin}/#imports`;
      let response = await page.goto(target, {waitUntil: 'domcontentloaded', timeout: remaining(15000)});
      const reloaded = response === null;
      // Hash navigation can have no HTTP response. Reload to retain a real
      // document response and render fresh durable job state before capture.
      if (reloaded) response = await page.reload({waitUntil: 'domcontentloaded', timeout: remaining(15000)});
      assert.ok(response, 'Imports navigation must obtain an HTTP document response');
      assert.equal(response.status(), 200); assert.equal(page.url(), target);
      (report.importsNavigations ??= []).push({target, reloaded, status: response.status()});
      await page.locator('#main[aria-busy="false"]').waitFor({state: 'visible'});
    };
    const shot = async (label) => {
      const bytes = await page.screenshot({type: 'png', fullPage: false, animations: 'disabled', timeout: remaining(10000)});
      assert.ok(bytes.length <= report.limits.screenshotBytes);
      const filename = `${label}.png`; await writeFile(join(output, filename), bytes, {flag: 'wx'});
      report.screenshots.push({filename, sha256: sha(bytes), bytes: bytes.length, width: 1365, height: 900});
    };
    const clickPost = async (selector, route) => {
      await page.locator(selector).waitFor({state: 'visible'}); assert.equal(await page.locator(selector).isEnabled(), true);
      const wait = page.waitForResponse((response) => response.url() === origin + route && response.request().method() === 'POST');
      const [response] = await Promise.all([wait, page.locator(selector).click()]);
      assert.equal(response.status(), 202); await response.finished();
    };
    let jobId; let interrupted; let imported; let partial; let partialVector;
    await step('hard-kill-during-second-managed-source', async (row) => {
      const server = await startServer('import-initial');
      server.child.send({command: 'start-fixture-import'});
      const held = await poll('second staged extraction', () => {
        assert.ok(!server.closed && !server.error, server.error || server.stderr);
        const failure = server.messages.find((item) => item.type === 'failure'); assert.ok(!failure, failure?.message);
        return server.messages.find((item) => item.type === 'import-held');
      }, Boolean);
      jobId = held.jobId; const before = snapshot();
      assert.equal(before.jobs.length, 1); assert.equal(before.jobs[0].id, jobId);
      assert.equal(before.jobs[0].state, 'running');
      assert.deepEqual(before.files.map((file) => file.status), ['completed', 'running']);
      assert.deepEqual(before.files.map((file) => file.staged), [1, 1]);
      assert.deepEqual(before.receipts.map((receipt) => receipt.status), ['completed', 'running']);
      assert.equal(before.receipts[1].finished_at, null); assert.equal(before.chunks.length, 2);
      await stopServer(true); interrupted = snapshot(); assert.deepEqual(interrupted, before);
      for (const [index, fixture] of report.fixtures.entries()) {
        assert.equal(sha(await readFile(interrupted.files[index].path)), fixture.sha256);
        assert.equal(interrupted.files[index].sha256, fixture.sha256);
        await unlink(join(workspace, 'originals', fixture.filename));
      }
      assert.deepEqual(await readdir(join(workspace, 'originals')), []);
      row.jobId = jobId; row.firstCompletedChunks = 2; row.deletedOriginals = 2;
      await retain('01-import-interrupted', interrupted);
    });
    await step('browser-resumes-import-from-managed-bytes', async (row) => {
      await startServer('import-resume');
      const job = await request(`/api/imports/${encodeURIComponent(jobId)}`);
      assert.equal(job.canResume, true); assert.equal(job.canPause, false); assert.ok(job.recoveryNotice);
      await importsPage(); await shot('01-import-resume-available');
      await clickPost('[data-testid="import-resume"]', `/api/imports/${encodeURIComponent(jobId)}/resume`);
      await poll('resumed import completion', () => request(`/api/imports/${encodeURIComponent(jobId)}`), (value) => value.status === 'completed');
      imported = snapshot();
      assert.deepEqual(imported.files.map((file) => file.status), ['completed', 'completed']);
      assert.deepEqual(imported.files.map((file) => file.attempts), [1, 2]);
      assert.deepEqual(imported.receipts.map((receipt) => receipt.status), ['completed', 'failed', 'completed']);
      assert.deepEqual(imported.receipts.map((receipt) => receipt.attempt), [1, 1, 2]);
      assert.ok(imported.receipts.every((receipt) => receipt.finished_at));
      assert.match(imported.receipts[1].error, /interrupt|recover|restart/i);
      assert.deepEqual(imported.receipts[0], interrupted.receipts[0]);
      const firstBook = interrupted.files[0].book_id;
      assert.deepEqual(imported.sections.filter((section) => section.book_id === firstBook), interrupted.sections);
      assert.deepEqual(imported.chunks.filter((chunk) => chunk.book_id === firstBook), interrupted.chunks);
      assert.equal(imported.sections.length, 4); assert.equal(imported.chunks.length, 4); assert.equal(imported.fts.length, 4);
      assert.equal(new Set(imported.chunks.map((chunk) => chunk.id)).size, 4);
      assert.deepEqual(imported.fts.map((item) => item.chunk_id), imported.chunks.map((chunk) => chunk.id));
      assert.deepEqual(imported.fts.map((item) => [item.book_id, item.text]), imported.chunks.map((chunk) => [chunk.book_id, chunk.text]));
      assert.deepEqual(imported.lexicalLantern.map((item) => item.chunk_id),
        imported.chunks.filter((chunk) => chunk.book_id === report.fixtures[0].sha256).map((chunk) => chunk.id));
      assert.deepEqual(imported.integrity.map((item) => item.integrity_check), ['ok']);
      for (const [index, fixture] of report.fixtures.entries()) {
        assert.equal(imported.files[index].book_id, fixture.sha256);
        assert.equal(imported.files[index].sha256, fixture.sha256);
        const sections = imported.sections.filter((section) => section.book_id === fixture.sha256);
        assert.deepEqual(sections.map((section) => section.text), BOOKS[index].sections.map(expectedSection));
        const response = await fetch(`${origin}/api/files/${encodeURIComponent(imported.files[index].id)}/source`, {signal: AbortSignal.timeout(remaining(5000))});
        assert.equal(response.status, 200); assert.equal(sha(Buffer.from(await response.arrayBuffer())), fixture.sha256);
      }
      for (const chunk of imported.chunks) {
        const section = imported.sections.find((item) => item.id === chunk.section_id); const locator = JSON.parse(chunk.locator_json);
        assert.equal(locator.sourceSha256, chunk.book_id); assert.equal(locator.offsetBasis, 'section_utf16_code_units');
        assert.equal(section.text.slice(locator.charStart, locator.charEnd), chunk.text);
      }
      row.finalFiles = 2; row.finalSections = 4; row.finalChunks = 4; row.receipts = 3;
      await retain('02-import-resumed', imported); await importsPage(); await shot('02-import-completed');
      await stopServer();
    });
    await step('hard-kill-after-one-persisted-embedding', async (row) => {
      modelPhase = 'embedding-initial'; await startServer(modelPhase); await importsPage();
      assert.equal(report.modelRequests.length, 0, 'Embedding must not start implicitly on server startup');
      await clickPost('[data-testid="embedding-resume"]', '/api/embeddings');
      await poll('held second embedding request', () => heldModelRequest, Boolean);
      partial = snapshot(); const embedded = partial.chunks.filter((chunk) => chunk.embedding_json !== null);
      assert.equal(embedded.length, 1); partialVector = embedded[0];
      assert.equal(partialVector.embedding_json, '[3,4]'); assert.equal(partialVector.embedding_dimension, 2); assert.equal(partialVector.embedding_norm, 5);
      assert.equal(report.modelRequests.filter((item) => item.kind === 'document').length, 2);
      assert.equal(report.modelRequests.filter((item) => item.kind === 'document' && item.fulfilled).length, 1);
      await stopServer(true); assert.deepEqual(snapshot(), partial);
      row.persistedChunkId = partialVector.id; row.pendingChunks = 3;
      await retain('03-embedding-interrupted', partial);
    });
    await step('browser-resumes-only-pending-embeddings', async (row) => {
      modelPhase = 'embedding-resume'; const beforeRequests = report.modelRequests.length;
      await startServer(modelPhase); assert.deepEqual(snapshot(), partial); await importsPage();
      assert.equal(report.modelRequests.length, beforeRequests, 'Restart must await the explicit Resume button');
      const status = await request('/api/status'); assert.equal(status.models.embedding.embeddedChunks, 1);
      await shot('03-embedding-resume-available');
      await clickPost('[data-testid="embedding-resume"]', '/api/embeddings');
      await poll('all persisted embeddings', () => request('/api/status'), (value) => value.models.embedding.embeddedChunks === 4 && !value.models.embedding.indexing);
      const final = snapshot(); assert.equal(final.chunks.length, 4);
      assert.deepEqual(final.chunks.find((chunk) => chunk.id === partialVector.id), partialVector);
      assert.deepEqual(final.sections, imported.sections); assert.deepEqual(final.receipts, imported.receipts); assert.deepEqual(final.fts, imported.fts);
      assert.ok(final.chunks.every((chunk) => chunk.embedding_json === '[3,4]' && chunk.embedding_dimension === 2
        && chunk.embedding_norm === 5 && chunk.embedding_model === partialVector.embedding_model));
      const resumed = report.modelRequests.filter((item) => item.phase === modelPhase && item.kind === 'document');
      const expected = partial.chunks.filter((chunk) => chunk.embedding_json === null).map((chunk) => sha(`document: ${chunk.text}`)).sort();
      assert.equal(resumed.length, 3); assert.ok(resumed.every((item) => item.fulfilled));
      assert.deepEqual(resumed.map((item) => item.inputSha256).sort(), expected);
      assert.ok(!resumed.some((item) => item.inputSha256 === sha(`document: ${partialVector.text}`)));
      row.resumedRequests = 3; row.previouslyPersistedRequests = 0;
      await retain('04-embedding-resumed', final); await importsPage(); await shot('04-embedding-completed');
    });
    await step('recovered-citation-opens-exact-source-section', async (row) => {
      modelPhase = 'citation-check';
      const bookId = report.fixtures[0].sha256;
      await page.goto(`${origin}/#book/${bookId}`, {waitUntil: 'domcontentloaded'});
      await page.getByRole('button', {name: 'Ask this book', exact: true}).click();
      await page.locator('#ask-question').fill(QUESTION);
      const wait = page.waitForResponse((response) => response.url() === origin + '/api/ask' && response.request().method() === 'POST');
      const [response] = await Promise.all([wait, page.locator('.composer button[type="submit"]').click()]);
      assert.equal(response.status(), 200); const answer = await response.json();
      assert.equal(answer.abstained, false); assert.match(answer.answer, /42 minutes/); assert.ok(answer.citations.length > 0);
      for (const citation of answer.citations) {
        assert.equal(citation.bookId, bookId);
        const section = imported.sections.find((item) => item.id === citation.sectionId); assert.ok(section);
        assert.equal(citation.locator.sourceSha256, bookId);
        assert.equal(section.text.slice(citation.locator.charStart, citation.locator.charEnd), citation.text);
      }
      const evidence = answer.citations.find((citation) => citation.text.includes(BOOKS[0].sections[0].paragraph)); assert.ok(evidence);
      await page.locator(`[data-testid="citation-link"][data-section-id="${evidence.sectionId}"]`).first().click();
      await page.waitForFunction((id) => document.querySelector('[data-testid="reader-content"]')?.getAttribute('data-section-id') === id,
        evidence.sectionId, {timeout: remaining(10000)});
      assert.equal(await page.locator('.section-text').textContent(), expectedSection(BOOKS[0].sections[0]));
      assert.equal(await page.locator('.citation-highlight').textContent(), evidence.text);
      const route = new URL(page.url()); assert.ok(route.hash.startsWith(`#read/${encodeURIComponent(evidence.sectionId)}`));
      const range = new URLSearchParams(route.hash.split('?')[1]);
      assert.equal(Number(range.get('start')), evidence.locator.charStart);
      assert.equal(Number(range.get('end')), evidence.locator.charEnd);
      row.bookId = bookId; row.sectionId = evidence.sectionId; row.citations = answer.citations.length;
      await retain('05-cited-answer', answer); await shot('05-recovered-source');
    });
    await step('desktop-mobile-browser-smoke-on-recovered-library', async (row) => {
      await bounded(browser.close(), remaining(10000), 'Recovery Chromium did not close before browser-smoke');
      browser = null; context = null; page = null;
      report.recoveryBrowserClosedBeforeSmoke = true;
      const smokeOutput = join(output, 'browser-smoke');
      const env = Object.fromEntries(['PATH', 'HOME', 'TMPDIR', 'LANG', 'PLAYWRIGHT_BROWSERS_PATH', 'XDG_RUNTIME_DIR']
        .filter((key) => process.env[key] !== undefined).map((key) => [key, process.env[key]]));
      Object.assign(env, {PLAYWRIGHT_MODULE: PLAYWRIGHT, LIBRARIAN_DEMO_ORIGIN: origin,
        LIBRARIAN_DEMO_OUTPUT: smokeOutput, LIBRARIAN_DEMO_DATASET: 'synthetic',
        LIBRARIAN_DEMO_BOOK_ID: report.fixtures[0].sha256, LIBRARIAN_DEMO_QUESTION: QUESTION,
        LIBRARIAN_DEMO_REQUIRE_CITATION: '1'});
      const child = spawn(process.execPath, [fileURLToPath(new URL('./browser-smoke.mjs', import.meta.url))],
        {env, detached: true, stdio: ['ignore', 'pipe', 'pipe']});
      smoke = {child, closed: false, diagnostics: '', bytes: 0};
      smoke.kill = () => {
        if (!child.pid || smoke.closed) return;
        try { process.kill(-child.pid, 'SIGKILL'); }
        catch (error) { if (error.code !== 'ESRCH') child.kill('SIGKILL'); }
      };
      smoke.exit = new Promise((done) => child.once('close', (code, signal) => {
        smoke.closed = true; smoke.code = code; smoke.signal = signal; done({code, signal});
      }));
      child.on('error', (error) => { smoke.error = short(error.message); });
      const log = (part) => {
        smoke.bytes += part.length;
        if (smoke.bytes > 32768) { smoke.error = 'Browser-smoke diagnostics exceeded their bound'; smoke.kill(); return; }
        smoke.diagnostics += part.toString('utf8');
      };
      child.stdout.on('data', log); child.stderr.on('data', log);
      const exit = await bounded(smoke.exit, remaining(90000), 'Browser-smoke exceeded its shared recovery deadline');
      assert.equal(smoke.error, undefined, smoke.error); assert.equal(exit.code, 0, short(smoke.diagnostics)); assert.equal(exit.signal, null);
      const bytes = await readFile(join(smokeOutput, 'report.json')); assert.ok(bytes.length <= 512 * 1024);
      const result = JSON.parse(bytes.toString('utf8'));
      assert.equal(result.completed, true); assert.equal(result.passed, true); assert.equal(result.browser_closed, true);
      assert.equal(result.dataset.declared, 'synthetic'); assert.equal(result.dataset.real_book_validation_claimed, false);
      assert.equal(result.require_citation, true); assert.equal(result.requested_book_id, report.fixtures[0].sha256);
      assert.deepEqual(result.journeys.map((journey) => journey.name), ['desktop', 'mobile']);
      assert.ok(result.journeys.every((journey) => journey.completed && journey.context_closed && journey.steps.every((item) => item.passed)));
      for (const key of ['console_errors', 'page_errors', 'http_failures', 'request_failures', 'guarded_requests']) assert.deepEqual(result[key], []);
      const images = result.journeys.flatMap((journey) => journey.screenshots); assert.ok(images.length > 0 && images.length <= 20);
      for (const image of images) {
        assert.match(image.filename, /^[a-z0-9-]+\.png$/);
        const raw = await readFile(join(smokeOutput, image.filename));
        assert.ok(raw.length <= report.limits.screenshotBytes); assert.equal(raw.length, image.bytes); assert.equal(sha(raw), image.sha256);
      }
      row.report = 'browser-smoke/report.json'; row.reportSha256 = sha(bytes); row.screenshots = images.length;
      row.journeys = result.journeys.map((journey) => ({name: journey.name, completed: journey.completed, steps: journey.steps.map((item) => item.name)}));
      row.scope = 'Existing read/ask/progress desktop/mobile smoke, including catalog filters and content-search return; no metadata or grouping mutations';
      await checkpoint();
      await stopServer();
    });
    remaining(); assert.equal(report.hardKills.length, 2); assert.equal(fixtureFailure, undefined);
    assert.deepEqual(report.browserErrors, []); assert.deepEqual(report.unexpectedRequests, []);
    report.completed = true; report.passed = true;
  } catch (error) {
    report.error = {name: error.name, message: short(error.message), stack: String(error.stack || '').slice(0, 2000)}; process.exitCode = 1;
  } finally {
    if (smoke && !smoke.closed) {
      try {
        smoke.kill();
        await bounded(smoke.exit, 7000, 'Browser-smoke child cleanup exceeded its deadline');
      } catch (error) { report.cleanup.smokeError = short(error.message); }
    }
    if (smoke) report.browserSmokeProcess = {pid: smoke.child.pid, closed: smoke.closed,
      exitCode: smoke.code, exitSignal: smoke.signal, diagnostics: short(smoke.diagnostics)};
    for (const server of owned) {
      if (!server.closed) {
        const children = await ownedChildren(server).catch(() => []);
        server.child.kill('SIGKILL');
        try { await bounded(server.exit, 7000, 'Fixture cleanup exceeded its deadline'); await awaitParserWatchdog(children); }
        catch (error) { report.cleanup.serverError = short(error.message); }
      }
      const entry = report.servers.find((item) => item.pid === server.pid);
      if (server.cleanupError) report.cleanup.serverError = server.cleanupError;
      if (entry) Object.assign(entry, {closed: server.closed, exitCode: server.code, exitSignal: server.signal,
        observedChildPids: server.observedChildPids || [], diagnostics: short(server.stderr)});
    }
    try { if (browser) await bounded(browser.close(), 10000, 'Browser cleanup exceeded its deadline'); report.cleanup.browserClosed = true; }
    catch (error) { report.cleanup.browserError = short(error.message); }
    for (const socket of modelSockets) socket.destroy();
    if (modelServer?.listening) await bounded(new Promise((done) => modelServer.close(done)), 5000, 'Mock model cleanup exceeded its deadline')
      .catch((error) => { report.cleanup.modelError = short(error.message); });
    report.cleanup.serversClosed = owned.every((server) => server.closed);
    if (workspace && report.cleanup.serversClosed && !report.cleanup.serverError) await rm(workspace, {recursive: true, force: true}).then(() => { report.cleanup.workspaceRemoved = true; })
      .catch((error) => { report.cleanup.workspaceError = short(error.message); });
    else if (workspace) report.cleanup.workspaceRetainedForProcessCleanup = true;
    if (Object.keys(report.cleanup).some((key) => key.endsWith('Error')) || !report.cleanup.serversClosed) { report.passed = false; process.exitCode = 1; }
    report.finishedAt = new Date().toISOString(); await checkpoint();
  }
}

await main().catch((error) => { process.stderr.write(`Recovery driver: ${short(error.message)}\n`); process.exitCode = 1; });
