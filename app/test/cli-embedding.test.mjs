import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { LibraryStore } from '../src/store.mjs';

const cli = fileURLToPath(new URL('../src/cli.mjs', import.meta.url));
const texts = count => Array.from({ length: count }, (_, index) => `Synthetic durable passage ${index}.`);

async function fixture(t, sourceTexts, respond = () => ({})) {
  const root = await mkdtemp(join(tmpdir(), 'librarian-cli-embedding-'));
  const dataDir = join(root, 'data'), dbPath = join(dataDir, 'library.sqlite');
  const store = new LibraryStore(dbPath, { dataDir });
  try {
    store.transaction(() => {
      store.db.prepare('INSERT INTO books(id,title,created_at,updated_at) VALUES(?,?,?,?)')
        .run('synthetic-book', 'Synthetic embedding workflow', '2026-10-05', '2026-10-05');
      store.db.prepare('INSERT INTO sections(id,book_id,ordinal,title,text,locator_json) VALUES(?,?,0,?,?,?)')
        .run('synthetic-section', 'synthetic-book', 'Synthetic evidence', sourceTexts.join('\n'), '{"format":"epub","member":"evidence.xhtml"}');
      const insert = store.db.prepare('INSERT INTO chunks(id,book_id,section_id,ordinal,text,locator_json) VALUES(?,?,?,?,?,?)');
      let offset = 0;
      sourceTexts.forEach((text, index) => {
        insert.run(`chunk-${String(index).padStart(5, '0')}`, 'synthetic-book', 'synthetic-section', index, text,
          JSON.stringify({ format: 'epub', member: 'evidence.xhtml', charStart: offset,
            charEnd: offset + text.length, offsetBasis: 'section_utf16_code_units' }));
        offset += text.length + 1;
      });
    });
  } finally { store.close(); }

  const calls = [], children = new Set();
  let activeChild, handlerError;
  const server = createServer(async (request, response) => {
    try {
      assert.equal(request.method, 'POST');
      assert.equal(request.url, '/api/embed');
      const parts = [];
      for await (const part of request) parts.push(part);
      const body = JSON.parse(Buffer.concat(parts).toString('utf8'));
      assert.equal(body.model, 'synthetic-cli-embedding');
      assert.equal(body.truncate, false);
      calls.push(body.input);
      const reply = respond({ input: body.input, call: calls.length, child: activeChild });
      if (reply === null) return; // A cancellation test intentionally leaves its owned request pending.
      response.writeHead(reply.status ?? 200, { 'content-type': 'application/json' });
      response.end(JSON.stringify(reply.body ?? { embeddings: body.input.map(() => [3, 4]) }));
    } catch (error) {
      handlerError = error;
      response.writeHead(500, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ error: error.message }));
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const endpoint = `http://127.0.0.1:${server.address().port}`;
  t.after(async () => {
    await Promise.all([...children].map(child => new Promise(resolve => {
      child.once('close', resolve); child.kill('SIGKILL');
    })));
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  });

  const snapshot = () => {
    const db = new DatabaseSync(dbPath, { readOnly: true });
    try {
      return {
        chunks: db.prepare('SELECT * FROM chunks ORDER BY id').all().map(row => ({ ...row })),
        sections: db.prepare('SELECT * FROM sections ORDER BY id').all().map(row => ({ ...row })),
        failures: db.prepare("SELECT name FROM sqlite_master WHERE name='embedding_failures'").get()
          ? db.prepare('SELECT * FROM embedding_failures ORDER BY chunk_id').all().map(row => ({ ...row })) : [],
      };
    } finally { db.close(); }
  };
  const run = async args => {
    const child = spawn(process.execPath, [cli, 'embed', ...args], {
      env: { LIBRARIAN_DATA_DIR: dataDir, LIBRARIAN_EMBEDDING_MODEL: 'synthetic-cli-embedding',
        LIBRARIAN_EMBEDDING_ENDPOINT: endpoint, LIBRARIAN_EMBEDDING_PROVIDER: 'ollama',
        LIBRARIAN_EMBEDDING_REVISION: 'synthetic-fixed-revision',
        LIBRARIAN_EMBEDDING_QUERY_PREFIX: 'query: ', LIBRARIAN_EMBEDDING_DOCUMENT_PREFIX: 'passage: ',
        NODE_NO_WARNINGS: '1' }, stdio: ['ignore', 'pipe', 'pipe'],
    });
    activeChild = child; children.add(child);
    let stdout = '', stderr = '', exceeded = false;
    const collect = (stream, destination) => stream.on('data', data => {
      if (destination === 'stdout') stdout += data;
      else stderr += data;
      if (Buffer.byteLength(stdout) + Buffer.byteLength(stderr) > 2 * 1024 * 1024) {
        exceeded = true; child.kill('SIGKILL');
      }
    });
    collect(child.stdout, 'stdout'); collect(child.stderr, 'stderr');
    let expired = false;
    const timer = setTimeout(() => { expired = true; child.kill('SIGKILL'); }, 30000);
    let exit;
    try {
      exit = await new Promise((resolve, reject) => {
        child.once('error', reject);
        child.once('close', (code, signal) => resolve({ code, signal }));
      });
    } finally { clearTimeout(timer); children.delete(child); activeChild = undefined; }
    assert.equal(expired, false, 'Owned CLI child exceeded its test deadline');
    assert.equal(exceeded, false, 'Owned CLI output exceeded the test capture bound');
    assert.ifError(handlerError);
    assert.equal(exit.signal, null, `CLI did not exit gracefully: ${stderr}`);
    const records = stdout.trim() ? stdout.trim().split('\n').map(line => JSON.parse(line)) : [];
    return { ...exit, records, final: records.at(-1), stderr };
  };
  return { calls, run, snapshot, endpoint };
}

function assertSourceUnchanged(before, after) {
  assert.deepEqual(after.sections, before.sections);
  for (const [index, row] of after.chunks.entries()) {
    const source = Object.fromEntries(Object.entries(row).filter(([key]) => !key.startsWith('embedding_')));
    const original = before.chunks[index];
    assert.deepEqual(source, Object.fromEntries(Object.entries(original).filter(([key]) => !key.startsWith('embedding_'))));
  }
}

test('embed --once yields after one real slice and a fresh process resumes without replacing saved vectors', async t => {
  const sourceTexts = texts(2001), owned = await fixture(t, sourceTexts);
  const before = owned.snapshot();
  const first = await owned.run(['--once']);
  assert.equal(first.code, 0, first.stderr);
  assert.equal(first.final.status, 'bounded');
  assert.deepEqual(first.final.execution, { mode: 'once', completed: false, yielded: true,
    resumeRequired: true, retryFailedRequested: false, exitCode: 0 });
  assert.equal(first.final.processed, 2000);
  assert.equal(first.final.attempted, 2000);
  assert.equal(first.final.pendingChunks, 1);
  assert.equal(first.final.coverage.totalChunks, 2001);
  assert.equal(first.final.coverage.embeddedChunks, 2000);
  assert.equal(first.final.coverage.failedChunks, 0);
  assert.equal(first.final.coverage.identity, first.final.identity);
  assert.deepEqual(JSON.parse(first.final.identity), { provider: 'ollama', endpoint: owned.endpoint,
    model: 'synthetic-cli-embedding', revision: 'synthetic-fixed-revision', queryPrefix: 'query: ', documentPrefix: 'passage: ' });
  const saved = owned.snapshot();
  assert.equal(saved.chunks.filter(row => row.embedding_json !== null).length, 2000);
  const second = await owned.run(['--once']);
  assert.equal(second.code, 0, second.stderr);
  assert.equal(second.final.status, 'complete');
  assert.equal(second.final.execution.completed, true);
  assert.equal(second.final.execution.yielded, false);
  assert.equal(second.final.execution.resumeRequired, false);
  assert.equal(second.final.processed, 1);
  assert.equal(second.final.coverage.embeddedChunks, 2001);
  assert.equal(second.final.identity, first.final.identity);
  const after = owned.snapshot();
  assert.deepEqual(after.chunks.slice(0, 2000), saved.chunks.slice(0, 2000));
  assert.deepEqual(owned.calls.flat(), sourceTexts.map(text => `passage: ${text}`));
  assertSourceUnchanged(before, after);
});

test('default embed still drains consecutive bounded slices to completion', async t => {
  const owned = await fixture(t, texts(2001));
  const result = await owned.run([]);
  assert.equal(result.code, 0, result.stderr);
  const summaries = result.records.filter(row => row.event !== 'progress');
  assert.deepEqual(summaries.map(row => row.status), ['bounded', 'complete']);
  assert.deepEqual(summaries.map(row => row.processed), [2000, 1]);
  assert.ok(summaries.every(row => !Object.hasOwn(row, 'execution')));
  assert.equal(owned.snapshot().chunks.filter(row => row.embedding_json !== null).length, 2001);
  assert.equal(owned.calls.flat().length, 2001);
});

test('once preserves quarantined failures and retries them only with the explicit flag', async t => {
  let reject = true;
  const owned = await fixture(t, ['durable good first', 'reject this input', 'durable good last'], ({ input }) =>
    reject && input.some(text => text.includes('reject')) ? { status: 413, body: { error: 'Synthetic input rejection' } } : {});
  const first = await owned.run(['--once']);
  assert.equal(first.code, 1);
  assert.equal(first.final.status, 'complete_with_errors');
  assert.equal(first.final.execution.completed, false);
  assert.equal(first.final.execution.yielded, false);
  assert.equal(first.final.coverage.embeddedChunks, 2);
  assert.equal(first.final.coverage.failedChunks, 1);
  assert.equal(first.final.coverage.pendingChunks, 0);
  const failed = owned.snapshot(), calls = owned.calls.length;
  assert.equal(failed.failures.length, 1);
  reject = false;
  const unchanged = await owned.run(['--once']);
  assert.equal(unchanged.code, 1);
  assert.equal(unchanged.final.attempted, 0);
  assert.equal(owned.calls.length, calls);
  assert.deepEqual(owned.snapshot(), failed);
  const retried = await owned.run(['--retry-failed', '--once']);
  assert.equal(retried.code, 0, retried.stderr);
  assert.equal(retried.final.status, 'complete');
  assert.equal(retried.final.execution.retryFailedRequested, true);
  assert.equal(retried.final.processed, 1);
  assert.equal(retried.final.identity, first.final.identity);
  assert.deepEqual(owned.calls.slice(calls), [['passage: reject this input']]);
  const after = owned.snapshot();
  assert.equal(after.failures.length, 0);
  for (const index of [0, 2]) assert.deepEqual(after.chunks[index], failed.chunks[index]);
});

test('SIGTERM during the next request reports cancellation and leaves completed batches resumable', async t => {
  const sourceTexts = texts(33);
  const owned = await fixture(t, sourceTexts, ({ call, child }) => {
    if (call === 2) { assert.equal(child.kill('SIGTERM'), true); return null; }
    return {};
  });
  const before = owned.snapshot();
  const cancelled = await owned.run(['--once']);
  assert.equal(cancelled.code, 1);
  assert.equal(cancelled.final.status, 'cancelled');
  assert.equal(cancelled.final.execution.completed, false);
  assert.equal(cancelled.final.execution.yielded, false);
  assert.equal(cancelled.final.execution.resumeRequired, true);
  assert.equal(cancelled.final.coverage.embeddedChunks, 16);
  assert.equal(cancelled.final.coverage.pendingChunks, 17);
  assert.equal(cancelled.final.coverage.failedChunks, 0);
  const saved = owned.snapshot(), requestsBeforeResume = owned.calls.length;
  const resumed = await owned.run(['--once']);
  assert.equal(resumed.code, 0, resumed.stderr);
  assert.equal(resumed.final.status, 'complete');
  assert.equal(resumed.final.processed, 17);
  assert.equal(resumed.final.identity, cancelled.final.identity);
  assert.deepEqual(owned.calls.slice(requestsBeforeResume).flat(), sourceTexts.slice(16).map(text => `passage: ${text}`));
  const after = owned.snapshot();
  assert.deepEqual(after.chunks.slice(0, 16), saved.chunks.slice(0, 16));
  assertSourceUnchanged(before, after);
});

test('a bounded slice with saved vectors and a quarantined input still exits unsuccessfully', async t => {
  const sourceTexts = texts(2001); sourceTexts[0] = 'reject this input';
  const owned = await fixture(t, sourceTexts, ({ input }) => input.some(text => text.includes('reject'))
    ? { status: 413, body: { error: 'Synthetic input rejection' } } : {});
  const result = await owned.run(['--once']);
  assert.equal(result.code, 1);
  assert.equal(result.final.status, 'bounded');
  assert.equal(result.final.processed, 1999);
  assert.equal(result.final.quarantined, 1);
  assert.equal(result.final.execution.yielded, false);
  assert.equal(result.final.execution.completed, false);
  assert.equal(result.final.execution.resumeRequired, true);
  assert.equal(result.final.coverage.embeddedChunks, 1999);
  assert.equal(result.final.coverage.pendingChunks, 1);
  assert.equal(result.final.coverage.failedChunks, 1);
  assert.equal(owned.snapshot().failures.length, 1);
});

test('backend failure is not a successful once yield and leaves inputs pending', async t => {
  const owned = await fixture(t, texts(2), () => ({ status: 503, body: { error: 'Synthetic offline backend' } }));
  const result = await owned.run(['--once']);
  assert.equal(result.code, 1);
  assert.equal(result.final.status, 'local_ai_unavailable');
  assert.equal(result.final.execution.yielded, false);
  assert.equal(result.final.execution.completed, false);
  assert.equal(result.final.execution.exitCode, result.code);
  assert.equal(result.final.coverage.pendingChunks, 2);
  assert.equal(result.final.coverage.embeddedChunks, 0);
  assert.equal(result.final.coverage.failedChunks, 0);
  assert.equal(owned.calls.length, 1);
});

test('unknown and repeated embedding options fail before any inference request', async t => {
  const owned = await fixture(t, texts(1));
  for (const args of [['--once', '--once'], ['--retry-failed', '--retry-failed'], ['--unknown']]) {
    const result = await owned.run(args);
    assert.equal(result.code, 1);
    assert.match(result.stderr, /Usage: embed \[--once\] \[--retry-failed\]/);
    assert.deepEqual(result.records, []);
  }
  assert.equal(owned.calls.length, 0);
  assert.equal(owned.snapshot().chunks[0].embedding_json, null);
});
