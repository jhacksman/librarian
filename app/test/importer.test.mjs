import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, open, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import test from 'node:test';
import { createReindexJob, importLibrary, inventoryLibrary, runImport, runParserWorker } from '../src/importer.mjs';
import { processIdentity } from '../src/process-lock.mjs';
import { LibraryStore, PIPELINE_VERSION, chunkText, isWithin } from '../src/store.mjs';

const sha = (value) => createHash('sha256').update(value).digest('hex');

async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), 'librarian-import-'));
  const root = join(directory, 'immutable-source');
  await mkdir(root);
  const store = new LibraryStore(join(directory, 'state', 'library.sqlite'));
  t.after(async () => { store.close(); await rm(directory, { recursive: true, force: true }); });
  return { directory, root, store };
}

async function source(root, name, bytes = name) {
  const path = join(root, name);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, bytes);
  return path;
}

function extracted(text = 'Original 🧭 evidence for the amber lantern.', format = 'epub') {
  return { title: 'Original synthetic book', authors: ['Test author'], metadata: {}, warnings: [],
    sections: [{ ordinal: 1, title: 'Evidence', text, locator: format === 'pdf' ? { format, page: 1 } : { format, member: 'one.xhtml' } }] };
}

async function fakeExtractor(request) {
  if (request.operation === 'inventoryArchive') return { ok: true, archive: {
    format: 'zip', entries: [{ name: 'book.epub', compressedBytes: 10, uncompressedBytes: 20, directory: false }],
    totalUncompressedBytes: 20, warnings: [] } };
  return { ok: true, book: extracted(undefined, request.format) };
}

test('actual transfer schema inventories all 491 sources and excludes both control manifests', async (t) => {
  const { root, store } = await fixture(t);
  const formats = { epub: 221, pdf: 227, mobi: 2, prc: 4, zip: 15, torrent: 22 };
  const results = [];
  let total = 0;
  for (const [format, count] of Object.entries(formats)) for (let number = 0; number < count; number++) {
    const name = `Original collection/${format}-${number}.${format}`;
    const bytes = Buffer.from(`Synthetic ${name}`);
    await source(root, name, bytes);
    results.push({ id: `source-${results.length}`, relative_path: name, source_bytes: bytes.length,
      destination_bytes: bytes.length, sha256: sha(bytes), status: 'verified' });
    total += bytes.length;
  }
  await source(root, 'SOURCE-MANIFEST.json', JSON.stringify({ source_count: 491 }));
  await source(root, 'TRANSFER-MANIFEST.json', JSON.stringify({ source_count: 491, source_bytes: total,
    verified_count: 491, verified_bytes: total, complete: true, unexpected: [], blocked_count: 0, results }));
  const job = await inventoryLibrary(store, root);
  assert.equal(job.total, 491);
  assert.equal(job.summary.inventoryComplete, true);
  assert.deepEqual(job.summary.formats, formats);
  assert.equal(store.listFiles({ jobId: job.id, limit: 1000 }).length, 491);
  assert.equal(store.db.prepare('SELECT count(*) AS count FROM import_receipts').get().count, 0);
  assert.equal(store.summary().books, 0);
});

test('copies every format locally, deduplicates bytes, and never merges editions by stem', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'guide.epub', 'EPUB edition bytes');
  await source(root, 'guide.pdf', 'Different PDF edition bytes');
  await source(root, 'copy.epub', 'EPUB edition bytes');
  for (const format of ['mobi', 'prc', 'zip', 'torrent']) await source(root, `retained.${format}`);
  let calls = 0;
  const job = await importLibrary(store, root, { extractor: async (request) => { calls++; return fakeExtractor(request); } });
  assert.equal(job.state, 'completed');
  assert.equal(job.processed, 7);
  assert.deepEqual(job.summary.statuses, { queued: 0, running: 0, completed: 2, duplicate: 1, unsupported: 3, auxiliary: 1, failed: 0 });
  assert.equal(calls, 3);
  const files = store.listFiles();
  const editions = files.filter((file) => file.relative_path.startsWith('guide.'));
  assert.equal(editions[0].group_key, editions[1].group_key);
  assert.notEqual(editions[0].book_id, editions[1].book_id);
  assert.ok(editions.every((file) => file.metadata.candidateGroup.conclusive === false));
  for (const file of files) {
    assert.equal(file.staged, 1);
    assert.ok(isWithin(join(store.dataDir, 'sources'), file.path));
    assert.ok(isWithin(root, file.source_path));
    assert.equal(sha(await readFile(file.path)), file.sha256);
  }
  await rm(root, { recursive: true });
  assert.equal(store.listBooks().length, 2);
  const again = await runImport(store, job.id, { extractor: () => assert.fail('Completed file was reparsed') });
  assert.equal(again.processed, 7);
  assert.equal(store.db.prepare('SELECT count(*) AS count FROM import_receipts').get().count, 7);
  assert.equal((await readdir(join(store.dataDir, 'sources'))).filter((name) => name.endsWith('.part')).length, 0);
});

test('a source larger than the old 64 MiB cap is copied without whole-file buffering', async (t) => {
  const { root, store } = await fixture(t);
  const path = join(root, 'large.pdf');
  const file = await open(path, 'w');
  await file.truncate(65 * 1024 * 1024 + 17);
  await file.close();
  const job = await importLibrary(store, root, { extractor: fakeExtractor });
  assert.equal(job.summary.statuses.completed, 1);
  assert.equal(store.listFiles()[0].size, 65 * 1024 * 1024 + 17);
});

test('checksum mismatch fails before parsing and retains an explicit receipt', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.pdf', 'wrong');
  await source(root, 'TRANSFER-MANIFEST.json', JSON.stringify({ source_count: 1, source_bytes: 5, complete: true,
    results: [{ relative_path: 'one.pdf', source_bytes: 5, destination_bytes: 5, sha256: sha('right'), status: 'verified' }] }));
  const job = await importLibrary(store, root, { extractor: () => assert.fail('Unverified source reached parser') });
  assert.equal(job.state, 'completed_with_errors');
  assert.equal(job.summary.statuses.failed, 1);
  const file = store.listFiles()[0];
  assert.equal(file.staged, 0);
  assert.match(file.error, /hash\/size/);
  assert.equal(store.db.prepare('SELECT status FROM import_receipts').get().status, 'failed');
  assert.deepEqual(await readdir(join(store.dataDir, 'sources')), []);
});

test('failed extraction resumes explicitly from managed bytes even after original sources disappear', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.epub');
  const first = await importLibrary(store, root, { extractor: async () => { throw new Error('synthetic parser fault'); } });
  assert.equal(first.state, 'completed_with_errors');
  assert.equal(store.listFiles()[0].staged, 1);
  await rm(root, { recursive: true });
  await runImport(store, first.id, { extractor: () => assert.fail('Failure retried without explicit request') });
  const second = await runImport(store, first.id, { retryFailed: true, extractor: fakeExtractor });
  assert.equal(second.state, 'completed');
  assert.deepEqual(store.db.prepare('SELECT status FROM import_receipts ORDER BY id').all().map((row) => row.status), ['failed', 'completed']);
});

test('interrupted running receipt is retained while a new attempt completes', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.epub');
  const job = await inventoryLibrary(store, root);
  const file = store.listFiles()[0];
  store.beginFile(job.id, file.id);
  const resumed = await runImport(store, job.id, { extractor: fakeExtractor });
  assert.equal(resumed.state, 'completed');
  const receipts = store.db.prepare('SELECT status,error,attempt FROM import_receipts ORDER BY id').all();
  assert.deepEqual(receipts.map((row) => row.status), ['failed', 'completed']);
  assert.match(receipts[0].error, /Interrupted/);
  assert.equal(receipts[1].attempt, 2);
});

test('pause queues only its interrupted file and does not retry earlier failures', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'a.epub');
  await source(root, 'b.epub');
  const controller = new AbortController();
  let calls = 0;
  const job = await importLibrary(store, root, { signal: controller.signal, extractor: async () => {
    if (++calls === 1) throw new Error('first file fails');
    controller.abort();
    return { ok: true, book: extracted() };
  } });
  assert.equal(job.state, 'paused');
  assert.equal(job.summary.statuses.failed, 1);
  assert.equal(job.summary.statuses.queued, 1);
  const resumed = await runImport(store, job.id, { extractor: fakeExtractor });
  assert.equal(resumed.summary.statuses.failed, 1);
  assert.equal(resumed.summary.statuses.completed, 1);
});

test('same-source reimport preserves human metadata, reading position and unchanged vector identity', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.epub');
  const job = await importLibrary(store, root, { extractor: fakeExtractor });
  const file = store.listFiles()[0];
  const section = store.db.prepare('SELECT * FROM sections').get();
  const chunk = store.db.prepare('SELECT * FROM chunks').get();
  store.db.prepare('UPDATE books SET title=?,metadata_json=? WHERE id=?').run('My title', JSON.stringify({ tags: ['reviewed'], editedAt: '2026-10-04' }), file.book_id);
  store.db.prepare('INSERT INTO reading_progress(book_id,section_id,updated_at) VALUES(?,?,?)').run(file.book_id, section.id, '2026-10-04');
  store.db.prepare('UPDATE chunks SET embedding_json=?,embedding_model=?,embedding_dimension=2,embedding_norm=1 WHERE id=?').run('[1,0]', 'synthetic', chunk.id);
  const attempt = store.beginFile(job.id, file.id);
  store.commitBook(job.id, file.id, attempt.receiptId, extracted());
  assert.equal(store.listBooks()[0].title, 'My title');
  assert.deepEqual(store.listBooks()[0].metadata.tags, ['reviewed']);
  assert.equal(store.listBooks()[0].metadata.pipelineVersion, PIPELINE_VERSION);
  assert.equal(store.db.prepare('SELECT section_id FROM reading_progress').get().section_id, section.id);
  assert.equal(store.db.prepare('SELECT embedding_json FROM chunks WHERE id=?').get(chunk.id).embedding_json, '[1,0]');
  const next = store.beginFile(job.id, file.id);
  store.commitBook(job.id, file.id, next.receiptId, extracted('Changed parser text.'));
  assert.equal(store.db.prepare('SELECT id FROM sections WHERE id=?').get(section.id), undefined);
  assert.equal(store.db.prepare('SELECT id FROM chunks WHERE id=?').get(chunk.id), undefined);
  assert.equal(store.db.prepare('SELECT section_id FROM reading_progress').get().section_id, null);
});

test('book and FTS replacement rollback together on malformed later section', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.epub');
  const job = await importLibrary(store, root, { extractor: fakeExtractor });
  const file = store.listFiles()[0];
  const before = store.db.prepare('SELECT * FROM chunks').all();
  const attempt = store.beginFile(job.id, file.id);
  const malformed = extracted('Replacement text must not survive rollback.');
  malformed.sections.push({ ordinal: 2, text: null, locator: {} });
  assert.throws(() => store.commitBook(job.id, file.id, attempt.receiptId, malformed), /Invalid source section/);
  assert.deepEqual(store.db.prepare('SELECT * FROM chunks').all(), before);
  assert.equal(store.db.prepare("SELECT count(*) AS count FROM chunks_fts WHERE chunks_fts MATCH 'amber'").get().count, 1);
  assert.equal(store.db.prepare("SELECT count(*) AS count FROM chunks_fts WHERE chunks_fts MATCH 'Replacement'").get().count, 0);
});

test('chunks preserve Unicode source slices, overlap and section bounds', () => {
  const text = `  ${Array.from({ length: 725 }, (_, i) => `🧭word${i}`).join(' \n')}  `;
  const chunks = [...chunkText(text)];
  assert.equal(chunks.length, 3);
  for (const chunk of chunks) assert.equal(chunk.text, text.slice(chunk.charStart, chunk.charEnd));
  assert.ok(chunks[1].charStart < chunks[0].charEnd);
  assert.ok(chunks[2].charEnd <= text.length);
  assert.equal([...chunkText(' \n\t ')].length, 0);
});

test('unrelated databases and source symlinks fail without altering external content', async (t) => {
  const { directory, root, store } = await fixture(t);
  const legacy = join(directory, 'legacy.sqlite');
  const db = new DatabaseSync(legacy);
  db.exec('CREATE TABLE books(book_id TEXT PRIMARY KEY)');
  db.close();
  const before = await readFile(legacy);
  assert.throws(() => new LibraryStore(legacy), /unrelated/);
  assert.deepEqual(await readFile(legacy), before);
  const outside = await source(directory, 'outside.epub', 'Never follow this source symlink');
  await symlink(outside, join(root, 'linked.epub'));
  const job = await importLibrary(store, root, { extractor: () => assert.fail('Symlink reached parser') });
  assert.equal(job.summary.statuses.failed, 1);
  assert.equal(store.listFiles()[0].staged, 0);
  assert.equal((await readFile(outside)).toString(), 'Never follow this source symlink');
});

test('worker deadline, output bound and invalid protocol reject owned child processes', async (t) => {
  const { directory } = await fixture(t);
  const slow = await source(directory, 'slow.mjs', "setInterval(() => {}, 1000);\n");
  await assert.rejects(runParserWorker({}, { workerPath: slow, timeoutMs: 100 }), (error) => error.code === 'PARSER_TIMEOUT');
  const noisy = await source(directory, 'noisy.mjs', "process.stdout.write('x'.repeat(8192));\n");
  await assert.rejects(runParserWorker({}, { workerPath: noisy, maxWorkerOutputBytes: 64 }), (error) => error.code === 'PARSER_OUTPUT_LIMIT');
  const malformed = await source(directory, 'malformed.mjs', "process.stdout.write('{}\\n{}\\n');\n");
  await assert.rejects(runParserWorker({}, { workerPath: malformed }), (error) => error.code === 'PARSER_PROTOCOL');
});

async function transfer(root, bytes, name = 'one.epub') {
  await source(root, name, bytes);
  await source(root, 'TRANSFER-MANIFEST.json', JSON.stringify({ complete: true, source_count: 1, source_bytes: Buffer.byteLength(bytes),
    results: [{ relative_path: name, source_bytes: Buffer.byteLength(bytes), destination_bytes: Buffer.byteLength(bytes), sha256: sha(bytes), status: 'verified' }] }));
}

test('a later B manifest cannot resume against retained A bytes or change A global status', async (t) => {
  const { root, store } = await fixture(t);
  const firstBytes = 'Original A edition', laterBytes = 'Modified B edition';
  await transfer(root, firstBytes);
  const first = await importLibrary(store, root, { extractor: fakeExtractor });
  const original = store.listFiles()[0];
  const chunks = store.db.prepare('SELECT * FROM chunks').all();
  await transfer(root, laterBytes);
  const conflict = await inventoryLibrary(store, root);
  const identity = store.getJobFile(conflict.id, original.id);
  assert.equal(identity.expected_sha256, sha(laterBytes));
  assert.equal(identity.expected_size, Buffer.byteLength(laterBytes));
  assert.equal(identity.retryable, 0);
  assert.match(identity.inventory_error, /Immutable source inventory changed/);
  for (let attempt = 0; attempt < 3; attempt++) {
    const resumed = await runImport(store, conflict.id, { retryFailed: true, extractor: () => assert.fail('Conflicting inventory reached extraction') });
    assert.equal(resumed.state, 'completed_with_errors');
    assert.equal(resumed.summary.statuses.failed, 1);
    assert.equal(resumed.summary.statuses.completed, 0);
    assert.deepEqual(store.getJobFile(conflict.id, original.id), identity);
    assert.deepEqual(store.listFiles()[0], original);
  }
  assert.equal(store.getJob(first.id).summary.statuses.completed, 1);
  assert.equal(store.getJobFile(first.id, original.id).expected_sha256, sha(firstBytes));
  assert.equal(sha(await readFile(original.path)), sha(firstBytes));
  assert.deepEqual(store.db.prepare('SELECT * FROM chunks').all(), chunks);
  const receipts = store.db.prepare('SELECT attempt,status,metadata_json FROM import_receipts WHERE job_id=?').all(conflict.id);
  assert.equal(receipts.length, 1);
  assert.equal(receipts[0].attempt, 0);
  assert.equal(receipts[0].status, 'failed');
  assert.equal(JSON.parse(receipts[0].metadata_json).expectedIdentity.sha256, sha(laterBytes));
});

test('an unstaged first manifest identity cannot be replaced by a second manifest', async (t) => {
  const { root, store } = await fixture(t);
  await transfer(root, 'A original');
  const first = await inventoryLibrary(store, root);
  const original = store.listFiles()[0];
  assert.equal(original.sha256, null);
  await transfer(root, 'B modified');
  const second = await inventoryLibrary(store, root);
  await runImport(store, second.id, { retryFailed: true, extractor: () => assert.fail('Conflicting unstaged identity reached extraction') });
  assert.deepEqual(store.listFiles()[0], original);
  assert.equal(store.getJobFile(first.id, original.id).expected_sha256, sha('A original'));
  assert.equal(store.getJobFile(second.id, original.id).expected_sha256, sha('B modified'));
  assert.equal(store.getJobFile(second.id, original.id).retryable, 0);
});

test('character-bounded chunks cover oversized tokens exactly without splitting astral pairs', () => {
  const text = ` \n${'x'.repeat(3999)}🧭${'y'.repeat(12001)}\t tail  `;
  const chunks = [...chunkText(text)];
  assert.ok(chunks.length >= 5 && chunks.length <= 8);
  const covered = new Uint8Array(text.length);
  for (const chunk of chunks) {
    assert.ok(chunk.text.length > 0 && chunk.text.length <= 4000);
    assert.equal(chunk.text, text.slice(chunk.charStart, chunk.charEnd));
    assert.equal(chunk.text.isWellFormed(), true);
    assert.ok(chunk.text.match(/\S+/gu).length <= 400);
    covered.fill(1, chunk.charStart, chunk.charEnd);
  }
  for (const match of text.matchAll(/\S+/gu)) for (let index = match.index; index < match.index + match[0].length; index++) assert.equal(covered[index], 1);
  assert.equal(chunks[0].charStart, 2);
  assert.equal(chunks.at(-1).charEnd, text.trimEnd().length);
  const spaced = `alpha${' '.repeat(16001)}beta`;
  const separated = [...chunkText(spaced, { maxChars: 10, words: 2, overlap: 1 })];
  assert.deepEqual(separated.map((chunk) => chunk.text), ['alpha', 'beta']);
  assert.equal(separated[1].charStart, spaced.indexOf('beta'));
  assert.throws(() => [...chunkText('🧭', { maxChars: 1 })], /Invalid/);
});

test('managed reindex needs no original folder and preserves unchanged citations, vectors, edits and reading progress', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.epub');
  await importLibrary(store, root, { extractor: fakeExtractor });
  const original = store.listFiles()[0];
  const section = store.db.prepare('SELECT * FROM sections').get();
  const chunk = store.db.prepare('SELECT * FROM chunks').get();
  const legacyChunkId = `${section.id}:c0`;
  store.db.prepare('UPDATE chunks SET id=?,locator_json=? WHERE id=?').run(legacyChunkId,
    JSON.stringify({ ...JSON.parse(chunk.locator_json), pipelineVersion: 'node-text-v1-words400-overlap80-utf16' }), chunk.id);
  store.db.prepare('UPDATE books SET title=?,metadata_json=? WHERE id=?').run('Keep my title', JSON.stringify({ tags: ['kept'], editedAt: '2026-10-04' }), original.book_id);
  store.db.prepare('INSERT INTO reading_progress(book_id,section_id,updated_at) VALUES(?,?,?)').run(original.book_id, section.id, '2026-10-04');
  store.db.prepare('UPDATE chunks SET embedding_json=?,embedding_model=?,embedding_dimension=2,embedding_norm=1 WHERE id=?').run('[1,0]', 'synthetic', legacyChunkId);
  await rm(root, { recursive: true });
  const queued = createReindexJob(store, { bookIds: [original.book_id] });
  assert.equal(queued.summary.mode, 'reindex');
  assert.equal(queued.state, 'queued');
  assert.equal(queued.total, 1);
  assert.equal(store.getFile(original.id).status, 'completed');
  const complete = await runImport(store, queued.id, { extractor: async (request) => {
    assert.equal(request.path, original.path);
    assert.equal(sha(await readFile(request.path)), original.sha256);
    return fakeExtractor(request);
  } });
  assert.equal(complete.state, 'completed');
  assert.equal(store.listBooks()[0].title, 'Keep my title');
  assert.deepEqual(store.listBooks()[0].metadata.tags, ['kept']);
  assert.equal(store.db.prepare('SELECT section_id FROM reading_progress').get().section_id, section.id);
  assert.equal(store.db.prepare('SELECT id FROM chunks').get().id, legacyChunkId);
  assert.equal(store.db.prepare('SELECT embedding_json FROM chunks').get().embedding_json, '[1,0]');
  const failed = createReindexJob(store);
  await runImport(store, failed.id, { extractor: () => { throw new Error('synthetic reindex failure'); } });
  assert.equal(store.getJob(failed.id).state, 'completed_with_errors');
  assert.equal(store.getFile(original.id).status, 'completed');
  assert.equal(store.getFile(original.id).book_id, original.book_id);
  assert.equal(store.db.prepare('SELECT id FROM chunks').get().id, legacyChunkId);
});

test('v1 oversized chunk is replaced without retargeting its citation or changing its unchanged section', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.epub');
  const text = 'q'.repeat(16001);
  await importLibrary(store, root, { extractor: () => ({ ok: true, book: extracted(text) }) });
  const file = store.listFiles()[0], section = store.db.prepare('SELECT * FROM sections').get();
  store.db.prepare('DELETE FROM chunks WHERE book_id=?').run(file.book_id);
  const oldId = `${section.id}:c0`;
  store.db.prepare('INSERT INTO chunks(id,book_id,section_id,ordinal,text,locator_json,embedding_json) VALUES(?,?,?,0,?,?,?)')
    .run(oldId, file.book_id, section.id, text, JSON.stringify({ format: 'epub', member: 'one.xhtml', sectionId: section.id,
      charStart: 0, charEnd: text.length, offsetBasis: 'section_utf16_code_units', sourceSha256: file.sha256, pipelineVersion: 'node-text-v1-words400-overlap80-utf16' }), '[1,0]');
  store.db.prepare('INSERT INTO reading_progress(book_id,section_id,updated_at) VALUES(?,?,?)').run(file.book_id, section.id, '2026-10-04');
  const job = createReindexJob(store);
  await runImport(store, job.id, { extractor: () => ({ ok: true, book: extracted(text) }) });
  assert.equal(store.db.prepare('SELECT id FROM chunks WHERE id=?').get(oldId), undefined);
  assert.equal(store.db.prepare('SELECT id FROM sections').get().id, section.id);
  assert.equal(store.db.prepare('SELECT section_id FROM reading_progress').get().section_id, section.id);
  const chunks = store.db.prepare('SELECT * FROM chunks ORDER BY ordinal').all();
  assert.equal(chunks.map((chunk) => chunk.text).join(''), text);
  assert.ok(chunks.every((chunk) => chunk.text.length <= 4000 && chunk.embedding_json === null));
  assert.equal(store.db.prepare('SELECT count(*) AS n FROM chunks_fts').get().n, chunks.length);
});

test('configured conversion is forwarded only through bounded worker requests and absent configuration stays explicit', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.mobi');
  const unsupported = await importLibrary(store, root, { extractor: () => assert.fail('Unconfigured conversion ran') });
  assert.equal(unsupported.summary.statuses.unsupported, 1);
  const file = store.listFiles()[0];
  const unsupportedReceipt = store.db.prepare('SELECT * FROM import_receipts WHERE job_id=?').get(unsupported.id);
  assert.match(file.metadata.reason, /MOBI text extraction is not supported/);
  assert.equal(JSON.parse(unsupportedReceipt.metadata_json).reason, file.metadata.reason);
  const converter = { executable: '/operator/chosen/ebook-convert', isolation: 'container' };
  const queued = createReindexJob(store, { fileIds: [file.id] });
  await runImport(store, queued.id, { converter, extractor: async (request) => {
    assert.equal(request.format, 'mobi');
    assert.deepEqual(request.converter, converter);
    assert.ok(isWithin(join(store.dataDir, 'derived'), request.derivedDir));
    assert.equal(request.path, file.path);
    const result = extracted();
    result.sections[0].locator = { format: 'mobi', anchors: { unused: { charStart: 0, charEnd: 4 } },
      derived: { format: 'epub', member: 'one.xhtml', sha256: sha('derived'), anchors: { first: { charStart: 0, charEnd: 4 } } } };
    return { ok: true, book: result };
  } });
  assert.equal(store.getJob(queued.id).state, 'completed');
  const completed = store.getFile(file.id);
  assert.equal(completed.metadata.reason, undefined);
  assert.equal(completed.metadata.extraction.searchable, true);
  assert.deepEqual(completed.metadata.candidateGroup, file.metadata.candidateGroup);
  assert.equal(completed.path, file.path);
  assert.equal(completed.sha256, sha(await readFile(completed.path)));
  assert.equal(completed.sha256, file.sha256);
  const completedReceipt = store.db.prepare('SELECT * FROM import_receipts WHERE job_id=?').get(queued.id);
  assert.equal(completedReceipt.status, 'completed');
  assert.equal(JSON.parse(completedReceipt.metadata_json).reason, undefined);
  assert.equal(JSON.parse(completedReceipt.metadata_json).extraction.searchable, true);
  assert.deepEqual(store.db.prepare('SELECT * FROM import_receipts WHERE id=?').get(unsupportedReceipt.id), unsupportedReceipt);
  const sectionLocator = JSON.parse(store.db.prepare('SELECT locator_json FROM sections').get().locator_json);
  const chunkLocator = JSON.parse(store.db.prepare('SELECT locator_json FROM chunks').get().locator_json);
  assert.ok(sectionLocator.anchors && sectionLocator.derived.anchors);
  assert.equal(chunkLocator.anchors, undefined);
  assert.equal(chunkLocator.derived.anchors, undefined);
  assert.equal(chunkLocator.derived.sha256, sha('derived'));
  const again = createReindexJob(store);
  await runImport(store, again.id, { extractor: () => assert.fail('Unconfigured re-conversion ran') });
  assert.equal(store.getJob(again.id).state, 'completed_with_errors');
  assert.equal(store.getFile(file.id).status, 'completed');
  assert.equal(store.getFile(file.id).book_id, file.sha256);
  assert.deepEqual(store.getFile(file.id).metadata, completed.metadata);
  assert.deepEqual(store.db.prepare('SELECT * FROM import_receipts WHERE id=?').get(completedReceipt.id), completedReceipt);
  assert.deepEqual(store.db.prepare('SELECT * FROM import_receipts WHERE id=?').get(unsupportedReceipt.id), unsupportedReceipt);
});

test('a live same-process lock cannot be stolen by a separately opened store', async (t) => {
  const { root, store } = await fixture(t);
  await source(root, 'one.epub');
  const job = await inventoryLibrary(store, root);
  store.db.prepare('INSERT INTO import_lock(id,owner,pid,process_identity,job_id,started_at) VALUES(1,?,?,?,?,?)')
    .run('other-owner', process.pid, processIdentity(), job.id, '2026-10-04');
  const second = new LibraryStore(store.path);
  try {
    await assert.rejects(runImport(second, job.id, { extractor: fakeExtractor }), (error) => error.code === 'IMPORT_BUSY');
    assert.equal(store.db.prepare('SELECT owner FROM import_lock').get().owner, 'other-owner');
    assert.equal(store.getJob(job.id).state, 'queued');
  } finally { second.close(); }
});

test('legacy inventory failures migrate fail-closed even after a prior retry cleared the job error', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'librarian-import-migration-'));
  const root = join(directory, 'source');
  await mkdir(root);
  let store = new LibraryStore(join(directory, 'state', 'library.sqlite'));
  t.after(async () => { store.close(); await rm(directory, { recursive: true, force: true }); });
  await transfer(root, 'original');
  await importLibrary(store, root, { extractor: fakeExtractor });
  const original = store.listFiles()[0];
  await transfer(root, 'modified');
  const conflict = await inventoryLibrary(store, root);
  store.db.prepare("UPDATE import_job_files SET status='completed',error=NULL WHERE job_id=?").run(conflict.id);
  for (const column of ['expected_sha256', 'expected_size', 'expected_mtime_ms', 'inventory_error', 'retryable']) {
    store.db.exec(`ALTER TABLE import_job_files DROP COLUMN ${column}`);
  }
  store.db.exec('ALTER TABLE import_lock DROP COLUMN process_identity');
  const path = store.path;
  store.close();
  store = new LibraryStore(path);
  const restored = store.getJobFile(conflict.id, original.id);
  assert.equal(restored.retryable, 0);
  assert.equal(restored.status, 'failed');
  assert.match(restored.inventory_error, /Immutable source inventory changed/);
  await runImport(store, conflict.id, { retryFailed: true, extractor: () => assert.fail('Historical inventory conflict retried') });
  assert.equal(store.getJob(conflict.id).state, 'completed_with_errors');
  assert.equal(store.getFile(original.id).sha256, original.sha256);
  assert.equal(store.getFile(original.id).status, 'completed');
  assert.ok(store.db.prepare('PRAGMA table_info(import_lock)').all().some((column) => column.name === 'process_identity'));
});
