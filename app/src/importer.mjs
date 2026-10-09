import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { constants, createReadStream, createWriteStream } from 'node:fs';
import { chmod, lstat, mkdir, open, readFile, readdir, realpath, rename, stat, unlink } from 'node:fs/promises';
import { basename, dirname, extname, join, relative, resolve } from 'node:path';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { canonicalDestination, digest, isWithin, now, PIPELINE_VERSION } from './store.mjs';
import { lockOwnerAlive, processIdentity } from './process-lock.mjs';

const activeStores = new Set();
const DEFAULT_WORKER = fileURLToPath(new URL('./extract-worker.mjs', import.meta.url));
const SUPPORTED = new Set(['epub', 'pdf']);
const CONTROL_NAME = /^(?:SOURCE|TRANSFER)(?:[-_][A-Za-z0-9_-]+)?\.json$/i;
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const MAX_INVENTORY_FILES = 100_000;

function errorText(error) { return String(error?.message ?? error).replace(/\0/g, '').slice(0, 4096); }
function fail(message, code = 'IMPORT_ERROR') { return Object.assign(new Error(message), { code }); }
function aborted(signal) { if (signal?.aborted) throw fail('Import paused by request', 'ABORT_ERR'); }
function acquire(store, jobId) {
  if (activeStores.has(store.path)) throw fail('Another import is already running', 'IMPORT_BUSY');
  const owner = randomUUID();
  store.transaction(() => {
    const existing = store.db.prepare('SELECT * FROM import_lock WHERE id=1').get();
    if (lockOwnerAlive(existing)) throw fail('Another import process is already running', 'IMPORT_BUSY');
    store.db.exec('DELETE FROM import_lock WHERE id=1');
    store.db.prepare('INSERT INTO import_lock(id,owner,pid,process_identity,job_id,started_at) VALUES(1,?,?,?,?,?)')
      .run(owner, process.pid, processIdentity(), jobId, now());
  });
  activeStores.add(store.path);
  return () => {
    store.db.prepare('DELETE FROM import_lock WHERE id=1 AND owner=?').run(owner);
    activeStores.delete(store.path);
  };
}

function relativeName(value) {
  if (typeof value !== 'string' || !value || value.includes('\0') || value.includes('\\')
      || value.startsWith('/') || value.split('/').some((part) => !part || part === '.' || part === '..')) {
    throw fail('Manifest contains an unsafe relative path', 'INVALID_MANIFEST');
  }
  return value;
}

function storage(store, dataDir = store.dataDir) {
  if (canonicalDestination(dataDir) !== store.dataDir) throw fail('Import storage must match the library data directory');
  return { sources: join(store.dataDir, 'sources'), covers: join(store.dataDir, 'covers'), derived: join(store.dataDir, 'derived') };
}

async function* walk(root, current = root, depth = 0) {
  if (depth > 64) throw fail('Source directory nesting exceeds the inventory bound');
  const entries = await readdir(current, { withFileTypes: true });
  entries.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  for (const entry of entries) {
    if (current === root && CONTROL_NAME.test(entry.name)) continue;
    const path = join(current, entry.name);
    if (entry.isDirectory()) yield* walk(root, path, depth + 1);
    else yield relative(root, path).split('\\').join('/');
  }
}

async function loadManifest(root, requested) {
  let path = requested ? resolve(requested) : null;
  if (!path) {
    const names = (await readdir(root)).filter((name) => /^TRANSFER(?:[-_][A-Za-z0-9_-]+)?\.json$/i.test(name));
    const preferred = names.find((name) => name === 'TRANSFER-MANIFEST.json');
    if (preferred) path = join(root, preferred);
    else if (names.length === 1) path = join(root, names[0]);
    else if (names.length > 1) throw fail('Multiple transfer manifests; specify manifestPath', 'INVALID_MANIFEST');
  }
  if (!path) return null;
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink() || info.size > MAX_MANIFEST_BYTES) throw fail('Invalid or oversized transfer manifest', 'INVALID_MANIFEST');
  const raw = await readFile(path);
  const document = JSON.parse(raw.toString('utf8'));
  const rows = Array.isArray(document) ? document : (document.results ?? document.files);
  if (!Array.isArray(rows) || rows.length > MAX_INVENTORY_FILES) throw fail('Manifest must contain a bounded results or files array', 'INVALID_MANIFEST');
  if (document.complete === false || (document.blocked_count ?? 0) > 0 || (document.unexpected?.length ?? 0) > 0) {
    throw fail('Transfer manifest reports an incomplete or inconsistent transfer', 'INVALID_MANIFEST');
  }
  const count = document.source_count ?? rows.length;
  if (!Number.isSafeInteger(count) || count !== rows.length) throw fail('Manifest source count does not match its entries', 'INVALID_MANIFEST');
  const seen = new Set();
  const entries = rows.map((row) => {
    const name = relativeName(row.relative_path ?? row.relativePath ?? row.path);
    const size = row.source_bytes ?? row.size ?? row.bytes;
    if (seen.has(name)) throw fail('Manifest repeats a source path', 'INVALID_MANIFEST');
    seen.add(name);
    if (!Number.isSafeInteger(size) || size < 0 || !/^[a-f0-9]{64}$/.test(row.sha256 ?? '')) {
      throw fail('Manifest requires a valid size and SHA-256 for every source', 'INVALID_MANIFEST');
    }
    if (row.destination_bytes != null && row.destination_bytes !== size) throw fail('Manifest source and destination sizes disagree', 'INVALID_MANIFEST');
    return { relativePath: name, expectedSize: size, expectedSha256: row.sha256,
      error: row.status && row.status !== 'verified' ? 'Transfer manifest entry is not verified' : null,
      metadata: { transferSourceId: row.id ?? null, transferVerification: row.verification ?? null } };
  });
  return { entries, path, sha256: digest(raw), expectedCount: count, sourceBytes: document.source_bytes ?? null };
}

export async function inventoryLibrary(store, inputRoot, options = {}) {
  const root = await realpath(resolve(inputRoot));
  if (!(await stat(root)).isDirectory()) throw fail('Import root must be a directory');
  storage(store, options.dataDir);
  if (isWithin(root, store.dataDir) || isWithin(store.dataDir, root)) throw fail('Immutable sources and managed library data must be separate');
  const job = store.createJob(root, { inventoryComplete: false, managedDataDir: store.dataDir });
  let release;
  try {
    release = acquire(store, job.id);
    const manifest = await loadManifest(root, options.manifestPath);
    const discovered = [];
    for await (const name of walk(root)) {
      aborted(options.signal);
      discovered.push(name);
      if (discovered.length > MAX_INVENTORY_FILES) throw fail('Source inventory exceeds its file-count bound');
    }
    const declared = new Set(manifest?.entries.map((entry) => entry.relativePath));
    const extras = manifest ? discovered.filter((name) => !declared.has(name)) : [];
    const entries = manifest ? [...manifest.entries, ...extras.map((name) => ({ relativePath: name, error: 'File is absent from the transfer manifest' }))]
      : discovered.map((relativePath) => ({ relativePath }));
    for (const [ordinal, entry] of entries.entries()) {
      aborted(options.signal);
      const sourcePath = resolve(root, entry.relativePath);
      let info, error = entry.error;
      try {
        info = await lstat(sourcePath);
        if (!info.isFile() || info.isSymbolicLink() || !isWithin(root, await realpath(sourcePath))) throw fail('Source is not a contained regular file');
        if (!Number.isSafeInteger(info.size)) throw fail('Source size cannot be represented safely');
        if (entry.expectedSize != null && info.size !== entry.expectedSize) throw fail('Source size differs from transfer manifest');
      } catch (caught) { error = errorText(caught); }
      const format = extname(entry.relativePath).slice(1).toLowerCase() || 'unknown';
      const stem = basename(entry.relativePath, extname(entry.relativePath)).normalize('NFC').toLowerCase();
      const groupKey = digest(`${root}\0${dirname(entry.relativePath).normalize('NFC')}\0${stem}`);
      store.registerFile(job.id, { ...entry, sourcePath, size: info?.size ?? entry.expectedSize ?? 0, format,
        mtimeMs: info?.mtimeMs ?? null, groupKey, error,
        metadata: { ...entry.metadata, candidateGroup: { key: groupKey, reason: 'same-folder-stem', conclusive: false } } }, ordinal);
    }
    const totalBytes = entries.reduce((sum, entry) => sum + (entry.expectedSize ?? 0), 0);
    if (manifest?.sourceBytes != null && manifest.sourceBytes !== totalBytes) throw fail('Transfer manifest byte total does not match its entries', 'INVALID_MANIFEST');
    return store.refreshJob(job.id, { state: extras.length ? 'failed' : 'queued', inventoryComplete: extras.length === 0,
      manifest: manifest ? { filename: basename(manifest.path), sha256: manifest.sha256, expectedCount: manifest.expectedCount, expectedBytes: manifest.sourceBytes } : null,
      unexpectedFiles: extras, accounting: 'Every declared source receives an individual durable file record' });
  } catch (error) {
    store.refreshJob(job.id, { state: options.signal?.aborted ? 'paused' : 'failed', inventoryComplete: false, error: errorText(error) });
    throw error;
  } finally { release?.(); }
}

async function hashFile(path, signal) {
  aborted(signal);
  const hash = createHash('sha256');
  let bytes = 0;
  const input = createReadStream(path, { flags: constants.O_RDONLY | constants.O_NOFOLLOW, signal });
  for await (const chunk of input) { hash.update(chunk); bytes += chunk.length; }
  return { sha256: hash.digest('hex'), bytes };
}

async function stageFile(store, file, options, progress, sourceRoot, expected) {
  const { sources } = storage(store, options.dataDir);
  await mkdir(sources, { recursive: true, mode: 0o700 });
  if (file.staged) {
    const managed = await realpath(file.path);
    if (!isWithin(await realpath(sources), managed)) throw fail('Managed source escaped its storage directory');
    const verified = await hashFile(managed, options.signal);
    if (verified.sha256 !== file.sha256 || verified.bytes !== file.size
        || (expected.expected_sha256 && expected.expected_sha256 !== verified.sha256)
        || (expected.expected_size != null && expected.expected_size !== verified.bytes)) throw fail('Managed source integrity or job identity check failed', 'SOURCE_INTEGRITY');
    return file;
  }
  const sourceInfo = await lstat(file.source_path);
  if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink() || sourceInfo.size !== file.size
      || !isWithin(sourceRoot, await realpath(file.source_path))
      || (file.mtime_ms != null && sourceInfo.mtimeMs !== file.mtime_ms)) throw fail('Immutable source changed after inventory', 'SOURCE_INTEGRITY');
  const temporary = join(sources, `.${randomUUID()}.part`);
  const hash = createHash('sha256');
  let bytes = 0, lastProgress = 0;
  try {
    const observer = new Transform({ transform(chunk, encoding, callback) {
      hash.update(chunk); bytes += chunk.length;
      if (Date.now() - lastProgress >= 500) { lastProgress = Date.now(); progress(bytes); }
      callback(null, chunk);
    } });
    await pipeline(createReadStream(file.source_path, { flags: constants.O_RDONLY | constants.O_NOFOLLOW }), observer,
      createWriteStream(temporary, { flags: 'wx', mode: 0o600 }), { signal: options.signal });
    const sha256 = hash.digest('hex');
    const after = await lstat(file.source_path);
    if (bytes !== file.size || after.size !== sourceInfo.size || after.mtimeMs !== sourceInfo.mtimeMs
        || (file.expected_sha256 && file.expected_sha256 !== sha256)
        || (expected.expected_sha256 && expected.expected_sha256 !== sha256)
        || (expected.expected_size != null && expected.expected_size !== bytes)) throw fail('Source hash/size changed or differs from transfer manifest', 'SOURCE_INTEGRITY');
    const handle = await open(temporary, 'r+');
    try { await handle.sync(); } finally { await handle.close(); }
    const destination = join(sources, `${sha256}.${file.format.replace(/[^a-z0-9]/g, '') || 'bin'}`);
    let exists = false;
    try {
      const target = await lstat(destination);
      if (!target.isFile() || target.isSymbolicLink()) throw fail('Managed source destination is not a regular file');
      exists = true;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (exists) {
      const verified = await hashFile(destination, options.signal);
      if (verified.sha256 !== sha256 || verified.bytes !== bytes) throw fail('Existing managed source is corrupt', 'SOURCE_INTEGRITY');
      await unlink(temporary);
    } else {
      await rename(temporary, destination);
      await chmod(destination, 0o444);
      const directory = await open(sources, 'r');
      try { await directory.sync(); } finally { await directory.close(); }
    }
    store.db.prepare('UPDATE files SET path=?,sha256=?,staged=1,updated_at=? WHERE id=?').run(destination, sha256, now(), file.id);
    progress(bytes);
    return store.getFile(file.id);
  } finally { await unlink(temporary).catch((error) => { if (error.code !== 'ENOENT') throw error; }); }
}

export function runParserWorker(request, { signal, timeoutMs = 180_000, maxWorkerOutputBytes = 64 * 1024 * 1024,
  workerPath = DEFAULT_WORKER } = {}) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || !Number.isSafeInteger(maxWorkerOutputBytes) || maxWorkerOutputBytes < 1) {
    return Promise.reject(fail('Invalid parser resource limits'));
  }
  const input = Buffer.from(`${JSON.stringify(request)}\n`);
  if (input.length > 16 * 1024) return Promise.reject(fail('Parser request exceeds its input bound'));
  return new Promise((resolvePromise, reject) => {
    if (signal?.aborted) { reject(fail('Import paused by request', 'ABORT_ERR')); return; }
    const child = spawn(process.execPath, ['--max-old-space-size=1536', '--disallow-code-generation-from-strings', workerPath], {
      stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
      env: { ...process.env, LIBRARIAN_PARENT_PID: String(process.pid) },
    });
    let failure = null, total = 0, stderr = '';
    const output = [];
    const stop = (error) => { failure ??= error; child.kill('SIGKILL'); };
    const abort = () => stop(fail('Import paused by request', 'ABORT_ERR'));
    const timer = setTimeout(() => stop(fail('Parser exceeded its per-file deadline', 'PARSER_TIMEOUT')), timeoutMs);
    timer.unref();
    signal?.addEventListener('abort', abort, { once: true });
    child.on('error', (error) => { failure ??= error; });
    child.stdin.on('error', (error) => { if (error.code !== 'EPIPE') stop(error); });
    child.stdout.on('data', (chunk) => {
      total += chunk.length;
      if (total > maxWorkerOutputBytes) stop(fail('Parser output exceeded its byte bound', 'PARSER_OUTPUT_LIMIT'));
      else output.push(chunk);
    });
    child.stderr.on('data', (chunk) => { if (stderr.length < 8192) stderr += chunk.toString('utf8').slice(0, 8192 - stderr.length); });
    child.on('close', (code, terminationSignal) => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      if (failure) { reject(failure); return; }
      try {
        const text = Buffer.concat(output).toString('utf8').trim();
        if (!text || text.split(/\r?\n/).length !== 1) throw fail('Parser did not return exactly one JSON response', 'PARSER_PROTOCOL');
        const result = JSON.parse(text);
        if (!result || result.ok !== true) throw fail(errorText(result?.error?.message ?? 'Parser rejected this file'), result?.error?.code ?? 'PARSE_FAILED');
        if (code !== 0 || terminationSignal) throw fail(`Parser terminated unexpectedly (${code ?? terminationSignal}): ${stderr}`, 'PARSER_EXIT');
        resolvePromise(result);
      } catch (error) { reject(error); }
    });
    child.stdin.end(input);
  });
}

function recover(store, jobId, retryFailed, reindex) {
  store.transaction(() => {
    const rows = store.db.prepare(`SELECT f.id FROM files f JOIN import_job_files j ON j.file_id=f.id WHERE j.job_id=? AND j.retryable=1 AND j.inventory_error IS NULL
      AND (j.status='running' ${retryFailed ? "OR j.status='failed'" : ''})`).all(jobId);
    for (const row of rows) {
      store.db.prepare("UPDATE import_receipts SET status='failed',error='Interrupted before completion; queued for resume',finished_at=? WHERE job_id=? AND file_id=? AND finished_at IS NULL")
        .run(now(), jobId, row.id);
      if (!reindex) store.db.prepare("UPDATE files SET status='queued',error=NULL,updated_at=? WHERE id=?").run(now(), row.id);
      store.db.prepare("UPDATE import_job_files SET status='queued',error=NULL WHERE job_id=? AND file_id=?").run(jobId, row.id);
    }
  });
}

export async function runImport(store, jobId, options = {}) {
  const initial = store.getJob(jobId);
  if (!initial) throw fail('Import job not found');
  if (!initial.summary.inventoryComplete) throw fail('Inventory must complete before parsing');
  storage(store, options.dataDir);
  const release = acquire(store, jobId);
  const reindex = initial.summary.mode === 'reindex';
  const supported = (format) => SUPPORTED.has(format) || (['mobi', 'prc'].includes(format) && options.converter != null);
  const notify = (details = {}) => {
    const job = store.refreshJob(jobId, details);
    try { options.onProgress?.(job); } catch { /* A disconnected observer must not interrupt durable import. */ }
    return job;
  };
  try {
    recover(store, jobId, options.retryFailed === true, reindex);
    notify({ state: 'running', error: null, recoveryNotice: null, current: null, pipelineVersion: PIPELINE_VERSION });
    const pending = store.db.prepare("SELECT file_id FROM import_job_files WHERE job_id=? AND status='queued' AND retryable=1 AND inventory_error IS NULL ORDER BY ordinal").all(jobId);
    for (const { file_id: fileId } of pending) {
      aborted(options.signal);
      const { file: initialFile, receiptId } = store.beginFile(jobId, fileId);
      let file = initialFile;
      try {
        const expected = store.assertJobIdentity(jobId, file);
        notify({ current: { fileId, relativePath: file.relative_path, phase: 'copy', bytes: 0, totalBytes: file.size } });
        file = await stageFile(store, file, options, (bytes) => notify({ current: {
          fileId, relativePath: file.relative_path, phase: 'copy', bytes, totalBytes: file.size } }), initial.root, expected);
        store.assertJobIdentity(jobId, file);
        store.db.prepare('UPDATE import_job_files SET expected_sha256=coalesce(expected_sha256,?) WHERE job_id=? AND file_id=?').run(file.sha256, jobId, fileId);
        aborted(options.signal);
        if (reindex && file.book_id && !supported(file.format)) throw fail('Reindex requires a configured extractor for this source format', 'UNSUPPORTED_FORMAT');
        const duplicate = store.db.prepare(`SELECT id,book_id,status FROM files WHERE sha256=? AND id<>? AND staged=1
          AND status IN ('completed','duplicate','unsupported','auxiliary') ORDER BY CASE WHEN book_id IS NOT NULL THEN 0 ELSE 1 END,id LIMIT 1`)
          .get(file.sha256, file.id);
        if (!reindex && duplicate && (duplicate.book_id || !supported(file.format))) {
          store.finishFile(jobId, fileId, receiptId, { status: 'duplicate', bookId: duplicate.book_id,
            duplicateOf: duplicate.id, metadata: { duplicateReason: 'identical-sha256', sourceSha256: file.sha256 } });
        } else if (file.format === 'torrent') {
          store.finishFile(jobId, fileId, receiptId, { status: 'auxiliary', metadata: { reason: 'Torrent retained as auxiliary metadata; no download performed' } });
        } else if (!supported(file.format) && file.format !== 'zip') {
          store.finishFile(jobId, fileId, receiptId, { status: 'unsupported', metadata: { reason: `Source retained; ${file.format.toUpperCase()} text extraction is not supported` } });
        } else {
          const outputDir = join(storage(store, options.dataDir).covers, file.sha256);
          await mkdir(outputDir, { recursive: true, mode: 0o700 });
          const request = { path: file.path, format: file.format, outputDir, limits: options.limits ?? {},
            operation: file.format === 'zip' ? 'inventoryArchive' : 'extractBook' };
          if (['mobi', 'prc'].includes(file.format)) {
            request.converter = options.converter;
            request.derivedDir = join(storage(store, options.dataDir).derived, file.sha256);
            if (!isWithin(store.dataDir, canonicalDestination(request.derivedDir))) throw fail('Derived conversion directory escaped managed storage');
            await mkdir(request.derivedDir, { recursive: true, mode: 0o700 });
          }
          notify({ current: { fileId, relativePath: file.relative_path, phase: file.format === 'zip' ? 'archive-inventory' : 'extract', bytes: file.size, totalBytes: file.size } });
          const result = options.extractor ? await options.extractor(request, options) : await runParserWorker(request, options);
          aborted(options.signal);
          if (!result || result.ok !== true) throw fail(errorText(result?.error?.message ?? 'Invalid parser result'), 'PARSER_PROTOCOL');
          if (file.format === 'zip') {
            if (!result.archive || !Array.isArray(result.archive.entries)) throw fail('Archive inventory is missing', 'PARSER_PROTOCOL');
            store.finishFile(jobId, fileId, receiptId, { status: 'unsupported', metadata: {
              reason: 'Archive contents inventoried only; archive members were not imported or executed', archive: result.archive } });
          } else {
            if (result.book?.title === file.sha256) {
              result.book.title = basename(file.relative_path, extname(file.relative_path));
            }
            if (result.book?.coverPath) {
              const cover = await realpath(result.book.coverPath);
              if (!isWithin(await realpath(outputDir), cover) || !(await stat(cover)).isFile()) throw fail('Parser cover escaped its derived-output directory', 'PARSER_PROTOCOL');
              result.book.coverPath = cover;
            }
            store.commitBook(jobId, fileId, receiptId, result.book);
          }
        }
      } catch (error) {
        store.finishFile(jobId, fileId, receiptId, { status: 'failed', error: errorText(error),
          metadata: { errorCode: error.code ?? 'IMPORT_ERROR', failurePhase: file.staged ? 'extract' : 'copy' } });
        if (options.signal?.aborted || error.code === 'ABORT_ERR') {
          store.transaction(() => {
            if (!reindex) store.db.prepare("UPDATE files SET status='queued',error=NULL,updated_at=? WHERE id=?").run(now(), fileId);
            store.db.prepare("UPDATE import_job_files SET status='queued',error=NULL WHERE job_id=? AND file_id=?").run(jobId, fileId);
          });
          throw error;
        }
      }
      notify({ current: null });
    }
    const final = notify({ current: null });
    return notify({ state: final.summary.statuses.failed ? 'completed_with_errors' : 'completed', completed: true });
  } catch (error) {
    notify({ state: options.signal?.aborted || error.code === 'ABORT_ERR' ? 'paused' : 'failed', completed: false, error: errorText(error), current: null });
    if (options.signal?.aborted || error.code === 'ABORT_ERR') return store.getJob(jobId);
    throw error;
  } finally { release(); }
}

export async function importLibrary(store, root, options = {}) {
  const job = await inventoryLibrary(store, root, options);
  return runImport(store, job.id, options);
}

/** Explicit re-extraction of verified managed bytes; never consult or require the original source folder. */
export function createReindexJob(store, { bookIds, fileIds } = {}) {
  if (bookIds !== undefined && fileIds !== undefined) throw fail('Select bookIds or fileIds, not both');
  const selection = bookIds ?? fileIds;
  if (selection !== undefined && (!Array.isArray(selection) || !selection.length || selection.length > MAX_INVENTORY_FILES
      || selection.some((id) => typeof id !== 'string' || !id || id.length > 512) || new Set(selection).size !== selection.length)) throw fail('Invalid reindex selection');
  let files;
  if (selection) {
    files = selection.map((id) => store.db.prepare(bookIds
      ? "SELECT * FROM files WHERE book_id=? AND staged=1 ORDER BY CASE WHEN status='completed' THEN 0 ELSE 1 END,id LIMIT 1"
      : 'SELECT * FROM files WHERE id=? AND staged=1').get(id));
    if (files.some((file) => !file?.sha256)) throw fail('Every reindex selection must have a verified managed source');
  } else files = store.db.prepare("SELECT * FROM files WHERE staged=1 AND book_id IS NOT NULL ORDER BY CASE WHEN status='completed' THEN 0 ELSE 1 END,id").all();
  const seen = new Set();
  files = files.filter((file) => { if (seen.has(file.sha256)) return false; seen.add(file.sha256); return true; });
  const job = store.createJob(storage(store).sources, { mode: 'reindex', inventoryComplete: false,
    managedDataDir: store.dataDir, pipelineVersion: PIPELINE_VERSION, sourcePolicy: 'Verified managed copies only; original files are not consulted' });
  let release;
  try {
    release = acquire(store, job.id);
    store.transaction(() => {
      for (const [ordinal, file] of files.entries()) {
        store.registerFile(job.id, { sourcePath: file.source_path, relativePath: file.relative_path, size: file.size,
          expectedSize: file.size, expectedSha256: file.sha256, mtimeMs: null, format: file.format, groupKey: file.group_key }, ordinal);
        store.db.prepare("UPDATE import_job_files SET status='queued',error=NULL WHERE job_id=? AND file_id=? AND inventory_error IS NULL")
          .run(job.id, file.id);
      }
      store.refreshJob(job.id, { state: 'queued', inventoryComplete: true });
    });
    return store.getJob(job.id);
  } catch (error) {
    store.refreshJob(job.id, { state: 'failed', inventoryComplete: false, error: errorText(error) });
    throw error;
  } finally { release?.(); }
}
