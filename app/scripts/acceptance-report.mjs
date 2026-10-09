/**
 * Spark-only, read-only collection accounting. No application/model imports.
 * Required: LIBRARIAN_DATA_DIR, LIBRARIAN_ACCEPTANCE_MANIFEST,
 * LIBRARIAN_ACCEPTANCE_JOB_ID, LIBRARIAN_ACCEPTANCE_OUTPUT (fresh JSON outside Git).
 * Optional: LIBRARIAN_ACCEPTANCE_EXPECTED_FILES (491),
 * LIBRARIAN_ACCEPTANCE_EMBEDDING_IDENTITY (exact configured identity JSON string).
 * Run after import/indexing stops. Reads and hashes managed copies, never NAS originals.
 * Exit 0: bounded accounting checks pass (capability gaps can remain).
 * Exit 2: report complete, accounting/storage/import checks fail. Exit 1: runner error.
 * No exit code certifies complete extraction, model quality, or full product acceptance.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { constants, createReadStream } from 'node:fs';
import { lstat, mkdir, readFile, realpath, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { performance } from 'node:perf_hooks';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'node:url';

const DRIVER = fileURLToPath(import.meta.url);
// readConfig supplies no override; match createRetrieval's configured default.
const MAX_DIMENSIONS = 8192;
const TERMINAL = new Set(['completed', 'duplicate', 'unsupported', 'auxiliary', 'failed']);
const hash = value => createHash('sha256').update(value).digest('hex');
const short = value => String(value ?? '').slice(0, 1000);
const within = (parent, child) => {
  const part = relative(parent, child);
  return part === '' || (part !== '..' && !part.startsWith(`..${sep}`) && !isAbsolute(part));
};
const parse = (value, fallback = {}) => { try { return JSON.parse(value); } catch { return fallback; } };
const sum = (rows, key) => rows.reduce((total, row) => total + (row[key] || 0), 0);
const distinct = values => new Set(values.filter(value => value != null)).size;
const counts = (rows, field) => Object.fromEntries([...new Set(rows.map(row => row[field]))].sort()
  .map(value => [value, rows.filter(row => row[field] === value).length]));

async function outsideGit(path) {
  for (let dir = path; ; dir = dirname(dir)) {
    try { await lstat(join(dir, '.git')); throw new Error('Report must be outside every Git checkout'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (dirname(dir) === dir) return;
  }
}

async function canonicalFuture(path) {
  try { return await realpath(path); }
  catch (error) {
    if (error.code !== 'ENOENT') throw error;
    return join(await canonicalFuture(dirname(path)), path.slice(dirname(path).length + 1));
  }
}

async function outputPath(value, dataDir) {
  assert.ok(value && isAbsolute(value), 'LIBRARIAN_ACCEPTANCE_OUTPUT must be an absolute new JSON path');
  const path = await canonicalFuture(resolve(value));
  await outsideGit(path);
  assert.ok(!within(resolve(dirname(DRIVER), '..'), path), 'Report must be outside app source');
  assert.ok(!within(dataDir, path), 'Acceptance output must be outside managed data');
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, '', { flag: 'wx', mode: 0o600 });
  return path;
}

function manifestRows(document, expected) {
  const rows = Array.isArray(document) ? document : (document.results ?? document.files);
  assert.ok(Array.isArray(rows) && rows.length === expected, `Manifest must contain exactly ${expected} source entries`);
  assert.ok(document.complete !== false && !(document.blocked_count > 0) && !document.unexpected?.length, 'Transfer manifest is not complete');
  assert.equal(document.source_count ?? rows.length, rows.length);
  const seen = new Set();
  const result = rows.map(row => {
    const path = row.relative_path ?? row.relativePath ?? row.path;
    const bytes = row.source_bytes ?? row.size ?? row.bytes;
    assert.ok(typeof path === 'string' && path && !path.includes('\\') && !path.includes('\0')
      && !path.startsWith('/') && !path.split('/').some(part => !part || part === '.' || part === '..'), 'Unsafe manifest path');
    assert.ok(!seen.has(path), 'Manifest repeats a source path'); seen.add(path);
    assert.ok(Number.isSafeInteger(bytes) && bytes >= 0 && /^[a-f0-9]{64}$/.test(row.sha256 ?? ''), 'Invalid manifest size/hash');
    assert.ok(!row.status || row.status === 'verified', 'Manifest entry was not verified');
    assert.equal(row.destination_bytes ?? bytes, bytes);
    return { relative_path: path, bytes, sha256: row.sha256, format: extname(path).slice(1).toLowerCase() || 'unknown' };
  });
  assert.ok(Number.isSafeInteger(sum(result, 'bytes')), 'Manifest byte total is unsafe');
  assert.equal(document.source_bytes ?? sum(result, 'bytes'), sum(result, 'bytes'));
  return result;
}

function mountFor(path, raw) {
  const unescape = value => value.replace(/\\([0-7]{3})/g, (_, octal) => String.fromCharCode(parseInt(octal, 8)));
  const mounts = raw.trim().split('\n').map(line => {
    const [before, after] = line.split(' - ');
    const fields = before.split(' '); const trailing = after?.split(' ') || [];
    return { mount_point: unescape(fields[4] || ''), filesystem: trailing[0] || 'unknown' };
  }).filter(row => row.mount_point && within(row.mount_point, path)).sort((a, b) => b.mount_point.length - a.mount_point.length);
  const selected = mounts[0] || { mount_point: null, filesystem: 'unknown' };
  return { ...selected, recognized_network_filesystem: /^(nfs4?|cifs|smb3?|smbfs|9p|fuse\.(sshfs|rclone|s3fs))$/i.test(selected.filesystem) };
}

async function managedFile(path, root, expectedHash, expectedBytes) {
  try {
    assert.ok(isAbsolute(path) && within(root, resolve(path)), 'Managed path is outside its required directory');
    const before = await lstat(path);
    assert.ok(before.isFile() && !before.isSymbolicLink(), 'Managed path is not an ordinary file');
    const canonical = await realpath(path);
    assert.ok(within(root, canonical), 'Managed path resolves outside its required directory');
    const digest = createHash('sha256'); let bytes = 0;
    for await (const part of createReadStream(canonical, { flags: constants.O_RDONLY | constants.O_NOFOLLOW })) { digest.update(part); bytes += part.length; }
    const after = await lstat(path);
    const sha256 = digest.digest('hex');
    const unchanged = before.dev === after.dev && before.ino === after.ino && before.size === after.size && before.mtimeMs === after.mtimeMs;
    return { path: canonical, bytes, sha256, unchanged_during_read: unchanged,
      verified: unchanged && bytes === expectedBytes && sha256 === expectedHash };
  } catch (error) { return { path, verified: false, error: short(error.message) }; }
}

function vectorValid(row) {
  if (typeof row.embedding_model !== 'string' || !row.embedding_model || typeof row.embedding_json !== 'string'
      || row.embedding_json.length > 512 * 1024) return false;
  const values = parse(row.embedding_json, null);
  if (!Array.isArray(values) || !values.length || values.length > MAX_DIMENSIONS || values.length !== row.embedding_dimension
      || values.some(value => typeof value !== 'number' || !Number.isFinite(value))) return false;
  const norm = Math.sqrt(values.reduce((total, value) => total + value * value, 0));
  return Number.isFinite(norm) && norm > 0 && Number.isFinite(row.embedding_norm)
    && Math.abs(row.embedding_norm - norm) <= Math.max(1e-8, norm * 1e-6);
}

async function main() {
  assert.equal(process.platform, 'linux', 'Run acceptance on Spark, never native macOS');
  assert.equal(process.arch, 'arm64', 'Use the qualified Spark ARM64 Node runtime');
  const started = performance.now();
  const dataInput = process.env.LIBRARIAN_DATA_DIR;
  assert.ok(dataInput && isAbsolute(dataInput), 'LIBRARIAN_DATA_DIR must be an absolute existing managed data directory');
  const dataDir = await realpath(dataInput);
  const output = await outputPath(process.env.LIBRARIAN_ACCEPTANCE_OUTPUT, dataDir);
  const report = { schema_version: 1, started_at: new Date().toISOString(), completed: false, checks_passed: false,
    full_collection_acceptance: false, runner_sha256: hash(await readFile(DRIVER)),
    environment: { node: process.version, platform: process.platform, arch: process.arch },
    scope: 'Read-only manifest/job/managed-source/text/vector accounting; no import, extraction, model calls or source-original reads',
    checks: {}, ledger: [], issues: [], limitations: [
      'A terminal receipt is a disposition, not proof that readable text exists.',
      'Unique hashes, catalog books, text-bearing books, chunks and compatible vectors are separate denominators.',
      'Vector validity checks shape/dimension/norm and exact identity; it does not verify model identity externally or semantic quality.',
      'Managed path/hash checks do not simulate removal of source roots or test browser navigation.',
      'Mount inspection excludes recognized network filesystems in this process namespace; physical host storage remains coordinator provenance.',
      'Archive member inventories do not mean archive payloads were imported. MOBI/PRC conversion and OCR are not performed here.',
      'Extracted text and nonempty pages do not prove complete or faithful extraction. No full-collection indexing claim is made.',
      'Durations reconstructed from receipt timestamps include waiting/retries; they are not benchmark or throughput measurements.',
    ] };
  let db;
  try {
    const manifestInput = process.env.LIBRARIAN_ACCEPTANCE_MANIFEST;
    const jobId = process.env.LIBRARIAN_ACCEPTANCE_JOB_ID;
    assert.ok(dataInput && isAbsolute(dataInput) && manifestInput && isAbsolute(manifestInput) && jobId, 'Data directory, absolute transfer manifest and exact import job ID are required');
    const expected = Number(process.env.LIBRARIAN_ACCEPTANCE_EXPECTED_FILES || 491);
    assert.ok(Number.isSafeInteger(expected) && expected > 0 && expected <= 100000, 'Invalid expected source count');
    const sourceDir = join(dataDir, 'sources');
    const dbPath = join(dataDir, 'library.sqlite');
    assert.ok(!within(dataDir, output), 'Acceptance output must be outside managed data');
    await outsideGit(dataDir);
    const manifestInfo = await lstat(manifestInput);
    assert.ok(manifestInfo.isFile() && !manifestInfo.isSymbolicLink() && manifestInfo.size <= 16 * 1024 * 1024, 'Invalid manifest file');
    const manifestRaw = await readFile(manifestInput);
    const expectedRows = manifestRows(JSON.parse(manifestRaw), expected);
    report.manifest = { path: manifestInput, sha256: hash(manifestRaw), bytes: manifestRaw.length, expected_files: expected,
      source_bytes: sum(expectedRows, 'bytes'), unique_hashes: distinct(expectedRows.map(row => row.sha256)), formats: counts(expectedRows, 'format') };
    const targetIdentity = process.env.LIBRARIAN_ACCEPTANCE_EMBEDDING_IDENTITY || null;
    if (targetIdentity) {
      const identity = JSON.parse(targetIdentity);
      assert.ok(identity && typeof identity === 'object' && typeof identity.model === 'string' && identity.model, 'Embedding identity must be the exact configured JSON identity with a model');
    }
    const dbInfo = await lstat(dbPath);
    assert.ok(dbInfo.isFile() && !dbInfo.isSymbolicLink() && within(dataDir, await realpath(dbPath)), 'Database must be an ordinary managed file');
    const mountRaw = await readFile('/proc/self/mountinfo', 'utf8');
    report.storage = { data_directory: dataDir, source_directory: sourceDir, mount: mountFor(dataDir, mountRaw),
      source_originals_read: false, managed_files: [], covers: [], footprint: {} };
    db = new DatabaseSync(dbPath, { readOnly: true, timeout: 5000 });
    db.exec('PRAGMA query_only=ON');
    assert.equal(db.prepare('PRAGMA application_id').get().application_id, 0x4c42524a, 'Not a Librarian JS database');
    assert.equal(db.prepare('PRAGMA user_version').get().user_version, 1, 'Unsupported schema');
    report.environment.sqlite = db.prepare('SELECT sqlite_version() AS version').get().version;
    const versionBefore = db.prepare('PRAGMA data_version').get().data_version;
    db.exec('BEGIN');
    const job = db.prepare('SELECT * FROM import_jobs WHERE id=?').get(jobId);
    assert.ok(job, 'Exact import job not found');
    const summary = parse(job.summary_json);
    const identityFields = ['expected_sha256', 'expected_size', 'expected_mtime_ms', 'inventory_error', 'retryable'];
    const jobColumns = new Set(db.prepare('PRAGMA table_info(import_job_files)').all().map(row => row.name));
    const jobIdentityAvailable = identityFields.every(name => jobColumns.has(name));
    const identitySQL = identityFields.map(name => `${jobColumns.has(name) ? `jf.${name}` : 'NULL'} AS job_${name}`).join(', ');
    const files = db.prepare(`SELECT f.*, jf.status AS job_status, jf.error AS job_error, jf.ordinal AS job_ordinal, ${identitySQL}
      FROM import_job_files jf JOIN files f ON f.id=jf.file_id WHERE jf.job_id=? ORDER BY jf.ordinal`).all(jobId);
    const manifestByPath = new Map(expectedRows.map(row => [row.relative_path, row]));
    const identityMatches = file => {
      const expectedRow = manifestByPath.get(file.relative_path);
      return Boolean(jobIdentityAvailable && !summary.identityMigrationNotice && expectedRow && !file.job_inventory_error
        && file.job_retryable === 1 && file.staged === 1 && file.job_expected_sha256 === expectedRow.sha256
        && file.job_expected_size === expectedRow.bytes && file.sha256 === expectedRow.sha256 && file.size === expectedRow.bytes);
    };
    const selectedBooks = new Set(files.filter(identityMatches).map(row => row.book_id).filter(Boolean));
    const books = db.prepare(`SELECT b.id,b.cover_path,b.metadata_json,
      (SELECT count(*) FROM sections s WHERE s.book_id=b.id) AS sections,
      (SELECT count(*) FROM chunks c WHERE c.book_id=b.id) AS chunks,
      (SELECT count(DISTINCT c.section_id) FROM chunks c WHERE c.book_id=b.id) AS sections_with_chunks
      FROM books b ORDER BY b.id`).all();
    const bookMap = new Map(books.map(row => [row.id, row]));
    const selected = books.filter(row => selectedBooks.has(row.id));
    const receipts = db.prepare(`SELECT r.id,r.job_id,r.file_id,r.attempt,r.status,r.error,r.started_at,r.finished_at
      FROM import_receipts r JOIN import_job_files jf ON jf.file_id=r.file_id
      WHERE jf.job_id=? ORDER BY r.id LIMIT 100001`).all(jobId);
    assert.ok(receipts.length <= 100000, 'Receipt history exceeds report bound');
    const currentReceipts = receipts.filter(row => row.job_id === jobId);
    const receiptMap = new Map();
    for (const row of receipts) { const list = receiptMap.get(row.file_id) || []; list.push(row); receiptMap.set(row.file_id, list); }
    const chunkCount = db.prepare('SELECT count(*) AS n FROM chunks').get().n;
    assert.ok(chunkCount <= 1000000, 'Vector validation exceeds the one-million-chunk bound');
    const identities = new Map(); const validByBook = new Map();
    const failuresAvailable = Boolean(db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='embedding_failures'").get());
    const metadataPendingSQL = `(c.embedding_json IS NULL OR c.embedding_model IS NULL OR c.embedding_model != ?
      OR c.embedding_dimension IS NULL OR c.embedding_dimension <= 0 OR c.embedding_norm IS NULL OR c.embedding_norm <= 0)`;
    const runtimePartition = targetIdentity ? { total_chunks: 0, embedded_metadata_chunks: 0, failed_chunks: 0, pending_chunks: 0 } : null;
    const targetFailureSQL = targetIdentity && failuresAvailable ? `EXISTS (SELECT 1 FROM embedding_failures ef
      WHERE ef.chunk_id=c.id AND ef.model_identity=? AND ef.source_text=c.text)` : '0';
    const vectorArgs = targetIdentity ? [targetIdentity, ...(failuresAvailable ? [targetIdentity] : [])] : [];
    const vectorRows = db.prepare(`SELECT c.book_id,c.embedding_json,c.embedding_model,c.embedding_dimension,c.embedding_norm,
      ${targetIdentity ? metadataPendingSQL : 'NULL'} AS target_pending_metadata, ${targetFailureSQL} AS current_target_failure FROM chunks c`);
    let noEmbedding = 0;
    for (const row of vectorRows.iterate(...vectorArgs)) {
      if (runtimePartition && selectedBooks.has(row.book_id)) {
        runtimePartition.total_chunks += 1;
        if (!row.target_pending_metadata) runtimePartition.embedded_metadata_chunks += 1;
        else runtimePartition[row.current_target_failure ? 'failed_chunks' : 'pending_chunks'] += 1;
      }
      if (row.embedding_json == null) { noEmbedding += 1; continue; }
      const key = row.embedding_model ?? '[missing identity]';
      const group = identities.get(key) || { identity: row.embedding_model, stored_chunks: 0, valid_chunks: 0, invalid_chunks: 0,
        selected_stored_chunks: 0, selected_valid_chunks: 0, dimensions: {} };
      group.stored_chunks += 1;
      if (selectedBooks.has(row.book_id)) group.selected_stored_chunks += 1;
      const valid = vectorValid(row);
      group[valid ? 'valid_chunks' : 'invalid_chunks'] += 1;
      if (valid) {
        group.dimensions[row.embedding_dimension] = (group.dimensions[row.embedding_dimension] || 0) + 1;
        if (selectedBooks.has(row.book_id)) group.selected_valid_chunks += 1;
        if (key === targetIdentity) validByBook.set(row.book_id, (validByBook.get(row.book_id) || 0) + 1);
      }
      identities.set(key, group);
    }
    const activeImport = db.prepare('SELECT job_id FROM import_lock').all();
    report.job = { id: job.id, state: job.state, root: job.root, total: job.total, processed: job.processed,
      started_at: job.started_at, updated_at: job.updated_at, manifest: summary.manifest ?? null,
      mode: summary.mode || 'import', immutable_identity_columns_available: jobIdentityAvailable,
      identity_migration_notice: summary.identityMigrationNotice || null,
      inventory_complete: summary.inventoryComplete === true, statuses: counts(files, 'job_status'), active_imports: activeImport };
    report.counts = { job_files: files.length, staged_alias_rows: files.filter(row => row.staged === 1).length,
      unique_staged_hashes: distinct(files.filter(row => row.staged === 1).map(row => row.sha256)),
      retained_referenced_book_ids: distinct(files.map(row => row.book_id)),
      identity_matched_source_rows: files.filter(identityMatches).length,
      identity_matched_unique_hashes: distinct(files.filter(identityMatches).map(row => row.sha256)),
      selected_book_scope: 'Only retained files matching both immutable per-job identity and this exact manifest; conflicting retained books excluded',
      cataloged_books: selected.length, books_with_chunks: selected.filter(row => row.chunks > 0).length,
      sections: sum(selected, 'sections'), sections_with_chunks: sum(selected, 'sections_with_chunks'), chunks: sum(selected, 'chunks'),
      entire_store: { files: db.prepare('SELECT count(*) AS n FROM files').get().n, books: books.length, chunks: chunkCount } };
    report.vectors = { configured_identity_supplied: targetIdentity !== null, target_identity: targetIdentity,
      max_dimensions: MAX_DIMENSIONS, dimension_cap_basis: 'Application config supplies no override to the retrieval default of 8192',
      identities: [...identities.values()], entire_store_chunks_without_vector: noEmbedding,
      failure_table_available: failuresAvailable, selected_runtime_metadata_partition: runtimePartition,
      partition_basis: 'Matches retrieval metadata counts; active failures require exact model identity, current source text and pending metadata. Parsed vector validity is checked separately.',
      selected_chunks: report.counts.chunks, selected_valid_target_chunks: targetIdentity ? sum(selected.map(row => ({ n: validByBook.get(row.id) || 0 })), 'n') : null,
      target_coverage_complete: null, semantic_quality_verified: false };
    if (targetIdentity) report.vectors.target_coverage_complete = report.counts.chunks > 0 && report.vectors.selected_valid_target_chunks === report.counts.chunks
      && Object.keys(identities.get(targetIdentity)?.dimensions || {}).length === 1;
    report.vectors.active_selected_failures = [];
    report.vectors.active_selected_failure_count = targetIdentity ? 0 : null;
    report.vectors.failure_examples_limit = 100;
    if (targetIdentity && failuresAvailable) {
      const failures = db.prepare(`SELECT ef.chunk_id,c.book_id,ef.code,ef.message,ef.failed_at
        FROM embedding_failures ef JOIN chunks c ON c.id=ef.chunk_id
        WHERE ef.model_identity=? AND ef.source_text=c.text AND ${metadataPendingSQL} ORDER BY ef.chunk_id`);
      for (const failure of failures.iterate(targetIdentity, targetIdentity)) {
        if (selectedBooks.has(failure.book_id)) {
          report.vectors.active_selected_failure_count += 1;
          if (report.vectors.active_selected_failures.length < 100) report.vectors.active_selected_failures.push({ ...failure, message: short(failure.message) });
        }
      }
    }
    if (runtimePartition) assert.equal(report.vectors.active_selected_failure_count, runtimePartition.failed_chunks, 'Failure ledger counts disagree');
    db.exec('COMMIT'); // Release the read snapshot before streaming managed source bytes.
    const byPath = new Map();
    for (const file of files) { const rows = byPath.get(file.relative_path) || []; rows.push(file); byPath.set(file.relative_path, rows); }
    const copies = new Map();
    for (const expectedRow of expectedRows) {
      const matches = byPath.get(expectedRow.relative_path) || [];
      const file = matches.length === 1 ? matches[0] : null;
      const row = { ...expectedRow, file_id: file?.id ?? null, issues: [] };
      if (!file) { row.issues.push(matches.length ? 'Ambiguous manifest/job path' : 'Manifest source absent from selected job'); report.ledger.push(row); continue; }
      const book = bookMap.get(file.book_id);
      const metadata = parse(file.metadata_json);
      const history = receiptMap.get(file.id) || [];
      const currentHistory = history.filter(receipt => receipt.job_id === jobId);
      const identityMatched = identityMatches(file);
      Object.assign(row, { file_status: file.status, job_status: file.job_status, error: short(file.job_error || file.error) || null,
        staged: file.staged === 1, identity_matches_manifest: identityMatched, book_id: identityMatched ? file.book_id : null, duplicate_of: file.duplicate_of,
        job_identity: { sha256: file.job_expected_sha256, size: file.job_expected_size, mtime_ms: file.job_expected_mtime_ms,
          inventory_error: file.job_inventory_error, retryable: file.job_retryable },
        retained_file_identity: { sha256: file.sha256, size: file.size, book_id: file.book_id },
        cataloged: identityMatched && Boolean(book), sections: identityMatched ? (book?.sections || 0) : 0, chunks: identityMatched ? (book?.chunks || 0) : 0,
        sections_with_chunks: identityMatched ? (book?.sections_with_chunks || 0) : 0, readable_text: identityMatched && Boolean(book?.chunks > 0),
        target_valid_vectors: targetIdentity && identityMatched && book ? (validByBook.get(book.id) || 0) : null,
        receipt_count_all_jobs: history.length, receipt_count_selected_job: currentHistory.length,
        latest_receipt: history.at(-1) || null, latest_selected_job_receipt: currentHistory.at(-1) || null,
        retained_disposition_reason: short(metadata.reason || metadata.duplicateReason) || null,
        warnings: Array.isArray(metadata.extraction?.warnings) ? metadata.extraction.warnings.map(short) : [],
        archive: metadata.archive ? { members: metadata.archive.entries?.length ?? null,
          ebook_candidates_by_extension: (metadata.archive.entries || []).filter(entry => /\.(epub|pdf|mobi|prc)$/i.test(entry.name || '')).length,
          total_uncompressed_bytes: metadata.archive.totalUncompressedBytes ?? null, contents_verified: metadata.archive.contentsVerified === true,
          members_imported_by_this_pipeline: false } : null });
      if (file.source_path !== resolve(job.root, expectedRow.relative_path)) row.issues.push('File provenance path does not match selected job root');
      if (!jobIdentityAvailable) row.issues.push('Legacy schema lacks immutable per-job identity; global file fields cannot attest this job');
      if (summary.identityMigrationNotice) row.issues.push('Historical inventory identity was not retained; migration cannot attest this job');
      if (file.job_inventory_error || file.job_retryable === 0) row.issues.push('Import inventory conflict is nonattestable');
      if (file.job_expected_sha256 !== expectedRow.sha256 || file.job_expected_size !== expectedRow.bytes || file.size !== expectedRow.bytes || file.sha256 !== expectedRow.sha256) row.issues.push('Manifest/per-job/retained hash or size mismatch');
      if (file.format !== expectedRow.format) row.issues.push('Manifest/store format mismatch');
      if (!TERMINAL.has(file.job_status)) row.issues.push('Selected job file is not terminal');
      if (file.job_status === 'failed') row.issues.push('Selected job file failed');
      if (TERMINAL.has(file.job_status) && !history.some(receipt => receipt.status === file.job_status && receipt.finished_at)) row.issues.push('No finished retained receipt supports the selected disposition');
      if (!row.staged) row.issues.push('No managed source was staged');
      else {
        const copyKey = JSON.stringify([file.path, expectedRow.sha256, expectedRow.bytes]);
        if (!copies.has(copyKey)) copies.set(copyKey, await managedFile(file.path, sourceDir, expectedRow.sha256, expectedRow.bytes));
        const copy = copies.get(copyKey);
        row.managed_path = copy.path; row.managed_bytes_verified = copy.verified;
        if (!copy.verified) row.issues.push('Managed source verification failed');
      }
      if (file.job_status === 'completed' && !book) row.issues.push('Completed source has no catalog book');
      if (file.book_id && !book) row.issues.push('Source refers to an absent catalog book');
      report.ledger.push(row);
    }
    report.storage.managed_files = [...copies.values()];
    for (const book of selected) {
      if (!book.cover_path) continue;
      const row = { book_id: book.id, path: book.cover_path, verified: false };
      try {
        const info = await lstat(book.cover_path); const canonical = await realpath(book.cover_path);
        row.verified = info.isFile() && !info.isSymbolicLink() && within(join(dataDir, 'covers'), canonical);
        row.bytes = info.size;
      } catch (error) { row.error = short(error.message); }
      report.storage.covers.push(row);
    }
    for (const suffix of ['', '-wal', '-shm']) {
      try { report.storage.footprint[`library.sqlite${suffix}`] = (await stat(`${dbPath}${suffix}`)).size; }
      catch (error) { if (error.code !== 'ENOENT') throw error; report.storage.footprint[`library.sqlite${suffix}`] = 0; }
    }
    report.storage.footprint.managed_source_bytes_distinct_paths = sum([...new Map(report.storage.managed_files.map(row => [row.path, row])).values()], 'bytes');
    report.storage.footprint.cover_bytes_selected_books = sum(report.storage.covers, 'bytes');
    report.storage.footprint.scope = 'Database/WAL/SHM observed sizes and selected-job referenced source/cover files; not recursive disk usage';
    report.formats = Object.fromEntries([...new Set(report.ledger.map(row => row.format))].sort().map(format => {
      const rows = report.ledger.filter(row => row.format === format);
      return [format, { sources: rows.length, bytes: sum(rows, 'bytes'), statuses: counts(rows, 'job_status'),
        unique_hashes: distinct(rows.map(row => row.sha256)), cataloged_unique_books: distinct(rows.filter(row => row.cataloged).map(row => row.book_id)),
        readable_unique_books: distinct(rows.filter(row => row.readable_text).map(row => row.book_id)) }];
    }));
    const elapsed = Date.parse(job.updated_at) - Date.parse(job.started_at);
    report.timing = { job_observed_elapsed_ms: Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : null,
      completed_receipts_in_selected_job: currentReceipts.filter(row => row.finished_at).length,
      sum_selected_receipt_elapsed_ms: currentReceipts.reduce((total, row) => {
        const ms = Date.parse(row.finished_at) - Date.parse(row.started_at); return total + (Number.isFinite(ms) && ms >= 0 ? ms : 0);
      }, 0), benchmark: false };
    report.gaps = { failed_sources: report.ledger.filter(row => row.job_status === 'failed').map(row => row.relative_path),
      unparsed_mobi_prc_sources: report.ledger.filter(row => ['mobi', 'prc'].includes(row.format) && !row.readable_text).map(row => row.relative_path),
      archive_sources_not_imported: report.ledger.filter(row => row.format === 'zip').map(row => row.relative_path),
      cataloged_books_without_chunks: selected.filter(row => !row.chunks).map(row => row.id),
      pdf_books_with_textless_pages: selected.flatMap(row => {
        const metadata = parse(row.metadata_json);
        return Array.isArray(metadata.pagesWithoutText) && metadata.pagesWithoutText.length ? [{ book_id: row.id, physical_pages: metadata.pagesWithoutText, reason: 'May be blank or image-only; OCR not run by this pipeline' }] : [];
      }), extraction_completeness_verified: false, full_collection_indexed: false };
    const versionAfter = db.prepare('PRAGMA data_version').get().data_version;
    report.snapshot = { database_read_only: true, data_version_before: versionBefore, data_version_after: versionAfter,
      unchanged_during_verification: versionBefore === versionAfter, source_rows_read_in_one_transaction: true };
    report.checks = { exact_manifest_job_count: files.length === expectedRows.length && job.total === expectedRows.length,
      immutable_per_job_identity_available: jobIdentityAvailable && !summary.identityMigrationNotice,
      exact_manifest_bound_to_job: summary.manifest?.sha256 === report.manifest.sha256 && summary.manifest?.expectedCount === expectedRows.length,
      inventory_complete: summary.inventoryComplete === true,
      every_manifest_row_verified: report.ledger.every(row => !row.issues.length),
      no_unlisted_job_rows: files.every(file => expectedRows.some(row => row.relative_path === file.relative_path)),
      import_finished_without_errors: job.state === 'completed' && files.every(file => TERMINAL.has(file.job_status) && file.job_status !== 'failed') && job.processed === files.length,
      no_active_import: activeImport.length === 0,
      no_unfinished_selected_job_receipts: currentReceipts.every(row => row.finished_at != null),
      managed_paths_not_on_recognized_network_mount: report.storage.mount.filesystem !== 'unknown' && !report.storage.mount.recognized_network_filesystem
        && report.storage.managed_files.every(row => { const mount = mountFor(row.path, mountRaw); return mount.filesystem !== 'unknown' && !mount.recognized_network_filesystem; }),
      referenced_covers_contained: report.storage.covers.every(row => row.verified),
      database_unchanged_during_verification: versionBefore === versionAfter };
    report.issues = Object.entries(report.checks).filter(([, passed]) => !passed).map(([name]) => name);
    report.checks_passed = report.issues.length === 0;
    report.completed = true;
    process.exitCode = report.checks_passed ? 0 : 2;
  } catch (error) {
    report.error = { name: error.name, message: short(error.message) }; process.exitCode = 1;
  } finally {
    if (db) { try { db.close(); report.database_closed = true; } catch (error) { report.cleanup_error = short(error.message); report.checks_passed = false; process.exitCode = 1; } }
    report.finished_at = new Date().toISOString(); report.runner_elapsed_ms = Math.round(performance.now() - started);
    const raw = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
    assert.ok(raw.length <= 64 * 1024 * 1024, 'Acceptance report exceeds size bound');
    await writeFile(`${output}.tmp`, raw, { flag: 'wx', mode: 0o600 }); await rename(`${output}.tmp`, output);
  }
  console.log(JSON.stringify({ report: output, completed: report.completed, checks_passed: report.checks_passed,
    full_collection_acceptance: false, issues: report.issues, error: report.error || null }));
}

main().catch(error => { console.error(`${error.name}: ${short(error.message)}`); process.exitCode = 1; });
