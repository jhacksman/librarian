import { createHash, randomUUID } from 'node:crypto';
import { mkdirSync, realpathSync, existsSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'node:url';
import { ensureBookAssets, saveBookAssets, decodeThumbnail } from './book-assets.mjs';

const APPLICATION_ID = 0x4c42524a;
const SCHEMA_VERSION = 1;
export const PIPELINE_VERSION = 'node-text-v2-words400-overlap80-chars4000-utf16';
const SOURCE_TREE = fileURLToPath(new URL('../../', import.meta.url));
export const FILE_STATUSES = Object.freeze(['queued', 'running', 'completed', 'duplicate', 'unsupported', 'auxiliary', 'failed']);
export const TERMINAL_STATUSES = Object.freeze(['completed', 'duplicate', 'unsupported', 'auxiliary', 'failed']);
export const now = () => new Date().toISOString();
export const digest = (value) => createHash('sha256').update(value).digest('hex');
export const parseJSON = (value, fallback = {}) => {
  try { return JSON.parse(value); } catch { return fallback; }
};

export function isWithin(parent, child) {
  const path = relative(resolve(parent), resolve(child));
  return path === '' || (!isAbsolute(path) && path !== '..' && !path.startsWith(`..${sep}`));
}

// Resolve existing ancestors too: a symlink must not move derived files into the checkout.
export function canonicalDestination(path) {
  const absolute = resolve(path);
  if (existsSync(absolute)) return realpathSync(absolute);
  return resolve(canonicalDestination(dirname(absolute)), relative(dirname(absolute), absolute));
}

export function assertExternalState(path) {
  const target = canonicalDestination(path);
  if (isWithin(realpathSync(SOURCE_TREE), target)) throw new Error('Library data must be outside the application checkout');
  return target;
}

export function* chunkText(text, { words = 400, overlap = 80, maxChars = 4000 } = {}) {
  if (typeof text !== 'string' || !Number.isSafeInteger(words) || !Number.isSafeInteger(overlap)
      || !Number.isSafeInteger(maxChars) || maxChars < 2
      || words < 1 || overlap < 0 || overlap >= words) throw new TypeError('Invalid chunk input or window');
  let window = [];
  let lastEnd = -1;
  for (const match of text.matchAll(/\S+/gu)) {
    const tokenEnd = match.index + match[0].length;
    for (let tokenStart = match.index; tokenStart < tokenEnd;) {
      let end = Math.min(tokenEnd, tokenStart + maxChars);
      // JavaScript offsets remain exact UTF-16 units, but never cut a valid astral pair.
      if (end < tokenEnd && text.charCodeAt(end - 1) >= 0xd800 && text.charCodeAt(end - 1) <= 0xdbff
          && text.charCodeAt(end) >= 0xdc00 && text.charCodeAt(end) <= 0xdfff) end--;
      if (window.length && (window.length >= words || end - window[0][0] > maxChars)) {
        const start = window[0][0], previousEnd = window.at(-1)[1];
        if (previousEnd > lastEnd) {
          lastEnd = previousEnd;
          yield { text: text.slice(start, lastEnd), charStart: start, charEnd: lastEnd };
        }
        window = overlap ? window.slice(-overlap) : [];
        while (window.length && (window.length >= words || end - window[0][0] > maxChars)) window.shift();
      }
      window.push([tokenStart, end]);
      tokenStart = end;
    }
  }
  if (window.length && window.at(-1)[1] !== lastEnd) {
    const start = window[0][0];
    const end = window.at(-1)[1];
    yield { text: text.slice(start, end), charStart: start, charEnd: end };
  }
}

const SCHEMA = `
CREATE TABLE books (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, authors_json TEXT NOT NULL DEFAULT '[]',
  description TEXT NOT NULL DEFAULT '', language TEXT NOT NULL DEFAULT '', publisher TEXT NOT NULL DEFAULT '',
  cover_path TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE files (
  id TEXT PRIMARY KEY, path TEXT NOT NULL, source_path TEXT NOT NULL UNIQUE, relative_path TEXT NOT NULL,
  sha256 TEXT, size INTEGER NOT NULL CHECK(size>=0), format TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued','running','completed','duplicate','unsupported','auxiliary','failed')),
  error TEXT, book_id TEXT REFERENCES books(id) ON DELETE SET NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL, mtime_ms REAL, expected_sha256 TEXT, expected_size INTEGER,
  group_key TEXT NOT NULL, duplicate_of TEXT REFERENCES files(id) ON DELETE SET NULL,
  staged INTEGER NOT NULL DEFAULT 0 CHECK(staged IN (0,1)), attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX files_sha ON files(sha256);
CREATE INDEX files_status ON files(status);
CREATE INDEX files_book ON files(book_id);
CREATE INDEX files_group ON files(group_key);
CREATE TABLE sections (
  id TEXT PRIMARY KEY, book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, locator_json TEXT NOT NULL,
  UNIQUE(book_id,ordinal)
);
CREATE TABLE chunks (
  id TEXT PRIMARY KEY, book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
  section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL,
  text TEXT NOT NULL, locator_json TEXT NOT NULL, embedding_json TEXT,
  embedding_model TEXT, embedding_dimension INTEGER, embedding_norm REAL, UNIQUE(section_id,ordinal)
);
CREATE INDEX chunks_book ON chunks(book_id);
CREATE INDEX chunks_section ON chunks(section_id,ordinal);
CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED,book_id UNINDEXED,text,tokenize='unicode61');
CREATE TRIGGER chunks_insert AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(chunk_id,book_id,text) VALUES(new.id,new.book_id,new.text);
END;
CREATE TRIGGER chunks_delete AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE chunk_id=old.id;
END;
CREATE TRIGGER chunks_text_update AFTER UPDATE OF id,book_id,text ON chunks
WHEN old.id<>new.id OR old.book_id<>new.book_id OR old.text<>new.text BEGIN
  DELETE FROM chunks_fts WHERE chunk_id=old.id;
  INSERT INTO chunks_fts(chunk_id,book_id,text) VALUES(new.id,new.book_id,new.text);
  UPDATE chunks SET embedding_json=NULL,embedding_model=NULL,embedding_dimension=NULL,embedding_norm=NULL WHERE id=new.id;
END;
CREATE TABLE reading_progress (
  book_id TEXT PRIMARY KEY REFERENCES books(id) ON DELETE CASCADE,
  section_id TEXT REFERENCES sections(id) ON DELETE SET NULL, updated_at TEXT NOT NULL
);
CREATE TABLE import_jobs (
  id TEXT PRIMARY KEY, root TEXT NOT NULL, state TEXT NOT NULL, total INTEGER NOT NULL DEFAULT 0,
  processed INTEGER NOT NULL DEFAULT 0, started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  summary_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE import_job_files (
  job_id TEXT NOT NULL REFERENCES import_jobs(id) ON DELETE CASCADE,
  file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, status TEXT NOT NULL, error TEXT,
  expected_sha256 TEXT, expected_size INTEGER, expected_mtime_ms REAL,
  inventory_error TEXT, retryable INTEGER NOT NULL DEFAULT 1 CHECK(retryable IN (0,1)),
  PRIMARY KEY(job_id,file_id), UNIQUE(job_id,ordinal)
);
CREATE TABLE import_receipts (
  id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES import_jobs(id),
  file_id TEXT NOT NULL REFERENCES files(id), attempt INTEGER NOT NULL,
  status TEXT NOT NULL, error TEXT, started_at TEXT NOT NULL, finished_at TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX receipts_file ON import_receipts(file_id,id);
CREATE TABLE import_lock (id INTEGER PRIMARY KEY CHECK(id=1),owner TEXT NOT NULL,pid INTEGER NOT NULL,process_identity TEXT,job_id TEXT NOT NULL,started_at TEXT NOT NULL);
`;

export class LibraryStore {
  constructor(dbPath, { dataDir = dirname(resolve(dbPath)) } = {}) {
    if (dbPath === ':memory:') throw new Error('The library requires a durable database path');
    this.path = assertExternalState(dbPath);
    this.dataDir = assertExternalState(dataDir);
    if (!isWithin(this.dataDir, this.path)) throw new Error('Database must be inside its data directory');
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
    mkdirSync(this.dataDir, { recursive: true, mode: 0o700 });
    this.db = new DatabaseSync(this.path, { timeout: 5000, enableForeignKeyConstraints: true });
    this.depth = 0;
    try {
      const tables = this.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").all();
      const application = this.db.prepare('PRAGMA application_id').get().application_id;
      const version = this.db.prepare('PRAGMA user_version').get().user_version;
      if (tables.length && (application !== APPLICATION_ID || version !== SCHEMA_VERSION)) {
        throw new Error('Refusing to modify an unrelated or unsupported database (including the Python index)');
      }
      if (!tables.length) this.transaction(() => {
        this.db.exec(SCHEMA);
        this.db.exec(`PRAGMA application_id=${APPLICATION_ID}; PRAGMA user_version=${SCHEMA_VERSION}`);
      });
      this.transaction(() => {
        const jobColumns = new Set(this.db.prepare('PRAGMA table_info(import_job_files)').all().map((row) => row.name));
        for (const [name, type] of [['expected_sha256', 'TEXT'], ['expected_size', 'INTEGER'], ['expected_mtime_ms', 'REAL'],
          ['inventory_error', 'TEXT'], ['retryable', 'INTEGER NOT NULL DEFAULT 1 CHECK(retryable IN (0,1))']]) {
          if (!jobColumns.has(name)) this.db.exec(`ALTER TABLE import_job_files ADD COLUMN ${name} ${type}`);
        }
        if (!jobColumns.has('expected_sha256')) {
          // Old inventory conflicts did not retain their own identity. Never retry or attest them using global bytes.
          this.db.exec(`UPDATE import_job_files SET expected_sha256=(SELECT coalesce(f.expected_sha256,f.sha256) FROM files f WHERE f.id=file_id),
            expected_size=(SELECT coalesce(f.expected_size,f.size) FROM files f WHERE f.id=file_id),
            expected_mtime_ms=(SELECT f.mtime_ms FROM files f WHERE f.id=file_id),
            inventory_error=(SELECT r.error FROM import_receipts r
              WHERE r.job_id=import_job_files.job_id AND r.file_id=import_job_files.file_id AND r.attempt=0 AND r.error IS NOT NULL ORDER BY r.id DESC LIMIT 1)`);
          this.db.exec("UPDATE import_job_files SET status='failed',error=inventory_error,retryable=0 WHERE inventory_error IS NOT NULL");
          for (const row of this.db.prepare('SELECT DISTINCT job_id FROM import_job_files WHERE inventory_error IS NOT NULL').all()) {
            this.refreshJob(row.job_id, { state: 'completed_with_errors', identityMigrationNotice: 'Historical inventory errors remain nonretryable; their original expected hashes were not retained by the older schema' });
          }
        }
        if (!this.db.prepare('PRAGMA table_info(import_lock)').all().some((row) => row.name === 'process_identity')) {
          this.db.exec('ALTER TABLE import_lock ADD COLUMN process_identity TEXT');
        }
        ensureBookAssets(this);
      });
      this.db.exec('PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL');
    } catch (error) {
      this.db.close();
      throw error;
    }
  }

  close() { this.db.close(); }

  transaction(fn) {
    const depth = this.depth;
    const savepoint = `library_transaction_${depth}`;
    this.db.exec(depth ? `SAVEPOINT ${savepoint}` : 'BEGIN IMMEDIATE');
    this.depth++;
    try {
      const result = fn();
      if (result && typeof result.then === 'function') throw new TypeError('SQLite transactions must be synchronous');
      this.db.exec(depth ? `RELEASE ${savepoint}` : 'COMMIT');
      return result;
    } catch (error) {
      this.db.exec(depth ? `ROLLBACK TO ${savepoint}; RELEASE ${savepoint}` : 'ROLLBACK');
      throw error;
    } finally { this.depth--; }
  }

  getFile(id) {
    const row = this.db.prepare('SELECT * FROM files WHERE id=?').get(id);
    return row ? { ...row, metadata: parseJSON(row.metadata_json) } : null;
  }

  getJob(id) {
    const row = this.db.prepare('SELECT * FROM import_jobs WHERE id=?').get(id);
    return row ? { ...row, summary: parseJSON(row.summary_json) } : null;
  }

  getJobFile(jobId, fileId) {
    return this.db.prepare('SELECT * FROM import_job_files WHERE job_id=? AND file_id=?').get(jobId, fileId) ?? null;
  }

  assertJobIdentity(jobId, file) {
    const expected = this.getJobFile(jobId, file.id);
    if (!expected) throw new Error('File does not belong to this import job');
    if (expected.inventory_error || expected.retryable === 0) throw new Error(expected.inventory_error || 'Inventory conflict cannot be retried');
    if ((expected.expected_size != null && expected.expected_size !== file.size)
        || (expected.expected_sha256 && file.sha256 && expected.expected_sha256 !== file.sha256)
        || (!file.staged && expected.expected_mtime_ms != null && expected.expected_mtime_ms !== file.mtime_ms)) {
      throw new Error('Managed file identity does not match this import job');
    }
    return expected;
  }

  listFiles({ jobId, status, limit = 100, offset = 0 } = {}) {
    if (!Number.isInteger(limit) || limit < 1 || limit > 1000 || !Number.isInteger(offset) || offset < 0) throw new RangeError('Invalid file page');
    if (status && !FILE_STATUSES.includes(status)) throw new RangeError('Invalid file status');
    const where = [], args = [];
    if (jobId) { where.push('jf.job_id=?'); args.push(jobId); }
    if (status) { where.push(`${jobId ? 'jf' : 'f'}.status=?`); args.push(status); }
    const sql = `SELECT f.*${jobId ? ',jf.status AS job_status,jf.error AS job_error,jf.inventory_error,jf.retryable,jf.expected_sha256 AS job_expected_sha256,jf.expected_size AS job_expected_size' : ''} FROM files f ${jobId ? 'JOIN import_job_files jf ON jf.file_id=f.id' : ''}
      ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY ${jobId ? 'jf.ordinal' : 'f.relative_path,f.id'} LIMIT ? OFFSET ?`;
    return this.db.prepare(sql).all(...args, limit, offset).map((row) => ({ ...row, metadata: parseJSON(row.metadata_json) }));
  }

  listBooks({ limit = 50, offset = 0 } = {}) {
    if (!Number.isInteger(limit) || limit < 1 || limit > 1000 || !Number.isInteger(offset) || offset < 0) throw new RangeError('Invalid book page');
    return this.db.prepare('SELECT * FROM books ORDER BY title,id LIMIT ? OFFSET ?').all(limit, offset)
      .map((row) => ({ ...row, authors: parseJSON(row.authors_json, []), metadata: parseJSON(row.metadata_json) }));
  }

  summary() {
    const counts = Object.fromEntries(FILE_STATUSES.map((status) => [status, 0]));
    for (const row of this.db.prepare('SELECT status,count(*) AS count FROM files GROUP BY status').all()) counts[row.status] = row.count;
    return { files: Object.values(counts).reduce((a, b) => a + b, 0), statuses: counts,
      books: this.db.prepare('SELECT count(*) AS count FROM books').get().count,
      sections: this.db.prepare('SELECT count(*) AS count FROM sections').get().count,
      chunks: this.db.prepare('SELECT count(*) AS count FROM chunks').get().count,
      stagedBytes: this.db.prepare('SELECT coalesce(sum(size),0) AS bytes FROM files WHERE staged=1').get().bytes,
      formats: Object.fromEntries(this.db.prepare('SELECT format,count(*) AS count FROM files GROUP BY format').all().map((row) => [row.format, row.count])) };
  }

  createJob(root, metadata = {}) {
    const id = randomUUID(), stamp = now();
    this.db.prepare('INSERT INTO import_jobs(id,root,state,total,processed,started_at,updated_at,summary_json) VALUES(?,?,?,0,0,?,?,?)')
      .run(id, root, 'queued', stamp, stamp, JSON.stringify(metadata));
    return this.getJob(id);
  }

  registerFile(jobId, entry, ordinal) {
    const id = digest(entry.sourcePath);
    const previous = this.getFile(id);
    let error = entry.error ?? null;
    if (previous && ((entry.expectedSha256 && (previous.sha256 || previous.expected_sha256) && entry.expectedSha256 !== (previous.sha256 || previous.expected_sha256))
        || entry.size !== previous.size || (entry.mtimeMs != null && previous.mtime_ms != null && entry.mtimeMs !== previous.mtime_ms))) {
      error = 'Immutable source inventory changed since its first import';
    }
    const status = error ? 'failed' : (previous?.status ?? 'queued');
    this.transaction(() => {
      if (!previous) this.db.prepare(`INSERT INTO files(id,path,source_path,relative_path,sha256,size,format,status,error,metadata_json,updated_at,
        mtime_ms,expected_sha256,expected_size,group_key) VALUES(?,?,?,?,NULL,?,?,?,?,?,?,?,?,?,?)`)
        .run(id, entry.sourcePath, entry.sourcePath, entry.relativePath, entry.size, entry.format, status, error,
          JSON.stringify(entry.metadata ?? {}), now(), entry.mtimeMs ?? null, entry.expectedSha256 ?? null,
          entry.expectedSize ?? null, entry.groupKey);
      this.db.prepare(`INSERT INTO import_job_files(job_id,file_id,ordinal,status,error,expected_sha256,expected_size,expected_mtime_ms,inventory_error,retryable)
        VALUES(?,?,?,?,?,?,?,?,?,?)`)
        .run(jobId, id, ordinal, status, error, entry.expectedSha256 ?? previous?.sha256 ?? previous?.expected_sha256 ?? null,
          entry.expectedSize ?? entry.size, entry.mtimeMs ?? null, error, error ? 0 : 1);
      if (error) this.db.prepare('INSERT INTO import_receipts(job_id,file_id,attempt,status,error,started_at,finished_at,metadata_json) VALUES(?,?,0,?,?,?,?,?)')
        .run(jobId, id, 'failed', error, now(), now(), JSON.stringify({ phase: 'inventory', retryable: false,
          expectedIdentity: { sha256: entry.expectedSha256 ?? null, size: entry.expectedSize ?? entry.size, mtimeMs: entry.mtimeMs ?? null },
          retainedIdentity: previous ? { sha256: previous.sha256, size: previous.size, status: previous.status } : null }));
    });
    return this.getFile(id);
  }

  refreshJob(jobId, { state, ...extra } = {}) {
    const job = this.getJob(jobId);
    if (!job) throw new Error('Import job not found');
    const counts = Object.fromEntries(FILE_STATUSES.map((name) => [name, 0]));
    const formats = {};
    for (const row of this.db.prepare('SELECT status,count(*) AS count FROM import_job_files WHERE job_id=? GROUP BY status').all(jobId)) counts[row.status] = row.count;
    for (const row of this.db.prepare('SELECT f.format,count(*) AS count FROM files f JOIN import_job_files j ON j.file_id=f.id WHERE j.job_id=? GROUP BY f.format').all(jobId)) formats[row.format] = row.count;
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const processed = TERMINAL_STATUSES.reduce((count, status) => count + counts[status], 0);
    const summary = { ...job.summary, ...extra, statuses: counts, formats };
    this.db.prepare('UPDATE import_jobs SET state=?,total=?,processed=?,updated_at=?,summary_json=? WHERE id=?')
      .run(state ?? job.state, total, processed, now(), JSON.stringify(summary), jobId);
    return this.getJob(jobId);
  }

  beginFile(jobId, fileId) {
    return this.transaction(() => {
      const file = this.getFile(fileId);
      if (!file) throw new Error('Import file not found');
      this.assertJobIdentity(jobId, file);
      const attempt = file.attempts + 1, stamp = now();
      if (this.getJob(jobId)?.summary.mode === 'reindex') {
        this.db.prepare('UPDATE files SET attempts=?,updated_at=? WHERE id=?').run(attempt, stamp, fileId);
      } else this.db.prepare("UPDATE files SET status='running',error=NULL,attempts=?,updated_at=? WHERE id=?").run(attempt, stamp, fileId);
      this.db.prepare("UPDATE import_job_files SET status='running',error=NULL WHERE job_id=? AND file_id=?").run(jobId, fileId);
      const receipt = this.db.prepare("INSERT INTO import_receipts(job_id,file_id,attempt,status,started_at) VALUES(?,?,?,'running',?)")
        .run(jobId, fileId, attempt, stamp);
      return { file: this.getFile(fileId), receiptId: Number(receipt.lastInsertRowid) };
    });
  }

  finishFile(jobId, fileId, receiptId, { status, error = null, bookId = null, duplicateOf = null, metadata = {} }) {
    if (!TERMINAL_STATUSES.includes(status)) throw new Error('Invalid terminal file state');
    this.transaction(() => {
      const file = this.getFile(fileId);
      const details = { ...file.metadata };
      // A disposition reason belongs to this attempt, not a prior outcome.
      // Historical receipts retain the reason that applied when they finished.
      delete details.reason;
      Object.assign(details, metadata);
      if (!(status === 'failed' && this.getJob(jobId)?.summary.mode === 'reindex')) {
        this.db.prepare('UPDATE files SET status=?,error=?,book_id=?,duplicate_of=?,metadata_json=?,updated_at=? WHERE id=?')
          .run(status, error, bookId, duplicateOf, JSON.stringify(details), now(), fileId);
      }
      this.db.prepare('UPDATE import_job_files SET status=?,error=? WHERE job_id=? AND file_id=?').run(status, error, jobId, fileId);
      this.db.prepare('UPDATE import_receipts SET status=?,error=?,finished_at=?,metadata_json=? WHERE id=? AND job_id=? AND file_id=?')
        .run(status, error, now(), JSON.stringify(details), receiptId, jobId, fileId);
    });
  }

  commitBook(jobId, fileId, receiptId, book) {
    const file = this.getFile(fileId);
    if (!file?.sha256 || !file.staged) throw new Error('Cannot index an unverified source');
    this.assertJobIdentity(jobId, file);
    if (!book || typeof book.title !== 'string' || !Array.isArray(book.sections)) throw new TypeError('Invalid extracted book');
    const id = file.sha256;
    const authors = Array.isArray(book.authors) ? book.authors : [];
    if (authors.some((author) => typeof author !== 'string')) throw new TypeError('Invalid authors');
    const previous = this.db.prepare('SELECT * FROM books WHERE id=?').get(id);
    const previousMetadata = parseJSON(previous?.metadata_json);
    const preserved = Object.fromEntries(Object.entries(previousMetadata).filter(([key]) => ['tags', 'editedAt', 'userEditedFields', 'notes'].includes(key)));
    const metadata = { ...(book.metadata ?? {}), ...preserved, toc: book.toc ?? [], warnings: book.warnings ?? [],
      sourceSha256: id, format: file.format, searchable: book.sections.some((section) => typeof section.text === 'string' && section.text.trim()),
      pipelineVersion: PIPELINE_VERSION, sourceOffsetBasis: 'section_utf16_code_units',
      grouping: { candidate: file.group_key, conclusive: false } };
    const edited = previous && previousMetadata.editedAt;
    return this.transaction(() => {
      const stamp = now();
      this.db.prepare(`INSERT INTO books(id,title,authors_json,description,language,publisher,cover_path,metadata_json,created_at,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,authors_json=excluded.authors_json,
        description=excluded.description,language=excluded.language,publisher=excluded.publisher,cover_path=excluded.cover_path,
        metadata_json=excluded.metadata_json,updated_at=excluded.updated_at`)
        .run(id, edited ? previous.title : book.title || file.relative_path, edited ? previous.authors_json : JSON.stringify(authors),
          edited ? previous.description : book.description ?? '', edited ? previous.language : book.language ?? '',
          edited ? previous.publisher : book.publisher ?? '', edited ? previous.cover_path : book.coverPath ?? null, JSON.stringify(metadata), stamp, stamp);
      saveBookAssets(this, id, { sourceMetadata: book.metadata?.sourceMetadata || {},
        thumbnail: decodeThumbnail(book.thumbnail), preserveCover: true,
        provenance: { sourceSha256: id, format: file.format, extractionVersion: book.metadata?.extractionVersion || null } });
      const sectionInsert = this.db.prepare(`INSERT INTO sections(id,book_id,ordinal,title,text,locator_json) VALUES(?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET title=excluded.title,text=excluded.text,locator_json=excluded.locator_json`);
      const chunkInsert = this.db.prepare(`INSERT INTO chunks(id,book_id,section_id,ordinal,text,locator_json) VALUES(?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET text=excluded.text,locator_json=excluded.locator_json`);
      const retainedSections = new Set();
      const retainedOrdinals = new Set();
      const retainedChunks = new Set();
      let count = 0;
      for (const [index, section] of book.sections.entries()) {
        const ordinal = section.ordinal ?? index + 1;
        if (!Number.isSafeInteger(ordinal) || ordinal < 0 || typeof section.text !== 'string' || !section.locator || typeof section.locator !== 'object') {
          throw new TypeError('Invalid source section');
        }
        if (retainedOrdinals.has(ordinal)) throw new TypeError('Duplicate extracted section ordinal');
        retainedOrdinals.add(ordinal);
        const fingerprint = digest(JSON.stringify([PIPELINE_VERSION, section.text, section.locator]));
        const oldSection = this.db.prepare('SELECT id,text,locator_json FROM sections WHERE book_id=? AND ordinal=?').get(id, ordinal);
        const sectionId = oldSection?.text === section.text && oldSection.locator_json === JSON.stringify(section.locator)
          ? oldSection.id : `${id}:s${ordinal}:${fingerprint}`;
        if (retainedSections.has(sectionId)) throw new TypeError('Duplicate extracted section ordinal');
        retainedSections.add(sectionId);
        this.db.prepare('DELETE FROM sections WHERE book_id=? AND ordinal=? AND id<>?').run(id, ordinal, sectionId);
        sectionInsert.run(sectionId, id, ordinal, section.title ?? '', section.text, JSON.stringify(section.locator));
        const sourceLocator = { ...section.locator };
        delete sourceLocator.anchors;
        if (sourceLocator.derived && typeof sourceLocator.derived === 'object') {
          sourceLocator.derived = { ...sourceLocator.derived };
          delete sourceLocator.derived.anchors;
        }
        let chunkOrdinal = 0;
        for (const chunk of chunkText(section.text)) {
          const locator = { ...sourceLocator, sectionId, sectionOrdinal: ordinal, charStart: chunk.charStart,
            charEnd: chunk.charEnd, offsetBasis: 'section_utf16_code_units', sourceSha256: id, pipelineVersion: PIPELINE_VERSION };
          const oldChunk = this.db.prepare('SELECT id,text,locator_json FROM chunks WHERE section_id=? AND ordinal=?').get(sectionId, chunkOrdinal);
          const oldLocator = parseJSON(oldChunk?.locator_json);
          const sameMeaning = oldChunk?.text === chunk.text && oldLocator.charStart === chunk.charStart && oldLocator.charEnd === chunk.charEnd
            && oldLocator.sectionId === sectionId && oldLocator.sourceSha256 === id && oldLocator.offsetBasis === 'section_utf16_code_units';
          const chunkId = sameMeaning ? oldChunk.id : `${sectionId}:c${chunkOrdinal}:${digest(JSON.stringify([PIPELINE_VERSION, chunk.charStart, chunk.charEnd, chunk.text]))}`;
          retainedChunks.add(chunkId);
          this.db.prepare('DELETE FROM chunks WHERE section_id=? AND ordinal=? AND id<>?').run(sectionId, chunkOrdinal, chunkId);
          chunkInsert.run(chunkId, id, sectionId, chunkOrdinal++, chunk.text, JSON.stringify(locator));
          count++;
        }
      }
      for (const row of this.db.prepare('SELECT id FROM sections WHERE book_id=?').all(id)) {
        if (!retainedSections.has(row.id)) this.db.prepare('DELETE FROM sections WHERE id=?').run(row.id);
      }
      for (const row of this.db.prepare('SELECT id FROM chunks WHERE book_id=?').all(id)) {
        if (!retainedChunks.has(row.id)) this.db.prepare('DELETE FROM chunks WHERE id=?').run(row.id);
      }
      this.finishFile(jobId, fileId, receiptId, { status: 'completed', bookId: id,
        metadata: { extraction: { sections: book.sections.length, chunks: count, warnings: book.warnings ?? [], searchable: metadata.searchable } } });
      return id;
    });
  }
}
