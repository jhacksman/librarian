import { createServer } from 'node:http';
import { createReadStream, realpathSync, statSync, accessSync, constants } from 'node:fs';
import { mkdir, readFile, realpath, stat } from 'node:fs/promises';
import { resolve, relative, sep, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { LibraryStore } from './store.mjs';
import { inventoryLibrary, runImport, createReindexJob } from './importer.mjs';
import { importOwner, reconcileInterruptedImports } from './import-lifecycle.mjs';
import { createRetrieval } from './retrieval.mjs';
import { resolveReaderContext } from './answer-context.mjs';
import { createOriginalReader } from './original-reader.mjs';
import { readConfig } from './config.mjs';
import { ensureCatalogSchema, catalogEntries, groupSuggestions, linkEditions, unlinkEdition } from './catalog.mjs';
import { metadataQuality } from './source-metadata-recovery.mjs';
import { bookAssetSummary, fallbackThumbnail, saveBookAssets, presentationIdentity } from './book-assets.mjs';

const publicDir = resolve(dirname(fileURLToPath(import.meta.url)), '../public');
const parseJSON = (value, fallback = {}) => { try { return JSON.parse(value || 'null') ?? fallback; } catch { return fallback; } };
const boundedInt = (value, fallback, max) => {
  if (value === null || value === undefined || value === '') return fallback;
  const n = Number(value);
  if (!Number.isInteger(n) || n < 0 || n > max) throw Object.assign(new Error('Invalid pagination value'), { statusCode: 400 });
  return n;
};
const fail = (message, statusCode = 400) => Object.assign(new Error(message), { statusCode });
const within = (root, file) => { const rel = relative(root, file); return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !rel.startsWith(sep)); };
const coverMime = file => ({ '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.gif': 'image/gif' })[extname(file || '').toLowerCase()];

function selectedCover(store, row, thumbnail) {
  if (thumbnail && thumbnail.kind !== 'fallback') return { ...thumbnail, source: 'stored_thumbnail' };
  if (row.cover_path && coverMime(row.cover_path)) {
    try {
      const root = realpathSync(store.dataDir), covers = realpathSync(resolve(store.dataDir, 'covers'));
      const file = realpathSync(row.cover_path);
      if (within(root, covers) && within(covers, file) && statSync(file).isFile()) {
        accessSync(file, constants.R_OK);
        return { url: `/api/books/${encodeURIComponent(row.id)}/cover`, mime: coverMime(row.cover_path),
          kind: 'cover', source: 'legacy_file' };
      }
    } catch { /* A missing or unavailable legacy image leaves the stored fallback visible. */ }
  }
  return thumbnail ? { ...thumbnail, source: 'stored_thumbnail' } : null;
}

async function bodyJSON(req) {
  if (!(req.headers['content-type'] || '').toLowerCase().startsWith('application/json')) throw fail('Send application/json', 415);
  let size = 0; const chunks = [];
  for await (const part of req) {
    size += part.length;
    if (size > 65536) throw fail('Request is too large', 413);
    chunks.push(part);
  }
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error();
    return value;
  } catch { throw fail('Expected a JSON object'); }
}

function json(res, value, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(value));
}

function bookView(store, row) {
  const formats = store.db.prepare('SELECT id,format,status FROM files WHERE book_id=? ORDER BY format,id').all(row.id);
  const count = store.db.prepare('SELECT count(*) AS n FROM sections WHERE book_id=?').get(row.id).n;
  const chunks = store.db.prepare('SELECT count(*) AS n FROM chunks WHERE book_id=?').get(row.id).n;
  const thumbnail = bookAssetSummary(store, row.id);
  const cover = selectedCover(store, row, thumbnail);
  const sourceMetadata = parseJSON(store.db.prepare('SELECT source_metadata_json FROM book_assets WHERE book_id=?').get(row.id)?.source_metadata_json);
  return {
    id: row.id, title: row.title, authors: parseJSON(row.authors_json, []), description: row.description,
    language: row.language, publisher: row.publisher, metadata: parseJSON(row.metadata_json),
    coverUrl: cover?.url || null, cover, thumbnail,
    metadataQuality: metadataQuality({title: row.title, authors: parseJSON(row.authors_json, []), metadata: parseJSON(row.metadata_json)}),
    sourceMetadata,
    formats: [...new Set(formats.map(f => f.format))], sectionCount: count, chunkCount: chunks, textSearchable: chunks > 0,
    status: count ? 'completed' : (formats[0]?.status || 'queued'),
  };
}

function jobView(row, controls = {}) {
  const summary = row.summary || parseJSON(row.summary_json);
  return { ...row, status: row.state, totalFiles: row.total, processedFiles: row.processed,
    startedAt: row.started_at, updatedAt: row.updated_at, summary, current: summary.current,
    error: summary.error || null, recoveryNotice: summary.recoveryNotice || null, ...controls };
}

function fileView(row) {
  const metadata = row.metadata || parseJSON(row.metadata_json);
  const status = row.job_status || row.status;
  return { id: row.id, path: row.relative_path, name: row.relative_path, format: row.format,
    status, error: row.job_error || row.error,
    bookId: row.book_id, sha256: row.sha256, size: row.size, staged: !!row.staged,
    // Older completed rows can retain a previous unsupported-format reason.
    reason: ['unsupported', 'auxiliary'].includes(status) ? metadata.reason || null : null, metadata };
}

export async function createApp({ config = readConfig(), store: providedStore, retrieval: providedRetrieval } = {}) {
  const maintenanceEnabled = config.maintenanceEnabled !== false;
  await mkdir(config.dataDir, { recursive: true, mode: 0o700 });
  const store = providedStore || new LibraryStore(config.dbPath, { dataDir: config.dataDir });
  const retrieval = providedRetrieval || createRetrieval(store, config.retrieval);
  ensureCatalogSchema(store);
  const activeImports = new Map();
  let embeddingTask = null;
  const embeddingState = { state: 'idle', error: null };
  let closing = false;
  if (maintenanceEnabled) reconcileInterruptedImports(store);

  function requireMaintenance() {
    if (!maintenanceEnabled) throw fail('Library maintenance is disabled for this session.', 403);
  }

  function viewJob(row) {
    const owner = importOwner(store);
    const summary = row.summary || parseJSON(row.summary_json);
    const retryableFiles = store.db.prepare("SELECT count(*) n FROM import_job_files WHERE job_id=? AND status IN ('queued','running','failed') AND retryable=1 AND inventory_error IS NULL").get(row.id).n;
    return jobView(row, {
      canPause: maintenanceEnabled && activeImports.has(row.id),
      canResume: maintenanceEnabled && retryableFiles > 0 && !owner && !activeImports.size && summary.inventoryComplete === true &&
        ['paused', 'failed', 'completed_with_errors'].includes(row.state),
      retryableFiles,
      externalOwner: owner?.job_id === row.id && !activeImports.has(row.id),
    });
  }

  function startEmbedding({ retryFailed = false } = {}) {
    if (!maintenanceEnabled || embeddingTask || !config.retrieval.embeddingModel) return;
    embeddingState.state = 'running'; embeddingState.error = null;
    const controller = new AbortController();
    const promise = (async () => {
      while (!controller.signal.aborted && !closing) {
        const result = await retrieval.embedPending({ signal: controller.signal, retryFailed, onProgress: progress => Object.assign(embeddingState, { progress }) });
        retryFailed = false;
        Object.assign(embeddingState, { state: result.status, result });
        if (result.status !== 'bounded' || !((result.pendingChunks ?? result.remaining) > 0) || !((result.processed + (result.quarantined || 0)) > 0)) break;
        embeddingState.state = 'running';
        await new Promise(done => setImmediate(done));
      }
      if (controller.signal.aborted) embeddingState.state = 'cancelled';
    })()
      .catch(error => Object.assign(embeddingState, { state: 'failed', error: error.message }))
      .finally(() => { embeddingTask = null; });
    embeddingTask = { promise, controller };
  }

  function startImport(id, retryFailed = false) {
    requireMaintenance();
    if (closing || activeImports.has(id)) return;
    if (activeImports.size) throw fail('An import is already running; wait or resume it', 409);
    if (importOwner(store)) throw fail('Another import process is running. Wait for it to finish.', 409);
    if (!store.getJob(id)?.summary.inventoryComplete) throw fail('Inventory did not finish. Import the source folder again.', 409);
    const controller = new AbortController();
    const promise = runImport(store, id, { ...config.importOptions, dataDir: config.dataDir, signal: controller.signal, retryFailed })
      .then(() => { if (!closing) startEmbedding(); })
      .catch(error => { console.error(`Import ${id} stopped: ${error.message}`); })
      .finally(() => activeImports.delete(id));
    activeImports.set(id, { promise, controller });
  }

  async function approvedRoot(input) {
    if (typeof input !== 'string' || !input.trim()) throw fail('Choose an import folder');
    const candidate = await realpath(resolve(input));
    let approved = false;
    for (const root of config.importRoots) {
      try { if (within(await realpath(root), candidate)) approved = true; } catch { /* unavailable configured root */ }
    }
    if (!approved) throw fail('Folder is outside the configured import locations', 403);
    if (!(await stat(candidate)).isDirectory()) throw fail('Choose a folder');
    return candidate;
  }

  async function sendManagedFile(req, res, file, mime, downloadName) {
    const root = await realpath(config.dataDir);
    const target = await realpath(file);
    if (!within(root, target)) throw fail('Source has not been copied into this library', 409);
    const info = await stat(target);
    if (!info.isFile()) throw fail('File unavailable', 404);
    let start = 0; let end = info.size - 1; let status = 200;
    if (req.headers.range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
      if (!match || (!match[1] && !match[2])) throw fail('Invalid range', 416);
      if (!match[1]) start = Math.max(0, info.size - Number(match[2]));
      else { start = Number(match[1]); if (match[2]) end = Math.min(Number(match[2]), end); }
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= info.size) {
        res.setHeader('Content-Range', `bytes */${info.size}`); throw fail('Range unavailable', 416);
      }
      status = 206; res.setHeader('Content-Range', `bytes ${start}-${end}/${info.size}`);
    }
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Content-Type', mime);
    if (downloadName) res.setHeader('Content-Disposition', `${mime === 'application/pdf' ? 'inline' : 'attachment'}; filename*=UTF-8''${encodeURIComponent(downloadName).replace(/'/g, '%27')}`);
    res.setHeader('Content-Length', Math.max(0, end - start + 1));
    res.writeHead(status);
    if (req.method === 'HEAD' || info.size === 0) return res.end();
    const stream = createReadStream(target, { start, end });
    stream.on('error', () => res.destroy());
    res.on('close', () => stream.destroy());
    stream.pipe(res);
  }

  const originalReader = createOriginalReader({ store, dataDir: config.dataDir, sendManagedFile });

  const server = createServer(async (req, res) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'no-referrer');
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-src 'self'; worker-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    try {
      const expected = new URL(config.origin);
      if (req.headers.host !== expected.host) throw fail('Unrecognized host', 403);
      if (req.headers.origin && req.headers.origin !== expected.origin) throw fail('Cross-origin request refused', 403);
      if (!['GET', 'HEAD'].includes(req.method) && req.headers.origin !== expected.origin) throw fail('Same-origin request required', 403);
      const url = new URL(req.url, expected);
      const path = url.pathname;
      if (!['GET', 'HEAD'].includes(req.method) &&
          (path === '/api/imports' || path.startsWith('/api/imports/') || ['/api/reindex', '/api/embeddings'].includes(path))) {
        requireMaintenance();
      }
      if (await originalReader.handle(req, res, url)) return;
      const matchBook = /^\/api\/books\/([^/]+)(?:\/(cover|thumbnail|progress))?$/.exec(path);
      if (req.method === 'GET' && path === '/api/status') {
        const totals = store.summary();
        totals.uniqueSources = store.db.prepare('SELECT count(DISTINCT sha256) AS n FROM files WHERE staged=1').get().n;
        totals.searchableBooks = store.db.prepare('SELECT count(*) AS n FROM books b WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.book_id=b.id)').get().n;
        totals.textIndexedFiles = store.db.prepare('SELECT count(*) AS n FROM files f WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.book_id=f.book_id)').get().n;
        const authors = store.db.prepare("SELECT DISTINCT j.value AS name FROM books, json_each(books.authors_json) j ORDER BY j.value COLLATE NOCASE").all().map(x => x.name);
        const formats = store.db.prepare('SELECT DISTINCT format FROM files ORDER BY format').all().map(x => x.format);
        const statuses = store.db.prepare('SELECT DISTINCT status FROM files ORDER BY status').all().map(x => x.status);
        const models = retrieval.status();
        totals.embeddedChunks = models.embedding.embeddedChunks;
        totals.vectorCoverage = { completed: models.embedding.embeddedChunks, total: models.embedding.totalChunks, status: models.embedding.status };
        return json(res, { totals, importRoots: config.importRoots, capabilities: { maintenance: maintenanceEnabled }, facets: { authors, formats, statuses }, models: { ...models, state: { ...embeddingState, active: !!embeddingTask } } });
      }
      if (req.method === 'GET' && path === '/api/catalog') {
        const filters = Object.fromEntries(['q', 'format', 'author', 'status', 'tag', 'sort'].flatMap(key => url.searchParams.has(key) ? [[key, url.searchParams.get(key)]] : []));
        filters.limit = boundedInt(url.searchParams.get('limit'), 36, 100) || 36;
        filters.offset = boundedInt(url.searchParams.get('offset'), 0, 1000000);
        let catalog;
        try { catalog = catalogEntries(store, filters); } catch (error) { throw fail(error.message); }
        catalog.items = catalog.items.map(card => ({ ...card, members: card.members.map(member => {
          const thumbnail = bookAssetSummary(store, member.id);
          const row = store.db.prepare('SELECT id,cover_path FROM books WHERE id=?').get(member.id);
          const cover = selectedCover(store, row, thumbnail);
          return { ...member, metadataQuality: metadataQuality(member), thumbnail, cover, coverUrl: cover?.url || null };
        }) }));
        return json(res, catalog);
      }
      if (req.method === 'GET' && path === '/api/catalog/suggestions') return json(res, groupSuggestions(store, {
        limit: boundedInt(url.searchParams.get('limit'), 25, 100) || 25, offset: boundedInt(url.searchParams.get('offset'), 0, 1000000) }));
      if (req.method === 'POST' && path === '/api/catalog/groups') {
        const data = await bodyJSON(req);
        if (Object.keys(data).some(key => !['bookIds', 'title'].includes(key))) throw fail('Unknown grouping field');
        try { return json(res, linkEditions(store, data)); } catch (error) { throw fail(error.message); }
      }
      const groupMatch = /^\/api\/catalog\/groups\/([^/]+)$/.exec(path);
      if (req.method === 'GET' && groupMatch) {
        const group = store.db.prepare('SELECT * FROM catalog_groups WHERE id=?').get(decodeURIComponent(groupMatch[1]));
        if (!group) throw fail('Catalog group not found', 404);
        const members = store.db.prepare('SELECT b.* FROM books b JOIN group_members m ON b.id=m.book_id WHERE m.group_id=? ORDER BY b.title,b.id').all(group.id).map(row => bookView(store, row));
        return json(res, { id: group.id, title: group.title, version: group.version, members });
      }
      if (req.method === 'POST' && path === '/api/catalog/unlink') {
        const data = await bodyJSON(req);
        if (Object.keys(data).some(key => key !== 'bookId') || typeof data.bookId !== 'string') throw fail('Choose one source book to unlink');
        try { return json(res, unlinkEdition(store, data.bookId)); } catch (error) { throw fail(error.message); }
      }
      if (req.method === 'GET' && path === '/api/books') {
        const limit = boundedInt(url.searchParams.get('limit'), 36, 100) || 36;
        const offset = boundedInt(url.searchParams.get('offset'), 0, 1000000);
        const clauses = []; const params = [];
        for (const field of ['q', 'format', 'author', 'status']) {
          const value = url.searchParams.get(field)?.trim();
          if (!value) continue;
          if (value.length > 500) throw fail('Filter is too long');
          if (field === 'q') { clauses.push("(instr(lower(b.title),lower(?))>0 OR instr(lower(b.authors_json),lower(?))>0 OR instr(lower(b.metadata_json),lower(?))>0)"); params.push(value, value, value); }
          if (field === 'format') { clauses.push('EXISTS (SELECT 1 FROM files f WHERE f.book_id=b.id AND f.format=?)'); params.push(value.toLowerCase().replace(/^\./, '')); }
          if (field === 'author') { clauses.push('EXISTS (SELECT 1 FROM json_each(b.authors_json) a WHERE a.value=?)'); params.push(value); }
          if (field === 'status') { clauses.push('EXISTS (SELECT 1 FROM files f WHERE f.book_id=b.id AND f.status=?)'); params.push(value); }
        }
        const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
        const sort = { title: 'b.title COLLATE NOCASE,b.id', author: 'b.authors_json COLLATE NOCASE,b.title COLLATE NOCASE,b.id', recent: 'b.rowid DESC' }[url.searchParams.get('sort') || 'title'];
        if (!sort) throw fail('Unknown sort order');
        const total = store.db.prepare(`SELECT count(*) AS n FROM books b ${where}`).get(...params).n;
        const items = store.db.prepare(`SELECT b.* FROM books b ${where} ORDER BY ${sort} LIMIT ? OFFSET ?`).all(...params, limit, offset).map(row => bookView(store, row));
        return json(res, { items, total, offset, limit });
      }
      if (matchBook) {
        const id = decodeURIComponent(matchBook[1]);
        const row = store.db.prepare('SELECT * FROM books WHERE id=?').get(id);
        if (!row) throw fail('Book not found', 404);
        if (req.method === 'GET' && !matchBook[2]) {
          const view = bookView(store, row);
          view.sections = store.db.prepare('SELECT id,ordinal,title,locator_json FROM sections WHERE book_id=? ORDER BY ordinal').all(id).map(s => ({ id: s.id, ordinal: s.ordinal, title: s.title, locator: parseJSON(s.locator_json) }));
          const nativeToc = Array.isArray(view.metadata.toc) ? view.metadata.toc : [];
          const mappedToc = nativeToc.flatMap(item => {
            const target = item.locator?.derived || item.locator || {};
            const section = view.sections.find(s => {
              const source = s.locator.derived || s.locator;
              return item.locator?.format === s.locator.format &&
                (target.page != null ? target.page === source.page : target.member === source.member);
            });
            if (!section) return [];
            const anchors = (section.locator.derived || section.locator).anchors || {};
            const range = Number.isSafeInteger(target.charStart) && Number.isSafeInteger(target.charEnd) ? target :
              target.heading && Object.hasOwn(anchors, target.heading) ? anchors[target.heading] : null;
            return [{ ...section, title: item.title || section.title, depth: item.depth || 0, locator: item.locator,
              unresolvedFragment: !!target.heading && !range, ...(range ? { charStart: range.charStart, charEnd: range.charEnd } : {}) }];
          });
          view.toc = mappedToc.length ? mappedToc : view.sections;
          view.relatedBooks = store.db.prepare(`SELECT DISTINCT b.id,b.title,f.format FROM files f JOIN books b ON b.id=f.book_id
            WHERE f.group_key IN (SELECT group_key FROM files WHERE book_id=?) AND b.id<>? ORDER BY b.title,f.format`).all(id, id);
          view.formats = store.db.prepare('SELECT id,format,status,relative_path FROM files WHERE book_id=? ORDER BY format,id').all(id).map(f => ({ id: f.id, format: f.format, status: f.status, name: f.relative_path, sourceUrl: `/api/files/${encodeURIComponent(f.id)}/source` }));
          const progress = store.db.prepare('SELECT section_id,updated_at FROM reading_progress WHERE book_id=?').get(id);
          view.readingProgress = progress ? { sectionId: progress.section_id, updatedAt: progress.updated_at } : null;
          view.warnings = view.metadata.warnings || [];
          return json(res, view);
        }
        if (['GET', 'HEAD'].includes(req.method) && matchBook[2] === 'cover') {
          if (!row.cover_path) throw fail('No cover available', 404);
          const mime = coverMime(row.cover_path);
          if (!mime) throw fail('Unsupported cover', 415);
          return await sendManagedFile(req, res, row.cover_path, mime);
        }
        if (['GET', 'HEAD'].includes(req.method) && matchBook[2] === 'thumbnail') {
          const asset = store.db.prepare('SELECT thumbnail,mime,thumbnail_sha256 FROM book_assets WHERE book_id=?').get(id);
          if (!asset) throw fail('Thumbnail has not been stored for this book', 404);
          const etag = `"${asset.thumbnail_sha256}"`;
          if (req.headers['if-none-match'] === etag) { res.writeHead(304, { ETag: etag }); return res.end(); }
          res.writeHead(200, { 'Content-Type': asset.mime, 'Content-Length': asset.thumbnail.length,
            'Cache-Control': 'private, max-age=3600', ETag: etag, 'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "default-src 'none'; style-src 'none'; sandbox" });
          return res.end(req.method === 'HEAD' ? undefined : asset.thumbnail);
        }
        if (req.method === 'PUT' && !matchBook[2]) {
          const data = await bodyJSON(req);
          const fields = ['title', 'authors', 'description', 'publisher', 'language', 'tags'];
          if (Object.keys(data).some(k => !fields.includes(k))) throw fail('Unknown metadata field');
          const updated = { title: row.title, authors: parseJSON(row.authors_json, []), description: row.description || '', publisher: row.publisher || '', language: row.language || '', tags: parseJSON(row.metadata_json).tags || [], ...data };
          for (const key of ['title', 'description', 'publisher', 'language']) if (typeof updated[key] !== 'string' || updated[key].length > (key === 'description' ? 20000 : 500)) throw fail(`Invalid ${key}`);
          if (!updated.title.trim()) throw fail('Title is required');
          for (const key of ['authors', 'tags']) if (!Array.isArray(updated[key]) || updated[key].length > 100 || updated[key].some(x => typeof x !== 'string' || x.length > 500)) throw fail(`Invalid ${key}`);
          store.db.prepare('UPDATE books SET title=?,authors_json=?,description=?,publisher=?,language=?,metadata_json=? WHERE id=?').run(updated.title.trim(), JSON.stringify(updated.authors), updated.description, updated.publisher, updated.language, JSON.stringify({ ...parseJSON(row.metadata_json), tags: updated.tags, editedAt: new Date().toISOString() }), id);
          const asset = store.db.prepare('SELECT kind,source_metadata_json,provenance_json FROM book_assets WHERE book_id=?').get(id);
          if (asset?.kind === 'fallback') saveBookAssets(store, id, { sourceMetadata: parseJSON(asset.source_metadata_json),
            thumbnail: fallbackThumbnail(updated.title), provenance: parseJSON(asset.provenance_json) });
          return json(res, { ok: true, id });
        }
        if (req.method === 'POST' && matchBook[2] === 'progress') {
          const data = await bodyJSON(req);
          if (typeof data.sectionId !== 'string' || !store.db.prepare('SELECT id FROM sections WHERE id=? AND book_id=?').get(data.sectionId, id)) throw fail('Section does not belong to this book');
          store.db.prepare('INSERT INTO reading_progress(book_id,section_id,updated_at) VALUES(?,?,?) ON CONFLICT(book_id) DO UPDATE SET section_id=excluded.section_id,updated_at=excluded.updated_at').run(id, data.sectionId, new Date().toISOString());
          return json(res, { ok: true });
        }
      }
      const sectionMatch = /^\/api\/sections\/([^/]+)$/.exec(path);
      if (req.method === 'GET' && sectionMatch) {
        const row = store.db.prepare('SELECT * FROM sections WHERE id=?').get(decodeURIComponent(sectionMatch[1]));
        if (!row) throw fail('Section not found', 404);
        const previous = store.db.prepare('SELECT id FROM sections WHERE book_id=? AND ordinal<? ORDER BY ordinal DESC LIMIT 1').get(row.book_id, row.ordinal);
        const next = store.db.prepare('SELECT id FROM sections WHERE book_id=? AND ordinal>? ORDER BY ordinal LIMIT 1').get(row.book_id, row.ordinal);
        return json(res, { id: row.id, bookId: row.book_id, title: row.title, text: row.text, locator: parseJSON(row.locator_json), previousId: previous?.id || null, nextId: next?.id || null });
      }
      const fileMatch = /^\/api\/files\/([^/]+)\/source$/.exec(path);
      if (['GET', 'HEAD'].includes(req.method) && fileMatch) {
        const row = store.db.prepare('SELECT * FROM files WHERE id=?').get(decodeURIComponent(fileMatch[1]));
        if (!row) throw fail('Source not found', 404);
        return await sendManagedFile(req, res, row.path, row.format === 'pdf' ? 'application/pdf' : 'application/octet-stream', row.relative_path.split('/').pop());
      }
      if (req.method === 'GET' && path === '/api/imports') {
        if (maintenanceEnabled) reconcileInterruptedImports(store);
        return json(res, { items: store.db.prepare('SELECT * FROM import_jobs ORDER BY started_at DESC LIMIT 100').all().map(viewJob) });
      }
      const importMatch = /^\/api\/imports\/([^/]+)(?:\/(resume|pause))?$/.exec(path);
      if (importMatch) {
        const id = decodeURIComponent(importMatch[1]); const job = store.getJob(id);
        if (!job) throw fail('Import not found', 404);
        if (req.method === 'GET' && !importMatch[2]) {
          const offset = boundedInt(url.searchParams.get('offset'), 0, 1000000);
          const limit = boundedInt(url.searchParams.get('limit'), 1000, 1000) || 1000;
          const files = store.listFiles({ jobId: id, offset, limit, status: url.searchParams.get('status') || undefined });
          return json(res, { ...viewJob(job), files: (Array.isArray(files) ? files : files.items).map(fileView), offset, limit });
        }
        if (req.method === 'POST' && importMatch[2] === 'resume') { await bodyJSON(req); startImport(id, true); return json(res, viewJob(store.getJob(id)), 202); }
        if (req.method === 'POST' && importMatch[2] === 'pause') {
          await bodyJSON(req);
          const task = activeImports.get(id);
          if (!task) {
            reconcileInterruptedImports(store);
            throw fail(importOwner(store)?.job_id === id ? 'This import is owned by another process; pause it there.' : 'This import is already stopped. Refresh to resume it.', 409);
          }
          task.controller.abort(); return json(res, { ok: true }, 202);
        }
      }
      if (req.method === 'POST' && path === '/api/imports') {
        if (activeImports.size) throw fail('An import is already running', 409);
        const data = await bodyJSON(req); const root = await approvedRoot(data.root);
        const job = await inventoryLibrary(store, root, { dataDir: config.dataDir });
        startImport(job.id); return json(res, viewJob(store.getJob(job.id)), 202);
      }
      if (req.method === 'POST' && path === '/api/reindex') {
        if (activeImports.size || importOwner(store)) throw fail('An import is already running', 409);
        const data = await bodyJSON(req);
        if (Object.keys(data).some(key => !['bookIds', 'fileIds'].includes(key))) throw fail('Unknown reindex field');
        let job;
        try { job = createReindexJob(store, data); } catch (error) { throw fail(error.message); }
        startImport(job.id); return json(res, viewJob(store.getJob(job.id)), 202);
      }
      if (req.method === 'POST' && path === '/api/embeddings') {
        const data = await bodyJSON(req);
        if (Object.keys(data).some(key => key !== 'retryFailed') || (data.retryFailed !== undefined && typeof data.retryFailed !== 'boolean')) throw fail('Invalid embedding request');
        if (!config.retrieval.embeddingModel) throw fail('Configure an installed local embedding model first.', 409);
        startEmbedding({ retryFailed: data.retryFailed === true }); return json(res, { ...embeddingState, active: !!embeddingTask }, 202);
      }
      if (req.method === 'POST' && ['/api/search', '/api/ask'].includes(path)) {
        const data = await bodyJSON(req);
        const value = path === '/api/ask' ? data.question : data.query;
        if (typeof value !== 'string' || !value.trim() || value.length > 4000) throw fail('Enter a question or search of 1–4000 characters');
        if (data.bookId !== undefined && data.bookId !== null && (typeof data.bookId !== 'string' || !store.db.prepare('SELECT id FROM books WHERE id=?').get(data.bookId))) throw fail('Unknown book scope');
        if (data.limit !== undefined && (!Number.isInteger(data.limit) || data.limit < 1 || data.limit > 20)) throw fail('Limit must be 1–20');
        if (path === '/api/ask' && data.responseMode !== undefined && !['auto', 'excerpts'].includes(data.responseMode)) throw fail('Response mode must be auto or excerpts');
        if (path === '/api/search' && data.readerContext !== undefined) throw fail('Reader context is only supported for Ask');
        let readerContext;
        if (data.readerContext !== undefined) {
          if (data.readerContext === null) throw fail('Reader context must identify a source section');
          try { readerContext = resolveReaderContext(store, data.readerContext, data.bookId); } catch (error) { throw fail(error.message); }
        }
        const result = await (path === '/api/ask' ? retrieval.ask({ question: value, bookId: data.bookId || readerContext?.bookId || undefined, responseMode: data.responseMode || 'auto', ...(readerContext ? { readerContext } : {}) }) : retrieval.search({ query: value, bookId: data.bookId || undefined, limit: data.limit || 10 }));
        const ids = [...new Set([...(result.hits || []), ...(result.citations || []), ...(result.extractive?.passages || [])].map(hit => hit.bookId).filter(Boolean))];
        result.books = ids.flatMap(id => {
          const row = store.db.prepare('SELECT * FROM books WHERE id=?').get(id);
          if (!row) return [];
          const view = bookView(store, row);
          const membership = store.db.prepare('SELECT group_id FROM group_members WHERE book_id=?').get(id);
          return [{ id: view.id, title: view.title, authors: view.authors, formats: view.formats,
            thumbnail: view.thumbnail, cover: view.cover, coverUrl: view.coverUrl, identity: presentationIdentity(view),
            metadata: { identifiers: view.metadata.identifiers || [], edition: view.metadata.edition || '', publicationDate: view.metadata.publicationDate || '' },
            groupId: membership?.group_id || null }];
        });
        return json(res, result);
      }
      if (['GET', 'HEAD'].includes(req.method) && ['/', '/index.html', '/app.js', '/styles.css', '/reader-views.mjs', '/reader-views.css'].includes(path)) {
        const filename = path === '/' ? 'index.html' : path.slice(1);
        const content = await readFile(resolve(publicDir, filename));
        res.writeHead(200, { 'Content-Type': (/\.m?js$/.test(filename)) ? 'text/javascript; charset=utf-8' : filename.endsWith('.css') ? 'text/css; charset=utf-8' : 'text/html; charset=utf-8' });
        return res.end(req.method === 'HEAD' ? undefined : content);
      }
      throw fail('Not found', 404);
    } catch (error) {
      if (res.headersSent) return res.destroy();
      const status = error.statusCode || (error.code === 'ENOENT' ? 404 : 500);
      if (status === 500) console.error(error);
      json(res, { error: status === 500 ? 'The library could not complete this request. Check the local server log.' : error.message }, status);
    }
  });
  server.requestTimeout = 30000; server.headersTimeout = 15000;
  return { server, store, config, async close() {
    closing = true;
    for (const item of activeImports.values()) item.controller.abort();
    embeddingTask?.controller.abort();
    await Promise.allSettled([...activeImports.values()].map(x => x.promise).concat(embeddingTask ? [embeddingTask.promise] : []));
    if (server.listening) await new Promise(done => { server.close(done); server.closeIdleConnections(); });
    if (!providedStore) store.close();
  } };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const app = await createApp();
  app.server.listen(app.config.port, app.config.host, () => console.log(`Librarian: ${app.config.origin}\nLocal data: ${app.config.dataDir}`));
  for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, async () => { await app.close(); process.exit(0); });
}
