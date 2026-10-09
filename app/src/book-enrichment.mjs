import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { lstat, mkdir, realpath } from 'node:fs/promises';
import { join } from 'node:path';
import { runParserWorker } from './importer.mjs';
import { decodeThumbnail, fallbackThumbnail, saveBookAssets } from './book-assets.mjs';
import { isWithin, parseJSON } from './store.mjs';
import { importOwner } from './import-lifecycle.mjs';
import { recoverReviewedMetadata } from './source-metadata-recovery.mjs';
import { recoverFrozenGold } from './frozen-gold-recovery.mjs';

const abort = signal => signal?.throwIfAborted();
async function verifiedSource(store, file, metadata, signal) {
  let path = file.path, format = file.format, expectedSha = file.sha256, expectedSize = file.size;
  if (['mobi', 'prc'].includes(format)) {
    const converted = metadata.conversion?.derived;
    if (!converted?.path || !converted.sha256) throw new Error('No verified converted EPUB is available; no new conversion was started');
    path = converted.path; format = 'epub'; expectedSha = converted.sha256; expectedSize = converted.bytes;
  }
  if (!['pdf', 'epub'].includes(format)) throw new Error('This format has no local metadata/cover extractor');
  const stat = await lstat(path), canonical = await realpath(path);
  if (!stat.isFile() || stat.isSymbolicLink() || !isWithin(await realpath(store.dataDir), canonical)) throw new Error('Enrichment requires a regular managed source');
  const hash = createHash('sha256'); let bytes = 0;
  for await (const chunk of createReadStream(canonical)) { abort(signal); hash.update(chunk); bytes += chunk.length; }
  const after = await lstat(canonical), actual = hash.digest('hex');
  if (actual !== expectedSha || (expectedSize != null && bytes !== expectedSize) || after.size !== stat.size || after.mtimeMs !== stat.mtimeMs) throw new Error('Managed source identity changed');
  return { path: canonical, format, sha256: actual, bytes, originalSourceSha256: file.sha256 };
}

// Only the coordinator invokes this under a quiescent DATA lease. This updates
// descriptive metadata and assets, never sections/chunks/vectors/progress/groups.
export async function enrichBooks(store, { limit = 25, bookIds, retryFailed = false, recoverMetadata = false, reviewedMetadata = {}, frozenGoldRecords = {}, signal, worker = runParserWorker, onProgress = () => {} } = {}) {
  if (!Number.isInteger(limit) || limit < 1 || limit > 500) throw new RangeError('Enrichment limit must be 1–500');
  if (bookIds && (!Array.isArray(bookIds) || bookIds.some(id => typeof id !== 'string'))) throw new TypeError('Book IDs must be strings');
  if (typeof recoverMetadata !== 'boolean' || !reviewedMetadata || typeof reviewedMetadata !== 'object' || Array.isArray(reviewedMetadata)) throw new TypeError('Invalid metadata recovery request');
  if (recoverMetadata && (!bookIds?.length || bookIds.length > limit)) throw new TypeError('Recovery requires explicit book IDs within the batch limit');
  if (importOwner(store)) throw new Error('An importer owns this DATA; enrichment cannot run concurrently');
  const wanted = bookIds ? new Set(bookIds) : null;
  const rows = store.db.prepare(`SELECT b.*,a.provenance_json FROM books b LEFT JOIN book_assets a ON a.book_id=b.id ORDER BY b.id`).all()
    .filter(row => wanted ? wanted.has(row.id) : !['source_enriched', ...(retryFailed ? [] : ['source_enrichment_failed'])].includes(parseJSON(row.provenance_json).state)).slice(0, limit);
  const result = { processed: 0, completed: 0, failed: 0, rows: [], textOrVectorsChanged: false };
  for (const row of rows) {
    abort(signal);
    const previous = parseJSON(row.metadata_json);
    const file = store.db.prepare(`SELECT * FROM files WHERE book_id=? AND staged=1 AND status IN ('completed','duplicate')
      ORDER BY CASE WHEN format IN ('epub','pdf') THEN 0 ELSE 1 END,id LIMIT 1`).get(row.id);
    let status;
    try {
      if (!file) throw new Error('No verified managed file exists for this book');
      const source = await verifiedSource(store, file, previous, signal);
      if (recoverMetadata && (file.format !== 'pdf' || source.sha256 !== row.id || (!reviewedMetadata[row.id] && !frozenGoldRecords[row.id]))) throw new Error('Recovery requires this original PDF and an independent identity-bound review');
      const outputDir = join(store.dataDir, 'covers', row.id);
      if (!recoverMetadata) await mkdir(outputDir, { recursive: true, mode: 0o700 });
      const extraction = await worker({ operation: 'extractMetadata', path: source.path, format: source.format,
        ...(recoverMetadata ? {frontMatter: true} : {outputDir}) },
        { signal, timeoutMs: 45000, maxWorkerOutputBytes: 2 * 1024 * 1024 });
      abort(signal);
      if (!extraction?.ok || !extraction.book?.metadata?.sourceMetadata) throw new Error('Metadata worker returned an incomplete result');
      const book = extraction.book;
      if (book.metadata.sourceMetadata.captureError) throw new Error(`Source metadata capture failed: ${book.metadata.sourceMetadata.captureError}`);
      const descriptive = { title: book.title, authors: book.authors, description: book.description, publisher: book.publisher, language: book.language };
      let sourceMetadata = { ...book.metadata.sourceMetadata, descriptive,
        ...(['mobi', 'prc'].includes(file.format) ? { format: file.format, derivedFormat: 'epub', conversion: previous.conversion } : {}) };
      const stamp = new Date().toISOString();
      const provenance = { state: 'source_enriched', sourceSha256: row.id, managedSource: source,
        capturedAt: stamp, mode: recoverMetadata ? 'reviewed_native_page_metadata_only' : 'metadata_and_cover_only', warnings: book.warnings || [] };
      store.transaction(() => {
        // Re-read inside the write transaction: edits may have arrived while the
        // disposable parser was running. Their display fields and notes win.
        const current = store.db.prepare('SELECT * FROM books WHERE id=?').get(row.id);
        if (!current) throw new Error('Source book disappeared during extraction');
        const latest = parseJSON(current.metadata_json), edited = Boolean(latest.editedAt);
        const title = book.title && book.title !== source.sha256 ? book.title : current.title;
        let display = {title: edited ? current.title : title,
          authors: edited ? parseJSON(current.authors_json, []) : book.authors?.length ? book.authors : parseJSON(current.authors_json, [])};
        let metadataUpdates = {};
        if (recoverMetadata) {
          // Embedded source fields remain untouched. The separate receipt keeps
          // exact native page quotes, the reviewer and every applied/preserved field.
          const currentDisplay = {title: current.title, authors: parseJSON(current.authors_json, []),
            editedAt: latest.editedAt, edition: latest.edition, creators: latest.creators};
          const recovered = frozenGoldRecords[row.id]
            ? recoverFrozenGold(currentDisplay, sourceMetadata.frontMatter, frozenGoldRecords[row.id])
            : recoverReviewedMetadata(currentDisplay, sourceMetadata.frontMatter, reviewedMetadata[row.id]);
          metadataUpdates = recovered.metadataUpdates || {};
          sourceMetadata = {...sourceMetadata, ...latest.sourceMetadata, frontMatter: sourceMetadata.frontMatter};
          display = recovered.display; sourceMetadata.recovery = recovered.receipt; provenance.recovery = recovered.receipt;
        }
        const metadata = recoverMetadata ? {...latest, ...metadataUpdates, sourceMetadata, sourceEnrichment: provenance} : {...latest, sourceMetadata,
          identifiers: [...new Set([...(latest.identifiers || []), ...(book.metadata.identifiers || [])])],
          publicationDate: book.metadata.publicationDate || latest.publicationDate || '',
          edition: book.metadata.edition || latest.edition || '', sourceEnrichment: provenance};
        store.db.prepare(`UPDATE books SET title=?,authors_json=?,description=?,language=?,publisher=?,metadata_json=?,updated_at=? WHERE id=?`)
          .run(display.title, JSON.stringify(display.authors), edited || recoverMetadata ? current.description : book.description || current.description,
            edited || recoverMetadata ? current.language : book.language || current.language, edited || recoverMetadata ? current.publisher : book.publisher || current.publisher,
            JSON.stringify(metadata), stamp, row.id);
        let thumbnail;
        if (recoverMetadata) {
          const asset = store.db.prepare('SELECT * FROM book_assets WHERE book_id=?').get(row.id);
          if (!asset) throw new Error('Reviewed metadata recovery requires a retained asset');
          thumbnail = {bytes: asset.thumbnail, mime: asset.mime, width: asset.width, height: asset.height,
            kind: asset.kind, provenance: parseJSON(asset.provenance_json).thumbnail};
        } else thumbnail = decodeThumbnail(book.thumbnail) || fallbackThumbnail(display.title, book.warnings?.join('; ') || 'No embedded cover image was found');
        saveBookAssets(store, row.id, {sourceMetadata, thumbnail, preserveCover: true, provenance});
      });
      result.completed++; status = { bookId: row.id, status: 'completed', thumbnailKind: store.db.prepare('SELECT kind FROM book_assets WHERE book_id=?').get(row.id).kind };
    } catch (error) {
      if (signal?.aborted) throw error;
      result.failed++;
      status = { bookId: row.id, status: 'failed', error: String(error.message).slice(0, 1000) };
      if (!recoverMetadata) {
        const asset = store.db.prepare('SELECT source_metadata_json FROM book_assets WHERE book_id=?').get(row.id);
        saveBookAssets(store, row.id, { sourceMetadata: parseJSON(asset?.source_metadata_json),
        thumbnail: fallbackThumbnail(row.title, status.error), preserveCover: true,
        provenance: { state: 'source_enrichment_failed', sourceSha256: row.id, error: status.error } });
      }
    }
    result.processed++; result.rows.push(status); onProgress(status);
  }
  result.remaining = store.db.prepare('SELECT count(*) AS count FROM books b LEFT JOIN book_assets a ON a.book_id=b.id WHERE coalesce(json_extract(a.provenance_json,\'$.state\'),\'\') <> \'source_enriched\'').get().count;
  result.pending = store.db.prepare('SELECT count(*) AS count FROM books b LEFT JOIN book_assets a ON a.book_id=b.id WHERE coalesce(json_extract(a.provenance_json,\'$.state\'),\'\') NOT IN (\'source_enriched\',\'source_enrichment_failed\')').get().count;
  result.failedBooks = store.db.prepare('SELECT count(*) AS count FROM book_assets WHERE json_extract(provenance_json,\'$.state\')=\'source_enrichment_failed\'').get().count;
  result.thumbnailRecords = store.db.prepare('SELECT count(*) AS count FROM book_assets').get().count;
  return result;
}
