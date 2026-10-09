import { createHash } from 'node:crypto';

export const THUMBNAIL_VERSION = 'book-thumbnail-v1';
export const THUMBNAIL_LIMIT = 256 * 1024;
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const parse = value => { try { return JSON.parse(value); } catch { return {}; } };
const escaped = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[char]));

export function fallbackThumbnail(title, reason = 'No cover image is available') {
  const lines = String(title || 'Untitled book').replace(/\s+/g, ' ').match(/.{1,22}(?:\s|$)|.{1,22}/gu)?.slice(0, 5) || ['Untitled book'];
  const bytes = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="256" height="384" viewBox="0 0 256 384"><rect width="256" height="384" fill="#45584b"/><path d="M22 24v336" stroke="#829385"/><text x="42" y="62" fill="#d3dbd1" font-family="sans-serif" font-size="11">LIBRARIAN</text><g fill="#f4ead5" font-family="serif" font-size="20">${lines.map((line, i) => `<text x="42" y="${117 + i * 29}">${escaped(line.trim())}</text>`).join('')}</g><text x="42" y="332" fill="#d3dbd1" font-family="sans-serif" font-size="12">No cover available</text></svg>`);
  return { bytes, mime: 'image/svg+xml', width: 256, height: 384, kind: 'fallback', provenance: { version: THUMBNAIL_VERSION, method: 'generated_placeholder', reason } };
}

export function ensureBookAssets(store) {
  store.db.exec(`CREATE TABLE IF NOT EXISTS book_assets (
    book_id TEXT PRIMARY KEY REFERENCES books(id) ON DELETE CASCADE,
    source_metadata_json TEXT NOT NULL DEFAULT '{}', thumbnail BLOB NOT NULL,
    mime TEXT NOT NULL CHECK(mime IN ('image/png','image/svg+xml')),
    thumbnail_sha256 TEXT NOT NULL, width INTEGER NOT NULL CHECK(width BETWEEN 1 AND 256),
    height INTEGER NOT NULL CHECK(height BETWEEN 1 AND 384),
    kind TEXT NOT NULL CHECK(kind IN ('cover','first_page','fallback')),
    provenance_json TEXT NOT NULL, updated_at TEXT NOT NULL,
    CHECK(length(thumbnail) BETWEEN 1 AND 262144)
  )`);
  for (const row of store.db.prepare('SELECT b.id,b.title,b.metadata_json FROM books b LEFT JOIN book_assets a ON a.book_id=b.id WHERE a.book_id IS NULL').all()) {
    const metadata = parse(row.metadata_json);
    saveBookAssets(store, row.id, { sourceMetadata: metadata.sourceMetadata || {}, thumbnail: fallbackThumbnail(row.title, 'Source cover has not been backfilled yet'), provenance: { sourceSha256: row.id, state: 'pending_source_enrichment' } });
  }
}

export function decodeThumbnail(value) {
  if (!value) return null;
  const bytes = Buffer.from(value.base64 || '', 'base64');
  if (bytes.toString('base64') !== value.base64 || value.mime !== 'image/png' || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw new TypeError('Invalid parser thumbnail');
  if (bytes.length < 24 || bytes.toString('ascii', 12, 16) !== 'IHDR' || bytes.readUInt32BE(16) !== value.width || bytes.readUInt32BE(20) !== value.height) throw new TypeError('Thumbnail dimensions disagree with its PNG header');
  return { ...value, bytes };
}

export function saveBookAssets(store, bookId, { sourceMetadata = {}, thumbnail, provenance = {}, preserveCover = false } = {}) {
  const book = store.db.prepare('SELECT title FROM books WHERE id=?').get(bookId);
  if (!book) throw new Error('Book assets require an existing source book');
  const existing = store.db.prepare('SELECT * FROM book_assets WHERE book_id=?').get(bookId);
  const metadata = JSON.stringify(sourceMetadata);
  if (Buffer.byteLength(metadata) > 1024 * 1024) throw new RangeError('Source metadata exceeds 1 MiB; it was not silently truncated');
  let image = thumbnail || fallbackThumbnail(book.title);
  if (preserveCover && existing && existing.kind !== 'fallback' && image.kind === 'fallback') image = { bytes: existing.thumbnail, mime: existing.mime, width: existing.width, height: existing.height, kind: existing.kind, provenance: parse(existing.provenance_json).thumbnail };
  if (!Buffer.isBuffer(image.bytes) && !(image.bytes instanceof Uint8Array)) throw new TypeError('Thumbnail bytes are required');
  if (!image.bytes.length || image.bytes.length > THUMBNAIL_LIMIT || !Number.isInteger(image.width) || image.width < 1 || image.width > 256 || !Number.isInteger(image.height) || image.height < 1 || image.height > 384 || !['cover', 'first_page', 'fallback'].includes(image.kind) || !['image/png', 'image/svg+xml'].includes(image.mime)) throw new RangeError('Invalid bounded thumbnail');
  if (image.mime === 'image/svg+xml' && image.kind !== 'fallback') throw new TypeError('Only generated placeholders may use SVG');
  store.db.prepare(`INSERT INTO book_assets(book_id,source_metadata_json,thumbnail,mime,thumbnail_sha256,width,height,kind,provenance_json,updated_at)
    VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(book_id) DO UPDATE SET source_metadata_json=excluded.source_metadata_json,
    thumbnail=excluded.thumbnail,mime=excluded.mime,thumbnail_sha256=excluded.thumbnail_sha256,width=excluded.width,
    height=excluded.height,kind=excluded.kind,provenance_json=excluded.provenance_json,updated_at=excluded.updated_at`)
    .run(bookId, metadata, image.bytes, image.mime, sha(image.bytes), image.width, image.height, image.kind,
      JSON.stringify({ ...provenance, thumbnail: image.provenance || {}, version: THUMBNAIL_VERSION }), new Date().toISOString());
}

export function bookAssetSummary(store, bookId) {
  const row = store.db.prepare('SELECT mime,thumbnail_sha256,width,height,kind,provenance_json,updated_at FROM book_assets WHERE book_id=?').get(bookId);
  return row ? { url: `/api/books/${encodeURIComponent(bookId)}/thumbnail`, mime: row.mime, sha256: row.thumbnail_sha256, width: row.width, height: row.height, kind: row.kind, provenance: parse(row.provenance_json), updatedAt: row.updated_at } : null;
}

export function presentationIdentity(book) {
  const normalized = value => String(value || '').normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim();
  const isbns = (book.metadata?.identifiers || []).flatMap(value => {
    const text = String(value).replace(/^(?:urn:)?isbn\s*:?\s*/i, '');
    if (!/^[\dXx -]+$/.test(text)) return [];
    let digits = text.replace(/[ -]/g, '').toUpperCase();
    if (/^\d{9}[\dX]$/.test(digits)) {
      if ([...digits].reduce((sum, value, i) => sum + (value === 'X' ? 10 : Number(value)) * (10 - i), 0) % 11) return [];
      const prefix = `978${digits.slice(0, 9)}`;
      digits = prefix + String((10 - [...prefix].reduce((sum, value, i) => sum + Number(value) * (i % 2 ? 3 : 1), 0) % 10) % 10);
    }
    if (!/^(?:978|979)\d{10}$/.test(digits) || [...digits].reduce((sum, value, i) => sum + Number(value) * (i % 2 ? 3 : 1), 0) % 10) return [];
    return [digits];
  });
  // Only an unambiguous ISBN plus matching title/authors/edition/date can link
  // presentation across formats. Source IDs and all citation locators stay intact.
  const unique = [...new Set(isbns)];
  const authors = (book.authors || []).map(normalized).filter(Boolean).sort();
  if (unique.length !== 1 || !normalized(book.title) || !authors.length) return { id: book.id, basis: 'source_book' };
  const key = [unique[0], normalized(book.title), authors, normalized(book.metadata?.edition), normalized(book.metadata?.publicationDate)];
  return { id: `edition-${sha(JSON.stringify(key))}`, basis: 'validated_isbn_title_authors_edition_date', isbn: unique[0] };
}

export async function rasterThumbnail(bytes, { kind = 'cover', provenance = {} } = {}) {
  if (bytes.length > 8 * 1024 * 1024) throw new RangeError('Cover source exceeds 8 MiB');
  const raster = bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ||
    (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) || /^GIF8[79]a/.test(bytes.subarray(0, 6).toString('ascii')) ||
    (bytes.subarray(0, 4).toString('ascii') === 'RIFF' && bytes.subarray(8, 12).toString('ascii') === 'WEBP');
  if (!raster) throw new TypeError('Only embedded PNG/JPEG/GIF/WebP raster covers are decoded; external or executable image markup is not loaded');
  const { createCanvas, loadImage } = await import('@napi-rs/canvas');
  const image = await loadImage(bytes);
  if (!Number.isFinite(image.width) || !Number.isFinite(image.height) || image.width < 1 || image.height < 1 || image.width * image.height > 40_000_000) throw new RangeError('Cover dimensions exceed their bound');
  const scale = Math.min(1, 256 / image.width, 384 / image.height);
  const width = Math.max(1, Math.round(image.width * scale)), height = Math.max(1, Math.round(image.height * scale));
  const canvas = createCanvas(width, height);
  canvas.getContext('2d').drawImage(image, 0, 0, width, height);
  const output = canvas.toBuffer('image/png');
  if (output.length > THUMBNAIL_LIMIT) throw new RangeError('Thumbnail exceeds 256 KiB');
  return { base64: output.toString('base64'), mime: 'image/png', width, height, kind,
    provenance: { ...provenance, version: THUMBNAIL_VERSION, method: 'bounded_raster_resize', originalImageSha256: sha(bytes), originalWidth: image.width, originalHeight: image.height } };
}
