import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, mkdir, writeFile, unlink, symlink, chmod } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createCanvas } from '@napi-rs/canvas';
import { readConfig } from '../src/config.mjs';
import { createApp } from '../src/server.mjs';
import { LibraryStore } from '../src/store.mjs';
import { decodeThumbnail, fallbackThumbnail, presentationIdentity, rasterThumbnail, saveBookAssets } from '../src/book-assets.mjs';

async function directory(t) {
  const root = await mkdtemp(join(tmpdir(), 'librarian-assets-'));
  t.after(() => rm(root, { recursive: true, force: true })); return root;
}
function seed(store, id, title, metadata = {}) {
  store.db.prepare('INSERT INTO books(id,title,authors_json,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?)')
    .run(id, title, '["Source author"]', JSON.stringify(metadata), '2026-10-05', '2026-10-05');
}

test('existing books get persistent honest fallback thumbnails without changing book metadata', async t => {
  const root = await directory(t), dataDir = join(root, 'data'), database = join(dataDir, 'library.sqlite');
  let store = new LibraryStore(database, { dataDir });
  seed(store, 'no-cover', '<Book & edition>', { tags: ['keep'], editedAt: 'user', notes: 'User note' });
  const before = store.db.prepare('SELECT * FROM books').all(); store.close();
  store = new LibraryStore(database, { dataDir });
  t.after(() => store.close());
  assert.deepEqual(store.db.prepare('SELECT * FROM books').all(), before);
  const row = store.db.prepare('SELECT * FROM book_assets WHERE book_id=?').get('no-cover');
  assert.equal(row.kind, 'fallback'); assert.equal(row.mime, 'image/svg+xml');
  assert.match(Buffer.from(row.thumbnail).toString(), /No cover available/);
  assert.match(Buffer.from(row.thumbnail).toString(), /&lt;Book &amp; edition&gt;/);
  assert.ok(row.thumbnail.length <= 256 * 1024); assert.equal(row.width, 256); assert.equal(row.height, 384);
});

test('a bounded raster thumbnail is served from SQLite after the original cover file is removed', async t => {
  const root = await directory(t), config = readConfig({ LIBRARIAN_DATA_DIR: join(root, 'data'), LIBRARIAN_PORT: '0', LIBRARIAN_MAINTENANCE: 'disabled' });
  const app = await createApp({ config }); await new Promise(resolve => app.server.listen(0, '127.0.0.1', resolve));
  config.origin = `http://127.0.0.1:${app.server.address().port}`;
  t.after(() => app.close());
  seed(app.store, 'real-cover', 'A real cover');
  const canvas = createCanvas(400, 600); canvas.getContext('2d').fillStyle = '#cc5533'; canvas.getContext('2d').fillRect(0, 0, 400, 600);
  const path = join(root, 'original.png'), original = canvas.toBuffer('image/png'); await writeFile(path, original);
  const thumbnail = await rasterThumbnail(original, { provenance: { sourceSha256: 'real-cover', member: 'OPS/cover.png' } });
  assert.equal(thumbnail.width, 256); assert.equal(thumbnail.height, 384);
  const metadata = { format: 'epub', metadataTree: { children: [{ tag: 'dc:rights', attrs: { lang: 'en' }, children: ['All retained rights'] }, { tag: 'meta', attrs: { property: 'belongs-to-collection', refines: '#title' }, children: ['Collection'] }] } };
  saveBookAssets(app.store, 'real-cover', { thumbnail: decodeThumbnail(thumbnail), sourceMetadata: metadata, provenance: { sourceSha256: 'real-cover' } });
  await unlink(path);
  const response = await fetch(`${config.origin}/api/books/real-cover/thumbnail`); assert.equal(response.status, 200);
  assert.equal(response.headers.get('content-type'), 'image/png');
  const bytes = Buffer.from(await response.arrayBuffer()); assert.deepEqual(bytes, decodeThumbnail(thumbnail).bytes);
  assert.equal(bytes.readUInt32BE(16), 256); assert.equal(bytes.readUInt32BE(20), 384);
  const head = await fetch(`${config.origin}/api/books/real-cover/thumbnail`, { method: 'HEAD' }); assert.equal(head.status, 200); assert.equal((await head.arrayBuffer()).byteLength, 0);
  const cached = await fetch(`${config.origin}/api/books/real-cover/thumbnail`, { headers: { 'if-none-match': response.headers.get('etag') } }); assert.equal(cached.status, 304);
  const detail = await (await fetch(`${config.origin}/api/books/real-cover`)).json(); assert.deepEqual(detail.sourceMetadata, metadata); assert.equal(detail.thumbnail.kind, 'cover');
});

test('catalog, book list and detail choose available managed covers without changing stored assets', async t => {
  const root = await directory(t), config = readConfig({ LIBRARIAN_DATA_DIR: join(root, 'data'), LIBRARIAN_PORT: '0', LIBRARIAN_MAINTENANCE: 'disabled' });
  const retrieval = { ask: async ({ bookId }) => ({ hits: [{ bookId }], citations: [] }) };
  const app = await createApp({ config, retrieval }); await new Promise(resolve => app.server.listen(0, '127.0.0.1', resolve));
  t.after(() => app.close()); const base = `http://127.0.0.1:${app.server.address().port}`;
  config.origin = base;
  const covers = join(config.dataDir, 'covers'); await mkdir(covers, { recursive: true });
  const canvas = createCanvas(2, 3), png = canvas.toBuffer('image/png');
  const legacy = join(covers, 'retained.png'), outside = join(root, 'outside.png');
  await writeFile(legacy, png); await writeFile(outside, png);
  const escape = join(covers, 'escape.png'); await symlink(outside, escape);
  const directoryCover = join(covers, 'directory.png'); await mkdir(directoryCover);
  const unsupported = join(covers, 'unsupported.svg'); await writeFile(unsupported, '<svg/>');
  const unreadable = join(covers, 'unreadable.png'); await writeFile(unreadable, png); await chmod(unreadable, 0o000);
  const scenarios = [
    ['legacy-fallback', legacy, 'fallback', 'legacy_file'],
    ['real-thumbnail', legacy, 'cover', 'stored_thumbnail'],
    ['first-page', legacy, 'first_page', 'stored_thumbnail'],
    ['missing-legacy', join(covers, 'missing.png'), 'fallback', 'stored_thumbnail'],
    ['outside-legacy', outside, 'fallback', 'stored_thumbnail'],
    ['symlink-escape', escape, 'fallback', 'stored_thumbnail'],
    ['directory-legacy', directoryCover, 'fallback', 'stored_thumbnail'],
    ['unsupported-legacy', unsupported, 'fallback', 'stored_thumbnail'],
    ['legacy-no-asset', legacy, null, 'legacy_file'],
    ['no-image', null, null, null],
    ...(process.getuid() !== 0 ? [['unreadable-legacy', unreadable, 'fallback', 'stored_thumbnail']] : []),
  ];
  for (const [id, path, kind] of scenarios) {
    seed(app.store, id, id, { notes: 'Retained note', tags: ['keep'] });
    app.store.db.prepare('UPDATE books SET cover_path=? WHERE id=?').run(path, id);
    if (kind) saveBookAssets(app.store, id, { thumbnail: kind === 'fallback' ? fallbackThumbnail(id) :
      { bytes: png, mime: 'image/png', width: 2, height: 3, kind }, provenance: { state: 'retained' } });
  }
  const before = ['books', 'book_assets'].map(table => app.store.db.prepare(`SELECT * FROM ${table} ORDER BY ${table === 'books' ? 'id' : 'book_id'}`).all());
  const catalog = await (await fetch(`${base}/api/catalog?limit=100`)).json();
  const listed = await (await fetch(`${base}/api/books?limit=100`)).json();
  for (const [id, , kind, source] of scenarios) {
    const detail = await (await fetch(`${base}/api/books/${id}`)).json();
    const response = await fetch(`${base}/api/ask`, { method: 'POST', headers: { origin: base, 'content-type': 'application/json' },
      body: JSON.stringify({ question: 'Source cover', bookId: id }) });
    assert.equal(response.status, 200); const answerBook = (await response.json()).books[0];
    const member = catalog.items.flatMap(card => card.members).find(book => book.id === id);
    const list = listed.items.find(book => book.id === id);
    const url = source ? `/api/books/${id}/${source === 'legacy_file' ? 'cover' : 'thumbnail'}` : null;
    for (const view of [detail, member, list, answerBook]) {
      assert.equal(view.coverUrl, url, id); assert.equal(view.cover?.source || null, source, id);
      assert.equal(view.cover?.kind || null, source === 'legacy_file' ? 'cover' : kind, id);
      assert.equal(view.thumbnail?.kind || null, kind, 'Stored asset identity is independent of selected cover');
    }
    if (url) {
      const response = await fetch(`${base}${url}`); assert.equal(response.status, 200);
      const bytes = Buffer.from(await response.arrayBuffer());
      assert.equal(response.headers.get('content-type'), detail.cover.mime);
      assert.deepEqual(bytes, source === 'legacy_file' || kind !== 'fallback' ? png : fallbackThumbnail(id).bytes);
    }
  }
  assert.deepEqual(['books', 'book_assets'].map(table => app.store.db.prepare(`SELECT * FROM ${table} ORDER BY ${table === 'books' ? 'id' : 'book_id'}`).all()), before);
});

test('format identities require a valid unambiguous ISBN and matching edition facts', () => {
  const base = { title: 'The C Programming Language, second edition', authors: ['Kernighan', 'Ritchie'], metadata: { publicationDate: '1988', identifiers: ['ISBN 0-13-110362-8'] } };
  const a = presentationIdentity({ ...base, id: 'epub' });
  const b = presentationIdentity({ ...base, id: 'pdf', metadata: { ...base.metadata, identifiers: ['urn:isbn:9780131103627'] } });
  assert.equal(a.id, b.id); assert.equal(a.isbn, '9780131103627');
  assert.notEqual(a.id, presentationIdentity({ ...base, id: 'other', title: 'The C Programming Language, first edition' }).id);
  assert.notEqual(a.id, presentationIdentity({ ...base, id: 'other', metadata: { ...base.metadata, publicationDate: '1999' } }).id);
  assert.equal(presentationIdentity({ ...base, id: 'invalid', metadata: { identifiers: ['9780131103628'] } }).id, 'invalid');
  assert.equal(presentationIdentity({ ...base, id: 'missing', authors: [] }).id, 'missing');
});

test('thumbnail bounds and mismatched PNG dimensions reject before a DB write', async t => {
  const root = await directory(t), store = new LibraryStore(join(root, 'data/library.sqlite'), { dataDir: join(root, 'data') }); t.after(() => store.close());
  seed(store, 'bound', 'Bounded book');
  const fallback = fallbackThumbnail('Bounded book');
  assert.throws(() => saveBookAssets(store, 'bound', { thumbnail: { ...fallback, width: 257 } }), /bounded thumbnail/);
  assert.throws(() => saveBookAssets(store, 'bound', { sourceMetadata: { tooLarge: 'x'.repeat(1024 * 1024) }, thumbnail: fallback }), /not silently truncated/);
  const canvas = createCanvas(2, 3), thumbnail = await rasterThumbnail(canvas.toBuffer('image/png'));
  assert.throws(() => decodeThumbnail({ ...thumbnail, width: 9 }), /dimensions disagree/);
  assert.equal(store.db.prepare('SELECT count(*) AS n FROM book_assets').get().n, 0);
});
