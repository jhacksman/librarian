import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { LibraryStore, digest } from '../src/store.mjs';
import { ensureCatalogSchema } from '../src/catalog.mjs';
import { enrichBooks } from '../src/book-enrichment.mjs';
import { fallbackThumbnail, saveBookAssets } from '../src/book-assets.mjs';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'librarian-enrichment-')), dataDir = join(root, 'data');
  const store = new LibraryStore(join(dataDir, 'library.sqlite'), { dataDir }); ensureCatalogSchema(store);
  t.after(async () => { store.close(); await rm(root, { recursive: true, force: true }); });
  await mkdir(join(dataDir, 'sources'));
  const raw = Buffer.from('Original synthetic source'), id = digest(raw), path = join(dataDir, 'sources', `${id}.epub`); await writeFile(path, raw);
  const metadata = { tags: ['user tag'], notes: 'User notes', editedAt: 'user edit', pipelineVersion: 'original text pipeline', toc: ['original TOC'] };
  store.db.prepare('INSERT INTO books(id,title,authors_json,description,language,publisher,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)')
    .run(id, 'User title', '["User author"]', 'User description', 'user language', 'User publisher', JSON.stringify(metadata), '2026-10-05', '2026-10-05');
  store.db.prepare('INSERT INTO files(id,path,source_path,relative_path,sha256,size,format,status,book_id,updated_at,group_key,staged) VALUES(?,?,?,?,?,?,?,?,?,?,?,1)')
    .run('file', path, '/original/not-opened.epub', 'source.epub', id, raw.length, 'epub', 'completed', id, '2026-10-05', 'group candidate');
  store.db.prepare('INSERT INTO sections(id,book_id,ordinal,title,text,locator_json) VALUES(?,?,?,?,?,?)').run('section', id, 1, 'Source', 'Exact preserved 🦊 text', '{"format":"epub","member":"chapter.xhtml"}');
  store.db.prepare('INSERT INTO chunks(id,book_id,section_id,ordinal,text,locator_json,embedding_json,embedding_model,embedding_dimension,embedding_norm) VALUES(?,?,?,?,?,?,?,?,?,?)')
    .run('chunk', id, 'section', 0, 'Exact preserved 🦊 text', '{"charStart":0,"charEnd":24}', '[1,0]', 'fixed embedding', 2, 1);
  store.db.prepare('INSERT INTO reading_progress(book_id,section_id,updated_at) VALUES(?,?,?)').run(id, 'section', 'saved progress');
  saveBookAssets(store, id, { thumbnail: fallbackThumbnail('User title'), provenance: { state: 'pending_source_enrichment' } });
  const protectedRows = () => Object.fromEntries(['sections', 'chunks', 'chunks_fts', 'reading_progress', 'files', 'group_members', 'catalog_groups', 'catalog_group_events'].map(table => [table, store.db.prepare(`SELECT * FROM ${table}`).all()]));
  return { store, id, raw, path, metadata, protectedRows };
}

test('metadata-only enrichment preserves edited display fields, source text, vectors, reading state and groups', async t => {
  const f = await fixture(t), before = f.protectedRows(); let calls = 0;
  const result = await enrichBooks(f.store, { worker: async request => {
    calls++; assert.equal(request.operation, 'extractMetadata'); assert.equal(request.path, f.path);
    return { ok: true, book: { title: 'Source title', authors: ['Source author'], description: 'Source description', language: 'en', publisher: 'Source publisher', warnings: [],
      metadata: { identifiers: ['9780131103627'], publicationDate: '1988', sourceMetadata: { metadataTree: { children: [{ tag: 'dc:rights', children: ['Retained rights'] }] } } } } };
  } });
  assert.equal(calls, 1); assert.equal(result.completed, 1); assert.equal(result.failed, 0); assert.equal(result.remaining, 0);
  assert.deepEqual(f.protectedRows(), before);
  const row = f.store.db.prepare('SELECT * FROM books WHERE id=?').get(f.id); assert.equal(row.title, 'User title'); assert.equal(row.authors_json, '["User author"]'); assert.equal(row.description, 'User description'); assert.equal(row.publisher, 'User publisher'); assert.equal(row.language, 'user language');
  const metadata = JSON.parse(row.metadata_json); for (const key of ['tags', 'notes', 'editedAt', 'pipelineVersion', 'toc']) assert.deepEqual(metadata[key], f.metadata[key]);
  assert.equal(metadata.sourceMetadata.descriptive.title, 'Source title'); assert.equal(metadata.sourceMetadata.metadataTree.children[0].children[0], 'Retained rights');
  assert.equal(JSON.parse(f.store.db.prepare('SELECT provenance_json FROM book_assets WHERE book_id=?').get(f.id).provenance_json).sourceSha256, f.id);
  const second = await enrichBooks(f.store, { worker: async () => { throw new Error('Already-enriched source must not be parsed again'); } }); assert.equal(second.processed, 0);
});

test('changed source refuses extraction, records honest failure and does not starve later batches', async t => {
  const f = await fixture(t), before = f.protectedRows(); await writeFile(f.path, 'Changed bytes'); let calls = 0;
  const result = await enrichBooks(f.store, { worker: async () => { calls++; throw new Error('Must not reach parser'); } });
  assert.equal(calls, 0); assert.equal(result.failed, 1); assert.equal(result.pending, 0); assert.equal(result.failedBooks, 1); assert.equal(result.remaining, 1);
  assert.deepEqual(f.protectedRows(), before);
  const asset = f.store.db.prepare('SELECT * FROM book_assets WHERE book_id=?').get(f.id); assert.equal(asset.kind, 'fallback'); assert.match(JSON.parse(asset.provenance_json).error, /identity changed/);
  const next = await enrichBooks(f.store); assert.equal(next.processed, 0);
});


test('converted MOBI enrichment uses verified retained EPUB and retains original conversion provenance', async t => {
  const f = await fixture(t), derived = Buffer.from('Verified synthetic derived EPUB');
  const derivedPath = join(f.store.dataDir, 'sources', 'retained-conversion.epub'); await writeFile(derivedPath, derived);
  const conversion = { converter: { name: 'existing local converter' }, original: { format: 'mobi', sha256: f.id },
    derived: { path: derivedPath, sha256: digest(derived), bytes: derived.length } };
  f.store.db.prepare('UPDATE files SET format=? WHERE book_id=?').run('mobi', f.id);
  f.store.db.prepare('UPDATE books SET metadata_json=? WHERE id=?').run(JSON.stringify({ ...f.metadata, conversion }), f.id);
  const before = f.protectedRows();
  const result = await enrichBooks(f.store, { worker: async request => {
    assert.equal(request.operation, 'extractMetadata'); assert.equal(request.format, 'epub'); assert.equal(request.path, derivedPath);
    return { ok: true, book: { title: 'Derived title', authors: [], warnings: [], metadata: { sourceMetadata: { packagePath: 'book.opf' } } } };
  } });
  assert.equal(result.completed, 1); assert.deepEqual(f.protectedRows(), before);
  const source = JSON.parse(f.store.db.prepare('SELECT source_metadata_json FROM book_assets WHERE book_id=?').get(f.id).source_metadata_json);
  assert.equal(source.format, 'mobi'); assert.equal(source.derivedFormat, 'epub'); assert.deepEqual(source.conversion, conversion);
  const provenance = JSON.parse(f.store.db.prepare('SELECT provenance_json FROM book_assets WHERE book_id=?').get(f.id).provenance_json);
  assert.equal(provenance.managedSource.sha256, digest(derived)); assert.equal(provenance.managedSource.originalSourceSha256, f.id);
});

test('a user edit arriving during extraction survives the write transaction with its latest notes', async t => {
  const f = await fixture(t), before = f.protectedRows();
  const result = await enrichBooks(f.store, {worker: async () => {
    f.store.db.prepare('UPDATE books SET title=?,authors_json=?,metadata_json=? WHERE id=?')
      .run('Latest user title', '["Latest user author"]', JSON.stringify({...f.metadata, notes: 'Latest user notes', editedAt: 'edit during extraction'}), f.id);
    return {ok: true, book: {title: 'Extracted title', authors: ['Extracted author'], warnings: [], metadata: {sourceMetadata: {raw: 'retained'}}}};
  }});
  assert.equal(result.completed, 1); const row = f.store.db.prepare('SELECT * FROM books WHERE id=?').get(f.id);
  assert.equal(row.title, 'Latest user title'); assert.equal(row.authors_json, '["Latest user author"]');
  assert.equal(JSON.parse(row.metadata_json).notes, 'Latest user notes'); assert.deepEqual(f.protectedRows(), before);
});

test('reviewed page recovery changes only deficient display fields and keeps asset bytes and protected data', async t => {
  const {captureFrontMatterPage, frontMatterEvidence} = await import('../src/source-metadata-recovery.mjs');
  const f = await fixture(t); f.store.db.prepare('UPDATE files SET format=? WHERE book_id=?').run('pdf', f.id);
  f.store.db.prepare('UPDATE books SET title=?,authors_json=?,metadata_json=? WHERE id=?').run('9780760350799.pdf', '[]', JSON.stringify({notes: 'Keep notes'}), f.id);
  const text = 'Home Repair\nExample Editor', capture = frontMatterEvidence(f.id, [captureFrontMatterPage(1, text, [])], 1);
  const field = value => ({value, evidence: [{page: 1, pageTextSha256: capture.pages[0].textSha256,
    start: text.indexOf(value), end: text.indexOf(value) + value.length, quote: value}]});
  const review = {schema: 'librarian-reviewed-source-metadata/v1', sourceSha256: f.id, reviewer: 'Independent fixture reviewer',
    evidenceReceiptSha256: 'a'.repeat(64), fields: {title: field('Home Repair'), authors: [field('Example Editor')]}};
  const assetBefore = f.store.db.prepare('SELECT thumbnail,thumbnail_sha256,kind FROM book_assets WHERE book_id=?').get(f.id), before = f.protectedRows();
  const result = await enrichBooks(f.store, {bookIds: [f.id], recoverMetadata: true, reviewedMetadata: {[f.id]: review}, worker: async request => {
    assert.equal(request.frontMatter, true);
    return {ok: true, book: {title: '9780760350799.pdf', authors: [], warnings: [], metadata: {sourceMetadata: {info: {Title: '9780760350799.pdf'}, frontMatter: capture}}}};
  }});
  assert.equal(result.completed, 1); const row = f.store.db.prepare('SELECT * FROM books WHERE id=?').get(f.id);
  assert.equal(row.title, 'Home Repair'); assert.equal(row.authors_json, '["Example Editor"]');
  const metadata = JSON.parse(row.metadata_json); assert.equal(metadata.notes, 'Keep notes');
  assert.equal(metadata.sourceMetadata.info.Title, '9780760350799.pdf'); assert.equal(metadata.sourceMetadata.recovery.state, 'reviewed_complete');
  assert.deepEqual(f.protectedRows(), before);
  assert.deepEqual(f.store.db.prepare('SELECT thumbnail,thumbnail_sha256,kind FROM book_assets WHERE book_id=?').get(f.id), assetBefore);
});

test('invalid reviewed recovery leaves books and existing assets unchanged', async t => {
  const f = await fixture(t), before = f.protectedRows(), book = f.store.db.prepare('SELECT * FROM books WHERE id=?').get(f.id), asset = f.store.db.prepare('SELECT * FROM book_assets WHERE book_id=?').get(f.id);
  const result = await enrichBooks(f.store, {bookIds: [f.id], recoverMetadata: true, reviewedMetadata: {[f.id]: {sourceSha256: 'wrong source'}}, worker: async () => {throw new Error('Non-PDF recovery must not reach parser');}});
  assert.equal(result.failed, 1); assert.deepEqual(f.protectedRows(), before);
  assert.deepEqual(f.store.db.prepare('SELECT * FROM books WHERE id=?').get(f.id), book);
  assert.deepEqual(f.store.db.prepare('SELECT * FROM book_assets WHERE book_id=?').get(f.id), asset);
});

test('frozen gold application preserves unrelated metadata, cover bytes and late user edits', async t => {
  const {parseFrozenGold, verifyFrozenGoldRenders} = await import('../src/frozen-gold-recovery.mjs');
  const {captureFrontMatterPage, frontMatterEvidence} = await import('../src/source-metadata-recovery.mjs');
  const f = await fixture(t), text = 'BLACK Home Repair\nCreated by: Example Editors\nFourth Edition';
  const page = captureFrontMatterPage(1, text, []), capture = frontMatterEvidence(f.id, [page], 1);
  const field = (value, quote) => ({state: 'supported', value, support: [{source_sha256: f.id,
    physical_page_number: 1, capture_mime_type: 'application/json', native_text_sha256: page.textSha256, exact_quote: quote}]});
  const bytes = Buffer.from(JSON.stringify({schema: 'independent-native-page-metadata-gold-set/v1', records: [{
    source_sha256: f.id, catalog_book_id: f.id, physical_total_pages: 1, pages: [{physical_page_number: 1, native_text_utf8: {sha256: page.textSha256}}],
    title: field('Home Repair', 'BLACK Home Repair'), authors: {state: 'not_observed', value: null, support: []},
    creators: field([{name: 'Example Editors', role: 'Created by'}], 'Created by: Example Editors'),
    edition: field('Fourth Edition', 'Fourth Edition')}]}));
  const gold = parseFrozenGold(bytes, digest(bytes)); await verifyFrozenGoldRenders(gold, '/unused');
  f.store.db.prepare('UPDATE files SET format=? WHERE book_id=?').run('pdf', f.id);
  const metadataBefore = {notes: 'Keep notes', tags: ['Keep tag'], identifiers: ['Keep identifier'],
    sourceMetadata: {info: {Title: 'Keep embedded title'}, unknown: {keep: true}}};
  f.store.db.prepare('UPDATE books SET title=?,authors_json=?,metadata_json=? WHERE id=?')
    .run('9780760350799.pdf', '[]', JSON.stringify(metadataBefore), f.id);
  const assetsBefore = f.store.db.prepare('SELECT thumbnail,thumbnail_sha256,mime,width,height,kind FROM book_assets WHERE book_id=?').get(f.id);
  const protectedBefore = f.protectedRows(), bookBefore = f.store.db.prepare('SELECT * FROM books WHERE id=?').get(f.id);
  const worker = async request => {
    assert.equal(request.outputDir, undefined); assert.equal(request.frontMatter, true);
    return {ok: true, book: {title: 'Wrong parsed title', authors: ['Wrong parser author'], description: 'Wrong description',
      publisher: 'Wrong publisher', language: 'Wrong language', warnings: [], metadata: {edition: 'Wrong edition',
        sourceMetadata: {info: {Title: 'Wrong embedded title'}, frontMatter: capture}}}};
  };
  const options = {bookIds: [f.id], recoverMetadata: true, frozenGoldRecords: {[f.id]: gold.records[0]}, worker};
  assert.equal((await enrichBooks(f.store, options)).completed, 1);
  const actual = f.store.db.prepare('SELECT * FROM books WHERE id=?').get(f.id), metadata = JSON.parse(actual.metadata_json);
  assert.equal(actual.title, 'Home Repair'); assert.equal(actual.authors_json, '[]'); assert.equal(metadata.edition, 'Fourth Edition');
  assert.deepEqual(metadata.creators, [{name: 'Example Editors', role: 'Created by'}]);
  for (const name of ['notes', 'tags', 'identifiers']) assert.deepEqual(metadata[name], metadataBefore[name]);
  assert.deepEqual(metadata.sourceMetadata.info, metadataBefore.sourceMetadata.info);
  assert.deepEqual(metadata.sourceMetadata.unknown, metadataBefore.sourceMetadata.unknown);
  for (const name of ['description', 'publisher', 'language', 'created_at']) assert.equal(actual[name], bookBefore[name]);
  assert.deepEqual(f.protectedRows(), protectedBefore);
  assert.deepEqual(f.store.db.prepare('SELECT thumbnail,thumbnail_sha256,mime,width,height,kind FROM book_assets WHERE book_id=?').get(f.id), assetsBefore);
  assert.equal((await enrichBooks(f.store, {...options, worker: async request => {
    f.store.db.prepare('UPDATE books SET title=?,metadata_json=? WHERE id=?').run('Late user title',
      JSON.stringify({...metadata, editedAt: 'during parser', edition: 'User edition', creators: [{name: 'User', role: 'editor'}]}), f.id);
    return worker(request);
  }})).completed, 1);
  const edited = f.store.db.prepare('SELECT * FROM books WHERE id=?').get(f.id);
  assert.equal(edited.title, 'Late user title'); assert.equal(JSON.parse(edited.metadata_json).edition, 'User edition');
  assert.deepEqual(JSON.parse(edited.metadata_json).creators, [{name: 'User', role: 'editor'}]);
});
