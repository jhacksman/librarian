// Execute only in the existing approved Spark Node24 runtime.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {parseFrozenGold, verifyFrozenGoldRenders, recoverFrozenGold} from '../src/frozen-gold-recovery.mjs';
import {captureFrontMatterPage, frontMatterEvidence} from '../src/source-metadata-recovery.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex'), id = 'a'.repeat(64);
const text = 'BLACK & DECKER Home Repair\nCreated by: Editors of Example Press\nFourth Edition';
function fixture({visual = false, author = false} = {}) {
  const page = captureFrontMatterPage(1, text, []), image = Buffer.from([137,80,78,71,13,10,26,10,0]);
  const support = quote => ({source_sha256: id, physical_page_number: 1,
    capture_mime_type: 'application/json', native_text_sha256: page.textSha256, exact_quote: quote});
  const field = (value, quote) => ({state: 'supported', value, support: [support(quote)]});
  const record = {source_sha256: id, catalog_book_id: id, physical_total_pages: 1,
    pages: [{physical_page_number: 1, native_text_utf8: {sha256: page.textSha256},
      ...(visual ? {rendered_full_page: {sha256: hash(image), bytes: image.length}} : {})}],
    title: field('Home Repair', 'BLACK & DECKER Home Repair'),
    authors: {state: 'not_observed', value: null, support: []},
    creators: field([{name: 'Editors of Example Press', role: 'Created by'}], 'Created by: Editors of Example Press'),
    edition: field('Fourth Edition', 'Fourth Edition')};
  if (author) record.authors = {state: 'supported', value: ['Ada Example'], support: [{source_sha256: id,
    physical_page_number: 1, capture_mime_type: 'image/png', page_sha256: hash(image), exact_quote: 'Ada Example'}],
    role_basis: 'conventional cover byline; no literal role label'};
  const bytes = Buffer.from(JSON.stringify({schema: 'independent-native-page-metadata-gold-set/v1', records: [record]}));
  // Synthetic review authority is explicit; the actual CLI has no digest override.
  const gold = parseFrozenGold(bytes, hash(bytes));
  return {gold, record: gold.records[0], capture: frontMatterEvidence(id, [page], 1), bytes, image};
}
const before = {title: '9780760350799.pdf', authors: []};

test('frozen gold keeps normalized titles, creator roles and editions separate from unknown authors', async () => {
  const f = fixture(); await verifyFrozenGoldRenders(f.gold, '/unused');
  const result = recoverFrozenGold(before, f.capture, f.record);
  assert.deepEqual(result.display, {title: 'Home Repair', authors: []});
  assert.deepEqual(result.metadataUpdates, {creators: [{name: 'Editors of Example Press', role: 'Created by'}], edition: 'Fourth Edition'});
  assert.deepEqual(result.receipt.unresolvedFields, ['authors']);
  assert.equal(result.receipt.fields.creators.support[0].exact_quote, 'Created by: Editors of Example Press');
});

test('frozen gold preserves every user edit and existing meaningful edition and creator credit', async () => {
  const f = fixture(); await verifyFrozenGoldRenders(f.gold, '/unused');
  const existing = {...before, edition: 'User edition', creators: [{name: 'User creator', role: 'editor'}]};
  assert.deepEqual(recoverFrozenGold(existing, f.capture, f.record).metadataUpdates, {});
  const result = recoverFrozenGold({...existing, editedAt: 'late edit'}, f.capture, f.record);
  assert.deepEqual(result.display, before); assert.deepEqual(result.metadataUpdates, {});
  assert.deepEqual(result.receipt.preservedFields, ['title', 'creators', 'edition']);
});

test('changed gold bytes, unapproved records and changed native quotes are rejected', async () => {
  const f = fixture();
  assert.throws(() => parseFrozenGold(Buffer.concat([f.bytes, Buffer.from(' ')]), hash(f.bytes)), /digest/);
  assert.throws(() => recoverFrozenGold(before, f.capture, structuredClone(f.record)), /Verified/);
  await verifyFrozenGoldRenders(f.gold, '/unused');
  const changed = frontMatterEvidence(id, [captureFrontMatterPage(1, 'Different source page', [])], 1);
  assert.throws(() => recoverFrozenGold(before, changed, f.record), /page identity/);
  assert.throws(() => {f.record.title.value = 'Invented title';}, TypeError);
});

test('visual bylines require exact frozen PNG custody and retain their disclosed role interpretation', async t => {
  const f = fixture({visual: true, author: true}), root = await mkdtemp(join(tmpdir(), 'gold-render-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const path = join(root, `${hash(f.image)}.png`); await writeFile(path, Buffer.alloc(f.image.length));
  await assert.rejects(verifyFrozenGoldRenders(f.gold, root), /digest/);
  assert.throws(() => recoverFrozenGold(before, f.capture, f.record), /Verified/);
  await writeFile(path, f.image); await verifyFrozenGoldRenders(f.gold, root);
  const result = recoverFrozenGold(before, f.capture, f.record);
  assert.deepEqual(result.display.authors, ['Ada Example']);
  assert.equal(result.receipt.fields.authors.roleBasis, 'conventional cover byline; no literal role label');
});
