import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { CATALOG_POLICY_VERSION, catalogEntries, ensureCatalogSchema, groupSuggestions, linkEditions, unlinkEdition } from '../src/catalog.mjs';
import { LibraryStore, digest } from '../src/store.mjs';

async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), 'librarian-catalog-'));
  const context = { directory, store: new LibraryStore(join(directory, 'state', 'library.sqlite')) };
  t.after(async () => { context.store.close(); await rm(directory, { recursive: true, force: true }); });
  return context;
}

function book(store, key, { title = 'Practical Systems', authors = ['Ada Example'], metadata = {},
  format = 'epub', folderStem = key, copies = 1 } = {}) {
  const id = digest(key), stamp = '2026-10-04T12:00:00.000Z';
  store.db.prepare('INSERT INTO books(id,title,authors_json,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?)')
    .run(id, title, JSON.stringify(authors), JSON.stringify(metadata), stamp, stamp);
  for (let copy = 0; copy < copies; copy++) {
    const fileId = digest(`${key}:${copy}`);
    store.db.prepare(`INSERT INTO files(id,path,source_path,relative_path,sha256,size,format,status,book_id,updated_at,group_key,staged)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,1)`).run(fileId, join(store.dataDir, 'sources', `${id}.${format}`),
      `/immutable/${key}-${copy}.${format}`, `Bundle/${key}-${copy}.${format}`, id, 100, format, copy ? 'duplicate' : 'completed', id, stamp, folderStem);
  }
  const locator = format === 'pdf' ? { format, page: 3 } : { format, member: 'chapters/amber.xhtml' };
  const sectionId = `${id}:source-bound-section`;
  const text = `${key}: original 🧭 evidence.`;
  store.db.prepare('INSERT INTO sections(id,book_id,ordinal,title,text,locator_json) VALUES(?,?,1,?,?,?)')
    .run(sectionId, id, 'Source evidence', text, JSON.stringify(locator));
  store.db.prepare('INSERT INTO chunks(id,book_id,section_id,ordinal,text,locator_json,embedding_json) VALUES(?,?,?,0,?,?,?)')
    .run(`${sectionId}:chunk`, id, sectionId, text, JSON.stringify({ ...locator, charStart: 0, charEnd: text.length, offsetBasis: 'section_utf16_code_units' }), '[1,0]');
  store.db.prepare('INSERT INTO reading_progress(book_id,section_id,updated_at) VALUES(?,?,?)').run(id, sectionId, stamp);
  return id;
}

function sourceRows(store) {
  return Object.fromEntries(['books', 'files', 'sections', 'chunks', 'chunks_fts', 'reading_progress']
    .map((table) => [table, store.db.prepare(`SELECT * FROM ${table} ORDER BY rowid`).all()]));
}

function relationships(store) {
  return Object.fromEntries(['catalog_groups', 'group_members', 'catalog_group_events']
    .map((table) => [table, store.db.prepare(`SELECT * FROM ${table} ORDER BY rowid`).all()]));
}

test('same folder stem produces an explained suggestion, never an automatic identity claim', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'first', { title: 'Cache Operations', authors: ['A. Author'], folderStem: 'candidate-17' });
  const b = book(store, 'second', { title: 'Cooking with Steam', authors: ['B. Author'], folderStem: 'candidate-17', format: 'pdf' });
  const before = sourceRows(store);
  const suggestions = groupSuggestions(store);
  assert.equal(suggestions.automaticLinks, false);
  assert.equal(suggestions.total, 1);
  const candidate = suggestions.items[0];
  assert.deepEqual(new Set(candidate.bookIds), new Set([a, b]));
  assert.equal(candidate.signal, 'filename_only');
  assert.equal(candidate.requiresReview, true);
  assert.deepEqual(candidate.formats, { epub: 1, pdf: 1 });
  assert.deepEqual(candidate.evidence.map((item) => item.kind), ['same_folder_stem']);
  assert.ok(candidate.cautions.some((text) => text.includes('Titles differ')));
  assert.ok(candidate.cautions.some((text) => text.includes('Author metadata differs')));
  assert.equal(store.db.prepare('SELECT count(*) AS n FROM group_members').get().n, 0);
  assert.equal(store.db.prepare('SELECT count(*) AS n FROM catalog_group_events').get().n, 0);
  assert.equal(catalogEntries(store).total, 2);
  assert.deepEqual(sourceRows(store), before);
});

test('exact reported identifiers and normalized title/authors corroborate without asserting proof', async (t) => {
  const { store } = await fixture(t);
  const identifier = 'urn:isbn:9780132350884';
  const a = book(store, 'epub', { title: 'Practical: Systems', authors: ['Ada Example', 'Bo Reader'], metadata: { identifiers: [identifier, identifier] } });
  const b = book(store, 'pdf', { title: 'PRACTICAL systems', authors: ['bo reader', 'ada example'], metadata: { identifiers: [identifier] }, format: 'pdf' });
  const candidate = groupSuggestions(store).items[0];
  assert.deepEqual(new Set(candidate.bookIds), new Set([a, b]));
  assert.equal(candidate.signal, 'multiple_metadata_signals');
  assert.deepEqual(candidate.evidence.map((item) => item.kind), ['shared_identifier', 'title_and_authors']);
  assert.equal(candidate.evidence[0].value, identifier);
  assert.equal(candidate.agreement.title, true);
  assert.equal(candidate.agreement.authors, true);
  assert.ok(candidate.cautions.some((text) => text.includes('do not prove edition identity')));
  assert.equal(candidate.requiresReview, true);
  assert.equal(catalogEntries(store).counts.confirmedGroups, 0);
});

test('explicit edition differences remain visible when title and authors suggest a logical work', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'earlier', { title: 'Practical Systems, Second Edition' });
  const b = book(store, 'later', { title: 'Practical Systems, 3rd edition', format: 'pdf' });
  const suggestion = groupSuggestions(store).items[0];
  assert.deepEqual(suggestion.agreement.editionTokens, ['2', '3']);
  assert.ok(suggestion.cautions.some((text) => text.includes('Different explicit edition numbers')));
  assert.equal(suggestion.requiresReview, true);
  const linked = linkEditions(store, { bookIds: [a, b], title: 'Practical Systems — editions' });
  assert.equal(linked.group.sourceBookCount, 2);
  assert.deepEqual(new Set(linked.group.members.map((member) => member.title)), new Set(['Practical Systems, Second Edition', 'Practical Systems, 3rd edition']));
  assert.equal(groupSuggestions(store).total, 0);
});

test('missing or malformed authors and identifiers cannot corroborate identical generic titles', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'bad-one', { title: 'Guide', authors: [], metadata: { identifiers: ['unknown', '', null, 123, 'none'] } });
  const b = book(store, 'bad-two', { title: 'Guide', authors: [], metadata: { identifiers: { value: 'urn:example:shared' } } });
  store.db.prepare('UPDATE books SET metadata_json=?,authors_json=? WHERE id=?').run('{broken', 'null', a);
  store.db.prepare('UPDATE books SET authors_json=? WHERE id=?').run('{"name":"Ada"}', b);
  assert.equal(groupSuggestions(store).total, 0);
  assert.equal(catalogEntries(store).total, 2);
});

test('overlapping evidence stays separate instead of joining unrelated sources transitively', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'a', { title: 'First', authors: [], metadata: { identifiers: ['urn:example:shared'] }, folderStem: 'a' });
  const b = book(store, 'b', { title: 'Second', authors: [], metadata: { identifiers: ['urn:example:shared'] }, folderStem: 'shared-stem' });
  const c = book(store, 'c', { title: 'Third', authors: [], folderStem: 'shared-stem' });
  const result = groupSuggestions(store);
  assert.equal(result.total, 2);
  assert.ok(result.items.every((candidate) => candidate.bookIds.length === 2));
  assert.deepEqual(new Set(result.items.map((candidate) => [...candidate.bookIds].sort().join(','))),
    new Set([[a, b].sort().join(','), [b, c].sort().join(',')]));
  assert.equal(new Set(result.items.map((candidate) => candidate.id)).size, 2);
  assert.deepEqual(groupSuggestions(store, { limit: 1, offset: 1 }).items, [result.items[1]]);
  assert.equal(catalogEntries(store).total, 3);
});

test('accepted grouping persists, repeated acceptance is idempotent, and citations/edits/progress remain exact', async (t) => {
  const context = await fixture(t);
  const a = book(context.store, 'edited', { title: 'Human title', metadata: { editedAt: '2026-10-03', tags: ['keep'], notes: 'Reading note' } });
  const b = book(context.store, 'other-format', { format: 'pdf', copies: 2 });
  const before = sourceRows(context.store);
  const linked = linkEditions(context.store, { bookIds: [a, b], title: 'My confirmed work' });
  assert.equal(linked.changed, true);
  assert.equal(linked.group.version, 1);
  assert.equal(linked.group.formatEntryCount, 3);
  assert.deepEqual(linked.group.formats, { epub: 1, pdf: 2 });
  assert.equal(linked.group.uniqueFormatCount, 2);
  assert.equal(linked.group.members.find((member) => member.id === b).firstSection.locator.page, 3);
  assert.equal(linked.group.members.find((member) => member.id === a).firstSection.locator.member, 'chapters/amber.xhtml');
  const event = context.store.db.prepare('SELECT * FROM catalog_group_events').get();
  assert.equal(event.action, 'created');
  assert.equal(JSON.parse(event.details_json).origin, 'explicit_user_selection');
  assert.equal(JSON.parse(event.details_json).policyVersion, CATALOG_POLICY_VERSION);
  const repeat = linkEditions(context.store, { bookIds: [b, a], title: 'My confirmed work' });
  assert.equal(repeat.changed, false);
  assert.equal(repeat.group.id, linked.group.id);
  assert.equal(context.store.db.prepare('SELECT count(*) AS n FROM catalog_group_events').get().n, 1);
  assert.deepEqual(sourceRows(context.store), before);
  const path = context.store.path;
  context.store.close();
  context.store = new LibraryStore(path);
  ensureCatalogSchema(context.store);
  ensureCatalogSchema(context.store);
  const catalog = catalogEntries(context.store);
  assert.equal(catalog.items[0].id, linked.group.id);
  assert.equal(catalog.items[0].title, 'My confirmed work');
  assert.deepEqual(catalog.counts, { catalogEntries: 1, confirmedGroups: 1, ungroupedBooks: 0, sourceBooks: 2, matchingSourceBooks: 2, formatEntries: 3 });
  assert.deepEqual(sourceRows(context.store), before);
});

test('unlink only removes the requested membership and retains an auditable empty group', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'a'), b = book(store, 'b');
  const before = sourceRows(store);
  const { group } = linkEditions(store, { bookIds: [a, b] });
  const removed = unlinkEdition(store, a);
  assert.equal(removed.changed, true);
  assert.deepEqual(removed.group.bookIds, [b]);
  assert.equal(removed.group.version, 2);
  assert.equal(catalogEntries(store).total, 2);
  assert.equal(unlinkEdition(store, a).changed, false);
  const last = unlinkEdition(store, b);
  assert.deepEqual(last.group.bookIds, []);
  assert.equal(last.group.version, 3);
  assert.equal(catalogEntries(store).counts.confirmedGroups, 0);
  assert.equal(catalogEntries(store).total, 2);
  const events = store.db.prepare('SELECT version,action,book_ids_json,details_json FROM catalog_group_events WHERE group_id=? ORDER BY version').all(group.id);
  assert.deepEqual(events.map((event) => [event.version, event.action]), [[1, 'created'], [2, 'unlinked'], [3, 'unlinked']]);
  assert.deepEqual(JSON.parse(events[1].book_ids_json), [b]);
  assert.equal(JSON.parse(events[1].details_json).removedBookId, a);
  assert.deepEqual(JSON.parse(events[2].book_ids_json), []);
  assert.deepEqual(sourceRows(store), before);
});

test('moving a partial existing group fails before any relation or user data changes', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'a', { title: 'One' }), b = book(store, 'b', { title: 'Two' }), c = book(store, 'c', { title: 'Three' });
  linkEditions(store, { bookIds: [a, b] });
  const before = relationships(store), sources = sourceRows(store);
  assert.throws(() => linkEditions(store, { bookIds: [a, c] }), /every member/);
  assert.deepEqual(relationships(store), before);
  assert.deepEqual(sourceRows(store), sources);
  const updated = linkEditions(store, { bookIds: [a, b, c], title: 'Confirmed series' });
  assert.equal(updated.group.sourceBookCount, 3);
  assert.equal(updated.group.version, 2);
});

test('explicit complete-group merges preserve both histories and source identities', async (t) => {
  const { store } = await fixture(t);
  const ids = ['a', 'b', 'c', 'd'].map((key) => book(store, key));
  const first = linkEditions(store, { bookIds: ids.slice(0, 2), title: 'First accepted group' }).group;
  const second = linkEditions(store, { bookIds: ids.slice(2), title: 'Second accepted group' }).group;
  const before = sourceRows(store);
  const result = linkEditions(store, { bookIds: ids, title: 'Combined by the user' });
  assert.equal(result.mergedGroupIds.length, 1);
  assert.deepEqual(new Set([result.group.id, ...result.mergedGroupIds]), new Set([first.id, second.id]));
  assert.equal(result.group.version, 2);
  assert.equal(catalogEntries(store).total, 1);
  const retired = result.mergedGroupIds[0];
  const event = store.db.prepare('SELECT * FROM catalog_group_events WHERE group_id=? ORDER BY version DESC LIMIT 1').get(retired);
  assert.equal(event.action, 'merged_into');
  assert.equal(JSON.parse(event.details_json).targetGroupId, result.group.id);
  assert.equal(JSON.parse(event.details_json).previousBookIds.length, 2);
  assert.deepEqual(JSON.parse(event.book_ids_json), []);
  assert.equal(store.db.prepare('SELECT count(*) AS n FROM catalog_group_events').get().n, 4);
  assert.deepEqual(sourceRows(store), before);
});

test('a failed audit write rolls back the complete link transaction', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'a'), b = book(store, 'b'), c = book(store, 'c');
  linkEditions(store, { bookIds: [a, b] });
  const before = relationships(store), sources = sourceRows(store);
  store.db.exec("CREATE TRIGGER reject_catalog_event BEFORE INSERT ON catalog_group_events BEGIN SELECT RAISE(ABORT,'synthetic audit failure'); END");
  assert.throws(() => linkEditions(store, { bookIds: [a, b, c], title: 'Must roll back' }), /synthetic audit failure/);
  assert.deepEqual(relationships(store), before);
  assert.deepEqual(sourceRows(store), sources);
});

test('unknown, duplicate and malformed selections fail without changing accepted relationships', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'a'), b = book(store, 'b');
  linkEditions(store, { bookIds: [a, b] });
  const before = relationships(store);
  for (const bookIds of [[a], [a, a], [a, ''], [a, null], [a, digest('missing')]]) {
    assert.throws(() => linkEditions(store, { bookIds }));
  }
  assert.throws(() => linkEditions(store, { bookIds: [a, b], title: ' ' }));
  assert.throws(() => unlinkEdition(store, digest('missing')), /does not exist/);
  assert.deepEqual(relationships(store), before);
  assert.throws(() => groupSuggestions(store, { limit: 201 }), RangeError);
  assert.throws(() => catalogEntries(store, { offset: -1 }), RangeError);
  assert.throws(() => catalogEntries(store, { format: [] }), TypeError);
});

test('filtering keeps the complete group card while identifying matching formats and preserving pagination', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'epub-a', { title: 'First text', authors: ['Ada'], metadata: { tags: ['systems'] } });
  const b = book(store, 'pdf-b', { title: 'Second text', authors: ['Bo'], format: 'pdf' });
  const c = book(store, 'third', { title: 'Zebra Notes', authors: ['Ada'], format: 'pdf' });
  const { group } = linkEditions(store, { bookIds: [a, b], title: 'Amber Work' });
  const filtered = catalogEntries(store, { format: 'PDF', q: 'Amber' });
  assert.equal(filtered.total, 1);
  assert.equal(filtered.items[0].id, group.id);
  assert.deepEqual(filtered.items[0].matchingBookIds, [b]);
  assert.deepEqual(new Set(filtered.items[0].bookIds), new Set([a, b]));
  assert.equal(filtered.counts.sourceBooks, 2);
  assert.equal(filtered.counts.matchingSourceBooks, 1);
  assert.equal(catalogEntries(store, { format: 'pdf', author: 'Ada', q: 'Amber' }).total, 0);
  assert.deepEqual(catalogEntries(store, { tag: 'systems' }).items[0].matchingBookIds, [a]);
  const page = catalogEntries(store, { limit: 1, offset: 1 });
  assert.equal(page.total, 2);
  assert.equal(page.items[0].id, c);
  assert.equal(catalogEntries(store, { q: 'no such title' }).total, 0);
  const serialized = JSON.stringify(catalogEntries(store));
  assert.ok(!serialized.includes('/immutable/'));
  assert.ok(!serialized.includes(store.dataDir));
});

test('suggestions disclose extra existing group members required for an explicit complete merge', async (t) => {
  const { store } = await fixture(t);
  const a = book(store, 'a', { title: 'Alpha', authors: [], metadata: { identifiers: ['urn:example:shared'] } });
  const b = book(store, 'b', { title: 'Beta', authors: [] });
  const c = book(store, 'c', { title: 'Gamma', authors: [], metadata: { identifiers: ['urn:example:shared'] } });
  linkEditions(store, { bookIds: [a, b] });
  const candidate = groupSuggestions(store).items[0];
  assert.deepEqual(new Set(candidate.bookIds), new Set([a, c]));
  assert.deepEqual(candidate.additionalBookIds, [b]);
  assert.equal(candidate.requiresReview, true);
  assert.throws(() => linkEditions(store, { bookIds: candidate.bookIds }), /every member/);
});
