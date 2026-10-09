import { randomUUID } from 'node:crypto';
import { basename, extname } from 'node:path';
import { digest, now, parseJSON } from './store.mjs';

export const CATALOG_POLICY_VERSION = 'explicit-catalog-links-v1';

// Groups are presentation relationships. Never replace source book, section or chunk IDs.
const SCHEMA = `
CREATE TABLE IF NOT EXISTS catalog_groups (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, version INTEGER NOT NULL CHECK(version>=1),
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS group_members (
  book_id TEXT PRIMARY KEY REFERENCES books(id) ON DELETE CASCADE,
  group_id TEXT NOT NULL REFERENCES catalog_groups(id), linked_at TEXT NOT NULL,
  linked_version INTEGER NOT NULL CHECK(linked_version>=1)
);
CREATE INDEX IF NOT EXISTS group_members_group ON group_members(group_id);
CREATE TABLE IF NOT EXISTS catalog_group_events (
  id INTEGER PRIMARY KEY, group_id TEXT NOT NULL REFERENCES catalog_groups(id),
  version INTEGER NOT NULL, action TEXT NOT NULL, book_ids_json TEXT NOT NULL,
  details_json TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(group_id,version)
);
`;

export function ensureCatalogSchema(store) {
  store.transaction(() => store.db.exec(SCHEMA));
}

const strings = (value) => Array.isArray(value) ? value.filter((item) => typeof item === 'string' && item.trim()).map((item) => item.trim()) : [];
const object = (value) => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const normalized = (value) => value.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
const compare = (a, b) => a < b ? -1 : a > b ? 1 : 0;
const unique = (values) => [...new Set(values)].sort(compare);

function page({ limit = 50, offset = 0 } = {}) {
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200 || !Number.isSafeInteger(offset) || offset < 0) {
    throw new RangeError('Catalog limit must be 1–200 and offset must be a nonnegative integer');
  }
  return { limit, offset };
}

function editionInfo(value) {
  const words = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'];
  const pattern = /\b(?:(\d{1,3})(?:st|nd|rd|th)?|(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth))\s+(?:edition|ed)\b|\bedition\s+(\d{1,3})\b/gu;
  const tokens = [];
  const base = normalized(value).replace(pattern, (_, digits, word, after) => {
    tokens.push(String(Number(digits || after || words.indexOf(word) + 1)));
    return ' ';
  }).replace(/\s+/g, ' ').trim();
  return { base, tokens: unique(tokens) };
}

function formatCounts(members) {
  const counts = new Map();
  for (const member of members) for (const file of member.files) counts.set(file.format, (counts.get(file.format) ?? 0) + 1);
  return Object.fromEntries([...counts].sort(([a], [b]) => compare(a, b)));
}

function snapshot(store) {
  const memberships = new Map(store.db.prepare('SELECT book_id,group_id FROM group_members').all().map((row) => [row.book_id, row.group_id]));
  const files = new Map();
  for (const row of store.db.prepare('SELECT id,book_id,format,status,relative_path,group_key,sha256 FROM files WHERE book_id IS NOT NULL ORDER BY format,relative_path,id').all()) {
    if (!files.has(row.book_id)) files.set(row.book_id, []);
    files.get(row.book_id).push({ id: row.id, format: row.format, status: row.status,
      name: row.relative_path, candidateKey: row.group_key, sha256: row.sha256 });
  }
  const sectionCounts = new Map(store.db.prepare('SELECT book_id,count(*) AS count FROM sections GROUP BY book_id').all().map((row) => [row.book_id, row.count]));
  const firstSections = new Map(store.db.prepare(`SELECT s.book_id,s.id,s.title,s.locator_json FROM sections s
    WHERE s.ordinal=(SELECT min(first_section.ordinal) FROM sections first_section WHERE first_section.book_id=s.book_id)`).all()
    .map((row) => [row.book_id, { id: row.id, title: row.title, locator: object(parseJSON(row.locator_json)) }]));
  const members = store.db.prepare('SELECT * FROM books ORDER BY id').all().map((row) => {
    const metadata = object(parseJSON(row.metadata_json));
    return { id: row.id, title: row.title, authors: strings(parseJSON(row.authors_json, [])),
      metadata: {editedAt: metadata.editedAt, edition: metadata.edition, creators: metadata.creators,
        sourceMetadata: {recovery: metadata.sourceMetadata?.recovery ? {state: metadata.sourceMetadata.recovery.state, confidence: metadata.sourceMetadata.recovery.confidence} : undefined}},
      language: row.language, publisher: row.publisher, tags: strings(metadata.tags),
      identifiers: unique(strings(metadata.identifiers).filter((id) => id.length >= 6 && id.length <= 512 && !/^(unknown|undefined|unspecified|none|not available)$/i.test(id))),
      groupId: memberships.get(row.id) ?? null, files: files.get(row.id) ?? [],
      firstSection: firstSections.get(row.id) ?? null, sectionCount: sectionCounts.get(row.id) ?? 0,
      updatedAt: row.updated_at };
  });
  const groups = store.db.prepare('SELECT * FROM catalog_groups ORDER BY created_at,id').all();
  return { members, groups };
}

function card(group, members) {
  const formats = formatCounts(members);
  return { id: group?.id ?? members[0].id, type: group ? 'group' : 'book',
    title: group?.title ?? members[0].title, version: group?.version ?? null,
    authors: unique(members.flatMap((member) => member.authors)),
    bookIds: members.map((member) => member.id), members,
    sourceBookCount: members.length, formatEntryCount: members.reduce((sum, member) => sum + member.files.length, 0),
    formats, uniqueFormatCount: Object.keys(formats).length,
    updatedAt: group?.updated_at ?? members[0].updatedAt };
}

/** Suggestions never write memberships or assert that two source files are the same edition. */
export function groupSuggestions(store, options = {}) {
  const { limit, offset } = page(options);
  ensureCatalogSchema(store);
  const { members } = snapshot(store);
  const buckets = new Map();
  const facts = new Map();
  const add = (kind, value, member) => {
    const key = JSON.stringify([kind, value]);
    if (!buckets.has(key)) buckets.set(key, { kind, value, members: new Map() });
    buckets.get(key).members.set(member.id, member);
  };
  for (const member of members) {
    const title = editionInfo(member.title);
    const authors = unique(member.authors.map(normalized).filter(Boolean));
    const editions = unique([...title.tokens, ...member.files.flatMap((file) => editionInfo(basename(file.name, extname(file.name))).tokens)]);
    facts.set(member.id, { title: normalized(member.title), baseTitle: title.base, authors, editions });
    for (const identifier of member.identifiers) add('shared_identifier', identifier, member);
    if (title.base && authors.length) add('title_and_authors', JSON.stringify([title.base, authors]), member);
    for (const key of unique(member.files.map((file) => file.candidateKey).filter(Boolean))) add('same_folder_stem', key, member);
  }
  const candidates = new Map();
  for (const bucket of buckets.values()) {
    if (bucket.members.size < 2) continue;
    const books = [...bucket.members.values()].sort((a, b) => compare(a.id, b.id));
    if (books[0].groupId && books.every((member) => member.groupId === books[0].groupId)) continue;
    const key = JSON.stringify(books.map((member) => member.id));
    if (!candidates.has(key)) candidates.set(key, { books, evidence: [] });
    const value = bucket.kind === 'title_and_authors' ? JSON.parse(bucket.value) : bucket.value;
    candidates.get(key).evidence.push({ kind: bucket.kind, value });
  }
  const items = [...candidates.values()].map(({ books, evidence }) => {
    // Bucket membership is not transitive: an ISBN match A/B and a stem match B/C stay separate.
    const info = books.map((member) => facts.get(member.id));
    const titleAgreement = unique(info.map((item) => item.baseTitle)).length === 1;
    const authorAgreement = info.every((item) => item.authors.length) && unique(info.map((item) => JSON.stringify(item.authors))).length === 1;
    const editionTokens = unique(info.flatMap((item) => item.editions));
    const cautions = ['Suggestions require review; metadata and filenames do not prove edition identity.'];
    if (editionTokens.length > 1) cautions.push('Different explicit edition numbers were found; keep the individual editions distinguishable.');
    if (info.some((item) => !item.editions.length)) cautions.push('An explicit edition number is missing from at least one source.');
    if (!titleAgreement) cautions.push('Titles differ.');
    if (!authorAgreement) cautions.push('Author metadata differs or is missing.');
    const kinds = unique(evidence.map((item) => item.kind));
    const metadataSignals = kinds.filter((kind) => kind !== 'same_folder_stem').length;
    const linkedGroups = new Set(books.map((member) => member.groupId).filter(Boolean));
    const selected = new Set(books.map((member) => member.id));
    const additionalBookIds = members.filter((member) => linkedGroups.has(member.groupId) && !selected.has(member.id)).map((member) => member.id);
    return { ...card(null, books), id: digest(JSON.stringify([CATALOG_POLICY_VERSION, books.map((member) => member.id)])),
      type: 'suggestion', requiresReview: true, policyVersion: CATALOG_POLICY_VERSION,
      evidence: evidence.sort((a, b) => compare(JSON.stringify(a), JSON.stringify(b))),
      signal: metadataSignals > 1 ? 'multiple_metadata_signals' : metadataSignals ? 'metadata_signal' : 'filename_only',
      agreement: { title: titleAgreement, authors: authorAgreement, editionTokens }, cautions,
      additionalBookIds, metadataSignals };
  }).sort((a, b) => b.metadataSignals - a.metadataSignals || compare(normalized(a.title), normalized(b.title)) || compare(a.id, b.id));
  return { items: items.slice(offset, offset + limit).map(({ metadataSignals, ...item }) => item), total: items.length, limit, offset,
    policyVersion: CATALOG_POLICY_VERSION, automaticLinks: false };
}

function selectedIds(bookIds) {
  if (!Array.isArray(bookIds) || bookIds.length < 2 || bookIds.length > 200
      || bookIds.some((id) => typeof id !== 'string' || !id.trim() || id.length > 512 || id.includes('\0'))
      || new Set(bookIds).size !== bookIds.length) throw new TypeError('Select 2–200 distinct existing source book IDs');
  return [...bookIds].sort(compare);
}

function groupMemberIds(store, groupId) {
  return store.db.prepare('SELECT book_id FROM group_members WHERE group_id=? ORDER BY book_id').all(groupId).map((row) => row.book_id);
}

function event(store, groupId, version, action, bookIds, details, stamp) {
  store.db.prepare('INSERT INTO catalog_group_events(group_id,version,action,book_ids_json,details_json,created_at) VALUES(?,?,?,?,?,?)')
    .run(groupId, version, action, JSON.stringify(bookIds), JSON.stringify({ policyVersion: CATALOG_POLICY_VERSION, ...details }), stamp);
}

function groupCard(store, groupId) {
  const { members, groups } = snapshot(store);
  return card(groups.find((group) => group.id === groupId), members.filter((member) => member.groupId === groupId));
}

/** Explicit acceptance only. Every member of an affected existing group must be selected. */
export function linkEditions(store, { bookIds, title } = {}) {
  const ids = selectedIds(bookIds);
  if (title !== undefined && (typeof title !== 'string' || !title.trim() || title.length > 500 || title.includes('\0'))) throw new TypeError('Group title must contain 1–500 characters');
  ensureCatalogSchema(store);
  return store.transaction(() => {
    const books = ids.map((id) => store.db.prepare('SELECT id,title FROM books WHERE id=?').get(id));
    if (books.some((book) => !book)) throw new RangeError('A selected source book does not exist');
    const selected = new Set(ids);
    const existingIds = unique(ids.flatMap((id) => {
      const membership = store.db.prepare('SELECT group_id FROM group_members WHERE book_id=?').get(id);
      return membership ? [membership.group_id] : [];
    }));
    for (const id of existingIds) if (groupMemberIds(store, id).some((bookId) => !selected.has(bookId))) {
      throw new RangeError('Select every member of an existing group before combining it; unlink an individual edition first to move it');
    }
    const existing = existingIds.map((id) => store.db.prepare('SELECT * FROM catalog_groups WHERE id=?').get(id))
      .sort((a, b) => compare(a.created_at, b.created_at) || compare(a.id, b.id));
    const target = existing[0];
    const groupId = target?.id ?? `catalog-${randomUUID()}`;
    const groupTitle = title?.trim() ?? target?.title ?? books[0].title;
    if (target && existing.length === 1 && JSON.stringify(groupMemberIds(store, groupId)) === JSON.stringify(ids) && target.title === groupTitle) {
      return { changed: false, group: groupCard(store, groupId), mergedGroupIds: [] };
    }
    const stamp = now(), version = (target?.version ?? 0) + 1;
    if (target) store.db.prepare('UPDATE catalog_groups SET title=?,version=?,updated_at=? WHERE id=?').run(groupTitle, version, stamp, groupId);
    else store.db.prepare('INSERT INTO catalog_groups(id,title,version,created_at,updated_at) VALUES(?,?,?,?,?)').run(groupId, groupTitle, version, stamp, stamp);
    for (const source of existing.slice(1)) {
      const previousBookIds = groupMemberIds(store, source.id);
      store.db.prepare('DELETE FROM group_members WHERE group_id=?').run(source.id);
      store.db.prepare('UPDATE catalog_groups SET version=version+1,updated_at=? WHERE id=?').run(stamp, source.id);
      event(store, source.id, source.version + 1, 'merged_into', [], { targetGroupId: groupId, previousBookIds, origin: 'explicit_user_selection' }, stamp);
    }
    for (const id of ids) store.db.prepare('INSERT INTO group_members(book_id,group_id,linked_at,linked_version) VALUES(?,?,?,?) ON CONFLICT(book_id) DO NOTHING')
      .run(id, groupId, stamp, version);
    event(store, groupId, version, target ? 'updated' : 'created', ids,
      { origin: 'explicit_user_selection', title: groupTitle, previousTitle: target?.title ?? null, mergedGroupIds: existing.slice(1).map((group) => group.id) }, stamp);
    return { changed: true, group: groupCard(store, groupId), mergedGroupIds: existing.slice(1).map((group) => group.id) };
  });
}

/** Removing the last member keeps the empty group and its audit events, but hides its card. */
export function unlinkEdition(store, bookId) {
  if (typeof bookId !== 'string' || !bookId || bookId.length > 512 || bookId.includes('\0')) throw new TypeError('A source book ID is required');
  ensureCatalogSchema(store);
  return store.transaction(() => {
    if (!store.db.prepare('SELECT id FROM books WHERE id=?').get(bookId)) throw new RangeError('Source book does not exist');
    const membership = store.db.prepare('SELECT group_id FROM group_members WHERE book_id=?').get(bookId);
    if (!membership) return { changed: false, bookId, group: null };
    const groupId = membership.group_id, stamp = now();
    store.db.prepare('DELETE FROM group_members WHERE book_id=?').run(bookId);
    store.db.prepare('UPDATE catalog_groups SET version=version+1,updated_at=? WHERE id=?').run(stamp, groupId);
    const { version } = store.db.prepare('SELECT version FROM catalog_groups WHERE id=?').get(groupId);
    event(store, groupId, version, 'unlinked', groupMemberIds(store, groupId), { removedBookId: bookId, origin: 'explicit_user_selection' }, stamp);
    return { changed: true, bookId, group: groupCard(store, groupId) };
  });
}

/** Filters select cards; all members remain visible, with matchingBookIds marking the filter scope. */
export function catalogEntries(store, filters = {}) {
  const { limit, offset } = page(filters);
  const values = {};
  for (const key of ['q', 'format', 'author', 'status', 'tag']) {
    if (filters[key] !== undefined && (typeof filters[key] !== 'string' || filters[key].length > 500)) throw new TypeError(`Invalid catalog ${key} filter`);
    values[key] = filters[key]?.trim() ?? '';
  }
  const sort = filters.sort ?? 'title';
  if (!['title', 'author', 'recent'].includes(sort)) throw new TypeError('Unknown catalog sort order');
  values.format = values.format.toLowerCase().replace(/^\./, '');
  ensureCatalogSchema(store);
  const { members, groups } = snapshot(store);
  const byGroup = new Map();
  for (const member of members) if (member.groupId) {
    if (!byGroup.has(member.groupId)) byGroup.set(member.groupId, []);
    byGroup.get(member.groupId).push(member);
  }
  const cards = [
    ...groups.filter((group) => byGroup.has(group.id)).map((group) => card(group, byGroup.get(group.id))),
    ...members.filter((member) => !member.groupId).map((member) => card(null, [member])),
  ];
  const items = cards.map((item) => ({ ...item, matchingBookIds: item.members.filter((member) => {
    const searchable = [item.title, member.title, ...member.authors, ...member.tags].join(' ').toLowerCase();
    return (!values.q || searchable.includes(values.q.toLowerCase()))
      && (!values.author || member.authors.includes(values.author))
      && (!values.tag || member.tags.includes(values.tag))
      && ((!values.format && !values.status) || member.files.some((file) => (!values.format || file.format === values.format) && (!values.status || file.status === values.status)));
  }).map((member) => member.id) })).filter((item) => item.matchingBookIds.length);
  items.sort((a, b) => (sort === 'recent' ? compare(b.updatedAt, a.updatedAt) : sort === 'author' ? compare(normalized(a.authors.join(' ')), normalized(b.authors.join(' '))) : 0)
    || compare(normalized(a.title), normalized(b.title)) || compare(a.id, b.id));
  return { items: items.slice(offset, offset + limit), total: items.length, limit, offset,
    counts: { catalogEntries: items.length, confirmedGroups: items.filter((item) => item.type === 'group').length,
      ungroupedBooks: items.filter((item) => item.type === 'book').length,
      sourceBooks: items.reduce((sum, item) => sum + item.sourceBookCount, 0),
      matchingSourceBooks: items.reduce((sum, item) => sum + item.matchingBookIds.length, 0),
      formatEntries: items.reduce((sum, item) => sum + item.formatEntryCount, 0) },
    policyVersion: CATALOG_POLICY_VERSION };
}
