// Reader scope is resolved against source records, never supplied passage text.
const fields = new Set(['bookId', 'sectionId', 'scope', 'page', 'member', 'fragment', 'charStart', 'charEnd']);
const parse = text => { try { return JSON.parse(text) ?? {}; } catch { return {}; } };

export function resolveReaderContext(store, value, bookId = null) {
  if (value == null) return null;
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !fields.has(key))) {
    throw new TypeError('readerContext contains unsupported fields');
  }
  for (const key of ['bookId', 'sectionId']) {
    if (typeof value[key] !== 'string' || !value[key].trim() || value[key].length > 256) throw new TypeError(`readerContext.${key} must be a source identifier`);
  }
  if (bookId != null && value.bookId !== bookId) throw new TypeError('readerContext does not belong to the selected book');
  if (!['section', 'book'].includes(value.scope ?? 'section')) throw new TypeError('readerContext.scope must be section or book');
  const row = store.db.prepare('SELECT id,book_id,text,locator_json FROM sections WHERE id=? AND book_id=?').get(value.sectionId, value.bookId);
  if (!row) throw new TypeError('readerContext section does not belong to this source');
  const rawLocator = parse(row.locator_json);
  const locator = rawLocator.derived ?? rawLocator;
  for (const key of ['page', 'member']) {
    if (value[key] !== undefined && value[key] !== locator[key]) throw new TypeError(`readerContext.${key} does not match the source section`);
  }
  if (value.fragment !== undefined && (typeof value.fragment !== 'string' || value.fragment.length > 1000 ||
    (value.fragment !== locator.fragment && !Object.hasOwn(locator.anchors ?? {}, value.fragment)))) {
    throw new TypeError('readerContext.fragment is not a source anchor');
  }
  if ((value.charStart === undefined) !== (value.charEnd === undefined)) throw new TypeError('readerContext needs both source offsets');
  if (value.charStart !== undefined && (!Number.isSafeInteger(value.charStart) || !Number.isSafeInteger(value.charEnd) ||
    value.charStart < 0 || value.charEnd < value.charStart || value.charEnd > row.text.length)) throw new TypeError('readerContext offsets are outside the source section');
  const fragmentOffsets = value.fragment && value.charStart === undefined ? locator.anchors?.[value.fragment] : null;
  return { bookId: row.book_id, sectionId: row.id, scope: value.scope ?? 'section',
    ...(fragmentOffsets && Number.isSafeInteger(fragmentOffsets.charStart) && Number.isSafeInteger(fragmentOffsets.charEnd) ?
      { charStart: fragmentOffsets.charStart, charEnd: fragmentOffsets.charEnd } : {}),
    ...Object.fromEntries(['page', 'member'].filter(key => locator[key] !== undefined).map(key => [key, locator[key]])),
    ...Object.fromEntries(['fragment', 'charStart', 'charEnd'].filter(key => value[key] !== undefined).map(key => [key, value[key]])) };
}

export function verifySourceSpan(db, hit) {
  const section = db.prepare('SELECT book_id,text,locator_json FROM sections WHERE id=?').get(hit.sectionId);
  const { charStart, charEnd, offsetBasis } = hit.locator ?? {};
  if (!section || section.book_id !== hit.bookId || offsetBasis !== 'section_utf16_code_units' ||
    !Number.isSafeInteger(charStart) || charStart < 0 || charEnd !== charStart + hit.text.length ||
    charEnd > section.text.length || section.text.slice(charStart, charEnd) !== hit.text) return false;
  const locator = parse(section.locator_json);
  return ['page', 'member'].every(key => hit.locator[key] === locator[key]) &&
    ['page', 'member'].every(key => hit.locator.derived?.[key] === locator.derived?.[key]);
}

export function readerSourceGroups(db, readerContext, maxSourceChars = 24000) {
  if (!readerContext) return { complete: null, anchor: null };
  const selected = db.prepare('SELECT ordinal FROM sections WHERE id=? AND book_id=?').get(readerContext.sectionId, readerContext.bookId);
  const rows = db.prepare(`SELECT s.*,b.title AS book_title,b.authors_json FROM sections s
    JOIN books b ON b.id=s.book_id WHERE s.book_id=? AND s.ordinal BETWEEN ? AND ? ORDER BY s.ordinal,s.id`)
    .all(readerContext.bookId, selected.ordinal - 1, selected.ordinal + 1);
  const asHit = row => ({ id: `${row.id}:source`, chunkId: null, bookId: row.book_id, sectionId: row.id,
    ordinal: row.ordinal, title: row.book_title, authors: parse(row.authors_json), text: row.text, excerpt: row.text,
    locator: { ...parse(row.locator_json), sectionOrdinal: row.ordinal, charStart: 0, charEnd: row.text.length,
      offsetBasis: 'section_utf16_code_units' }, provenance: { branches: ['reader_source'], completeSection: true } });
  const selectedRow = rows.find(row => row.id === readerContext.sectionId);
  const previous = db.prepare(`SELECT s.*,b.title AS book_title,b.authors_json FROM sections s JOIN books b ON b.id=s.book_id
    WHERE s.book_id=? AND s.ordinal<? AND length(s.text)>0 ORDER BY s.ordinal DESC LIMIT 1`).get(readerContext.bookId, selected.ordinal);
  const next = db.prepare(`SELECT s.*,b.title AS book_title,b.authors_json FROM sections s JOIN books b ON b.id=s.book_id
    WHERE s.book_id=? AND s.ordinal>? AND length(s.text)>0 ORDER BY s.ordinal LIMIT 1`).get(readerContext.bookId, selected.ordinal);
  const anchor = [selectedRow, previous, next]
    .filter(row => row?.text.length).map((row, index) => ({ ...asHit(row), provenance: { branches: ['reader_source'], completeSection: true,
      ...(row.id !== readerContext.sectionId ? { direction: row.ordinal < selected.ordinal ? 'previous' : 'next' } :
        { readerAnchor: { charStart: readerContext.charStart ?? 0, charEnd: readerContext.charEnd ?? row.text.length } }) } }));
  let complete = selectedRow.text ? [asHit(selectedRow)] : [];
  if (readerContext.scope === 'book') {
    const size = db.prepare('SELECT SUM(length(text)) AS characters,COUNT(*) AS sections FROM sections WHERE book_id=?').get(readerContext.bookId);
    complete = size.characters <= maxSourceChars && size.sections <= 24 ? db.prepare(`SELECT s.*,b.title AS book_title,b.authors_json FROM sections s
      JOIN books b ON b.id=s.book_id WHERE s.book_id=? ORDER BY s.ordinal,s.id`).all(readerContext.bookId).filter(row => row.text.length).map(asHit) : [];
  }
  const start = readerContext.charStart ?? 0, end = readerContext.charEnd ?? selectedRow.text.length;
  const focused = asHit(selectedRow);
  const focus = end > start ? [{ ...focused, text: selectedRow.text.slice(start, end), excerpt: selectedRow.text.slice(start, end),
    locator: { ...focused.locator, charStart: start, charEnd: end },
    provenance: { ...focused.provenance, completeSection: start === 0 && end === selectedRow.text.length,
      readerAnchor: { charStart: start, charEnd: end } } }] : [];
  const nearbySectionIds = db.prepare('SELECT id FROM sections WHERE book_id=? AND ordinal BETWEEN ? AND ?')
    .all(readerContext.bookId, selected.ordinal - 2, selected.ordinal + 2).map(row => row.id);
  return { complete, anchor, focus, nearbySectionIds, selectedEmpty: !selectedRow.text.length };
}
