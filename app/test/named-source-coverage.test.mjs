import test from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { createRetrieval } from '../src/retrieval.mjs';

// Synthetic catalog/text only. Real SQLite exercises FTS, source identity and
// context admission; the injected transport records prompts and never uses a
// model or network. These checks make no semantic answer-quality claim.
const TITLES = {
  amber: 'Amber Harbor Handbook',
  cobalt: 'Cobalt Valley Companion',
  revised: 'Amber Harbor Handbook Revised Edition',
};
const QUESTION = `Compare ${TITLES.amber} and ${TITLES.cobalt} on maintenance.`;

function fixture(t) {
  const db = new DatabaseSync(':memory:');
  t.after(() => db.close());
  db.exec(`
    CREATE TABLE books(id TEXT PRIMARY KEY, title TEXT, authors_json TEXT);
    CREATE TABLE sections(id TEXT PRIMARY KEY, book_id TEXT, ordinal INTEGER,
      title TEXT, text TEXT, locator_json TEXT);
    CREATE TABLE chunks(id TEXT PRIMARY KEY, book_id TEXT, section_id TEXT,
      ordinal INTEGER, text TEXT, locator_json TEXT, embedding_json TEXT,
      embedding_model TEXT, embedding_dimension INTEGER, embedding_norm REAL);
    CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, book_id UNINDEXED, text);
  `);
  return {
    db,
    transaction(fn) {
      db.exec('BEGIN');
      try { const result = fn(); db.exec('COMMIT'); return result; }
      catch (error) { db.exec('ROLLBACK'); throw error; }
    },
    add(id, title, texts) {
      db.prepare('INSERT INTO books VALUES (?, ?, ?)').run(id, title, '["Synthetic Author"]');
      const sectionId = `${id}:section`;
      const source = texts.join('\n');
      db.prepare('INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?)')
        .run(sectionId, id, 0, 'Synthetic section', source, '{"page":1}');
      let start = 0;
      for (const [ordinal, text] of texts.entries()) {
        const chunkId = `${id}:chunk:${ordinal}`;
        const locator = { format: 'pdf', page: 1, sectionOrdinal: 0,
          charStart: start, charEnd: start + text.length, offsetBasis: 'section_utf16_code_units' };
        db.prepare(`INSERT INTO chunks(id,book_id,section_id,ordinal,text,locator_json)
          VALUES (?,?,?,?,?,?)`).run(chunkId, id, sectionId, ordinal, text, JSON.stringify(locator));
        db.prepare('INSERT INTO chunks_fts VALUES (?, ?, ?)').run(chunkId, id, text);
        start += text.length + 1;
      }
    },
  };
}

function capturedChat() {
  const calls = [];
  return {
    calls,
    fetch: async (url, init) => {
      calls.push({ url, body: JSON.parse(init.body) });
      assert.ok(url.endsWith('/api/chat'), 'Named-source coverage must not add embedding requests');
      return new Response(JSON.stringify({ done: true,
        message: { content: JSON.stringify({ abstain: true, answer: '' }) } }),
      { headers: { 'content-type': 'application/json' } });
    },
  };
}

function retrieval(store, capture, config = {}) {
  return createRetrieval(store, { candidateLimit: 1, answerLimit: 2,
    chatModel: 'synthetic-capture-only', fetch: capture.fetch, ...config });
}

function addRequestedBooks(store, { cobaltText = 'Maintenance uses replaceable parts.' } = {}) {
  store.add('amber', TITLES.amber, ['Maintenance keeps independent components.']);
  store.add('cobalt', TITLES.cobalt, [cobaltText]);
}

function addLeadingNoise(store) {
  // Every query token occurs repeatedly here. The requested sources match only
  // the topic, so the test first verifies that the ordinary bounded pool omits
  // both before asserting that the named-source fallback recovers them.
  store.add('noise', 'Unrelated Catalog Notes', Array.from({ length: 4 }, () => `${QUESTION} `.repeat(8)));
}

const bookIds = rows => [...new Set(rows.map(row => row.bookId))].sort();
const requestedIds = result => result.sourceCoverage.requestedBooks.map(book => book.bookId).sort();
const prompt = call => JSON.parse(call.body.messages.find(message => message.role === 'user').content);

function assertExactCitations(store, result) {
  for (const citation of result.citations) {
    const row = store.db.prepare('SELECT book_id,text FROM sections WHERE id=?').get(citation.sectionId);
    assert.equal(row.book_id, citation.bookId);
    assert.equal(citation.text, row.text.slice(citation.locator.charStart, citation.locator.charEnd));
  }
}

test('named-source seeds recover both requested books outside the ordinary top K', async t => {
  const store = fixture(t);
  addRequestedBooks(store);
  addLeadingNoise(store);
  const capture = capturedChat();
  const api = retrieval(store, capture);

  const ordinary = await api.search({ query: QUESTION, bookId: null, limit: 2 });
  assert.equal(ordinary.metrics.lexicalCandidates, 2);
  assert.deepEqual(bookIds(ordinary.hits), ['noise'], 'Fixture must exclude both requested books before fallback');
  assert.equal(capture.calls.length, 0);

  const result = await api.ask({ question: QUESTION, bookId: null });
  assert.equal(result.bookId, null);
  assert.equal(result.question, QUESTION);
  assert.deepEqual(requestedIds(result), ['amber', 'cobalt']);
  assert.deepEqual(result.sourceCoverage.missingBooks, []);
  assert.deepEqual(result.sourceCoverage.contextMissingBooks, []);
  assert.deepEqual(bookIds(result.hits), ['amber', 'cobalt']);
  assert.ok(result.hits.every(hit => hit.provenance.branches.includes('named_source_lexical')));
  assert.equal(capture.calls.length, 1);
  assert.equal(prompt(capture.calls[0]).question, QUESTION);
  assert.deepEqual([...new Set(prompt(capture.calls[0]).passages.map(item => item.bookTitle))].sort(),
    [TITLES.amber, TITLES.cobalt].sort());
  assertExactCitations(store, result);
});

test('ordinary all-library search keeps its ranking before and after a named-source ask', async t => {
  const store = fixture(t);
  addRequestedBooks(store);
  addLeadingNoise(store);
  const capture = capturedChat();
  const api = retrieval(store, capture, { answerLimit: 3 });
  const before = await api.search({ query: QUESTION, limit: 2 });
  const comparison = await api.ask({ question: QUESTION, bookId: null });
  const after = await api.search({ query: QUESTION, bookId: null, limit: 2 });

  assert.equal(before.bookId, null);
  assert.equal(after.bookId, null);
  assert.deepEqual(before.hits.map(hit => hit.id), after.hits.map(hit => hit.id));
  assert.deepEqual(bookIds(after.hits), ['noise']);
  assert.deepEqual(after.sourceCoverage, {
    requestedBooks: [], missingBooks: [], ambiguousTitles: [], resolutionLimited: false,
  });
  assert.ok(after.hits.every(hit => !hit.provenance.branches.includes('named_source_lexical')));
  assert.equal(comparison.bookId, null);
  assert.ok(comparison.hits.some(hit => hit.bookId === 'noise'), 'Global candidates still fill unused answer slots');
  assert.equal(capture.calls.length, 1, 'Ordinary searches must not call chat');
});

test('a named catalog book without an eligible passage withholds chat', async t => {
  const store = fixture(t);
  addRequestedBooks(store, { cobaltText: 'Zephyrs orbit silently.' });
  addLeadingNoise(store);
  const capture = capturedChat();
  const result = await retrieval(store, capture).ask({ question: QUESTION, bookId: null });

  assert.deepEqual(requestedIds(result), ['amber', 'cobalt']);
  assert.deepEqual(result.sourceCoverage.missingBooks, ['cobalt']);
  assert.ok(result.sourceCoverage.contextMissingBooks.includes('cobalt'));
  assert.equal(result.status, 'abstained');
  assert.equal(result.abstained, true);
  assert.ok(result.warnings.some(item => item.code === 'named_source_coverage_missing'));
  assert.equal(capture.calls.length, 0);
});

test('a byte budget that admits either requested book alone withholds an incomplete pair', async t => {
  const store = fixture(t);
  const longText = `Maintenance ${'界'.repeat(700)}`;
  store.add('amber', TITLES.amber, [longText]);
  store.add('cobalt', TITLES.cobalt, [longText]);
  const wideCapture = capturedChat();
  const wideTokens = 8192;
  const wide = retrieval(store, wideCapture, { contextTokens: wideTokens });
  const singles = [];
  for (const bookId of ['amber', 'cobalt']) singles.push(await wide.ask({ question: QUESTION, bookId }));
  assert.equal(wideCapture.calls.length, 2);
  const singleBytes = Math.max(...wideCapture.calls.map(call =>
    Buffer.byteLength(JSON.stringify(call.body.messages), 'utf8')));
  const reservation = wideTokens - singles[0].context.inputByteBudget;
  const tightTokens = singleBytes + reservation + 16;
  assert.ok(tightTokens >= 2048 && tightTokens < wideTokens);

  const capture = capturedChat();
  const tight = retrieval(store, capture, { contextTokens: tightTokens });
  for (const bookId of ['amber', 'cobalt']) await tight.ask({ question: QUESTION, bookId });
  assert.equal(capture.calls.length, 2, 'The tighter byte budget must still admit either source alone');
  const result = await tight.ask({ question: QUESTION, bookId: null });

  assert.deepEqual(requestedIds(result), ['amber', 'cobalt']);
  assert.deepEqual(result.sourceCoverage.missingBooks, [], 'Both anchors exist; this failure must be context admission');
  assert.ok(result.sourceCoverage.contextMissingBooks.length > 0);
  assert.equal(result.status, 'abstained');
  assert.ok(result.warnings.some(item => item.code === 'named_source_coverage_missing'));
  assert.equal(capture.calls.length, 2, 'The comparison must not call chat with only one required source');
});

test('duplicate catalog titles remain ambiguous instead of selecting an edition', async t => {
  const store = fixture(t);
  store.add('amber-edition-one', TITLES.amber, ['Maintenance records first edition.']);
  store.add('amber-edition-two', TITLES.amber, ['Maintenance records second edition.']);
  store.add('cobalt', TITLES.cobalt, ['Maintenance records components.']);
  const capture = capturedChat();
  const result = await retrieval(store, capture).ask({ question: QUESTION, bookId: null });

  assert.ok(result.sourceCoverage.ambiguousTitles.includes(TITLES.amber.toLowerCase()));
  assert.ok(result.warnings.some(item => item.code === 'ambiguous_source_titles'));
  assert.ok(!requestedIds(result).some(id => id.startsWith('amber-edition-')));
  assert.equal(result.status, 'abstained');
  assert.equal(capture.calls.length, 0, 'Ambiguous editions must not be silently chosen by the model');
});

test('an explicit single-book scope never expands to another named source', async t => {
  const store = fixture(t);
  addRequestedBooks(store);
  addLeadingNoise(store);
  const capture = capturedChat();
  const api = retrieval(store, capture);
  const search = await api.search({ query: QUESTION, bookId: 'amber', limit: 2, ensureNamedSources: true });
  const result = await api.ask({ question: QUESTION, bookId: 'amber' });

  assert.equal(search.bookId, 'amber');
  assert.deepEqual(bookIds(search.hits), ['amber']);
  assert.equal(result.bookId, 'amber');
  assert.deepEqual(requestedIds(result), []);
  assert.deepEqual(bookIds(result.hits), ['amber']);
  assert.deepEqual(bookIds(result.citations), ['amber']);
  assert.equal(capture.calls.length, 1);
  assert.ok(prompt(capture.calls[0]).passages.every(item => item.bookTitle === TITLES.amber));
  assertExactCitations(store, result);
});

test('a shorter title contained only in a longer mention is not a second requested source', async t => {
  const store = fixture(t);
  store.add('amber', TITLES.amber, ['Maintenance replaces components.']);
  store.add('revised', TITLES.revised, ['Maintenance records components.']);
  const capture = capturedChat();
  const result = await retrieval(store, capture).ask({
    question: `Explain ${TITLES.revised} on maintenance.`, bookId: null,
  });

  assert.deepEqual(requestedIds(result), [], 'One explicit title must not fabricate a two-source comparison');
  assert.deepEqual(result.sourceCoverage.ambiguousTitles, []);
});

test('a separately mentioned shorter title survives overlap resolution', async t => {
  const store = fixture(t);
  store.add('amber', TITLES.amber, ['Maintenance replaces components.']);
  store.add('revised', TITLES.revised, ['Maintenance records components.']);
  const capture = capturedChat();
  const question = `Compare ${TITLES.amber} with ${TITLES.revised} on maintenance.`;
  const result = await retrieval(store, capture).ask({ question, bookId: null });

  assert.deepEqual(requestedIds(result), ['amber', 'revised']);
  assert.deepEqual(result.sourceCoverage.ambiguousTitles, []);
  assert.deepEqual(result.sourceCoverage.contextMissingBooks, []);
  assert.equal(capture.calls.length, 1);
  assert.deepEqual([...new Set(prompt(capture.calls[0]).passages.map(item => item.bookTitle))].sort(),
    [TITLES.amber, TITLES.revised].sort());
});

test('a standalone shared prefix remains ambiguous after another full title mention', async t => {
  const store = fixture(t);
  store.add('basics', `${TITLES.amber} Basics Edition`, ['Maintenance replaces components.']);
  store.add('revised', TITLES.revised, ['Maintenance records components.']);
  const capture = capturedChat();
  const result = await retrieval(store, capture).ask({
    question: `Compare ${TITLES.revised} with ${TITLES.amber} on maintenance.`, bookId: null,
  });

  assert.ok(result.sourceCoverage.ambiguousTitles.includes(TITLES.amber.toLowerCase()));
  assert.ok(result.warnings.some(item => item.code === 'ambiguous_source_titles'));
  assert.equal(result.status, 'abstained');
  assert.equal(capture.calls.length, 0, 'An earlier full title must not hide the later ambiguous prefix');
});
