import test from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { createRetrieval } from '../src/retrieval.mjs';

// Real SQLite/FTS and source offsets; only the chat transport is synthetic.
// Its fixed abstention lets these tests inspect allocation without pretending
// that a mocked answer establishes entailment or real-model answer quality.
function fixture(t, { longIds = true } = {}) {
  const db = new DatabaseSync(':memory:');
  t.after(() => db.close());
  db.exec(`
    CREATE TABLE books(id TEXT PRIMARY KEY, title TEXT, authors_json TEXT);
    CREATE TABLE sections(id TEXT PRIMARY KEY, book_id TEXT, ordinal INTEGER,
      title TEXT, text TEXT, locator_json TEXT);
    CREATE TABLE chunks(id TEXT PRIMARY KEY, book_id TEXT, section_id TEXT, ordinal INTEGER,
      text TEXT, locator_json TEXT, embedding_json TEXT, embedding_model TEXT,
      embedding_dimension INTEGER, embedding_norm REAL);
    CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, book_id UNINDEXED, text);
  `);
  const store = { db, transaction(fn) {
    db.exec('BEGIN');
    try { const result = fn(); db.exec('COMMIT'); return result; }
    catch (error) { db.exec('ROLLBACK'); throw error; }
  } };
  function add(key, texts, { richLocator = false } = {}) {
    const bookId = longIds ? key.repeat(64) : key;
    const sectionId = longIds ? `${bookId}:s0:${'d'.repeat(64)}` : `${bookId}:s0`;
    const title = `Synthetic ${key.toUpperCase()} manual`;
    const source = texts.join('\n');
    db.prepare('INSERT INTO books VALUES (?, ?, ?)').run(bookId, title, '["Synthetic Author"]');
    db.prepare('INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?)')
      .run(sectionId, bookId, 0, 'Instructions', source, JSON.stringify(richLocator ?
        { format: 'mobi', page: 1, derived: { format: 'epub', member: 'chapter.xhtml' } } : { format: 'pdf', page: 1 }));
    let start = 0;
    const ids = [];
    for (const [ordinal, text] of texts.entries()) {
      const chunkId = longIds ? `${sectionId}:c${ordinal}:${'e'.repeat(64)}` : `${bookId}:c${ordinal}`;
      const locator = { format: 'pdf', page: 1, sectionOrdinal: 0,
        charStart: start, charEnd: start + text.length, offsetBasis: 'section_utf16_code_units',
        sourceSha256: '1'.repeat(64), originalSha256: '2'.repeat(64), sha256: '3'.repeat(64) };
      if (richLocator) Object.assign(locator, {
        format: 'mobi',
        anchors: { opening: 0, inspection: 17 },
        derived: { format: 'epub', member: 'chapter.xhtml', fragment: 'inspection',
          sourceSha256: '4'.repeat(64), originalSha256: '5'.repeat(64), sha256: '6'.repeat(64),
          anchors: { inspection: 17 } },
      });
      db.prepare('INSERT INTO chunks(id,book_id,section_id,ordinal,text,locator_json) VALUES (?,?,?,?,?,?)')
        .run(chunkId, bookId, sectionId, ordinal, text, JSON.stringify(locator));
      db.prepare('INSERT INTO chunks_fts VALUES (?, ?, ?)').run(chunkId, bookId, text);
      ids.push(chunkId);
      start += text.length + 1;
    }
    return { bookId, sectionId, ids, texts };
  }
  return { ...store, add };
}

function chatCapture() {
  const calls = [];
  return { calls, fetch: async (url, init) => {
    assert.equal(url, 'http://127.0.0.1:11434/chat/completions');
    assert.equal(init.method, 'POST');
    calls.push(JSON.parse(init.body));
    return new Response(JSON.stringify({ choices: [{ finish_reason: 'stop',
      message: { content: '{"abstain":true,"answer":""}' } }] }),
    { headers: { 'content-type': 'application/json' } });
  } };
}

function retrieval(store, capture, options = {}) {
  return createRetrieval(store, { provider: 'openai', chatModel: 'synthetic-context-observer',
    fetch: capture.fetch, ...options });
}

function assertExactMap(store, result, call) {
  const sent = JSON.parse(call.messages[1].content);
  assert.equal(sent.question, result.question);
  assert.equal(sent.passages.length, result.citations.length);
  assert.equal(new Set(result.citations.map(item => item.citation)).size, result.citations.length);
  for (const [index, item] of result.citations.entries()) {
    const row = store.db.prepare(`SELECT c.book_id,c.section_id,c.text,c.locator_json,s.text AS section_text
      FROM chunks c JOIN sections s ON s.id=c.section_id WHERE c.id=?`).get(item.chunkId);
    assert.ok(row, 'Every public citation retains its real stored chunk identity');
    const original = JSON.parse(row.locator_json);
    assert.equal(item.bookId, row.book_id);
    assert.equal(item.sectionId, row.section_id);
    assert.ok(item.locator.charStart >= original.charStart);
    assert.ok(item.locator.charEnd <= original.charEnd);
    assert.ok(item.locator.charEnd > item.locator.charStart);
    assert.equal(item.text, row.section_text.slice(item.locator.charStart, item.locator.charEnd));
    assert.equal(item.excerpt, item.text);
    assert.deepEqual(item.locator, { ...original,
      charStart: item.locator.charStart, charEnd: item.locator.charEnd });
    assert.equal(item.citation, index + 1);
    assert.equal(item.label, `[${index + 1}]`);
    assert.equal(sent.passages[index].citation, item.citation);
    assert.equal(sent.passages[index].bookTitle, item.title);
    assert.equal(sent.passages[index].text.split('\n').map((line,at)=>{
      assert.ok(line.startsWith(`${at+1}|`));return line.slice(String(at+1).length+1);
    }).join('\n'), item.text);
    assert.equal(sent.passages[index].locator.charStart, item.locator.charStart);
    assert.equal(sent.passages[index].locator.charEnd, item.locator.charEnd);
    assert.ok(!/^[\uDC00-\uDFFF]/.test(item.text), 'No split low surrogate at a window start');
    assert.ok(!/[\uD800-\uDBFF]$/.test(item.text), 'No split high surrogate at a window end');
  }
  const inputBytes = Buffer.byteLength(JSON.stringify(call.messages), 'utf8');
  assert.equal(result.context.inputBytes, inputBytes);
  assert.ok(inputBytes <= result.context.inputByteBudget);
  assert.equal(result.metrics.contextCharacters, result.citations.reduce((sum, item) => sum + item.text.length, 0));
  assert.equal(result.metrics.contextPassages, result.citations.length);
  return sent;
}

function competingTexts(label, repetitions = 1) {
  const padding = 'neutral filler '.repeat(240);
  return [
    `${padding}Before ${label}, isolate the supply. ${'buffer '.repeat(48)}`,
    `${'quartz '.repeat(repetitions)}${'buffer '.repeat(46)}${label} requires the amber seal. ${padding}`,
    `${'buffer '.repeat(48)}After ${label}, release pressure slowly. ${padding}`,
  ];
}

function addPage(store, book, ordinal, text) {
  const sectionId = `${book.bookId}:page${ordinal + 1}`, chunkId = `${sectionId}:chunk`;
  const locator = { format: 'pdf', page: ordinal + 1, sectionOrdinal: ordinal,
    charStart: 0, charEnd: text.length, offsetBasis: 'section_utf16_code_units' };
  store.db.prepare('INSERT INTO sections VALUES (?,?,?,?,?,?)')
    .run(sectionId, book.bookId, ordinal, `Page ${ordinal + 1}`, text, JSON.stringify({ format: 'pdf', page: ordinal + 1 }));
  store.db.prepare('INSERT INTO chunks(id,book_id,section_id,ordinal,text,locator_json) VALUES (?,?,?,?,?,?)')
    .run(chunkId, book.bookId, sectionId, 0, text, JSON.stringify(locator));
  store.db.prepare('INSERT INTO chunks_fts VALUES (?,?,?)').run(chunkId, book.bookId, text);
  return { sectionId, chunkId, text };
}

test('a complete worked example precedes clipped secondary hits in the same source', async t => {
  const store = fixture(t);
  const core = `${'quartz '.repeat(50)}Worked example starts.\n${'neutral details '.repeat(130)}\n` +
    'Input values are 2, 4, 6 and one missing value. All-row count is 4; populated-value count is 3.\n' +
    `${'neutral details '.repeat(30)}\nThe conditional denominator is 1 when count is zero, otherwise the count. Worked example ends.`;
  const book = store.add('a', ['Before the example, check the input values.', core,
    'After the example, retain the stated conditions.',
    ...Array.from({ length: 4 }, (_, index) => `quartz secondary note ${index}. ${'padding '.repeat(320)}`)]);
  const capture = chatCapture();
  const result = await retrieval(store, capture).ask({ question: 'Explain the quartz worked example.', bookId: book.bookId });
  assert.equal(result.hits[0].chunkId, book.ids[1], 'The fixture ranks the worked example first');
  assert.equal(capture.calls.length, 1);
  assertExactMap(store, result, capture.calls[0]);
  assert.ok(result.citations.some(item => item.chunkId === book.ids[1] && item.text === core));
  assert.ok(result.warnings.some(item => item.code === 'context_limit'));
});

test('reader evidence retains the selected page and complete nearby continuation before a distant summary', async t => {
  const store = fixture(t);
  const book = store.add('a', [`Selected page introduces quartz. ${'background '.repeat(90)}`]);
  const example = addPage(store, book, 1, `${'quartz '.repeat(12)}Example inputs are 3, 42 and foo. ${'explanation '.repeat(70)}Output order is 3, 42, foo.`);
  const continuation = addPage(store, book, 2, `quartz rejection handling. ${'discussion '.repeat(100)}Use a catch handler or try/catch around await.`);
  addPage(store, book, 24, `${'quartz '.repeat(50)}Distant summary. ${'padding '.repeat(150)}`);
  const capture = chatCapture();
  const result = await retrieval(store, capture).ask({ question: 'Explain the nearby quartz example and rejection handling.',
    readerContext: { bookId: book.bookId, sectionId: book.sectionId, page: 1, scope: 'section' } });
  assert.equal(capture.calls.length, 1);
  const sent = JSON.parse(capture.calls[0].messages[1].content);
  for (const [sectionId, text] of [[book.sectionId, book.texts[0]], [example.sectionId, example.text], [continuation.sectionId, continuation.text]]) {
    assert.ok(result.citations.some(item => item.sectionId === sectionId && item.text === text), `Full reader unit ${sectionId} retained`);
  }
  assert.deepEqual(new Set(sent.passages.map(item => item.locator.page)), new Set([1, 2, 3]));
  assert.ok(result.context.inputBytes <= 6480);
});

test('physical-page scope constrains retrieval before top K and excludes neighboring outside pages', async t => {
  const store = fixture(t);
  const book = store.add('a', ['quartz outside page 1.']);
  const page2 = addPage(store, book, 1, 'quartz requested example on page 2.');
  const page3 = addPage(store, book, 2, 'quartz requested continuation on page 3.');
  for (let ordinal = 3; ordinal < 11; ordinal += 1) addPage(store, book, ordinal, `${'quartz '.repeat(60)}outside page.`);
  const capture = chatCapture();
  const result = await retrieval(store, capture, { candidateLimit: 1, answerLimit: 1 }).ask({
    question: 'Use only pages 2–3 to explain quartz.', bookId: book.bookId });
  assert.equal(capture.calls.length, 1);
  assert.ok(result.hits.every(hit => [2, 3].includes(hit.locator.page)));
  assert.deepEqual(new Set(result.citations.map(hit => hit.sectionId)), new Set([page2.sectionId, page3.sectionId]));
  assert.deepEqual(result.requestedSourceScope, { kind: 'physical_pages', bookId: book.bookId,
    firstPage: 2, lastPage: 3, completeness: 'not_established' });
  const absent = await retrieval(store, capture).ask({ question: 'Use only page 99 to explain quartz.', bookId: book.bookId });
  assert.equal(capture.calls.length, 1);
  assert.equal(absent.inference.requested, false);
  assert.equal(absent.responseKind, 'no_evidence');
  assert.equal(absent.hits.length, 0);
});

test('negated page restrictions remain broad and unsupported ranges fail before inference', async t => {
  const store = fixture(t), book = store.add('a', ['quartz source information.']);
  const capture = chatCapture();
  const result = await retrieval(store, capture).ask({ question: 'Explain quartz, not only pages 2–3.', bookId: book.bookId });
  assert.equal(capture.calls.length, 1);
  assert.equal(result.requestedSourceScope, undefined);
  assert.equal(result.hits[0].bookId, book.bookId);
  for (const question of ['Use only pages 2 and 3 for quartz.', 'Use only pages 3–2 for quartz.', 'Use only pages 2–3, 5 for quartz.']) {
    await assert.rejects(retrieval(store, capture).ask({ question, bookId: book.bookId }), TypeError);
  }
  assert.equal(capture.calls.length, 1);
});

test('page restrictions allow purpose prose but reject numeric range extensions before inference', async t => {
  const store = fixture(t), book = store.add('a', ['quartz outside page 1.']);
  addPage(store, book, 1, 'quartz example on page 2.');
  addPage(store, book, 2, 'quartz continuation on page 3.');
  const capture = chatCapture();
  for (const [question, lastPage] of [
    ['Use only page 2 to explain quartz.', 2],
    ['Use only pages 2–3 through a quartz example.', 3],
    ['Use only pages 2 to 3 to explain quartz.', 3],
    ['Use only pages 2 through 3 to explain quartz.', 3],
  ]) {
    const result = await retrieval(store, capture).ask({ question, bookId: book.bookId });
    assert.deepEqual(result.requestedSourceScope, { kind: 'physical_pages', bookId: book.bookId,
      firstPage: 2, lastPage, completeness: 'not_established' });
    assert.ok(result.hits.length > 0);
    assert.ok(result.citations.length > 0);
    assert.ok(result.hits.every(hit => hit.locator.page >= 2 && hit.locator.page <= lastPage));
    assert.ok(result.citations.every(hit => hit.locator.page >= 2 && hit.locator.page <= lastPage));
  }
  assert.equal(capture.calls.length, 4);
  for (const question of [
    'Use only pages 2–3 to 5 for quartz.',
    'Use only pages 2–3 through 5 for quartz.',
    'Use only pages 2–3 to page 5 for quartz.',
    'Use only pages 2 through 3 through pages 5 for quartz.',
  ]) {
    await assert.rejects(retrieval(store, capture).ask({ question, bookId: book.bookId }), TypeError);
  }
  assert.equal(capture.calls.length, 4);
});

test('a long selected section retains a verified focus even when retrieved hits are distant', async t => {
  const store = fixture(t), book = store.add('a', [`Visible section information. ${'background '.repeat(400)}`]);
  addPage(store, book, 24, `quartz distant discussion. ${'context '.repeat(300)}`);
  const capture = chatCapture();
  const result = await retrieval(store, capture).ask({ question: 'Explain quartz from this reader section.',
    readerContext: { bookId: book.bookId, sectionId: book.sectionId, page: 1, scope: 'section' } });
  assert.equal(capture.calls.length, 1);
  assert.ok(result.citations.some(hit => hit.sectionId === book.sectionId && hit.text.includes('Visible section information.')));
  assert.equal(result.context.sourceCoverage, 'partial_retrieved_passages');
});

test('a useful admitted selected range survives optional groups that cannot fit', async t => {
  const store = fixture(t);
  const book = store.add('a', [`Visible range with supported information. ${'selected '.repeat(425)}`]);
  addPage(store, book, 24, `quartz ${'漢'.repeat(3000)}`);
  const capture = chatCapture();
  const result = await retrieval(store, capture).ask({ question: 'Explain the quartz range.',
    readerContext: { bookId: book.bookId, sectionId: book.sectionId, scope: 'section', page: 1,
      charStart: 0, charEnd: book.texts[0].length } });
  assert.equal(capture.calls.length, 1);
  assert.ok(result.citations.some(hit => hit.sectionId === book.sectionId && hit.text === book.texts[0]));
  assert.ok(!result.warnings.some(item => item.code === 'context_group_unavailable'));
});

test('opaque source identity length cannot change sent evidence, while public maps retain full identities', async t => {
  const observations = [];
  for (const longIds of [false, true]) {
    const store = fixture(t, { longIds });
    store.add('a', competingTexts('Alpha', 6), { richLocator: true });
    store.add('b', competingTexts('Beta'), { richLocator: true });
    const before = store.db.prepare('SELECT id,text,locator_json FROM chunks ORDER BY id').all();
    const capture = chatCapture();
    const result = await retrieval(store, capture, { answerLimit: 2,
      maxContextChars: 4096, contextTokens: 12288 }).ask({ question: 'quartz' });
    assert.equal(capture.calls.length, 1);
    const sent = assertExactMap(store, result, capture.calls[0]);
    assert.deepEqual(new Set(result.citations.map(item => item.bookId)),
      new Set(longIds ? ['a'.repeat(64), 'b'.repeat(64)] : ['a', 'b']));
    for (const passage of sent.passages) {
      assert.deepEqual(Object.keys(passage).sort(), ['bookTitle', 'citation', 'locator', 'text']);
      for (const projected of [passage.locator, passage.locator.derived]) {
        for (const key of ['sourceSha256', 'originalSha256', 'sha256', 'anchors']) {
          assert.equal(projected[key], undefined, `Opaque ${key} stays out of the model context`);
        }
      }
      assert.equal(passage.locator.derived.member, 'chapter.xhtml');
      assert.equal(passage.locator.derived.fragment, 'inspection');
    }
    assert.deepEqual(store.db.prepare('SELECT id,text,locator_json FROM chunks ORDER BY id').all(), before);
    observations.push({ sent, inputBytes: result.context.inputBytes });
  }
  assert.deepEqual(observations[1], observations[0]);
});

test('a tight context retains competing anchors and both boundary clauses instead of spending everything on rank one', async t => {
  const store = fixture(t);
  const alpha = store.add('a', competingTexts('Alpha', 6));
  const beta = store.add('b', competingTexts('Beta'));
  const capture = chatCapture();
  const result = await retrieval(store, capture, { answerLimit: 2,
    maxContextChars: 4096, contextTokens: 12288 }).ask({ question: 'quartz' });
  assert.equal(result.hits.length, 2);
  assert.equal(result.hits[0].bookId, alpha.bookId, 'The fixture supplies a dominant first anchor');
  assert.equal(capture.calls.length, 1);
  const sent = assertExactMap(store, result, capture.calls[0]);
  for (const [book, label] of [[alpha, 'Alpha'], [beta, 'Beta']]) {
    const evidence = result.citations.filter(item => item.bookId === book.bookId);
    assert.deepEqual(new Set(evidence.map(item => item.chunkId)), new Set(book.ids));
    const text = evidence.map(item => item.text).join('\n');
    assert.ok(text.includes(`${label} requires the amber seal.`));
    assert.ok(text.includes(`Before ${label}, isolate the supply.`));
    assert.ok(text.includes(`After ${label}, release pressure slowly.`));
  }
  assert.equal(sent.passages.length, 6);
  assert.ok(result.metrics.contextCharacters <= 4096);
  assert.ok(result.citations.some(item => item.provenance.sourceWindow));
});

test('ample space expands a single group back to complete original text and offsets', async t => {
  const store = fixture(t);
  const book = store.add('a', competingTexts('Alpha'));
  const capture = chatCapture();
  const result = await retrieval(store, capture, { answerLimit: 1,
    maxContextChars: 24000, contextTokens: 32768 }).ask({ question: 'quartz', bookId: book.bookId });
  assert.equal(capture.calls.length, 1);
  assertExactMap(store, result, capture.calls[0]);
  assert.equal(result.citations.length, 3);
  for (const [index, chunkId] of book.ids.entries()) {
    const matches = result.citations.filter(item => item.chunkId === chunkId);
    assert.equal(matches.length, 1);
    assert.equal(matches[0].text, book.texts[index]);
    assert.deepEqual(matches[0].locator,
      JSON.parse(store.db.prepare('SELECT locator_json FROM chunks WHERE id=?').get(chunkId).locator_json));
  }
});

test('serialized Unicode and escaped source context obeys its byte budget with exact source slices', async t => {
  const store = fixture(t);
  const padding = '🧭漢字 e\u0301 "quoted"\\\n'.repeat(260);
  const book = store.add('a', [
    `${padding}Before starting, open the bypass.`,
    `quartz valve instructions: use the blue inlet. ${padding}`,
    `After stopping, close the bypass. ${padding}`,
  ]);
  const capture = chatCapture();
  const question = '  quartz valve 🧭  ';
  const result = await retrieval(store, capture, { answerLimit: 1 }).ask({ question, bookId: book.bookId });
  assert.equal(capture.calls.length, 1);
  const sent = assertExactMap(store, result, capture.calls[0]);
  assert.equal(sent.question, question);
  assert.equal(sent.passages.length, 3);
  assert.ok(sent.passages.some(item => item.text.includes('Before starting, open the bypass.')));
  assert.ok(sent.passages.some(item => item.text.includes('After stopping, close the bypass.')));
  assert.ok(sent.passages.some(item => item.text.includes('use the blue inlet.')));
  assert.ok(result.citations.some(item => item.provenance.sourceWindow));
  assert.ok(result.context.inputBytes > result.metrics.contextCharacters);
});

test('a budget unable to admit the leading group makes no model request and reports the missing context', async t => {
  const store = fixture(t);
  store.add('a', competingTexts('Alpha'));
  const capture = chatCapture();
  const result = await retrieval(store, capture, { answerLimit: 1, contextTokens: 2048 })
    .ask({ question: 'quartz' });
  assert.equal(result.hits.length, 1);
  assert.equal(capture.calls.length, 0);
  assert.equal(result.citations.length, 0);
  assert.equal(result.status, 'abstained');
  assert.ok(result.warnings.some(item => item.code === 'context_group_unavailable'));
});

test('overlapping chunks neither double-charge source text nor fill a gap between admitted windows', async t => {
  const marker = 'UNSENT GAP MARKER';
  const source = `quartz ${'a'.repeat(1693)}${marker}${'z'.repeat(2600)}`;
  assert.equal(source.indexOf(marker), 1700);
  for (const maxContextChars of [2048, 8192]) await t.test(`source budget ${maxContextChars}`, async inner => {
    const store = fixture(inner);
    const book = store.add('a', ['quartz placeholder', 'neighbor placeholder']);
    store.db.prepare('UPDATE sections SET text=? WHERE id=?').run(source, book.sectionId);
    for (const [index, [start, end]] of [[0, 2200], [1900, source.length]].entries()) {
      const id = book.ids[index];
      const locator = JSON.parse(store.db.prepare('SELECT locator_json FROM chunks WHERE id=?').get(id).locator_json);
      const text = source.slice(start, end);
      store.db.prepare('UPDATE chunks SET text=?,locator_json=? WHERE id=?')
        .run(text, JSON.stringify({ ...locator, charStart: start, charEnd: end }), id);
      store.db.prepare('UPDATE chunks_fts SET text=? WHERE chunk_id=?').run(text, id);
    }
    const capture = chatCapture();
    const result = await retrieval(store, capture, { answerLimit: 1,
      maxContextChars, contextTokens: 16384 }).ask({ question: 'quartz' });
    assert.equal(capture.calls.length, 1);
    assertExactMap(store, result, capture.calls[0]);
    const ordered = [...result.citations].sort((a, b) => a.locator.charStart - b.locator.charStart);
    for (let index = 1; index < ordered.length; index += 1) {
      assert.ok(ordered[index - 1].locator.charEnd <= ordered[index].locator.charStart,
        'Each original source position consumes the evidence budget at most once');
    }
    if (maxContextChars === 2048) {
      assert.equal(ordered.length, 2);
      assert.ok(ordered[0].locator.charEnd < source.indexOf(marker));
      assert.ok(ordered[1].locator.charStart > source.indexOf(marker) + marker.length);
      assert.ok(ordered.every(item => !item.text.includes(marker)));
      assert.ok(result.metrics.contextCharacters <= maxContextChars);
    } else {
      assert.equal(ordered[0].locator.charStart, 0);
      assert.equal(ordered.at(-1).locator.charEnd, source.length);
      assert.equal(ordered.map(item => item.text).join(''), source);
      assert.equal(result.metrics.contextCharacters, source.length);
    }
  });
});
