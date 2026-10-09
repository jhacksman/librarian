// Original synthetic fixtures. Execute only in the approved Spark Node 24 runtime.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {mkdtemp, mkdir, readFile, readdir, realpath, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';
import {crc32, deflateRawSync, deflateSync} from 'node:zlib';

import {extractBook, inventoryArchive} from '../src/extract.mjs';

const CODE = 'if (ready) {\n  run("a  b");\n\tfinish();\n}';

async function workspace(t) {
  const directory = await mkdtemp(path.join(tmpdir(), 'librarian-extract-'));
  t.after(() => rm(directory, {recursive: true, force: true}));
  return directory;
}

// Write ZIP records independently of the ZIP reader, including the central
// directory. Corruption cases change only the field named by their test.
function zip(entries) {
  const localRecords = [];
  const centralRecords = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name, 'utf8');
    const raw = Buffer.isBuffer(entry.data) ? entry.data : Buffer.from(entry.data ?? '', 'utf8');
    const method = entry.method ?? 0;
    const flags = 0x800 | (entry.flags ?? 0);
    const payload = method === 8 ? deflateRawSync(raw) : raw;
    // Traditional ZIP encryption adds a 12-byte header to the stored size.
    // Its bytes are opaque here: rejection must occur before any decryption.
    const compressed = flags & 1 ? Buffer.concat([Buffer.alloc(12, 0xa5), payload]) : payload;
    const checksum = entry.badCrc ? (crc32(raw) ^ 1) >>> 0 : crc32(raw);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(flags, 6);
    local.writeUInt16LE(method, 8);
    local.writeUInt32LE(checksum, 14);
    local.writeUInt32LE(compressed.length, 18);
    local.writeUInt32LE(raw.length, 22);
    local.writeUInt16LE(name.length, 26);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE((3 << 8) | 20, 4); // Unix creator and ZIP 2.0.
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(flags, 8);
    central.writeUInt16LE(method, 10);
    central.writeUInt32LE(checksum, 16);
    central.writeUInt32LE(compressed.length, 20);
    central.writeUInt32LE(raw.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(((entry.mode ?? 0o100644) << 16) >>> 0, 38);
    central.writeUInt32LE(offset, 42);
    localRecords.push(local, name, compressed);
    centralRecords.push(central, name);
    offset += local.length + name.length + compressed.length;
  }
  const central = Buffer.concat(centralRecords);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(central.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...localRecords, central, end]);
}

function png() {
  const chunk = (type, data) => {
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    const checksum = Buffer.alloc(4);
    checksum.writeUInt32BE(crc32(body));
    return Buffer.concat([length, body, checksum]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(1, 0);
  header.writeUInt32BE(1, 4);
  header[8] = 8;
  header[9] = 6; // One RGBA pixel; scanline starts with filter byte zero.
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header), chunk('IDAT', deflateSync(Buffer.from([0, 20, 80, 40, 255]))),
    chunk('IEND', Buffer.alloc(0))]);
}

function epubEntries({first, second, nav, cover = png()} = {}) {
  const firstMarkup = first ?? '<h1 id="opening">Opening &amp; tools</h1>'
    + '<p>Before <code>alpha&lt;beta &amp;&amp; gamma</code> after.</p>'
    + `<pre><code>${CODE}</code></pre>`
    + '<table><tr><th>Key</th><th>Value</th></tr><tr><td>port</td><td>443</td></tr></table>'
    + '<p>After the table.</p><p hidden="hidden">hidden-secret</p><script>script-secret</script>';
  const secondMarkup = second ?? '<h2 id="closing-note">Closing note</h2><p>Second source follows the opening.</p>';
  const xhtml = (body) => '<?xml version="1.0" encoding="UTF-8"?>'
    + '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>head-only</title></head>'
    + `<body>${body}</body></html>`;
  return [
    {name: 'mimetype', data: 'application/epub+zip'},
    // The archive deliberately presents chapter two before chapter one.
    {name: 'OPS/Text/second.xhtml', data: xhtml(secondMarkup), method: 8},
    {name: 'META-INF/container.xml', data: '<?xml version="1.0"?>'
      + '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
      + '<rootfiles><rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/></rootfiles></container>'},
    {name: 'OPS/package.opf', data: '<?xml version="1.0" encoding="UTF-8"?>'
      + '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">'
      + '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
      + '<dc:title>Harbor &amp; Tools</dc:title><dc:creator>Ada &amp; Bea</dc:creator><dc:creator>Ren&#233; Li</dc:creator>'
      + '<dc:description>Notes &lt;without scripts&gt; &amp; evidence.</dc:description>'
      + '<dc:language>en</dc:language><dc:publisher>Acme &amp; Co</dc:publisher>'
      + '<dc:identifier id="book-id">urn:example:harbor-tools</dc:identifier><dc:date>2026-10-04</dc:date>'
      + '<meta property="dcterms:modified">2026-10-04T00:00:00Z</meta></metadata>'
      + '<manifest><item id="first" href="Text/first.xhtml" media-type="application/xhtml+xml"/>'
      + '<item id="second" href="Text/second.xhtml" media-type="application/xhtml+xml"/>'
      + '<item id="nav" href="Nav/toc.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
      + '<item id="cover" href="Images/cover.png" media-type="image/png" properties="cover-image"/></manifest>'
      + '<spine><itemref idref="first"/><itemref idref="second"/></spine></package>', method: 8},
    {name: 'OPS/Nav/toc.xhtml', data: nav ?? '<?xml version="1.0"?>'
      + '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>'
      + '<nav epub:type="toc"><ol><li><a href="../Text/%66irst.xhtml#opening">Start &amp; tools</a>'
      + '<ol><li><a href="../Text/second.xhtml#closing%2Dnote">Nested close</a></li></ol></li></ol></nav></body></html>', method: 8},
    {name: 'OPS/Images/cover.png', data: cover},
    {name: 'OPS/Text/first.xhtml', data: xhtml(firstMarkup), method: 8},
  ];
}

function pdf(pages, {paddingBytes = 0} = {}) {
  const objects = [null];
  const add = (body) => { objects.push(body); return objects.length - 1; };
  const catalog = add('');
  const pageTree = add('');
  const font = add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
  const kids = [];
  for (const text of pages) {
    const escaped = text.replace(/[\\()]/g, '\\$&');
    const stream = text ? `BT /F1 12 Tf 72 700 Td (${escaped}) Tj ET` : '';
    const contents = add(`<< /Length ${Buffer.byteLength(stream, 'ascii')} >>\nstream\n${stream}\nendstream`);
    kids.push(add(`<< /Type /Page /Parent ${pageTree} 0 R /MediaBox [0 0 612 792]`
      + ` /Resources << /Font << /F1 ${font} 0 R >> >> /Contents ${contents} 0 R >>`));
  }
  objects[catalog] = `<< /Type /Catalog /Pages ${pageTree} 0 R >>`;
  objects[pageTree] = `<< /Type /Pages /Kids [${kids.map((id) => `${id} 0 R`).join(' ')}] /Count ${kids.length} >>`;
  const info = add('<< /Title (Physical Page Notebook) /Author (Lena Example)'
    + ' /Subject (Original three-page fixture) /Creator (Independent fixture writer) >>');
  // Inert header padding grows the source buffer without putting text outside the page.
  const padding = paddingBytes ? `%${'x'.repeat(paddingBytes)}\n` : '';
  const chunks = [Buffer.from(`%PDF-1.4\n%\xe2\xe3\xcf\xd3\n${padding}`, 'latin1')];
  const offsets = [0];
  let length = chunks[0].length;
  for (let id = 1; id < objects.length; id += 1) {
    offsets.push(length);
    const chunk = Buffer.from(`${id} 0 obj\n${objects[id]}\nendobj\n`, 'ascii');
    chunks.push(chunk);
    length += chunk.length;
  }
  const table = offsets.slice(1).map((offset) => `${String(offset).padStart(10, '0')} 00000 n \n`).join('');
  chunks.push(Buffer.from(`xref\n0 ${objects.length}\n0000000000 65535 f \n${table}`
    + `trailer\n<< /Size ${objects.length} /Root ${catalog} 0 R /Info ${info} 0 R >>\nstartxref\n${length}\n%%EOF\n`, 'ascii'));
  return Buffer.concat(chunks);
}

test('EPUB uses its spine, decodes metadata, and resolves nested relative navigation', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'book.epub');
  const bytes = zip(epubEntries());
  await writeFile(source, bytes);
  const book = await extractBook(source, {format: 'epub'});
  assert.equal(book.title, 'Harbor & Tools');
  assert.deepEqual(book.authors, ['Ada & Bea', 'René Li']);
  assert.equal(book.description, 'Notes <without scripts> & evidence.');
  assert.equal(book.publisher, 'Acme & Co');
  assert.equal(book.language, 'en');
  assert.deepEqual(book.metadata.identifiers, ['urn:example:harbor-tools']);
  assert.equal(book.metadata.publicationDate, '2026-10-04');
  assert.deepEqual(book.sections.map(({ordinal, title, locator}) => ({ordinal, title, locator})), [
    {ordinal: 1, title: 'Opening & tools', locator: {format: 'epub', member: 'OPS/Text/first.xhtml', heading: 'opening',
      anchors: {opening: {charStart: 0, charEnd: 15}}}},
    {ordinal: 2, title: 'Closing note', locator: {format: 'epub', member: 'OPS/Text/second.xhtml', heading: 'closing-note',
      anchors: {'closing-note': {charStart: 0, charEnd: 12}}}},
  ]);
  assert.deepEqual(book.toc, [
    {title: 'Start & tools', locator: {format: 'epub', member: 'OPS/Text/first.xhtml', heading: 'opening',
      charStart: 0, charEnd: 15, offsetBasis: 'section_utf16_code_units'}},
    {title: 'Nested close', locator: {format: 'epub', member: 'OPS/Text/second.xhtml', heading: 'closing-note',
      charStart: 0, charEnd: 12, offsetBasis: 'section_utf16_code_units'}},
  ]);
  assert.equal(book.coverPath, null);
  assert.deepEqual(await readFile(source), bytes);
  assert.deepEqual(await readdir(root), ['book.epub']);
});

test('EPUB fragments in one member select distinct normalized Unicode headings at exact UTF-16 offsets', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'fragments.epub');
  const first = ' \n<div>\n <h1 id="early">  Early &#x1F642; \n topic  </h1>\n </div>'
    + '<p>Repeated title: Later 🧭 topic.</p>'
    + '<pre>\r\n  x = 1;\r\n\t y = 2;\r\n</pre>'
    + '<table><tr><td> \t row &#x3B2;  </td><td>value</td></tr></table>'
    + '<div> \n <h2 id="later">\tLater <em>🧭</em> \n topic  </h2> \n </div>';
  const nav = '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>'
    + '<nav epub:type="toc"><ol><li><a href="../Text/first.xhtml#early">Early</a></li>'
    + '<li><a href="../Text/first.xhtml#later">Later</a></li></ol></nav></body></html>';
  await writeFile(source, zip(epubEntries({first, nav})));
  const book = await extractBook(source, {format: 'epub'});
  const section = book.sections[0];
  assert.equal(book.toc.length, 2);
  const expected = ['Early 🙂 topic', 'Later 🧭 topic'];
  for (const [index, entry] of book.toc.entries()) {
    const {member, heading, charStart, charEnd, offsetBasis} = entry.locator;
    assert.equal(member, 'OPS/Text/first.xhtml');
    assert.equal(offsetBasis, 'section_utf16_code_units');
    assert.ok(Number.isInteger(charStart) && Number.isInteger(charEnd) && charStart >= 0 && charEnd <= section.text.length);
    assert.deepEqual(section.locator.anchors[heading], {charStart, charEnd});
    assert.equal(section.text.slice(charStart, charEnd), expected[index]);
    assert.equal(charEnd - charStart, expected[index].length);
  }
  const early = book.toc[0].locator;
  const later = book.toc[1].locator;
  assert.equal(early.charStart, 0);
  assert.ok(later.charStart > early.charEnd);
  assert.equal(later.charStart, section.text.lastIndexOf(expected[1]));
  assert.notEqual(later.charStart, section.text.indexOf(expected[1]), 'An earlier repeated title is not the fragment destination');
  assert.notEqual(later.charStart, [...section.text.slice(0, later.charStart)].length, 'Offsets count UTF-16 units, not Unicode code points');
  assert.ok(section.text.includes('  x = 1;\n\t y = 2;'));
});

test('EPUB omits offsets for ambiguous or hidden fragments and preserves an empty legacy anchor destination', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'unresolved-fragments.epub');
  const first = '<h1>Visible chapter</h1><h2 id="duplicate">First duplicate</h2><p id="duplicate">Second duplicate</p>'
    + '<section hidden="hidden"><h2 id="hidden-fragment">Secret heading</h2></section>'
    + '<a name="legacy"></a><h2>🙂 Last visible heading</h2>';
  const nav = '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc"><ol>'
    + ['duplicate', 'hidden-fragment', 'missing', 'legacy'].map((id) => `<li><a href="../Text/first.xhtml#${id}">${id}</a></li>`).join('')
    + '</ol></nav></body></html>';
  await writeFile(source, zip(epubEntries({first, nav})));
  const book = await extractBook(source, {format: 'epub'});
  const section = book.sections[0];
  assert.doesNotMatch(section.text, /Secret heading/);
  for (const entry of book.toc.slice(0, 3)) {
    assert.equal(Object.hasOwn(entry.locator, 'charStart'), false);
    assert.equal(Object.hasOwn(entry.locator, 'charEnd'), false);
    assert.equal(Object.hasOwn(section.locator.anchors, entry.locator.heading), false);
  }
  assert.equal(book.warnings.filter((warning) => warning.includes('no unambiguous extracted-text location')).length, 3);
  const legacy = book.toc[3].locator;
  assert.equal(legacy.charStart, section.text.indexOf('🙂 Last visible heading'));
  assert.equal(legacy.charEnd - legacy.charStart, 2);
  assert.equal(section.text.slice(legacy.charStart, legacy.charEnd), '🙂');
});

test('EPUB retains inline code continuity, preformatted whitespace, and table cell boundaries', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'text.epub');
  await writeFile(source, zip(epubEntries()));
  const book = await extractBook(source, {format: 'epub'});
  const text = book.sections[0].text;
  assert.ok(text.includes('Before alpha<beta && gamma after.'));
  assert.ok(text.includes(CODE), 'Authored indentation, double spaces, and tab must survive');
  assert.match(text, /Key\tValue/);
  assert.match(text, /port\t443/);
  assert.ok(text.indexOf(CODE) < text.indexOf('Key\tValue'));
  assert.ok(text.indexOf('port\t443') < text.indexOf('After the table.'));
  assert.doesNotMatch(text, /head-only|hidden-secret|script-secret/);
});

test('EPUB retains original raster cover bytes only in the requested output directory', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'cover.epub');
  const outputDir = path.join(root, 'covers');
  const cover = png();
  const bytes = zip(epubEntries({cover}));
  await writeFile(source, bytes);
  const book = await extractBook(source, {format: 'epub', outputDir});
  assert.equal(typeof book.coverPath, 'string');
  assert.equal(path.dirname(book.coverPath), await realpath(outputDir));
  assert.equal(path.extname(book.coverPath), '.png');
  assert.deepEqual(await readFile(book.coverPath), cover);
  assert.deepEqual(await readdir(outputDir), [path.basename(book.coverPath)]);
  assert.deepEqual(await readFile(source), bytes);
});

test('EPUB refuses a symlink cover directory without failing usable book text', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'cover.epub');
  const realOutput = path.join(root, 'untouched');
  const outputDir = path.join(root, 'linked-output');
  await mkdir(realOutput);
  await symlink(realOutput, outputDir, 'dir');
  await writeFile(source, zip(epubEntries()));
  const book = await extractBook(source, {format: 'epub', outputDir});
  assert.equal(book.coverPath, null);
  assert.ok(book.sections[0].text.includes('After the table.'));
  assert.ok(book.warnings.some((warning) => warning.includes('Cover unavailable: INVALID_OUTPUT')));
  assert.deepEqual(await readdir(realOutput), []);
});

test('EPUB rejects a text member whose CRC disagrees with the retained bytes', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'bad-crc.epub');
  const entries = epubEntries().map((entry) => entry.name === 'OPS/Text/first.xhtml' ? {...entry, badCrc: true} : entry);
  await writeFile(source, zip(entries));
  await assert.rejects(extractBook(source, {format: 'epub'}), {code: 'CORRUPT_ARCHIVE'});
});

for (const [label, limits] of [
  ['individual section characters', {maxSectionChars: 5}],
  ['aggregate text characters across sections', {maxTextChars: 15}],
]) {
  test(`EPUB rejects excess ${label} instead of silently truncating`, async (t) => {
    const root = await workspace(t);
    const source = path.join(root, 'bounded.epub');
    await writeFile(source, zip(epubEntries({first: '<p>alpha beta</p>', second: '<p>gamma delta</p>'})));
    await assert.rejects(extractBook(source, {format: 'epub', limits}), {code: 'LIMIT_EXCEEDED'});
  });
}

for (const [label, limits] of [
  ['expanded member bytes', {maxEntryBytes: 3}],
  ['compression ratio', {maxCompressionRatio: 2}],
]) {
  test(`EPUB enforces its ${label} limit before reading member content`, async (t) => {
    const root = await workspace(t);
    const source = path.join(root, 'bounded-members.epub');
    await writeFile(source, zip(epubEntries({first: `<p>${'A'.repeat(512)}</p>`})));
    await assert.rejects(extractBook(source, {format: 'epub', limits}), {code: 'LIMIT_EXCEEDED'});
  });
}

test('PDF preserves physical pages around a blank page and reports that OCR was not performed', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'pages.pdf');
  const texts = ['First physical page signal.', '', 'Third physical page anchor.'];
  const bytes = pdf(texts);
  await writeFile(source, bytes);
  const book = await extractBook(source, {format: 'pdf'});
  assert.equal(book.title, 'Physical Page Notebook');
  assert.deepEqual(book.authors, ['Lena Example']);
  assert.equal(book.metadata.pageCount, 3);
  assert.deepEqual(book.metadata.pagesWithoutText, [2]);
  assert.equal(book.metadata.ocrPerformed, false);
  assert.equal(book.metadata.extractionCompleteness, 'not_assessed');
  assert.deepEqual(book.sections.map(({ordinal, locator, text}) => ({ordinal, locator, text})), texts.map((text, index) => ({
    ordinal: index + 1, locator: {format: 'pdf', page: index + 1}, text,
  })));
  assert.ok(book.warnings.some((warning) => /OCR/i.test(warning)));
  assert.ok(book.warnings.some((warning) => /blank|image.only|without.*text|no.*text/i.test(warning)));
  assert.equal(book.coverPath, null);
  assert.deepEqual(await readFile(source), bytes);
  assert.deepEqual(await readdir(root), ['pages.pdf']);
});

test('PDF reports an empty source distinctly from an oversized source', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'empty.pdf');
  await writeFile(source, Buffer.alloc(0));
  await assert.rejects(extractBook(source, {format: 'pdf'}), {
    code: 'EMPTY_SOURCE', message: /empty \(0 bytes\)/,
  });
  assert.equal((await readFile(source)).length, 0);
  assert.deepEqual(await readdir(root), ['empty.pdf']);
});

for (const [label, limits] of [
  ['physical page count', {maxPages: 2}],
  ['source input bytes', {maxSourceBytes: 100}],
  ['PDF input bytes', {maxPdfBytes: 100}],
]) {
  test(`PDF rejects excess ${label} explicitly`, async (t) => {
    const root = await workspace(t);
    const source = path.join(root, 'bounded.pdf');
    await writeFile(source, pdf(['One.', '', 'Three.']));
    await assert.rejects(extractBook(source, {format: 'pdf', limits}), {code: 'LIMIT_EXCEEDED'});
  });
}

test('ZIP inventory reports declared contents without claiming payload CRC verification or extracting files', async (t) => {
  const root = await workspace(t);
  const source = path.join(root, 'inventory.zip');
  const bytes = zip([
    {name: 'notes/', data: '', mode: 0o40755},
    {name: 'notes/readme.txt', data: 'alpha', method: 8, badCrc: true},
    {name: 'book.epub', data: 'bravo'},
  ]);
  await writeFile(source, bytes);
  const archive = await inventoryArchive(source);
  assert.equal(archive.format, 'zip');
  assert.equal(archive.contentsVerified, false);
  assert.equal(archive.totalUncompressedBytes, 10);
  assert.deepEqual(archive.entries.map(({name, uncompressedBytes, directory}) => ({name, uncompressedBytes, directory})), [
    {name: 'notes/', uncompressedBytes: 0, directory: true},
    {name: 'notes/readme.txt', uncompressedBytes: 5, directory: false},
    {name: 'book.epub', uncompressedBytes: 5, directory: false},
  ]);
  assert.ok(archive.warnings.some((warning) => /CRC.*not.*verified/i.test(warning)));
  assert.deepEqual(await readFile(source), bytes);
  assert.deepEqual(await readdir(root), ['inventory.zip']);
});

const unsafeArchives = [
  {name: 'parent traversal', entries: [{name: '../outside.txt', data: 'escape'}], message: /(?:relative|unsafe).*path/i},
  {name: 'absolute member', entries: [{name: '/outside.txt', data: 'escape'}], message: /(?:absolute|unsafe).*path/i},
  {name: 'duplicate members', entries: [{name: 'same.txt', data: 'one'}, {name: 'same.txt', data: 'two'}], code: 'UNSAFE_ARCHIVE'},
  {name: 'Unix symbolic link', entries: [{name: 'link', data: 'target.txt', mode: 0o120777}], code: 'UNSAFE_ARCHIVE'},
  {name: 'encrypted member', entries: [{name: 'secret.txt', data: 'opaque', flags: 1}], code: 'ENCRYPTED_ARCHIVE'},
  {name: 'unsupported compression', entries: [{name: 'packed.txt', data: 'opaque', method: 12}], code: 'UNSUPPORTED_COMPRESSION'},
];
for (const fixture of unsafeArchives) {
  test(`ZIP inventory explicitly rejects ${fixture.name}`, async (t) => {
    const root = await workspace(t);
    const source = path.join(root, 'unsafe.zip');
    await writeFile(source, zip(fixture.entries));
    await assert.rejects(inventoryArchive(source), fixture.code ? {code: fixture.code} : {message: fixture.message});
  });
}

const boundedArchives = [
  {name: 'source bytes', entries: [{name: 'one.txt', data: '1234'}], limits: {maxSourceBytes: 10}},
  {name: 'entry count', entries: [{name: 'one.txt', data: '1'}, {name: 'two.txt', data: '2'}], limits: {maxZipEntries: 1}},
  {name: 'aggregate expanded bytes', entries: [{name: 'one.txt', data: '1234'}, {name: 'two.txt', data: '5678'}], limits: {maxTotalUncompressedBytes: 7}},
];
for (const fixture of boundedArchives) {
  test(`ZIP inventory enforces its ${fixture.name} limit`, async (t) => {
    const root = await workspace(t);
    const source = path.join(root, 'bounded.zip');
    await writeFile(source, zip(fixture.entries));
    await assert.rejects(inventoryArchive(source, {limits: fixture.limits}), {code: 'LIMIT_EXCEEDED'});
  });
}

for (const [label, limits] of [
  ['expanded member bytes', {maxEntryBytes: 3}],
  ['compression ratio', {maxCompressionRatio: 2}],
]) {
  test(`ZIP inventory accepts declarations exceeding the extraction ${label} limit`, async (t) => {
    const root = await workspace(t);
    const source = path.join(root, 'inventory-only.zip');
    const bytes = zip([{name: 'repeated.txt', data: 'A'.repeat(512), method: 8, badCrc: true}]);
    await writeFile(source, bytes);
    const archive = await inventoryArchive(source, {limits});
    assert.equal(archive.contentsVerified, false);
    assert.equal(archive.totalUncompressedBytes, 512);
    assert.equal(archive.entries.length, 1);
    assert.equal(archive.entries[0].name, 'repeated.txt');
    assert.equal(archive.entries[0].uncompressedBytes, 512);
    assert.deepEqual(await readFile(source), bytes);
    assert.deepEqual(await readdir(root), ['inventory-only.zip']);
  });
}

for (const format of ['mobi', 'prc', 'torrent']) {
  test(`${format.toUpperCase()} is explicitly unsupported rather than interpreted as text`, async (t) => {
    const root = await workspace(t);
    const source = path.join(root, `unsupported.${format}`);
    const bytes = Buffer.from('Opaque original fixture; do not silently import this as prose.');
    await writeFile(source, bytes);
    await assert.rejects(extractBook(source, {format}), {code: 'UNSUPPORTED_FORMAT'});
    assert.deepEqual(await readFile(source), bytes);
  });
}

test('extraction worker returns exactly one structured JSON error for malformed input', () => {
  const worker = fileURLToPath(new URL('../src/extract-worker.mjs', import.meta.url));
  const child = spawnSync(process.execPath, [worker], {
    input: '{not valid JSON}\n', encoding: 'utf8', timeout: 10000, maxBuffer: 65536,
    env: {...process.env, DISABLE_SYSTEM_FONTS_LOAD: '1'},
  });
  assert.equal(child.error, undefined);
  assert.equal(child.signal, null);
  assert.match(child.stdout, /^\{[^\r\n]*\}\n$/);
  const result = JSON.parse(child.stdout);
  assert.equal(result.ok, false);
  assert.equal(result.error.code, 'INVALID_REQUEST');
  assert.equal(typeof result.error.message, 'string');
  assert.ok(result.error.message.length > 0);
});
// Append to app/test/extract.test.mjs, which supplies workspace/zip/epubEntries/png/pdf.
// This is authored source only. Execute through the Spark coordinator.

test('EPUB source metadata retains repeated values, creator roles, refinements and exact package bytes', async t => {
  const root = await workspace(t), source = path.join(root, 'rich.epub');
  const entries = epubEntries();
  const opf = entries.find(entry => entry.name === 'OPS/package.opf');
  opf.data = opf.data.replace('</metadata>', '<dc:subject id="subject-1">Computing</dc:subject>'
    + '<dc:subject id="subject-2">Networks</dc:subject>'
    + '<dc:creator id="editor" opf:role="edt" opf:file-as="Example, Pat">Pat Example</dc:creator>'
    + '<meta refines="#editor" property="role" scheme="marc:relators">edt</meta>'
    + '<meta property="vendor:edition">3</meta><dc:rights>Retained &amp; attributed</dc:rights></metadata>');
  await writeFile(source, zip(entries));
  const book = await extractBook(source, {format: 'epub', metadataOnly: true});
  const metadata = book.metadata.sourceMetadata;
  assert.equal(metadata.format, 'epub');
  assert.equal(metadata.packageAttributes['unique-identifier'], 'book-id');
  assert.equal(Buffer.from(metadata.rawPackage.value, 'base64').toString('utf8'), opf.data);
  assert.deepEqual(metadata.entries.filter(entry => entry.name === 'dc:subject').map(entry => entry.value), ['Computing', 'Networks']);
  const creator = metadata.entries.find(entry => entry.attributes.id === 'editor');
  assert.equal(creator.attributes['opf:role'], 'edt');
  assert.equal(creator.attributes['opf:file-as'], 'Example, Pat');
  assert.equal(metadata.entries.find(entry => entry.attributes.refines === '#editor').attributes.scheme, 'marc:relators');
  assert.equal(book.metadata.edition, '3');
  assert.ok(metadata.metadataTree.children.some(node => node?.tag === 'dc:rights'));
});

test('metadata-only EPUB captures source metadata and thumbnail without reading corrupt chapter or navigation content', async t => {
  const root = await workspace(t), source = path.join(root, 'metadata-only.epub');
  const entries = epubEntries().map(entry => /(?:Text|Nav)\//.test(entry.name) ? {...entry, badCrc: true} : entry);
  await writeFile(source, zip(entries));
  const book = await extractBook(source, {format: 'epub', metadataOnly: true, limits: {maxTextChars: 1, maxSectionChars: 1}});
  assert.deepEqual(book.sections, []); assert.deepEqual(book.toc, []);
  assert.equal(book.metadata.metadataOnly, true);
  assert.equal(Object.hasOwn(book.metadata, 'searchable'), false);
  assert.equal(book.metadata.sourceMetadata.packageMember, 'OPS/package.opf');
  assert.equal(book.thumbnail.kind, 'cover'); assert.equal(book.thumbnail.mime, 'image/png');
  assert.ok(book.thumbnail.width <= 256 && book.thumbnail.height <= 384);
  assert.ok(Buffer.from(book.thumbnail.base64, 'base64').length <= 256 * 1024);
  assert.equal(book.thumbnail.provenance.member, 'OPS/Images/cover.png');
  await assert.rejects(extractBook(source, {format: 'epub'}), {code: 'CORRUPT_ARCHIVE'});
});

test('source metadata uses its own byte budget and fails explicitly instead of truncating', async t => {
  const root = await workspace(t), source = path.join(root, 'metadata-budget.epub');
  const entries = epubEntries(), opf = entries.find(entry => entry.name === 'OPS/package.opf');
  opf.data = opf.data.replace('</metadata>', `<!--${'Original metadata comment '.repeat(4000)}--></metadata>`);
  await writeFile(source, zip(entries));
  const book = await extractBook(source, {format: 'epub', metadataOnly: true});
  assert.ok(Buffer.byteLength(JSON.stringify(book.metadata.sourceMetadata)) > 65536);
  assert.equal(Buffer.from(book.metadata.sourceMetadata.rawPackage.value, 'base64').toString('utf8'), opf.data);
  await assert.rejects(extractBook(source, {format: 'epub', metadataOnly: true, limits: {maxSourceMetadataBytes: 128}}), {code: 'LIMIT_EXCEEDED'});
});

test('metadata-only PDF retains complete Info and renders an honestly labelled first page without text extraction', async t => {
  const root = await workspace(t), source = path.join(root, 'metadata-only.pdf');
  await writeFile(source, pdf(['First physical page signal.', '', 'Third physical page anchor.']));
  const book = await extractBook(source, {format: 'pdf', metadataOnly: true, limits: {maxTextChars: 1, maxSectionChars: 1}});
  assert.deepEqual(book.sections, []); assert.deepEqual(book.toc, []);
  assert.equal(book.metadata.sourceMetadata.info.Subject, 'Original three-page fixture');
  assert.equal(book.metadata.sourceMetadata.info.Creator, 'Independent fixture writer');
  assert.ok(Object.hasOwn(book.metadata.sourceMetadata, 'xmp'));
  assert.ok(Object.hasOwn(book.metadata.sourceMetadata, 'rawXmp'));
  assert.equal(book.metadata.pageCount, 3); assert.equal(Object.hasOwn(book.metadata, 'searchable'), false);
  assert.equal(book.thumbnail.kind, 'first_page'); assert.equal(book.thumbnail.provenance.page, 1);
  assert.ok(book.thumbnail.width <= 256 && book.thumbnail.height <= 384);
  assert.ok(Buffer.from(book.thumbnail.base64, 'base64').length <= 256 * 1024);
});

test('unsupported EPUB cover bytes leave usable metadata and an explicit cover warning', async t => {
  const root = await workspace(t), source = path.join(root, 'invalid-cover.epub');
  await writeFile(source, zip(epubEntries({cover: Buffer.from('<svg xmlns="http://www.w3.org/2000/svg"/>')})));
  const book = await extractBook(source, {format: 'epub', metadataOnly: true});
  assert.equal(book.thumbnail, undefined); assert.equal(book.coverPath, null);
  assert.ok(book.warnings.some(warning => warning.includes('UNSUPPORTED_COVER')));
  assert.equal(book.metadata.sourceMetadata.format, 'epub');
});

test('EPUB guide cover document resolves one local raster image and records the declaration', async t => {
  const root = await workspace(t), source = path.join(root, 'guide-cover.epub');
  const entries = epubEntries(), opf = entries.find(entry => entry.name === 'OPS/package.opf');
  opf.data = opf.data.replace(' properties="cover-image"', '')
    .replace('</manifest>', '<item id="cover-document" href="Text/cover.xhtml" media-type="application/xhtml+xml"/></manifest>')
    .replace('</package>', '<guide><reference type="cover" href="Text/cover.xhtml"/></guide></package>');
  entries.push({name: 'OPS/Text/cover.xhtml', data: '<html><body><img src="../Images/cover.png"/></body></html>'});
  await writeFile(source, zip(entries));
  const book = await extractBook(source, {format: 'epub', metadataOnly: true});
  assert.equal(book.thumbnail.kind, 'cover');
  assert.equal(book.thumbnail.provenance.declaration, 'epub_guide_cover');
  assert.equal(book.thumbnail.provenance.coverDocument, 'OPS/Text/cover.xhtml');
  assert.equal(book.thumbnail.provenance.member, 'OPS/Images/cover.png');
});

// Rebuild the existing original synthetic PDF with a real XMP stream and custom
// Info fields. Offsets are regenerated rather than patched into a broken PDF.
function pdfWithRichMetadata({includeXmp = true, customPadding = ''} = {}) {
  const original = pdf(['Metadata fixture page.']).toString('latin1');
  const objects = [null];
  // Leave the separator newline available to the following object's match.
  for (const match of original.matchAll(/\n(\d+) 0 obj\n([\s\S]*?)\nendobj(?=\n)/g)) objects[Number(match[1])] = match[2];
  const originalSize = Number(/\/Size (\d+)/.exec(original)[1]);
  assert.equal(objects.length, originalSize, 'Synthetic PDF object table must be complete before rebuilding');
  for (let id = 1; id < originalSize; id++) assert.equal(typeof objects[id], 'string', `Missing synthetic PDF object ${id}`);
  const rootId = Number(/\/Root (\d+) 0 R/.exec(original)[1]);
  const infoId = Number(/\/Info (\d+) 0 R/.exec(original)[1]);
  const xmp = '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    + '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    + '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:vendor="urn:fixture:vendor">'
    + '<dc:identifier>urn:isbn:9780131103627</dc:identifier><dc:date>1988</dc:date>'
    + '<dc:subject><rdf:Bag><rdf:li>Computing</rdf:li><rdf:li>Tools</rdf:li></rdf:Bag></dc:subject>'
    + '<vendor:opaque vendor:attribute="preserve">Unknown custom value</vendor:opaque>'
    + '</rdf:Description></rdf:RDF></x:xmpmeta>';
  const xmpId = objects.length;
  objects.push(`<< /Type /Metadata /Subtype /XML /Length ${Buffer.byteLength(xmp)} >>\nstream\n${xmp}\nendstream`);
  if (includeXmp) objects[rootId] = objects[rootId].replace(/>>$/, `/Metadata ${xmpId} 0 R >>`);
  assert.match(customPadding, /^x*$/, 'Custom padding must be literal PDF-safe fixture text');
  objects[infoId] = objects[infoId].replace(/>>$/, '/ISBN (9780131103627) /Edition (2) /CustomFixtureKey (All Info retained)'
    + ' /CustomNumber 42 /CustomBoolean false /CustomName /FixtureName'
    + (customPadding ? ` /CustomPadding (${customPadding})` : '') + ' >>');
  const chunks = [Buffer.from('%PDF-1.4\n%\xe2\xe3\xcf\xd3\n', 'latin1')], offsets = [0];
  let length = chunks[0].length;
  for (let id = 1; id < objects.length; id++) {
    offsets.push(length);
    const chunk = Buffer.from(`${id} 0 obj\n${objects[id]}\nendobj\n`, 'utf8');
    chunks.push(chunk); length += chunk.length;
  }
  const table = offsets.slice(1).map(offset => `${String(offset).padStart(10, '0')} 00000 n \n`).join('');
  chunks.push(Buffer.from(`xref\n0 ${objects.length}\n0000000000 65535 f \n${table}`
    + `trailer\n<< /Size ${objects.length} /Root ${rootId} 0 R /Info ${infoId} 0 R >>\nstartxref\n${length}\n%%EOF\n`));
  return {bytes: Buffer.concat(chunks), xmp};
}

test('PDF capture keeps XMP arrays, unknown namespace attributes and complete custom Info', async t => {
  const root = await workspace(t), source = path.join(root, 'rich-metadata.pdf');
  const fixture = pdfWithRichMetadata(); await writeFile(source, fixture.bytes);
  // Verify the synthetic input independently of Librarian's metadata capture.
  const {getDocument} = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const parserTask = getDocument({data: Uint8Array.from(fixture.bytes), verbosity: 0, stopAtErrors: true,
    useSystemFonts: false, disableFontFace: true, enableXfa: false, useWasm: false});
  let expectedInfo, expectedXmp;
  try {
    const document = await parserTask.promise;
    const {info, metadata: parsed} = await document.getMetadata();
    assert.ok(info.Custom instanceof Map, 'Pinned PDF.js returns custom Info as a Map');
    assert.deepEqual(Object.fromEntries(info.Custom), {ISBN: '9780131103627', Edition: '2',
      CustomFixtureKey: 'All Info retained', CustomNumber: 42, CustomBoolean: false, CustomName: {name: 'FixtureName'}});
    expectedInfo = {...info, Custom: Object.fromEntries(info.Custom)};
    assert.equal(typeof parsed?.[Symbol.iterator], 'function', 'Pinned PDF.js metadata must be iterable');
    assert.deepEqual(parsed.get('dc:subject'), ['Computing', 'Tools'], 'The fixture must contain the expected XMP array');
    assert.equal(parsed.getRaw(), fixture.xmp, 'The parser must retain the fixture XMP stream');
    expectedXmp = Object.fromEntries(parsed);
  } finally { await parserTask.destroy(); }
  const book = await extractBook(source, {format: 'pdf', metadataOnly: true});
  const metadata = book.metadata.sourceMetadata;
  assert.deepEqual(metadata.xmp['dc:subject'], ['Computing', 'Tools']);
  assert.equal(metadata.xmp['vendor:opaque'], 'Unknown custom value');
  assert.match(metadata.rawXmp, /vendor:attribute="preserve"/);
  assert.match(metadata.rawXmp, /Unknown custom value/);
  assert.equal(metadata.info.Custom.CustomFixtureKey, 'All Info retained');
  assert.deepEqual(metadata.info, expectedInfo, 'Every PDF.js-reported Info field must be retained');
  const retained = JSON.parse(JSON.stringify(book));
  assert.deepEqual(retained.metadata.sourceMetadata.info, expectedInfo, 'Custom values must survive worker/storage JSON serialization');
  assert.deepEqual(retained.metadata.sourceMetadata.xmp, expectedXmp, 'Every parsed XMP field must survive JSON serialization');
  assert.ok(book.metadata.identifiers.includes('urn:isbn:9780131103627'));
  assert.ok(book.metadata.identifiers.includes('9780131103627'), 'Custom Info ISBN must supplement XMP identifiers');
  assert.equal(book.metadata.publicationDate, '1988');
  assert.equal(book.metadata.edition, '2');
  await assert.rejects(extractBook(source, {format: 'pdf', metadataOnly: true, limits: {maxSourceMetadataBytes: 128}}), {code: 'LIMIT_EXCEEDED'});
  const infoOnlySource = path.join(root, 'custom-info-only.pdf');
  await writeFile(infoOnlySource, pdfWithRichMetadata({includeXmp: false}).bytes);
  const infoOnly = await extractBook(infoOnlySource, {format: 'pdf', metadataOnly: true});
  assert.deepEqual(infoOnly.metadata.sourceMetadata.xmp, {});
  assert.equal(infoOnly.metadata.sourceMetadata.rawXmp, null);
  assert.deepEqual(infoOnly.metadata.identifiers, ['9780131103627']);
  assert.equal(infoOnly.metadata.edition, '2', 'Edition must be projected from custom Info without XMP');
  const customHeavySource = path.join(root, 'custom-info-heavy.pdf');
  await writeFile(customHeavySource, pdfWithRichMetadata({includeXmp: false, customPadding: 'x'.repeat(65536)}).bytes);
  await assert.rejects(extractBook(customHeavySource, {format: 'pdf', metadataOnly: true,
    limits: {maxSourceMetadataBytes: 65536}}), {code: 'LIMIT_EXCEEDED'}, 'The source metadata budget must count custom Info entries');
});

test('opt-in native front matter captures only first8physical pages without creating indexed text or replacing embedded fields', async t => {
  const root = await workspace(t), source = path.join(root, 'front-matter.pdf');
  const pages = Array.from({length: 10}, (_, i) => `Physical source page ${i + 1}`);
  // Keep the dedicated-buffer/source-hash regression independent of visible text length.
  const fixture = pdf(pages, {paddingBytes: 8192}); assert.ok(fixture.length > 8192);
  const expectedSourceSha256 = createHash('sha256').update(fixture).digest('hex');
  await writeFile(source, fixture);
  const observed = await extractBook(source, {format: 'pdf', metadataOnly: true, frontMatter: true,
    limits: {maxTextChars: 1, maxSectionChars: 1}});
  assert.deepEqual(observed.sections, []); assert.equal(observed.metadata.textCharacters, 0);
  const evidence = observed.metadata.sourceMetadata.frontMatter;
  assert.equal(evidence.sourceSha256, expectedSourceSha256);
  assert.equal(evidence.pages.length, 8); assert.equal(evidence.pages[7].text, pages[7]);
  assert.deepEqual(evidence.pages.map(page => page.text), pages.slice(0, 8));
  assert.equal(evidence.wholeBookInspected, false); assert.equal(evidence.ocrPerformed, false);
  assert.equal(observed.title, 'Physical Page Notebook'); assert.deepEqual(observed.authors, ['Lena Example']);
  const defaultResult = await extractBook(source, {format: 'pdf', metadataOnly: true});
  assert.equal(defaultResult.metadata.sourceMetadata.frontMatter, undefined);
});

test('front matter records blank native pages honestly and rejects malformed capture options', async t => {
  const root = await workspace(t), source = path.join(root, 'blank-front-matter.pdf');
  await writeFile(source, pdf(['', 'Copyright source signal']));
  const observed = await extractBook(source, {format: 'pdf', metadataOnly: true, frontMatter: true});
  assert.equal(observed.metadata.sourceMetadata.frontMatter.pages[0].text, '');
  assert.equal(observed.metadata.sourceMetadata.frontMatter.pages[1].text, 'Copyright source signal');
  await assert.rejects(extractBook(source, {format: 'pdf', frontMatter: 'yes'}), {code: 'INVALID_REQUEST'});
});
