// Authored fixtures; execute only in the approved Spark Linux Node 24 runtime.
// The fake bwrap below checks the wrapper protocol, not kernel sandbox isolation.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {chmod, mkdtemp, mkdir, readFile, readdir, realpath, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {test} from 'node:test';
import {crc32} from 'node:zlib';

import {convertBook} from '../src/convert.mjs';
import {extractBook} from '../src/extract.mjs';

const linuxOnly = {skip: process.platform !== 'linux'};
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// A stored ZIP with an independent central directory; no archive-writer package.
function derivedEpub() {
  const entries = [
    ['mimetype', 'application/epub+zip'],
    ['META-INF/container.xml', '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="OPS/book.opf" media-type="application/oebps-package+xml"/></rootfiles></container>'],
    ['OPS/book.opf', '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Retained conversion</dc:title><dc:creator>Original Author</dc:creator><dc:language>en</dc:language><dc:identifier id="id">urn:fixture:conversion</dc:identifier><meta property="dcterms:modified">2026-10-04T00:00:00Z</meta></metadata><manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/></manifest><spine><itemref idref="chapter"/></spine></package>'],
    ['OPS/chapter.xhtml', '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Chapter</title></head><body><h1 id="retained">Retained chapter</h1><p>Original-format evidence survives conversion.</p></body></html>'],
    ['OPS/nav.xhtml', '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc"><ol><li><a href="chapter.xhtml#retained">Retained chapter</a></li></ol></nav></body></html>'],
  ];
  const locals = []; const centrals = []; let offset = 0;
  for (const [filename, text] of entries) {
    const name = Buffer.from(filename); const bytes = Buffer.from(text); const checksum = crc32(bytes);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x800, 6);
    local.writeUInt32LE(checksum, 14); local.writeUInt32LE(bytes.length, 18); local.writeUInt32LE(bytes.length, 22);
    local.writeUInt16LE(name.length, 26);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE((3 << 8) | 20, 4);
    central.writeUInt16LE(20, 6); central.writeUInt16LE(0x800, 8); central.writeUInt32LE(checksum, 16);
    central.writeUInt32LE(bytes.length, 20); central.writeUInt32LE(bytes.length, 24);
    central.writeUInt16LE(name.length, 28); central.writeUInt32LE((0o100644 << 16) >>> 0, 38);
    central.writeUInt32LE(offset, 42);
    locals.push(local, name, bytes); centrals.push(central, name); offset += local.length + name.length + bytes.length;
  }
  const directory = Buffer.concat(centrals); const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, directory, end]);
}

async function fixture(t, {format = 'mobi', behavior = 'success', isolation = 'container'} = {}) {
  const root = await realpath(await mkdtemp(path.join(tmpdir(), 'librarian-convert-')));
  t.after(() => rm(root, {recursive: true, force: true}));
  const source = path.join(root, `source ; literal.${format}`);
  const derivedDir = path.join(root, 'derived');
  const executable = path.join(root, 'fake-converter.cjs');
  const sandboxExecutable = path.join(root, 'fake-bwrap.cjs');
  const report = path.join(root, 'converter-report.json');
  const protocol = path.join(root, 'sandbox-report.json');
  const original = Buffer.from(`Original ${format.toUpperCase()} fixture, retained byte for byte.`);
  const epub = derivedEpub();
  await writeFile(source, original); await mkdir(derivedDir);
  const converterScript = `#!${process.execPath}
const fs = require('node:fs');
const behavior = ${JSON.stringify(behavior)};
const [input, output, ...flags] = process.argv.slice(2);
fs.writeFileSync(${JSON.stringify(report)}, JSON.stringify({input, output, flags, envKeys: Object.keys(process.env), secret: process.env.LIBRARIAN_CONVERT_TEST_SECRET ?? null}));
if (behavior === 'timeout') { setInterval(() => {}, 1000); }
else if (behavior === 'drm') { process.stderr.write('DRMError: This book is locked by DRM and cannot be converted.'); process.exitCode = 2; }
else if (behavior === 'logs') { process.stderr.write('L'.repeat(131072)); }
else if (behavior === 'symlink') { fs.symlinkSync(input, output); }
else if (behavior === 'missing') { /* A zero exit alone does not make a valid conversion. */ }
else if (behavior === 'invalid') { fs.writeFileSync(output, 'not an EPUB ZIP'); }
else { fs.writeFileSync(output, Buffer.from(${JSON.stringify(epub.toString('base64'))}, 'base64')); }
`;
  // Resolve the declared bind mounts without attempting to emulate isolation.
  // Child processes stay in the launched process group for timeout cleanup.
  const sandboxScript = `#!${process.execPath}
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const args = process.argv.slice(2);
const mounts = [];
let env = {...process.env}; let cwd;
for (let i = 0; i < args.length; i += 1) {
  if (['--ro-bind', '--bind', '--ro-bind-try', '--bind-try'].includes(args[i])) { mounts.push([args[i + 1], args[i + 2]]); i += 2; }
  else if (args[i] === '--clearenv') { env = {}; }
  else if (args[i] === '--setenv') { env[args[i + 1]] = args[i + 2]; i += 2; }
  else if (args[i] === '--unsetenv') { delete env[args[i + 1]]; i += 1; }
  else if (args[i] === '--chdir') { cwd = args[++i]; }
}
mounts.sort((a, b) => b[1].length - a[1].length);
function mapped(value) {
  const bind = mounts.find(([, target]) => value === target || value.startsWith(target + '/'));
  return bind ? path.join(bind[0], value.slice(bind[1].length)) : value;
}
const boundary = args.indexOf('--');
const start = boundary >= 0 ? boundary + 1 : args.findIndex((arg) => mapped(arg) === ${JSON.stringify(executable)});
if (start < 0) throw new Error('Fake bwrap could not identify converter command');
const command = args.slice(start);
fs.writeFileSync(${JSON.stringify(protocol)}, JSON.stringify({args, command, envKeys: Object.keys(env), secret: env.LIBRARIAN_CONVERT_TEST_SECRET ?? null}));
const child = spawnSync(mapped(command[0]), command.slice(1).map(mapped), {stdio: 'inherit', env, cwd: cwd ? mapped(cwd) : undefined});
if (child.error) throw child.error;
process.exitCode = child.status ?? 1;
`;
  await writeFile(executable, converterScript, {mode: 0o700});
  await writeFile(sandboxExecutable, sandboxScript, {mode: 0o700});
  return {root, source, format, original, epub, derivedDir, report, protocol, executable, sandboxExecutable,
    converter: isolation === 'bubblewrap'
      ? {executable, sandboxExecutable, installationRoot: root}
      : {executable, isolation}};
}

const convert = (f, overrides = {}) => convertBook(f.source, {
  format: f.format, derivedDir: f.derivedDir, converter: f.converter, ...overrides,
});

async function noArtifacts(f) {
  assert.deepEqual(await readdir(f.derivedDir), []);
  assert.deepEqual(await readFile(f.source), f.original);
}

test('conversion retains a content-addressed EPUB, preserves source bytes, and records both hashes', linuxOnly, async (t) => {
  const f = await fixture(t, {isolation: 'bubblewrap'});
  const previous = process.env.LIBRARIAN_CONVERT_TEST_SECRET;
  process.env.LIBRARIAN_CONVERT_TEST_SECRET = 'must-not-reach-the-converter';
  let result;
  try { result = await convert(f); }
  finally {
    if (previous === undefined) delete process.env.LIBRARIAN_CONVERT_TEST_SECRET;
    else process.env.LIBRARIAN_CONVERT_TEST_SECRET = previous;
  }
  const originalHash = sha256(f.original); const derivedHash = sha256(f.epub);
  assert.equal(result.path, path.join(f.derivedDir, `${originalHash}-${derivedHash}.epub`));
  assert.deepEqual(await readFile(result.path), f.epub);
  assert.deepEqual(await readFile(f.source), f.original);
  assert.deepEqual(await readdir(f.derivedDir), [path.basename(result.path)]);
  assert.equal(result.provenance.schemaVersion, 1);
  assert.deepEqual(result.provenance.original, {format: 'mobi', sha256: originalHash, bytes: f.original.length});
  assert.deepEqual(result.provenance.derived, {format: 'epub', sha256: derivedHash, bytes: f.epub.length, path: result.path});
  assert.equal(result.provenance.converter.name, 'calibre ebook-convert');
  assert.equal(result.provenance.converter.executableSha256, sha256(await readFile(f.executable)));
  assert.equal(result.provenance.converter.sandboxSha256, sha256(await readFile(f.sandboxExecutable)));
  assert.equal(result.provenance.converter.isolation, 'bubblewrap');
  assert.ok(result.provenance.limitations.length > 0);
  const protocol = JSON.parse(await readFile(f.protocol, 'utf8'));
  const report = JSON.parse(await readFile(f.report, 'utf8'));
  assert.deepEqual(protocol.command, [f.executable, '/input/source.mobi', '/output/converted.epub']);
  assert.equal(report.input, f.source);
  assert.deepEqual(report.flags, []);
  assert.equal(report.secret, null); assert.equal(protocol.secret, null);
  assert.ok(!report.envKeys.includes('LIBRARIAN_CONVERT_TEST_SECRET'));
  assert.ok(!report.envKeys.includes('NODE_OPTIONS'));
  assert.ok(protocol.args.includes('--unshare-net') || protocol.args.includes('--unshare-all'));
  assert.ok(protocol.args.includes('--die-with-parent'));
});

for (const [behavior, limits, code] of [
  ['timeout', {timeoutMs: 1000}, 'CONVERSION_TIMEOUT'],
  ['logs', {maxLogBytes: 32}, 'LIMIT_EXCEEDED'],
  ['success', {maxOutputBytes: 64}, 'LIMIT_EXCEEDED'],
]) {
  test(`conversion enforces ${Object.keys(limits)[0]} and removes temporary output`, linuxOnly, async (t) => {
    const f = await fixture(t, {behavior});
    await assert.rejects(convert(f, {converter: {...f.converter, ...limits}}), {code});
    await noArtifacts(f);
  });
}

for (const [behavior, code] of [
  ['drm', 'ENCRYPTED_BOOK'], ['symlink', 'INVALID_CONVERSION'],
  ['missing', 'INVALID_CONVERSION'], ['invalid', 'INVALID_CONVERSION'],
]) {
  test(`conversion rejects ${behavior} output without retaining an artifact`, linuxOnly, async (t) => {
    const f = await fixture(t, {behavior});
    await assert.rejects(convert(f), {code});
    await noArtifacts(f);
  });
}

test('conversion rejects a symlink output directory without writing its target', linuxOnly, async (t) => {
  const f = await fixture(t);
  const outside = path.join(f.root, 'untouched');
  await mkdir(outside); await rm(f.derivedDir, {recursive: true}); await symlink(outside, f.derivedDir, 'dir');
  await assert.rejects(convert(f), {code: 'INVALID_OUTPUT'});
  assert.deepEqual(await readdir(outside), []);
  assert.deepEqual(await readFile(f.source), f.original);
});

test('conversion refuses conflicting retained bytes instead of overwriting an existing artifact', linuxOnly, async (t) => {
  const f = await fixture(t);
  const first = await convert(f);
  const conflict = Buffer.from('Existing artifact must not be overwritten.');
  await chmod(first.path, 0o600);
  await writeFile(first.path, conflict);
  await assert.rejects(convert(f), {code: 'OUTPUT_CONFLICT'});
  assert.deepEqual(await readFile(first.path), conflict);
  assert.deepEqual(await readdir(f.derivedDir), [path.basename(first.path)]);
  assert.deepEqual(await readFile(f.source), f.original);
});

for (const format of ['mobi', 'prc']) {
  test(`${format.toUpperCase()} extraction keeps its original format and derived EPUB citation mapping`, linuxOnly, async (t) => {
    const f = await fixture(t, {format});
    const book = await extractBook(f.source, {format, converter: f.converter, derivedDir: f.derivedDir});
    assert.equal(book.title, 'Retained conversion');
    assert.equal(book.metadata.format, format);
    assert.equal(book.sections.length, 1);
    assert.ok(book.sections[0].text.includes('Original-format evidence survives conversion.'));
    const locator = {format, sourceSha256: sha256(f.original), derived: {
      format: 'epub', member: 'OPS/chapter.xhtml', heading: 'retained', sha256: sha256(f.epub),
    }};
    assert.deepEqual(book.sections[0].locator, {...locator, derived: {...locator.derived,
      anchors: {retained: {charStart: 0, charEnd: 16}}}});
    assert.deepEqual(book.toc, [{title: 'Retained chapter', locator: {...locator, derived: {...locator.derived,
      charStart: 0, charEnd: 16, offsetBasis: 'section_utf16_code_units'}}}]);
    assert.deepEqual(book.metadata.conversion.original, {format, sha256: sha256(f.original), bytes: f.original.length});
    assert.equal(book.metadata.conversion.derived.sha256, sha256(f.epub));
    assert.equal(book.metadata.conversion.converter.isolation, 'container');
    assert.equal(book.metadata.conversion.converter.sandboxSha256, null);
    assert.ok(book.warnings.some((warning) => /derived EPUB sections/i.test(warning)));
    assert.deepEqual(await readFile(book.metadata.conversion.derived.path), f.epub);
    assert.deepEqual(await readFile(f.source), f.original);
  });
}
