// Read-only corpus observer. Run only in the approved Spark runtime after source
// review; the coordinator mounts the exact retained source tree read-only.
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {lstat, realpath, readFile, mkdir, readdir, writeFile} from 'node:fs/promises';
import {join, relative, isAbsolute} from 'node:path';
import {runParserWorker} from '../src/importer.mjs';

const [manifestPath, sourceRootArg, outputRootArg, ...extra] = process.argv.slice(2);
if (!manifestPath || !sourceRootArg || !outputRootArg || extra.length) throw new Error('Usage: node scripts/collect-metadata-evidence.mjs CASES.json READ_ONLY_SOURCES EMPTY_OWNED_OUTPUT');
const stat = await lstat(manifestPath);
if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 64 * 1024) throw new Error('Invalid case manifest');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
if (manifest.schema !== 'librarian-source-metadata-cases/v1' || !Array.isArray(manifest.cases)
    || !manifest.cases.length || manifest.cases.length > 25) throw new Error('Require1–25 pinned source cases');
const seen = new Set(); let sourceBytes = 0;
for (const row of manifest.cases) {
  if (!/^[a-f0-9]{64}$/.test(row.bookId || '') || row.sourceSha256 !== row.bookId || row.member !== `${row.bookId}.pdf`
      || !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > 512 * 1024 ** 2 || seen.has(row.bookId)) throw new Error('Invalid or repeated original PDF case');
  seen.add(row.bookId); sourceBytes += row.bytes;
}
if (sourceBytes > 4 * 1024 ** 3) throw new Error('Evidence source aggregate exceeds4GiB');
const sourceRoot = await realpath(sourceRootArg);
await mkdir(outputRootArg, {recursive: true, mode: 0o700});
const outputRoot = await realpath(outputRootArg), rel = relative(sourceRoot, outputRoot);
if ((!rel || (!rel.startsWith('..') && !isAbsolute(rel))) || (await readdir(outputRoot)).length) throw new Error('Evidence output must be empty and outside source root');
const stop = new AbortController(); process.once('SIGINT', () => stop.abort()); process.once('SIGTERM', () => stop.abort());
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const rows = []; let outputBytes = 0;
for (const row of manifest.cases) {
  stop.signal.throwIfAborted(); const filename = join(sourceRoot, row.member), before = await lstat(filename);
  if (!before.isFile() || before.isSymbolicLink() || before.size !== row.bytes || await realpath(filename) !== filename) throw new Error('Source file custody mismatch');
  const digest = createHash('sha256'); for await (const chunk of createReadStream(filename)) {stop.signal.throwIfAborted(); digest.update(chunk);}
  if (digest.digest('hex') !== row.sourceSha256) throw new Error('Source SHA mismatch');
  const response = await runParserWorker({operation: 'extractMetadata', path: filename, format: 'pdf', frontMatter: true},
    {signal: stop.signal, timeoutMs: 45000, maxWorkerOutputBytes: 2 * 1024 ** 2});
  const after = await lstat(filename), capture = response.book?.metadata?.sourceMetadata?.frontMatter;
  if (!response.ok || capture?.sourceSha256 !== row.bookId || after.size !== before.size || after.mtimeMs !== before.mtimeMs
      || after.ino !== before.ino || after.dev !== before.dev || response.book.sections?.length) throw new Error('Source observation incomplete or changed');
  const bytes = Buffer.from(JSON.stringify(response) + '\n'); outputBytes += bytes.length;
  if (bytes.length > 2 * 1024 ** 2 || outputBytes > 48 * 1024 ** 2) throw new Error('Evidence output cap exceeded');
  const member = `${row.bookId}.json`; await writeFile(join(outputRoot, member), bytes, {flag: 'wx', mode: 0o600});
  const result = {bookId: row.bookId, sourceSha256: row.sourceSha256, sourceBytes: row.bytes,
    observedTitle: response.book.title, observedAuthors: response.book.authors,
    capturedPages: capture.capturedPages, evidence: {member, bytes: bytes.length, sha256: hash(bytes)}};
  rows.push(result); console.log(JSON.stringify(result));
}
const report = {schema: 'librarian-native-front-matter-observation/v1', completed: true,
  sourceManifestSha256: hash(await readFile(manifestPath)), cases: rows, outputBytes,
  databaseOpened: false, corpusWrites: 0, conversions: 0, ocr: 0, network: 0, modelCalls: 0,
  appliedMetadata: false, behavioralAcceptanceVerdict: null};
await writeFile(join(outputRoot, 'report.json'), JSON.stringify(report, null, 2) + '\n', {flag: 'wx', mode: 0o600});
