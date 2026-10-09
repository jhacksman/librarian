// Execute only inside the existing Spark coordinator's isolated snapshot job.
// This helper starts no server, installs no model, and imports no external book.
import { readFile, writeFile, mkdir, lstat, realpath } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { resolve, join, dirname, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { LibraryStore } from '../src/store.mjs';
import { createRetrieval } from '../src/retrieval.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
if (process.env.LIBRARIAN_SPARK_COORDINATED !== '1') throw new Error('Run only through the existing Spark coordinator');
const inputPath = process.argv[2];
if (!inputPath) throw new Error('Supply the coordinator-bound inference input JSON');
const inputStat = await lstat(inputPath);
if (!inputStat.isFile() || inputStat.isSymbolicLink() || inputStat.size > 512 * 1024) throw new Error('Require an ordinary bounded inference input file');
const inputBytes = await readFile(inputPath);
if (inputBytes.length > 512 * 1024) throw new Error('Inference input exceeds its finite bound');
const input = JSON.parse(inputBytes);
if (!Array.isArray(input.cases) || !input.cases.length || input.cases.length > 12 ||
    typeof input.dbPath !== 'string' || typeof input.outputDir !== 'string' || input.isolatedSnapshot !== true) {
  throw new Error('Require an isolated SQLite clone, output directory and 1-12 bound cases');
}
if (new Set(input.cases.map(item => item.id)).size !== input.cases.length || input.cases.some(item =>
    Object.keys(item).some(key => !['id', 'question', 'bookId', 'readerContext'].includes(key)))) {
  throw new Error('Cases must have unique IDs and question/source scope only; gold and expected claims belong outside model inputs');
}
const config = input.retrieval;
if (!config || config.chatModel !== 'librarian-bonsai2-ptq1' || config.embeddingModel !== 'librarian-qwen3-embedding-q8' ||
    !config.chatRevision || !config.embeddingRevision || config.fetch !== undefined) {
  throw new Error('Require the explicitly approved Bonsai chat and Qwen embedding identities/revisions');
}
if (config.contextTokens !== 8192) throw new Error('Larger contexts require a separate same-model qualification first');
const seconds = input.maxSeconds ?? 600;
if (!Number.isInteger(seconds) || seconds < 30 || seconds > 1200) throw new Error('Invalid finite inference run bound');
if (typeof input.jobRoot !== 'string') throw new Error('Require a dedicated /tmp coordinator job root');
const jobRoot = resolve(input.jobRoot);
const jobStat = await lstat(jobRoot);
if (!jobRoot.startsWith('/tmp/') || !jobStat.isDirectory() || jobStat.isSymbolicLink() || await realpath(jobRoot) !== jobRoot) {
  throw new Error('The coordinator job root must be an ordinary dedicated /tmp directory');
}
const confined = path => path.startsWith(jobRoot + sep);
const dbPath = resolve(input.dbPath), outputDir = resolve(input.outputDir);
if (!confined(dbPath) || !confined(outputDir) || dbPath === outputDir) throw new Error('Snapshot and new output must be confined to the job root');
const dbStat = await lstat(dbPath);
if (!dbStat.isFile() || dbStat.isSymbolicLink() || dbStat.nlink !== 1 || await realpath(dbPath) !== dbPath ||
    !((await realpath(dirname(outputDir))) === jobRoot || confined(await realpath(dirname(outputDir))))) throw new Error('Require an ordinary isolated snapshot and ordinary output parent');
for (const suffix of ['-wal', '-shm', '-journal']) {
  try { await lstat(dbPath + suffix); throw new Error('Snapshot must be standalone without existing sidecars'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}
for (const kind of ['chat', 'embedding']) {
  const adapter = config[kind] ?? {};
  if (Object.keys(adapter).some(key => !['provider', 'endpoint', 'model', 'revision'].includes(key)) ||
      (adapter.model !== undefined && adapter.model !== config[`${kind}Model`]) ||
      (adapter.revision !== undefined && adapter.revision !== config[`${kind}Revision`]) ||
      (adapter.provider ?? config.provider) !== 'openai') throw new Error('Effective adapters must match the approved local identities');
  const endpoint = new URL(adapter.endpoint ?? config.endpoint);
  if (endpoint.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(endpoint.hostname) ||
      endpoint.username || endpoint.password || endpoint.search || endpoint.hash || !endpoint.pathname.endsWith('/v1')) {
    throw new Error('Inference adapters must use the explicitly bound local OpenAI-compatible API roots');
  }
}
const artifacts = [];
const save = async (name, bytes) => {
  bytes = typeof bytes === 'string' ? Buffer.from(bytes) : bytes;
  if (bytes.length > 2 * 1024 * 1024) throw new Error(`Artifact ${name} exceeds its byte bound`);
  await writeFile(join(outputDir, name), bytes, { flag: 'wx' });
  artifacts.push({ path: name, bytes: bytes.length, sha256: hash(bytes) });
};
// Coordinator binds the approved snapshot, original-source/native-gold receipts,
// runtime/model contracts and this source archive. Rehash returned file inputs;
// gold is never placed in a model prompt or used to rewrite its response.
if (!Array.isArray(input.bindings) || !input.bindings.length || input.bindings.length > 20) throw new Error('Require pinned input custody bindings');
if (!input.bindings.some(binding => binding.role === 'isolated-snapshot' && resolve(binding.path) === dbPath)) {
  throw new Error('Require an exact isolated-snapshot hash binding before opening SQLite');
}
let bindingBytes = 0;
for (const binding of input.bindings) {
  const stat = await lstat(binding.path);
  bindingBytes += stat.size;
  const maximum = binding.role === 'isolated-snapshot' ? 2 * 1024 ** 3 : 16 * 1024 ** 2;
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > maximum || bindingBytes > 3 * 1024 ** 3) throw new Error('Custody bindings must be bounded ordinary files');
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(binding.path, { highWaterMark: 256 * 1024 })) digest.update(chunk);
  if (digest.digest('hex') !== binding.sha256) throw new Error(`Input binding changed: ${binding.role}`);
}
// Create a fresh evidence directory only after all inputs have been validated.
await mkdir(outputDir);
await save('input.json', inputBytes);
const runStarted = performance.now();
const deadline = AbortSignal.timeout(seconds * 1000);
let currentCase = 'none', transportCount = 0;
const traces = [];
const tracedFetch = async (url, init) => {
  deadline.throwIfAborted();
  if (++transportCount > 32) throw new Error('Finite model transport bound exceeded');
  const sequence = transportCount;
  const prefix = `${currentCase}-${String(sequence).padStart(2, '0')}`;
  const request = Buffer.from(init.body);
  await save(`${prefix}-request.raw.json`, request);
  const started = performance.now();
  const trace = { caseId: currentCase, sequence, url, requestSha256: hash(request), dispatched: false,
    elapsedMs: 0, httpStatus: null, responseSha256: null, error: null };
  traces.push(trace);
  try {
    trace.dispatched = true; // Dispatch is not proof the engine executed.
    const response = await fetch(url, { ...init, signal: AbortSignal.any([deadline, init.signal]) });
    trace.httpStatus = response.status;
    const reader = response.body?.getReader();
    const chunks = [];
    let length = 0;
    try {
      if (reader) while (true) {
        const next = await reader.read();
        if (next.done) break;
        length += next.value.byteLength;
        if (length > 512 * 1024) { await reader.cancel(); throw new Error('Model response exceeded 512 KiB transport bound'); }
        chunks.push(Buffer.from(next.value));
      }
    } finally { reader?.releaseLock(); }
    const bytes = Buffer.concat(chunks);
    trace.responseSha256 = hash(bytes);
    await save(`${prefix}-response.raw.json`, bytes);
    try {
      const data = JSON.parse(bytes);
      trace.returnedModel = data.model ?? null;
      trace.usage = data.usage ?? null;
      trace.finishReason = data.choices?.[0]?.finish_reason ?? data.done_reason ?? null;
    } catch { /* Original malformed bytes are retained, not repaired. */ }
    return new Response(bytes, { status: response.status, headers: { 'content-type': response.headers.get('content-type') ?? 'application/json' } });
  } catch (error) {
    trace.error = { name: error.name, message: error.message };
    throw error;
  } finally { trace.elapsedMs = performance.now() - started; }
};
const store = new LibraryStore(dbPath);
const report = { schema: 'librarian-local-ask-inference/v1', inputSha256: hash(inputBytes),
  qualityAcceptance: false, semanticJudgment: 'requires_independent_native_source_review',
  isolatedSnapshot: true, configuredModels: { chat: config.chatModel, chatRevision: config.chatRevision,
    embedding: config.embeddingModel, embeddingRevision: config.embeddingRevision },
  cases: [], traces, artifacts, elapsedMs: 0, completed: false };
try {
  const retrieval = createRetrieval(store, { ...config, fetch: tracedFetch, maxResponseBytes: 512 * 1024 });
  for (const item of input.cases) {
    if (typeof item.id !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(item.id) ||
        typeof item.question !== 'string' || !item.bookId || !store.db.prepare('SELECT id FROM books WHERE id=?').get(item.bookId)) {
      throw new Error('Every finite case must bind an existing exact source and safe case identifier');
    }
    currentCase = item.id;
    const caseStarted = performance.now();
    const result = { id: item.id, question: item.question, bookId: item.bookId,
      judge: 'pending', error: null, elapsedMs: 0 };
    report.cases.push(result);
    try {
      deadline.throwIfAborted();
      const answer = await retrieval.ask({ question: item.question, bookId: item.bookId,
        readerContext: item.readerContext ?? null, responseMode: 'auto', signal: deadline });
      result.status = answer.status;
      result.answerStatus = answer.answerStatus;
      result.responseKind = answer.responseKind;
      result.inference = answer.inference;
      result.context = answer.context ?? null;
      result.warnings = answer.warnings;
      // This preserves retrieved hits, exact sent-context citation spans,
      // synthesized output and support receipts together for independent review.
      await save(`${item.id}-ask.json`, JSON.stringify(answer, null, 2));
    } catch (error) {
      result.error = { name: error.name, message: error.message };
    } finally { result.elapsedMs = performance.now() - caseStarted; }
    if (deadline.aborted) break;
  }
  report.completed = report.cases.length === input.cases.length && !deadline.aborted;
  report.status = retrieval.status();
} finally {
  store.close();
  report.elapsedMs = performance.now() - runStarted;
  await writeFile(join(outputDir, 'report.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
}
process.stdout.write(JSON.stringify({ completed: report.completed, cases: report.cases.length,
  transportCount, outputDir, elapsedMs: report.elapsedMs, qualityAcceptance: false }) + '\n');
