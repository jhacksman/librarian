/**
 * Test-only child of recovery-browser.mjs. No HTTP testing controls or production
 * environment switches are installed. IPC starts one initial import with the
 * actual extractor held before its second call. Restarted HTTP Resume uses the
 * application's ordinary importer and parser worker.
 */
import assert from 'node:assert/strict';
import {join, isAbsolute} from 'node:path';

assert.equal(process.platform, 'linux', 'Recovery fixture runs only on Spark Linux');
assert.ok(process.send, 'Recovery fixture must be an owned IPC child');
const [workspace, phase, modelOrigin] = process.argv.slice(2);
assert.ok(isAbsolute(workspace));
assert.ok(['import-initial', 'import-resume', 'embedding-initial', 'embedding-resume'].includes(phase));
const model = new URL(modelOrigin);
assert.equal(model.hostname, '127.0.0.1');
assert.equal(model.protocol, 'http:');
const [{createApp}, {LibraryStore}, {readConfig}, {inventoryLibrary, runImport}, {extractBook}] = await Promise.all([
  import('../src/server.mjs'), import('../src/store.mjs'), import('../src/config.mjs'),
  import('../src/importer.mjs'), import('../src/extract.mjs'),
]);
const config = readConfig({LIBRARIAN_DATA_DIR: join(workspace, 'state'), LIBRARIAN_PORT: '0',
  LIBRARIAN_IMPORT_ROOTS: join(workspace, 'originals'), LIBRARIAN_MODEL_ENDPOINT: model.origin});
if (phase.startsWith('embedding')) Object.assign(config.retrieval, {
  embeddingModel: 'synthetic-recovery-embedding', embeddingRevision: 'fixture-v1',
  chatModel: 'synthetic-recovery-chat', chatRevision: 'fixture-v1',
  queryPrefix: 'query: ', documentPrefix: 'document: ', batchSize: 1,
  timeoutMs: 60000, embeddingRunMs: 120000,
});
const store = new LibraryStore(config.dbPath, {dataDir: config.dataDir});
const app = await createApp({config, store});
await new Promise((resolve, reject) => { app.server.once('error', reject); app.server.listen(0, '127.0.0.1', resolve); });
config.origin = `http://127.0.0.1:${app.server.address().port}`;
process.send({type: 'ready', origin: config.origin, phase, pid: process.pid});
let started = false;
process.on('message', async (message) => {
  if (message?.command !== 'start-fixture-import' || phase !== 'import-initial' || started) {
    process.send({type: 'failure', message: 'Unexpected fixture IPC command'}); return;
  }
  started = true;
  try {
    const job = await inventoryLibrary(store, join(workspace, 'originals'), {dataDir: config.dataDir});
    process.send({type: 'import-started', jobId: job.id});
    let calls = 0;
    await runImport(store, job.id, {dataDir: config.dataDir, extractor: async (request) => {
      calls += 1;
      if (calls === 2) {
        process.send({type: 'import-held', jobId: job.id});
        await new Promise(() => {}); // The owning driver must use real SIGKILL.
      }
      assert.equal(calls, 1);
      return {ok: true, book: await extractBook(request.path, request)};
    }});
    process.send({type: 'failure', message: 'Fixture import completed without its required hard kill'});
  } catch (error) { process.send({type: 'failure', message: String(error.message).slice(0, 500)}); }
});
for (const signal of ['SIGTERM', 'SIGINT']) process.once(signal, async () => {
  await app.close(); store.close(); process.exit(0);
});
process.once('disconnect', () => process.exit(1));
