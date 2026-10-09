import { mkdir } from 'node:fs/promises';
import { readConfig } from './config.mjs';
import { LibraryStore } from './store.mjs';
import { inventoryLibrary, runImport, createReindexJob } from './importer.mjs';
import { createRetrieval } from './retrieval.mjs';
import { enrichBooks } from './book-enrichment.mjs';

const [command, ...args] = process.argv.slice(2);
const config = readConfig();
if (command === 'enrich-books' && process.env.LIBRARIAN_ENRICHMENT_EXCLUSIVE !== '1') throw new Error('Enrichment requires the coordinator exclusive DATA lease with all readers stopped');
await mkdir(config.dataDir, { recursive: true, mode: 0o700 });
const store = new LibraryStore(config.dbPath, { dataDir: config.dataDir });
const stop = new AbortController();
process.once('SIGINT', () => stop.abort());
process.once('SIGTERM', () => stop.abort());
const output = value => console.log(JSON.stringify(value));
try {
  if (command === 'enrich-books') {
    if (process.env.LIBRARIAN_ENRICHMENT_EXCLUSIVE !== '1') throw new Error('Enrichment requires the coordinator exclusive DATA lease with all readers stopped; set LIBRARIAN_ENRICHMENT_EXCLUSIVE=1 only under that lease');
    const limitIndex = args.indexOf('--limit');
    const limit = limitIndex >= 0 ? Number(args[limitIndex + 1]) : 25;
    if (limitIndex >= 0 && (args.indexOf('--limit', limitIndex + 1) >= 0 || args[limitIndex + 1] === undefined)) throw new Error('Usage: enrich-books [--limit 1–500] [BOOK_IDS...]');
    const bookIds = args.filter((value, index) => value !== '--retry-failed' && (limitIndex < 0 || (index !== limitIndex && index !== limitIndex + 1)));
    output(await enrichBooks(store, { limit, retryFailed: args.includes('--retry-failed'), ...(bookIds.length ? { bookIds } : {}), signal: stop.signal,
      onProgress: value => output({ event: 'book_enriched', ...value }) }));
  } else if (command === 'import') {
    if (args.length !== 1) throw new Error('Usage: node src/cli.mjs import /absolute/source/folder');
    const job = await inventoryLibrary(store, args[0], { dataDir: config.dataDir });
    output({ event: 'inventoried', job });
    output(await runImport(store, job.id, { ...config.importOptions, signal: stop.signal, onProgress: progress => output({ event: 'progress', ...progress }) }));
  } else if (command === 'resume') {
    if (args.length !== 1 || !store.getJob(args[0])) throw new Error('Usage: node src/cli.mjs resume EXISTING_JOB_ID');
    output(await runImport(store, args[0], { ...config.importOptions, signal: stop.signal, retryFailed: true, onProgress: progress => output({ event: 'progress', ...progress }) }));
  } else if (command === 'reindex' || command === 'reindex-files') {
    if (command === 'reindex-files' && !args.length) throw new Error('Supply managed file IDs to reindex-files');
    const job = createReindexJob(store, args.length ? { [command === 'reindex' ? 'bookIds' : 'fileIds']: args } : {});
    output({ event: 'reindex_queued', job });
    output(await runImport(store, job.id, { ...config.importOptions, signal: stop.signal, onProgress: progress => output({ event: 'progress', ...progress }) }));
  } else if (command === 'embed') {
    if (!config.retrieval.embeddingModel) throw new Error('Set LIBRARIAN_EMBEDDING_MODEL to an installed local model');
    if (args.some(arg => !['--retry-failed', '--once'].includes(arg)) || new Set(args).size !== args.length) {
      throw new Error('Usage: embed [--once] [--retry-failed]');
    }
    const retrieval = createRetrieval(store, config.retrieval);
    const once = args.includes('--once');
    let result, retryFailed = args.includes('--retry-failed');
    do {
      result = await retrieval.embedPending({ signal: stop.signal, retryFailed, onProgress: progress => output({ event: 'progress', ...progress }) });
      retryFailed = false;
      if (!once) output(result);
    } while (!once && !stop.signal.aborted && result.status === 'bounded' && (result.pendingChunks ?? result.remaining) > 0 && result.processed + (result.quarantined || 0) > 0);
    if (once) {
      // A successful slice can leave pending work; it is not complete coverage.
      // Failures and cancellation still require an operator to inspect the report.
      const yielded = !stop.signal.aborted && result.status === 'bounded' && result.processed > 0 && result.failedChunks === 0;
      const exitCode = !stop.signal.aborted && (result.status === 'complete' || yielded) ? 0 : 1;
      output({ ...result, coverage: retrieval.status().embedding,
        execution: { mode: 'once', completed: result.status === 'complete', yielded,
          resumeRequired: result.pendingChunks > 0, retryFailedRequested: args.includes('--retry-failed'), exitCode } });
      process.exitCode = exitCode;
    } else if (result.status !== 'complete') process.exitCode = 1;
  } else if (command === 'status') {
    output({ ...store.summary(), models: createRetrieval(store, config.retrieval).status() });
  } else if (command === 'receipts') {
    for (let offset = 0; ; offset += 1000) {
      const files = store.listFiles({ jobId: args[0], limit: 1000, offset });
      for (const file of files) output(file);
      if (files.length < 1000) break;
    }
  } else if (command === 'ask' || command === 'search') {
    const question = args.join(' ');
    if (!question.trim()) throw new Error('Supply a question');
    const retrieval = createRetrieval(store, config.retrieval);
    output(command === 'ask' ? await retrieval.ask({ question }) : await retrieval.search({ query: question }));
  } else {
    throw new Error('Commands: import FOLDER, resume JOB_ID, reindex [BOOK_IDS...], reindex-files FILE_IDS..., embed [--once] [--retry-failed], status, receipts [JOB_ID], search QUERY, ask QUESTION, enrich-books [--limit 1–500] [--retry-failed] [BOOK_IDS...]');
  }
} catch (error) {
  console.error(JSON.stringify({ error: error.message })); process.exitCode = 1;
} finally { store.close(); }
