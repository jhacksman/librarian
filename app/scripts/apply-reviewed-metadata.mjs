// Coordinator only, under the same exclusive DATA lease as enrich-books.
import {readFile, lstat} from 'node:fs/promises';
import {readConfig} from '../src/config.mjs';
import {LibraryStore} from '../src/store.mjs';
import {enrichBooks} from '../src/book-enrichment.mjs';

if (process.env.LIBRARIAN_ENRICHMENT_EXCLUSIVE !== '1') throw new Error('Reviewed recovery requires the coordinator exclusive DATA lease');
const [receiptPath, ...extra] = process.argv.slice(2);
if (!receiptPath || extra.length) throw new Error('Usage: node scripts/apply-reviewed-metadata.mjs REVIEWED_RECEIPTS.json');
const stat = await lstat(receiptPath);
if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 128 * 1024) throw new Error('Invalid or oversized reviewed metadata receipts');
const bundle = JSON.parse(await readFile(receiptPath, 'utf8'));
if (bundle.schema !== 'librarian-reviewed-source-metadata-batch/v1' || !Array.isArray(bundle.reviews)
    || !bundle.reviews.length || bundle.reviews.length > 25) throw new Error('Supply1–25 independent source reviews');
const reviewedMetadata = Object.fromEntries(bundle.reviews.map(review => [review.sourceSha256, review]));
const bookIds = Object.keys(reviewedMetadata);
if (bookIds.length !== bundle.reviews.length || bookIds.some(id => !/^[a-f0-9]{64}$/.test(id))) throw new Error('Reviews must identify unique original sources');
const config = readConfig(), store = new LibraryStore(config.dbPath, {dataDir: config.dataDir});
const stop = new AbortController();
process.once('SIGINT', () => stop.abort()); process.once('SIGTERM', () => stop.abort());
try {
  const result = await enrichBooks(store, {limit: 25, bookIds, recoverMetadata: true, reviewedMetadata, signal: stop.signal,
    onProgress: row => console.log(JSON.stringify({event: 'reviewed_metadata', ...row}))});
  console.log(JSON.stringify(result));
  if (result.failed || result.processed !== bookIds.length) process.exitCode = 1;
} finally {store.close();}
