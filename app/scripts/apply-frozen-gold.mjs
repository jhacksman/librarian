// Isolated coordinator-owned clone only. Frozen gold and page PNGs remain external inputs.
import {readFile, lstat, open} from 'node:fs/promises';
import {parseFrozenGold, verifyFrozenGoldRenders} from '../src/frozen-gold-recovery.mjs';
import {readConfig} from '../src/config.mjs';
import {LibraryStore} from '../src/store.mjs';
import {enrichBooks} from '../src/book-enrichment.mjs';

if (process.env.LIBRARIAN_ENRICHMENT_EXCLUSIVE !== '1') throw new Error('Frozen gold recovery requires exclusive isolated DATA custody');
const [goldPath, renderRoot, observationPath, ...extra] = process.argv.slice(2);
if (!goldPath || !renderRoot || !observationPath || extra.length) throw new Error('Usage: node scripts/apply-frozen-gold.mjs GOLD.json FROZEN_RENDER_ROOT FRESH_OBSERVATION.json');
const stat = await lstat(goldPath);
if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 2 * 1024 ** 2) throw new Error('Invalid frozen gold input');
const gold = parseFrozenGold(await readFile(goldPath));
if (gold.records.length !== 18) throw new Error('This frozen pilot requires exactly18 sources');
const renderedEvidence = await verifyFrozenGoldRenders(gold, renderRoot);
const frozenGoldRecords = Object.fromEntries(gold.records.map(record => [record.source_sha256, record]));
const observationFile = await open(observationPath, 'wx', 0o600);
const config = readConfig();
let store;
const observe = () => Object.keys(frozenGoldRecords).map(id => {
  const row = store.db.prepare('SELECT * FROM books WHERE id=?').get(id);
  if (!row) throw new Error(`Frozen gold book missing: ${id}`);
  const metadata = JSON.parse(row.metadata_json);
  const asset = store.db.prepare('SELECT thumbnail_sha256,mime,width,height,kind FROM book_assets WHERE book_id=?').get(id);
  return {id, title: row.title, authors: JSON.parse(row.authors_json), edition: metadata.edition || null,
    creators: metadata.creators || null, editedAt: metadata.editedAt || null, cover: asset,
    recovery: metadata.sourceMetadata?.recovery || null};
});
const stop = new AbortController();
process.once('SIGINT', () => stop.abort()); process.once('SIGTERM', () => stop.abort());
try {
  store = new LibraryStore(config.dbPath, {dataDir: config.dataDir});
  const before = observe();
  const result = await enrichBooks(store, {limit: 18, bookIds: Object.keys(frozenGoldRecords), recoverMetadata: true,
    frozenGoldRecords, signal: stop.signal, onProgress: row => console.log(JSON.stringify({event: 'frozen_gold_recovery', ...row}))});
  const observation = {schema: 'librarian-frozen-gold-application/v1', renderedEvidence, before, after: observe(), ...result};
  await observationFile.writeFile(JSON.stringify(observation, null, 2) + '\n');
  console.log(JSON.stringify({schema: observation.schema, renderedEvidence, observationPath, ...result}));
  if (result.failed || result.processed !== 18) process.exitCode = 1;
} finally {store?.close(); await observationFile.close();}
