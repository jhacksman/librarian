// Read-only coverage inventory; no application/store startup or source parsing.
import {DatabaseSync} from 'node:sqlite';
import {metadataQuality} from '../src/source-metadata-recovery.mjs';
const [dbPath, ...extra] = process.argv.slice(2);
if (!dbPath || extra.length) throw new Error('Usage: node scripts/audit-metadata.mjs READ_ONLY_LIBRARY.sqlite');
const db = new DatabaseSync(dbPath, {readOnly: true});
try {
  db.exec('PRAGMA query_only=ON');
  const total = db.prepare('SELECT count(*) AS n FROM books').get().n;
  if (!Number.isInteger(total) || total > 500) throw new Error('Metadata audit requires at most500books; increase only after bound review');
  const rows = db.prepare(`SELECT id,title,authors_json,json_extract(metadata_json,'$.editedAt') AS edited_at FROM books ORDER BY id`).all();
  const books = rows.map(row => {
    const authors = JSON.parse(row.authors_json);
    if (typeof row.title !== 'string' || row.title.length > 1000 || !Array.isArray(authors) || authors.length > 64
        || authors.some(value => typeof value !== 'string' || value.length > 1000)) throw new Error('Invalid bounded display metadata');
    return {id: row.id, title: row.title, authors, ...metadataQuality({title: row.title, authors, metadata: {editedAt: row.edited_at}})};
  });
  const report = {schema: 'librarian-display-metadata-audit/v1', total, covered: books.length,
    needsReview: books.filter(book => book.needsReview).length, filenameTitles: books.filter(book => book.reasons.includes('filename_title')).length,
    authorsNotIdentified: books.filter(book => book.reasons.includes('authors_not_identified')).length,
    userEdited: books.filter(book => book.userEdited).length, books, databaseReadOnly: true,
    sourceParsing: false, databaseWrites: 0, acceptanceVerdict: null};
  const output = JSON.stringify(report);
  if (Buffer.byteLength(output) > 1024 * 1024) throw new Error('Metadata audit output exceeds1MiB');
  console.log(output);
} finally {db.close();}
