import {createHash} from 'node:crypto';
import {readFile, lstat} from 'node:fs/promises';
import {join} from 'node:path';
import {filenameTitle, frontMatterEvidence} from './source-metadata-recovery.mjs';

// Independently frozen before this implementation; corpus evidence stays outside Git.
export const FROZEN_GOLD_SHA256 = 'd023dff9723d508062c90faff7f5757815f3cae56576fc40c7ddf2d008224f80';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const hex = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const meaningful = value => typeof value === 'string' && value.trim();
const approvals = new WeakMap();
const freeze = value => {
  if (value && typeof value === 'object') { for (const child of Object.values(value)) freeze(child); Object.freeze(value); }
  return value;
};

export function parseFrozenGold(bytes, expectedSha256 = FROZEN_GOLD_SHA256) {
  if (!Buffer.isBuffer(bytes) || bytes.length > 2 * 1024 ** 2 || !hex(expectedSha256)
      || sha(bytes) !== expectedSha256) throw new TypeError('Frozen independent gold digest mismatch');
  const gold = JSON.parse(bytes.toString('utf8'));
  if (gold.schema !== 'independent-native-page-metadata-gold-set/v1' || !Array.isArray(gold.records)
      || !gold.records.length || gold.records.length > 25) throw new TypeError('Invalid frozen gold set');
  const seen = new Set();
  for (const record of gold.records) {
    if (!hex(record.source_sha256) || record.catalog_book_id !== record.source_sha256
        || seen.has(record.source_sha256) || !Array.isArray(record.pages)) throw new TypeError('Gold source identity mismatch');
    seen.add(record.source_sha256);
    for (const name of ['title', 'authors', 'creators', 'edition']) {
      const field = record[name];
      if (!field || !Array.isArray(field.support) || field.support.length > 12
          || (field.value !== null && (field.state !== 'supported' || !field.support.length))) throw new TypeError('Invalid gold field support');
      if (field.value === null) continue;
      if (['title', 'edition'].includes(name) && (!meaningful(field.value) || field.value.length > 1000)) throw new TypeError('Invalid gold text');
      if (name === 'authors' && (!Array.isArray(field.value) || !field.value.length || field.value.length > 16
          || field.value.some(value => !meaningful(value) || value.length > 1000))) throw new TypeError('Invalid gold authors');
      if (name === 'creators' && (!Array.isArray(field.value) || !field.value.length || field.value.length > 16
          || field.value.some(value => !meaningful(value.name) || !meaningful(value.role)
            || value.name.length > 1000 || value.role.length > 200))) throw new TypeError('Invalid gold creator roles');
    }
    approvals.set(record, {sha256: expectedSha256, rendersVerified: false});
  }
  return freeze(gold);
}

// Root supplies immutable render files named by their frozen digest, never arbitrary gold paths.
export async function verifyFrozenGoldRenders(gold, evidenceRoot) {
  const expected = new Map();
  for (const record of gold.records) {
    if (!approvals.has(record)) throw new TypeError('Gold must be loaded from pinned bytes');
    for (const page of record.pages) if (page.rendered_full_page) {
      const image = page.rendered_full_page;
      if (!hex(image.sha256) || !Number.isSafeInteger(image.bytes) || image.bytes < 8 || image.bytes > 16 * 1024 ** 2) throw new TypeError('Invalid frozen render pin');
      expected.set(image.sha256, image.bytes);
    }
    for (const name of ['title', 'authors', 'creators', 'edition']) for (const ref of record[name].support) {
      if (ref.capture_mime_type === 'image/png' && !expected.has(ref.page_sha256)) throw new TypeError('Visual support has no frozen render pin');
    }
  }
  if (expected.size > 64) throw new RangeError('Too many frozen page renders');
  for (const [digest, bytes] of expected) {
    const path = join(evidenceRoot, `${digest}.png`), stat = await lstat(path);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size !== bytes) throw new TypeError('Frozen render size/type mismatch');
    const data = await readFile(path);
    if (sha(data) !== digest || !data.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) throw new TypeError('Frozen render digest mismatch');
  }
  for (const record of gold.records) approvals.get(record).rendersVerified = true;
  return {renders: expected.size, bytes: [...expected.values()].reduce((a, b) => a + b, 0)};
}

export function recoverFrozenGold(display, capture, record) {
  const approval = approvals.get(record);
  if (!approval?.rendersVerified || record.source_sha256 !== capture?.sourceSha256) throw new TypeError('Verified frozen gold for this source is required');
  frontMatterEvidence(capture.sourceSha256, capture.pages, capture.totalPages);
  if (record.physical_total_pages !== capture.totalPages) throw new TypeError('Frozen gold physical page count mismatch');
  const displayResult = {title: display.title, authors: [...(display.authors || [])]}, metadataUpdates = {};
  const receipt = {version: 'frozen-independent-gold-recovery-v1', sourceSha256: capture.sourceSha256,
    goldSha256: approval.sha256, confidence: 'independently_frozen_source_review', appliedFields: [], preservedFields: [],
    unresolvedFields: [], fields: {}};
  for (const name of ['title', 'authors', 'creators', 'edition']) {
    const field = record[name];
    if (field.value === null) { receipt.unresolvedFields.push(name); continue; }
    for (const ref of field.support) {
      const page = capture.pages.find(page => page.page === ref.physical_page_number);
      const frozenPage = record.pages.find(page => page.physical_page_number === ref.physical_page_number);
      if (ref.source_sha256 !== capture.sourceSha256 || !page || !frozenPage
          || page.textSha256 !== frozenPage.native_text_utf8?.sha256) throw new TypeError('Gold native page identity mismatch');
      if (ref.capture_mime_type === 'application/json'
          && (page.textSha256 !== ref.native_text_sha256 || typeof ref.exact_quote !== 'string'
            || !ref.exact_quote || !page.text.includes(ref.exact_quote))) throw new TypeError('Frozen gold quote no longer matches native source');
      if (!['application/json', 'image/png'].includes(ref.capture_mime_type)) throw new TypeError('Unsupported gold evidence');
    }
    receipt.fields[name] = {value: field.value, support: field.support,
      ...(field.credit_role ? {creditRole: field.credit_role} : {}),
      ...(field.role_basis ? {roleBasis: field.role_basis} : {}), ...(field.reason ? {reason: field.reason} : {})};
    const existing = name === 'title' ? !filenameTitle(display.title) : name === 'authors'
      ? displayResult.authors.some(meaningful) : name === 'edition' ? meaningful(display.edition)
        : Array.isArray(display.creators) && display.creators.length;
    if (display.editedAt || existing) {receipt.preservedFields.push(name); continue;}
    if (name === 'title' || name === 'authors') displayResult[name] = structuredClone(field.value);
    else metadataUpdates[name] = structuredClone(field.value);
    receipt.appliedFields.push(name);
  }
  receipt.state = display.editedAt ? 'user_edit_preserved' : receipt.unresolvedFields.length ? 'needs_review' : 'reviewed_complete';
  return {display: displayResult, metadataUpdates, receipt};
}
