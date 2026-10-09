// Execute only in the approved Spark Node24 runtime.
import test from 'node:test';
import assert from 'node:assert/strict';
import {FRONT_MATTER_LIMITS, captureFrontMatterPage, frontMatterEvidence,
  recoverReviewedMetadata, metadataQuality} from '../src/source-metadata-recovery.mjs';

const id = 'a'.repeat(64), text = 'THE COMPLETE GUIDE\nHome Repair\nFourth Edition\nEditors of Example Press\nISBN9780131103627 🦊';
const capture = () => frontMatterEvidence(id, [captureFrontMatterPage(1, text, [])], 1);
const field = (value, page = capture().pages[0]) => ({value, evidence: [{page: 1,
  pageTextSha256: page.textSha256, start: page.text.indexOf(value), end: page.text.indexOf(value) + value.length, quote: value}]});
const review = () => ({schema: 'librarian-reviewed-source-metadata/v1', sourceSha256: id,
  reviewer: 'Independent page reviewer', evidenceReceiptSha256: 'b'.repeat(64),
  fields: {title: field('Home Repair'), authors: [field('Editors of Example Press')]}});
const before = {title: '9780760350799.pdf', authors: []};

test('front matter retains physical pages, exact text, UTF16 quote offsets and geometry hints', () => {
  const item = {str: 'THE COMPLETE GUIDE', height: 40, transform: [1, 0, 0, 1, 10, 500]};
  const page = captureFrontMatterPage(1, text, [item]);
  assert.equal(page.lines[4].end, text.length); assert.equal(page.prominent[0].height, 40);
  assert.equal(page.offsetBasis, 'page_text_utf16_code_units'); assert.equal(frontMatterEvidence(id, [page], 1).wholeBookInspected, true);
});

test('reviewed exact source quotes recover filename title and missing corporate author with provenance', () => {
  const result = recoverReviewedMetadata(before, capture(), review());
  assert.deepEqual(result.display, {title: 'Home Repair', authors: ['Editors of Example Press']});
  assert.equal(result.receipt.state, 'reviewed_complete'); assert.equal(result.receipt.confidence, 'independently_reviewed_source_quotes');
  assert.deepEqual(result.receipt.appliedFields, ['title', 'authors']); assert.equal(result.receipt.review.fields.title.evidence[0].quote, 'Home Repair');
});

test('font prominence and ISBN alone never apply a guessed title or author', () => {
  const result = recoverReviewedMetadata(before, capture());
  assert.deepEqual(result.display, before); assert.equal(result.receipt.confidence, 'unreviewed');
  assert.deepEqual(result.receipt.unresolvedFields, ['title', 'authors']);
});

test('source and page digest mismatches reject borrowed or stale metadata', () => {
  const wrongSource = review(); wrongSource.sourceSha256 = 'c'.repeat(64);
  assert.throws(() => recoverReviewedMetadata(before, capture(), wrongSource), /identity-bound/);
  const wrongPage = review(); wrongPage.fields.title.evidence[0].pageTextSha256 = 'd'.repeat(64);
  assert.throws(() => recoverReviewedMetadata(before, capture(), wrongPage), /source page/);
});

test('a real quote cannot authorize invented text or an invalid span', () => {
  const invented = review(); invented.fields.title.value = 'Unobserved subtitle';
  assert.throws(() => recoverReviewedMetadata(before, capture(), invented), /exactly/);
  const badSpan = review(); badSpan.fields.title.evidence[0].end = text.length + 1;
  assert.throws(() => recoverReviewedMetadata(before, capture(), badSpan), /source page/);
  const noReceipt = review(); noReceipt.evidenceReceiptSha256 = '';
  assert.throws(() => recoverReviewedMetadata(before, capture(), noReceipt), /independent/);
});

test('existing meaningful embedded fields and every user edit take precedence', () => {
  const existing = {title: 'Existing meaningful title', authors: ['Existing author']};
  assert.deepEqual(recoverReviewedMetadata(existing, capture(), review()).display, existing);
  const edited = recoverReviewedMetadata({...before, editedAt: 'new user edit'}, capture(), review());
  assert.deepEqual(edited.display, before); assert.equal(edited.receipt.state, 'user_edit_preserved');
  assert.deepEqual(edited.receipt.preservedFields, ['title', 'authors']);
});

test('ordered multiline title quotes allow whitespace normalization without removing edition evidence', () => {
  const r = review(); r.fields.title = {value: 'Home Repair Fourth Edition', evidence: [...field('Home Repair').evidence, ...field('Fourth Edition').evidence]};
  assert.equal(recoverReviewedMetadata(before, capture(), r).display.title, 'Home Repair Fourth Edition');
});

test('partial review and image-only pages remain explicitly unresolved', () => {
  const r = review(); delete r.fields.authors;
  const recovered = recoverReviewedMetadata(before, capture(), r);
  assert.equal(recovered.receipt.state, 'needs_review'); assert.deepEqual(recovered.receipt.unresolvedFields, ['authors']);
  const empty = frontMatterEvidence(id, [captureFrontMatterPage(1, '', [])], 1);
  assert.deepEqual(recoverReviewedMetadata(before, empty).receipt.unresolvedFields, ['title', 'authors']);
  assert.equal(empty.ocrPerformed, false);
});

test('evidence bounds fail explicitly; incomplete or duplicated physical pages are rejected', () => {
  assert.throws(() => captureFrontMatterPage(1, 'x'.repeat(FRONT_MATTER_LIMITS.charsPerPage + 1), []), RangeError);
  assert.throws(() => captureFrontMatterPage(1, '', Array(FRONT_MATTER_LIMITS.itemsPerPage + 1).fill({})), RangeError);
  assert.throws(() => frontMatterEvidence(id, [captureFrontMatterPage(2, '', [])], 2), RangeError);
  assert.throws(() => frontMatterEvidence(id, [captureFrontMatterPage(1, '', [])], 10), RangeError);
});

test('catalog quality distinguishes indexing completion from filename titles and absent authors', () => {
  assert.deepEqual(metadataQuality(before).reasons, ['filename_title', 'authors_not_identified']);
  assert.equal(metadataQuality({title: 'Home Repair', authors: ['Editors of Example Press']}).needsReview, false);
  assert.deepEqual(metadataQuality({title: 'A valid anonymous work', authors: []}).reasons, ['authors_not_identified']);
});


test('blank author entries remain deficient and can be recovered unless user edited', () => {
  const blanks = {title: '9780760350799.pdf', authors: ['', '   ']};
  assert.equal(metadataQuality(blanks).needsReview, true);
  assert.deepEqual(recoverReviewedMetadata(blanks, capture(), review()).display.authors, ['Editors of Example Press']);
  const edited = recoverReviewedMetadata({...blanks, editedAt: 'user intentionally blanked authors'}, capture(), review());
  assert.deepEqual(edited.display.authors, ['', '   ']);
  assert.deepEqual(edited.receipt.unresolvedFields, ['title', 'authors']);
});
