import {createHash} from 'node:crypto';

export const FRONT_MATTER_LIMITS = Object.freeze({pages: 8, itemsPerPage: 4096, charsPerPage: 16384,
  totalChars: 49152, totalBytes: 256 * 1024});
const hash = text => createHash('sha256').update(text).digest('hex');
const clean = value => String(value ?? '').replace(/\s+/gu, ' ').trim();
const meaningfulAuthors = values => Array.isArray(values) && values.some(value => typeof value === 'string' && clean(value));
const hex = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
export const filenameTitle = title => !clean(title) || /\.(?:pdf|epub|mobi|prc)$/iu.test(clean(title))
  || /^(?:97[89]\d{10})(?:[_. -].*)?$/u.test(clean(title)) || /^[a-f0-9]{64}$/iu.test(clean(title));

// Evidence capture never applies a typography guess to the catalog. Font size
// helps a reviewer find the title; exact page text and source identity are retained.
export function captureFrontMatterPage(page, text, items) {
  if (!Number.isInteger(page) || page < 1 || page > FRONT_MATTER_LIMITS.pages || typeof text !== 'string'
      || text.length > FRONT_MATTER_LIMITS.charsPerPage || !Array.isArray(items)
      || items.length > FRONT_MATTER_LIMITS.itemsPerPage) throw new RangeError('Front matter evidence exceeds its page/item/text bound');
  const lines = []; let offset = 0;
  for (const value of text.split('\n')) {
    if (value.trim()) lines.push({text: value, start: offset, end: offset + value.length});
    offset += value.length + 1;
  }
  const prominent = items.filter(item => typeof item.str === 'string' && item.str.trim())
    .map(item => ({text: item.str, x: item.transform?.[4], y: item.transform?.[5], height: Math.abs(item.height || item.transform?.[3] || 0)}))
    .filter(item => [item.x, item.y, item.height].every(Number.isFinite))
    .sort((a, b) => b.height - a.height).slice(0, 24);
  return {page, text, textSha256: hash(text), offsetBasis: 'page_text_utf16_code_units', lines, prominent};
}

export function frontMatterEvidence(sourceSha256, pages, totalPages) {
  if (!hex(sourceSha256) || !Array.isArray(pages) || pages.length > FRONT_MATTER_LIMITS.pages
      || !Number.isInteger(totalPages) || totalPages < 1 || pages.length !== Math.min(totalPages, FRONT_MATTER_LIMITS.pages)
      || pages.some((page, index) => page.page !== index + 1 || typeof page.text !== 'string' || page.text.length > FRONT_MATTER_LIMITS.charsPerPage || page.textSha256 !== hash(page.text))
      || pages.reduce((sum, page) => sum + page.text.length, 0) > FRONT_MATTER_LIMITS.totalChars) {
    throw new RangeError('Invalid or oversized front matter capture');
  }
  const result = {version: 'native-pdf-front-matter-v1', sourceSha256, totalPages,
    capturedPages: pages.length, completeWithinBound: true, wholeBookInspected: pages.length === totalPages,
    ocrPerformed: false, networkUsed: false, modelUsed: false, pages,
    limitation: 'Native PDF text only; image-only or ambiguous title/author roles require independent review. Font size is a navigation hint, not a catalog value.'};
  if (Buffer.byteLength(JSON.stringify(result)) > FRONT_MATTER_LIMITS.totalBytes) throw new RangeError('Front matter evidence exceeds its aggregate byte bound');
  return result;
}

function reviewedValue(field, capture) {
  if (!field || typeof field.value !== 'string' || !field.value.trim() || field.value.length > 1000
      || !Array.isArray(field.evidence) || field.evidence.length < 1 || field.evidence.length > 12) throw new TypeError('Reviewed field needs a bounded value and source quotes');
  const quotes = field.evidence.map(ref => {
    const page = capture.pages.find(page => page.page === ref.page);
    if (!page || page.textSha256 !== ref.pageTextSha256 || !Number.isSafeInteger(ref.start)
        || !Number.isSafeInteger(ref.end) || ref.start < 0 || ref.end <= ref.start || ref.end > page.text.length
        || typeof ref.quote !== 'string' || page.text.slice(ref.start, ref.end) !== ref.quote) {
      throw new TypeError('Reviewed metadata quote does not match this source page');
    }
    return ref.quote;
  });
  const value = clean(field.value);
  if (value !== clean(quotes.join(' '))) throw new TypeError('Reviewed value must consist exactly of its source quotes');
  return {...field, value};
}

// Only an independently reviewed receipt can apply page text. An ISBN or a
// matching basename never authorizes borrowing metadata from a different book.
export function recoverReviewedMetadata(display, capture, review) {
  if (!capture || capture.version !== 'native-pdf-front-matter-v1') throw new TypeError('Verified native front matter is required');
  frontMatterEvidence(capture.sourceSha256, capture.pages, capture.totalPages);
  const before = {title: display.title, authors: [...(display.authors || [])]};
  const receipt = {version: 'source-page-metadata-recovery-v1', sourceSha256: capture.sourceSha256,
    state: 'needs_review', confidence: 'unreviewed', appliedFields: [], preservedFields: [], review: null,
    unresolvedFields: [filenameTitle(before.title) ? 'title' : null, meaningfulAuthors(before.authors) ? null : 'authors'].filter(Boolean)};
  if (!review) return {display: before, receipt};
  if (review.schema !== 'librarian-reviewed-source-metadata/v1' || review.sourceSha256 !== capture.sourceSha256
      || typeof review.reviewer !== 'string' || !review.reviewer.trim() || review.reviewer.length > 200
      || !hex(review.evidenceReceiptSha256) || !review.fields || typeof review.fields !== 'object'
      || Array.isArray(review.fields) || Object.keys(review.fields).some(key => !['title', 'authors'].includes(key))) {
    throw new TypeError('An identity-bound independent source review is required');
  }
  const fields = {};
  if (review.fields.title !== undefined) fields.title = reviewedValue(review.fields.title, capture);
  if (review.fields.authors !== undefined) {
    if (!Array.isArray(review.fields.authors) || !review.fields.authors.length || review.fields.authors.length > 16) throw new TypeError('Reviewed authors must be source-quoted individual values');
    fields.authors = review.fields.authors.map(value => reviewedValue(value, capture));
  }
  receipt.review = {reviewer: review.reviewer, evidenceReceiptSha256: review.evidenceReceiptSha256, fields};
  const result = {...before};
  for (const field of Object.keys(fields)) {
    if (display.editedAt || (field === 'title' ? !filenameTitle(before.title) : meaningfulAuthors(before.authors))) receipt.preservedFields.push(field);
    else {
      result[field] = field === 'title' ? fields.title.value : fields.authors.map(value => value.value);
      receipt.appliedFields.push(field);
    }
  }
  receipt.unresolvedFields = [filenameTitle(result.title) ? 'title' : null, meaningfulAuthors(result.authors) ? null : 'authors'].filter(Boolean);
  receipt.confidence = 'independently_reviewed_source_quotes';
  receipt.state = display.editedAt ? 'user_edit_preserved' : receipt.unresolvedFields.length ? 'needs_review' : 'reviewed_complete';
  return {display: result, receipt};
}

export function metadataQuality(book) {
  const reasons = [filenameTitle(book.title) ? 'filename_title' : null,
    meaningfulAuthors(book.authors) ? null : 'authors_not_identified'].filter(Boolean);
  return {needsReview: reasons.length > 0, reasons, userEdited: Boolean(book.metadata?.editedAt),
    recovery: book.metadata?.sourceMetadata?.recovery ? {state: book.metadata.sourceMetadata.recovery.state, confidence: book.metadata.sourceMetadata.recovery.confidence} : null};
}
