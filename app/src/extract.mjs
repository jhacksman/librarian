import {createHash} from 'node:crypto';
import {mkdir, lstat, readFile, realpath, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {crc32} from 'node:zlib';
import {convertBook, ConversionError} from './convert.mjs';
import {rasterThumbnail} from './book-assets.mjs';
import {FRONT_MATTER_LIMITS, captureFrontMatterPage, frontMatterEvidence} from './source-metadata-recovery.mjs';

// Extraction runs in a disposable child process; the importer owns timeouts,
// source custody and database writes. Limits fail explicitly, never truncate.
export const DEFAULT_LIMITS = Object.freeze({
  maxSourceBytes: 2 * 1024 ** 3, maxPdfBytes: 512 * 1024 ** 2,
  maxZipEntries: 10000, maxEntryBytes: 64 * 1024 ** 2,
  maxTotalUncompressedBytes: 2 * 1024 ** 3, maxCompressionRatio: 1000,
  maxPages: 5000, maxSections: 5000, maxSectionChars: 8 * 1024 ** 2,
  maxTextChars: 32 * 1024 ** 2, maxOutputBytes: 64 * 1024 ** 2,
  maxMetadataChars: 65536, maxCoverBytes: 8 * 1024 ** 2,
  maxSourceMetadataBytes: 1024 ** 2,
});

export class ExtractionError extends Error {
  constructor(code, message) { super(message); this.name = 'ExtractionError'; this.code = code; }
}

const fail = (code, message) => { throw new ExtractionError(code, message); };
const requireValue = (condition, code, message) => { if (!condition) fail(code, message); };
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const clean = (value) => String(value ?? '').replace(/\u0000/g, '').replace(/\s+/g, ' ').trim();

export function extractionLimits(overrides = {}) {
  requireValue(overrides && typeof overrides === 'object' && !Array.isArray(overrides), 'INVALID_LIMIT', 'Limits must be an object.');
  const result = {...DEFAULT_LIMITS};
  for (const [key, value] of Object.entries(overrides)) {
    requireValue(Object.hasOwn(result, key) && Number.isSafeInteger(value) && value > 0 && value <= result[key],
      'INVALID_LIMIT', `Invalid extraction limit: ${key}`);
    result[key] = value;
  }
  return result;
}

async function sourceInfo(filename, limits) {
  const stat = await lstat(filename);
  requireValue(stat.isFile() && !stat.isSymbolicLink(), 'INVALID_SOURCE', 'Source must be a regular managed file.');
  requireValue(stat.size > 0, 'EMPTY_SOURCE', 'Source file is empty (0 bytes); there is no content to extract.');
  requireValue(stat.size <= limits.maxSourceBytes, 'LIMIT_EXCEEDED', 'Source file exceeds the extraction size limit.');
  return stat;
}

function safeMember(name) {
  requireValue(typeof name === 'string' && name.length > 0 && name.length <= 4096 && !/[\u0000-\u001f\\]/.test(name)
    && !name.startsWith('/') && !/^[a-z][a-z0-9+.-]*:/i.test(name)
    && !name.split('/').some((part) => part === '..' || part === '.'), 'UNSAFE_ARCHIVE', 'Unsafe archive member path.');
  return name;
}

async function openArchive(filename, limits, {inventoryOnly = false} = {}) {
  await sourceInfo(filename, limits);
  const {default: yauzl} = await import('yauzl');
  const zip = await yauzl.openPromise(filename, {autoClose: false, strictFileNames: true, validateEntrySizes: true});
  const entries = new Map();
  let total = 0;
  let inflated = 0;
  try {
    requireValue(zip.entryCount <= limits.maxZipEntries, 'LIMIT_EXCEEDED', 'Archive contains too many entries.');
    for await (const entry of zip.eachEntry()) {
      const name = safeMember(entry.fileName);
      requireValue(!entries.has(name), 'UNSAFE_ARCHIVE', 'Archive contains duplicate member paths.');
      requireValue(entries.size < limits.maxZipEntries, 'LIMIT_EXCEEDED', 'Archive contains too many entries.');
      const unixType = (entry.externalFileAttributes >>> 16) & 0xf000;
      requireValue(unixType === 0 || unixType === 0x8000 || unixType === 0x4000, 'UNSAFE_ARCHIVE', 'Archive contains a link or special file.');
      requireValue(!(entry.generalPurposeBitFlag & 1), 'ENCRYPTED_ARCHIVE', 'Encrypted ZIP entries are not supported.');
      requireValue([0, 8].includes(entry.compressionMethod), 'UNSUPPORTED_COMPRESSION', 'Unsupported ZIP compression method.');
      requireValue(Number.isSafeInteger(entry.uncompressedSize) && Number.isSafeInteger(entry.compressedSize)
        && entry.uncompressedSize >= 0 && entry.compressedSize >= 0, 'UNSAFE_ARCHIVE', 'Invalid archive entry sizes.');
      if (!inventoryOnly) {
        requireValue(entry.uncompressedSize <= limits.maxEntryBytes, 'LIMIT_EXCEEDED', 'Archive member exceeds its size limit.');
        requireValue(entry.uncompressedSize <= Math.max(1, entry.compressedSize) * limits.maxCompressionRatio,
          'LIMIT_EXCEEDED', 'Archive member exceeds its compression ratio limit.');
      }
      total += entry.uncompressedSize;
      requireValue(total <= limits.maxTotalUncompressedBytes, 'LIMIT_EXCEEDED', 'Archive exceeds its total expansion limit.');
      entries.set(name, entry);
    }
    return {
      entries, total,
      close: () => zip.close(),
      async read(name, cap = limits.maxEntryBytes) {
        const entry = entries.get(safeMember(name));
        requireValue(entry && !name.endsWith('/'), 'MISSING_MEMBER', `Required archive member is missing: ${name}`);
        cap = Math.min(cap, limits.maxEntryBytes);
        requireValue(entry.uncompressedSize <= cap, 'LIMIT_EXCEEDED', 'Requested archive member exceeds its size limit.');
        requireValue(entry.uncompressedSize <= Math.max(1, entry.compressedSize) * limits.maxCompressionRatio,
          'LIMIT_EXCEEDED', 'Requested archive member exceeds its compression ratio limit.');
        const stream = await zip.openReadStreamPromise(entry);
        const buffers = [];
        let size = 0;
        let checksum = 0;
        try {
          for await (const chunk of stream) {
            size += chunk.length;
            inflated += chunk.length;
            requireValue(size <= cap && inflated <= limits.maxTotalUncompressedBytes,
              'LIMIT_EXCEEDED', 'Inflated archive data exceeds its limit.');
            checksum = crc32(chunk, checksum);
            buffers.push(chunk);
          }
        } finally { stream.destroy(); }
        requireValue(size === entry.uncompressedSize && checksum === entry.crc32, 'CORRUPT_ARCHIVE', 'Archive member failed its size or CRC check.');
        return Buffer.concat(buffers, size);
      },
    };
  } catch (error) { zip.close(); throw error; }
}

export async function inventoryArchive(filename, {limits: overrides} = {}) {
  const limits = extractionLimits(overrides);
  const archive = await openArchive(filename, limits, {inventoryOnly: true});
  try {
    return {format: 'zip', entries: [...archive.entries].map(([name, entry]) => ({name,
      compressedBytes: entry.compressedSize, uncompressedBytes: entry.uncompressedSize, directory: name.endsWith('/')})),
    totalUncompressedBytes: archive.total, contentsVerified: false,
    warnings: ['Archive inventory only. Entries were not extracted, executed or imported; content CRCs have not been verified.']};
  } finally { archive.close(); }
}

async function markup(raw, xmlMode = false) {
  const encoding = (raw[0] === 0xff && raw[1] === 0xfe) || (raw[0] === 0x3c && raw[1] === 0)
    ? 'utf-16le' : (raw[0] === 0xfe && raw[1] === 0xff) || (raw[0] === 0 && raw[1] === 0x3c) ? 'utf-16be' : 'utf-8';
  const text = new TextDecoder(encoding, {fatal: true}).decode(raw);
  requireValue(!/<!ENTITY\b|<!DOCTYPE[^>]*\[/i.test(text), 'UNSAFE_MARKUP', 'Custom XML entity declarations are not supported.');
  const {Parser} = await import('htmlparser2');
  const root = {tag: '#root', attrs: {}, children: []};
  const stack = [root];
  let nodes = 0;
  const parser = new Parser({
    onopentag(tag, attrs) {
      requireValue(++nodes <= 200000 && stack.length < 128, 'LIMIT_EXCEEDED', 'Document markup exceeds structural limits.');
      const node = {tag: tag.toLowerCase(), attrs, children: []};
      stack.at(-1).children.push(node);
      stack.push(node);
    },
    onclosetag() { if (stack.length > 1) stack.pop(); },
    ontext(value) { stack.at(-1).children.push(value); },
  }, {xmlMode, decodeEntities: true, lowerCaseTags: true, lowerCaseAttributeNames: true, recognizeSelfClosing: true});
  parser.end(text);
  return root;
}

const localName = (node) => node.tag?.split(':').at(-1);
function descendants(node, predicate) {
  const found = [];
  const pending = [...(node.children ?? [])].reverse();
  while (pending.length) {
    const next = pending.pop();
    if (typeof next === 'string') continue;
    if (predicate(next)) found.push(next);
    pending.push(...next.children.slice().reverse());
  }
  return found;
}
const named = (node, name) => descendants(node, (item) => localName(item) === name);
const textOf = (node) => typeof node === 'string' ? node : (node?.children ?? []).map(textOf).join('');

function resolveMember(base, href) {
  requireValue(typeof href === 'string' && href.length <= 4096 && !/[\u0000-\u001f\\]/.test(href)
    && !/^[a-z][a-z0-9+.-]*:/i.test(href) && !href.startsWith('/'), 'UNSAFE_ARCHIVE', 'External or unsafe EPUB reference.');
  const [rawPath, ...fragment] = href.split('#');
  let member;
  let heading;
  try {
    const decoded = decodeURIComponent(rawPath);
    requireValue(!decoded.startsWith('/') && !/^[a-z][a-z0-9+.-]*:/i.test(decoded) && !/[\u0000-\u001f\\]/.test(decoded),
      'UNSAFE_ARCHIVE', 'Unsafe decoded EPUB reference.');
    member = rawPath ? path.posix.normalize(path.posix.join(path.posix.dirname(base), decoded)) : base;
    heading = fragment.length ? decodeURIComponent(fragment.join('#')) : undefined;
  } catch { fail('UNSAFE_ARCHIVE', 'Invalid EPUB reference encoding.'); }
  safeMember(member);
  requireValue(!heading || !/[\u0000-\u001f]/.test(heading), 'UNSAFE_ARCHIVE', 'Invalid EPUB fragment.');
  return {member, ...(heading ? {heading} : {})};
}

function sectionText(root) {
  const body = named(root, 'body')[0] ?? root;
  const blocks = new Set(['p', 'div', 'section', 'article', 'header', 'footer', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote', 'figure', 'figcaption', 'pre', 'table', 'tr']);
  const skip = new Set(['head', 'script', 'style', 'noscript', 'template', 'iframe', 'object', 'svg']);
  const output = [];
  const ranges = new Map();
  let length = 0;
  const append = (value) => {
    const text = value.replace(/\u0000/g, '');
    output.push(text);
    length += text.length;
  };
  function visit(node, pre = false, omitted = false) {
    if (typeof node === 'string') {
      if (!omitted) append(pre ? node.replace(/\r\n?/g, '\n') : node.replace(/\s+/g, ' '));
      return;
    }
    const tag = localName(node);
    omitted ||= skip.has(tag) || Object.hasOwn(node.attrs, 'hidden') || node.attrs['aria-hidden'] === 'true';
    const ids = [...new Set([node.attrs.id, node.attrs['xml:id'], tag === 'a' ? node.attrs.name : undefined])]
      .filter((id) => typeof id === 'string' && id.length > 0);
    const range = {start: {offset: length}, end: {offset: length}};
    // Duplicate IDs are ambiguous even if one occurs inside hidden content.
    for (const id of ids) ranges.set(id, ranges.has(id) || omitted ? null : range);
    if (!omitted) {
      if (blocks.has(tag) || tag === 'br') append('\n');
      range.start.offset = length;
      if (tag === 'li') append('• ');
    }
    for (const child of node.children) visit(child, pre || tag === 'pre', omitted);
    range.end.offset = length;
    if (!omitted) {
      if (tag === 'td' || tag === 'th') append('\t');
      if (blocks.has(tag)) append('\n');
    }
  }
  visit(body);
  // Every remaining normalization only deletes characters. Move sorted anchor
  // boundaries through those deletions instead of searching for repeated text.
  const points = [...new Set([...ranges.values()].filter(Boolean).flatMap((range) => [range.start, range.end]))]
    .sort((a, b) => a.offset - b.offset);
  let text = output.join('');
  const remove = (pattern, keep = 0) => {
    let cursor = 0;
    let deleted = 0;
    text = text.replace(pattern, (match, offset) => {
      const start = offset + keep;
      const end = offset + match.length;
      while (cursor < points.length && points[cursor].offset <= start) points[cursor++].offset -= deleted;
      while (cursor < points.length && points[cursor].offset <= end) points[cursor++].offset = start - deleted;
      deleted += end - start;
      return match.slice(0, keep);
    });
    while (cursor < points.length) points[cursor++].offset -= deleted;
  };
  remove(/\n[ \t]+(?=\n)/g, 1);
  remove(/\n{3,}/g, 2);
  remove(/^\s+|\s+$/g);
  const anchors = [];
  for (const [id, range] of ranges) {
    if (!range) continue;
    let charStart = range.start.offset;
    let charEnd = range.end.offset;
    while (charStart < charEnd && /\s/u.test(text[charStart])) charStart += 1;
    while (charEnd > charStart && /\s/u.test(text[charEnd - 1])) charEnd -= 1;
    // Empty legacy <a name> markers select the next visible code point. Keep a
    // supplementary character intact; all stored offsets remain UTF-16 units.
    if (charStart === charEnd) {
      while (charStart < text.length && /\s/u.test(text[charStart])) charStart += 1;
      charEnd = charStart + (text.codePointAt(charStart) > 0xffff ? 2 : 1);
    }
    if (charStart < text.length && charEnd <= text.length) anchors.push([id, {charStart, charEnd}]);
  }
  return {text, anchors: Object.fromEntries(anchors)};
}

async function saveCover(bytes, outputDir, limits) {
  if (!outputDir) return null;
  requireValue(bytes.length <= limits.maxCoverBytes, 'LIMIT_EXCEEDED', 'Cover exceeds the image size limit.');
  let extension;
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) extension = 'png';
  else if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) extension = 'jpg';
  else if (/^GIF8[79]a/.test(bytes.subarray(0, 6).toString('ascii'))) extension = 'gif';
  else if (bytes.subarray(0, 4).toString('ascii') === 'RIFF' && bytes.subarray(8, 12).toString('ascii') === 'WEBP') extension = 'webp';
  else fail('UNSUPPORTED_COVER', 'Only raster PNG, JPEG, GIF or WebP covers are retained.');
  await mkdir(outputDir, {recursive: true});
  const directory = await realpath(outputDir);
  requireValue(!(await lstat(outputDir)).isSymbolicLink(), 'INVALID_OUTPUT', 'Cover directory must not be a symbolic link.');
  const destination = path.join(directory, `${sha256(bytes)}.${extension}`);
  try { await writeFile(destination, bytes, {flag: 'wx', mode: 0o600}); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    requireValue((await lstat(destination)).isFile() && !(await lstat(destination)).isSymbolicLink()
      && sha256(await readFile(destination)) === sha256(bytes), 'INVALID_OUTPUT', 'Cover destination conflicts with an existing file.');
  }
  return destination;
}

function addSection(book, section, limits) {
  requireValue(book.sections.length < limits.maxSections && section.text.length <= limits.maxSectionChars,
    'LIMIT_EXCEEDED', 'Extracted section exceeds its configured limit.');
  book.metadata.textCharacters += section.text.length;
  requireValue(book.metadata.textCharacters <= limits.maxTextChars, 'LIMIT_EXCEEDED', 'Extracted text exceeds its total limit.');
  book.sections.push({ordinal: book.sections.length + 1, ...section});
}

function blankBook(filename, format) {
  return {title: path.basename(filename, path.extname(filename)), authors: [], description: '', language: '', publisher: '',
    metadata: {format, extractionVersion: 'js-extract-v1', textCharacters: 0,
      textOffsetBasis: 'section_utf16_code_units', extractionCompleteness: 'not_assessed'},
    coverPath: null, sections: [], toc: [], warnings: []};
}

// Source metadata is independent of the editable display fields. Preserve its
// original hierarchy, repeated values and attributes; never silently truncate.
function retainSourceMetadata(book, metadata, limits) {
  requireValue(Buffer.byteLength(JSON.stringify(metadata)) <= limits.maxSourceMetadataBytes,
    'LIMIT_EXCEEDED', 'Complete source metadata exceeds its byte limit.');
  book.metadata.sourceMetadata = {captureVersion: 'source-metadata-v1', ...metadata};
  requireValue(Buffer.byteLength(JSON.stringify(book.metadata.sourceMetadata)) <= limits.maxSourceMetadataBytes,
    'LIMIT_EXCEEDED', 'Complete source metadata exceeds its byte limit.');
}

async function retainCover(book, bytes, {outputDir, limits, kind, provenance}) {
  // Normalization happens in the disposable extractor process. The original
  // cover remains available separately, while the API serves only the DB asset.
  requireValue(bytes.length <= limits.maxCoverBytes, 'LIMIT_EXCEEDED', 'Cover exceeds the image size limit.');
  const raster = bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
    || (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff)
    || /^GIF8[79]a/.test(bytes.subarray(0, 6).toString('ascii'))
    || (bytes.subarray(0, 4).toString('ascii') === 'RIFF' && bytes.subarray(8, 12).toString('ascii') === 'WEBP');
  requireValue(raster, 'UNSUPPORTED_COVER', 'Only raster PNG, JPEG, GIF or WebP covers are retained.');
  try { book.thumbnail = await rasterThumbnail(bytes, {kind, provenance}); }
  catch (error) { book.warnings.push(`Thumbnail unavailable: ${error.code ?? 'INVALID_COVER'}`); }
  if (outputDir) {
    try { book.coverPath = await saveCover(bytes, outputDir, limits); }
    catch (error) { book.warnings.push(`Cover unavailable: ${error.code ?? 'INVALID_COVER'}`); }
  }
}

async function extractEpub(filename, options) {
  const {limits, outputDir, metadataOnly = false} = options;
  const archive = await openArchive(filename, limits);
  const book = blankBook(filename, 'epub');
  try {
    if (archive.entries.has('META-INF/encryption.xml')) {
      const encryption = await markup(await archive.read('META-INF/encryption.xml', 1024 ** 2), true);
      const algorithms = named(encryption, 'encryptionmethod').map((node) => node.attrs.algorithm);
      requireValue(algorithms.every((value) => ['http://www.idpf.org/2008/embedding', 'http://ns.adobe.com/pdf/enc#RC'].includes(value)),
        'ENCRYPTED_BOOK', 'Encrypted EPUB content is not supported.');
      book.warnings.push('EPUB contains obfuscated fonts; text extraction does not load these fonts.');
    }
    const container = await markup(await archive.read('META-INF/container.xml', 1024 ** 2), true);
    const rootfiles = named(container, 'rootfile');
    requireValue(rootfiles.length > 0, 'INVALID_EPUB', 'EPUB container has no package document.');
    const packagePath = safeMember(rootfiles.find((node) => node.attrs['media-type'] === 'application/oebps-package+xml')?.attrs['full-path'] ?? rootfiles[0].attrs['full-path']);
    const packageBytes = await archive.read(packagePath, 4 * 1024 ** 2);
    const opf = await markup(packageBytes, true);
    const packageNode = named(opf, 'package')[0];
    const metadata = named(opf, 'metadata')[0];
    const manifest = named(opf, 'manifest')[0];
    const spine = named(opf, 'spine')[0];
    requireValue(packageNode && metadata && manifest && spine, 'INVALID_EPUB', 'EPUB package is missing metadata, manifest or spine.');
    const field = (name) => clean(textOf(named(metadata, name)[0]));
    book.title = field('title') || book.title;
    book.authors = named(metadata, 'creator').map((node) => clean(textOf(node))).filter(Boolean);
    book.description = field('description');
    book.language = field('language');
    book.publisher = field('publisher');
    book.metadata.identifiers = named(metadata, 'identifier').map((node) => clean(textOf(node))).filter(Boolean);
    book.metadata.publicationDate = field('date');
    book.metadata.packageMember = packagePath;
    retainSourceMetadata(book, {format: 'epub', packageMember: packagePath,
      packageSha256: sha256(packageBytes), packageAttributes: {...packageNode.attrs},
      metadataTree: metadata,
      // Exact package bytes preserve case, namespaces, whitespace, comments and
      // markup that the bounded semantic parser does not represent.
      rawPackage: {encoding: 'base64', value: packageBytes.toString('base64'), byteLength: packageBytes.length},
      entries: (metadata.children ?? []).filter((node) => typeof node !== 'string').map((node, ordinal) => ({
        ordinal, name: node.tag, attributes: {...node.attrs}, value: textOf(node)})),
    }, limits);
    const edition = named(metadata, 'meta').find((node) =>
      /(?:^|:)(?:edition|edition-number)$/i.test(node.attrs.property ?? node.attrs.name ?? ''));
    if (edition) book.metadata.edition = clean(textOf(edition) || edition.attrs.content);
    const items = new Map();
    for (const item of named(manifest, 'item')) {
      requireValue(item.attrs.id && !items.has(item.attrs.id), 'INVALID_EPUB', 'EPUB manifest has a missing or duplicate identifier.');
      if (/^[a-z][a-z0-9+.-]*:|^\/\//i.test(item.attrs.href ?? '')) {
        items.set(item.attrs.id, {...item.attrs, external: true});
        book.warnings.push('An external EPUB manifest resource was not fetched.');
        continue;
      }
      items.set(item.attrs.id, {...item.attrs, ...resolveMember(packagePath, item.attrs.href)});
    }
    const refs = named(spine, 'itemref');
    requireValue(refs.length > 0 && refs.length <= limits.maxSections, 'INVALID_EPUB', 'EPUB has no usable bounded reading order.');
    const seen = new Set();
    for (const ref of metadataOnly ? [] : refs) {
      const item = items.get(ref.attrs.idref);
      if (item?.external) {
        book.warnings.push('An external reading-order item was omitted; its text was not fetched.');
        continue;
      }
      requireValue(item && !seen.has(item.member), 'INVALID_EPUB', 'EPUB spine references a missing or duplicate item.');
      seen.add(item.member);
      if (!['application/xhtml+xml', 'text/html'].includes(item['media-type'])) {
        book.warnings.push(`Reading-order item has unsupported content type: ${item.member}`);
        continue;
      }
      const document = await markup(await archive.read(item.member));
      const heading = descendants(document, (node) => /^h[1-6]$/.test(localName(node)))[0];
      const title = clean(textOf(heading)) || clean(textOf(named(document, 'title')[0])) || path.posix.basename(item.member);
      const {text, anchors} = sectionText(document);
      addSection(book, {title, text, locator: {format: 'epub', member: item.member,
        ...(heading?.attrs.id ? {heading: heading.attrs.id} : {}),
        ...(Object.keys(anchors).length ? {anchors} : {})}, linear: ref.attrs.linear !== 'no'}, limits);
      if (!text) book.warnings.push(`No searchable text in EPUB member: ${item.member}`);
    }
    const nav = [...items.values()].find((item) => !item.external && (item.properties ?? '').split(/\s+/).includes('nav'));
    if (!metadataOnly && nav) {
      const document = await markup(await archive.read(nav.member));
      const navigation = named(document, 'nav').find((node) => (node.attrs['epub:type'] ?? '').split(/\s+/).includes('toc'));
      for (const link of named(navigation ?? document, 'a')) {
        if (!link.attrs.href) continue;
        try {
          const target = resolveMember(nav.member, link.attrs.href);
          if (archive.entries.has(target.member)) book.toc.push({title: clean(textOf(link)), locator: {format: 'epub', ...target}});
        } catch { book.warnings.push('An external or unsafe EPUB navigation target was omitted.'); }
      }
    } else if (!metadataOnly) {
      const ncx = items.get(spine.attrs.toc) ?? [...items.values()].find((item) => item['media-type'] === 'application/x-dtbncx+xml');
      if (ncx && !ncx.external) {
        const document = await markup(await archive.read(ncx.member), true);
        for (const point of named(document, 'navpoint')) {
          const label = named(point, 'navlabel')[0];
          const content = named(point, 'content')[0];
          if (!content?.attrs.src) continue;
          try {
            const target = resolveMember(ncx.member, content.attrs.src);
            if (archive.entries.has(target.member)) book.toc.push({title: clean(textOf(label)), locator: {format: 'epub', ...target}});
          } catch { book.warnings.push('An external or unsafe EPUB navigation target was omitted.'); }
        }
      }
    }
    const sectionByMember = new Map(book.sections.map((section) => [section.locator.member, section]));
    for (const entry of book.toc) {
      const {member, heading} = entry.locator;
      if (!heading) continue;
      const anchors = sectionByMember.get(member)?.locator.anchors;
      if (anchors && Object.hasOwn(anchors, heading)) {
        Object.assign(entry.locator, anchors[heading], {offsetBasis: 'section_utf16_code_units'});
      } else book.warnings.push(`EPUB navigation fragment has no unambiguous extracted-text location: ${member}#${heading}`);
    }
    const coverId = named(metadata, 'meta').find((node) => node.attrs.name === 'cover')?.attrs.content;
    let cover = [...items.values()].find((item) => (item.properties ?? '').split(/\s+/).includes('cover-image')) ?? items.get(coverId);
    let declaration = cover && (cover.properties ?? '').split(/\s+/).includes('cover-image') ? 'epub3_cover_image' : 'epub2_cover_meta';
    if (!cover) {
      const reference = named(named(opf, 'guide')[0] ?? {children: []}, 'reference')
        .find((node) => (node.attrs.type ?? '').split(/\s+/).includes('cover'));
      if (reference?.attrs.href) {
        try { cover = {...resolveMember(packagePath, reference.attrs.href)}; declaration = 'epub_guide_cover'; }
        catch { book.warnings.push('An external or unsafe EPUB cover guide target was omitted.'); }
      }
    }
    if (cover && !cover.external) {
      try {
        let member = cover.member;
        const manifestItem = [...items.values()].find(item => item.member === member);
        if (['application/xhtml+xml', 'text/html'].includes(manifestItem?.['media-type']) || /\.x?html?$/i.test(member)) {
          const document = await markup(await archive.read(member));
          const candidates = descendants(document, (node) => ['img', 'image'].includes(localName(node)));
          const targets = [];
          for (const node of candidates) {
            const href = node.attrs.src ?? node.attrs.href ?? node.attrs['xlink:href'];
            if (!href) continue;
            try {
              const target = resolveMember(member, href);
              if (archive.entries.has(target.member)) targets.push(target.member);
            } catch { book.warnings.push('An external or unsafe EPUB cover image target was omitted.'); }
          }
          const uniqueTargets = [...new Set(targets)];
          requireValue(uniqueTargets.length === 1, 'AMBIGUOUS_COVER', 'Declared EPUB cover document has no unambiguous local image.');
          member = uniqueTargets[0];
        }
        await retainCover(book, await archive.read(member, limits.maxCoverBytes), {outputDir, limits,
          kind: 'cover', provenance: {format: 'epub', packageMember: packagePath, member,
            ...(member !== cover.member ? {coverDocument: cover.member} : {}), declaration}});
      }
      catch (error) { book.warnings.push(`Cover unavailable: ${error.code ?? 'INVALID_COVER'}`); }
    }
    book.metadata.spineItems = refs.length;
    book.metadata.archiveEntries = archive.entries.size;
    if (!metadataOnly) book.warnings.push('EPUB text preserves reading order, block boundaries, code whitespace and table cells where represented in markup. Visual layout and image-only text are not reconstructed.');
    return book;
  } finally { archive.close(); }
}

function pdfPageText(items) {
  const parts = [];
  let previous;
  for (const item of items) {
    if (typeof item.str !== 'string') continue;
    const transform = item.transform ?? [1, 0, 0, 1, 0, 0];
    const x = transform[4];
    const y = transform[5];
    const height = Math.max(1, Math.abs(item.height || transform[3] || 10));
    if (previous && item.str) {
      const gap = x - previous.end;
      if (Math.abs(y - previous.y) > Math.max(2, height * 0.4)) { if (!previous.eol) parts.push('\n'); }
      else if (!previous.eol && !/\s$/.test(previous.text) && !/^\s/.test(item.str)) {
        if (gap > height * 2) parts.push('\t');
        else if (gap > height * 0.1) parts.push(' ');
      }
    }
    parts.push(item.str);
    if (item.hasEOL) parts.push('\n');
    previous = {y, end: x + (item.width || 0), text: item.str, eol: item.hasEOL};
  }
  return parts.join('').replace(/\u0000/g, '').replace(/\n{3,}/g, '\n\n').trim();
}

async function extractPdf(filename, {limits, outputDir, metadataOnly = false, frontMatter = false}) {
  const stat = await sourceInfo(filename, limits);
  requireValue(stat.size <= limits.maxPdfBytes, 'LIMIT_EXCEEDED', 'PDF exceeds its in-process size limit.');
  // Avoid system/home font scanning; only local package assets can be loaded.
  process.env.DISABLE_SYSTEM_FONTS_LOAD = '1';
  const require = createRequire(import.meta.url);
  const packageRoot = path.dirname(require.resolve('pdfjs-dist/package.json'));
  class PackagedBinaryData {
    async fetch({kind, filename: name}) {
      const folders = {cMapUrl: 'cmaps', standardFontDataUrl: 'standard_fonts'};
      requireValue(Object.hasOwn(folders, kind) && typeof name === 'string' && /^[a-zA-Z0-9_.-]+$/.test(name)
        && name !== '.' && name !== '..', 'UNSUPPORTED_PDF_RESOURCE', 'Only packaged PDF character maps and fonts may be loaded.');
      const resource = path.join(packageRoot, folders[kind], name);
      const info = await lstat(resource);
      requireValue(info.isFile() && !info.isSymbolicLink() && info.size <= 16 * 1024 ** 2,
        'UNSUPPORTED_PDF_RESOURCE', 'Invalid packaged PDF resource.');
      const bytes = await readFile(resource);
      return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    }
  }
  const {getDocument} = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const bytes = await readFile(filename);
  // PDF.js may transfer/detach the input ArrayBuffer during document loading.
  // Bind evidence to the original bytes before handing that buffer to PDF.js.
  const originalSourceSha256 = frontMatter ? sha256(bytes) : null;
  const data = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const task = getDocument({data, verbosity: 0, stopAtErrors: true,
    useWorkerFetch: false, BinaryDataFactory: PackagedBinaryData, cMapPacked: true,
    useSystemFonts: false, disableFontFace: true, enableXfa: false, useWasm: false,
    isOffscreenCanvasSupported: false, isImageDecoderSupported: false, maxImageSize: 40_000_000});
  const book = blankBook(filename, 'pdf');
  try {
    const document = await task.promise;
    requireValue(document.numPages <= limits.maxPages && document.numPages <= limits.maxSections,
      'LIMIT_EXCEEDED', 'PDF exceeds the page limit.');
    book.metadata.pageCount = document.numPages;
    book.metadata.pagesWithoutText = [];
    book.metadata.ocrPerformed = false;
    book.metadata.extractionCompleteness = 'not_assessed';
    try {
      const {info: parsedInfo, metadata, ...documentProperties} = await document.getMetadata();
      // Preserve every PDF.js-reported custom Info entry before JSON serialization.
      const info = {...parsedInfo, ...(parsedInfo.Custom instanceof Map ? {Custom: Object.fromEntries(parsedInfo.Custom)} : {})};
      // PDF.js 6 exposes every parsed XMP field through its metadata iterator.
      const xmp = metadata ? Object.fromEntries(metadata) : {};
      retainSourceMetadata(book, {format: 'pdf', fingerprints: document.fingerprints, info, xmp, rawXmp: metadata?.getRaw?.() ?? null,
        documentProperties}, limits);
      const value = (name, fallback = '') => metadata?.get(name) || fallback;
      book.title = clean(value('dc:title', info.Title)) || book.title;
      const creators = value('dc:creator', info.Author);
      book.authors = (Array.isArray(creators) ? creators : creators ? [creators] : []).map(clean).filter(Boolean);
      book.description = clean(value('dc:description', info.Subject));
      book.language = clean(value('dc:language'));
      book.publisher = clean(value('dc:publisher'));
      book.metadata.pdf = {version: clean(info.PDFFormatVersion), producer: clean(info.Producer),
        creator: clean(info.Creator), creationDate: clean(info.CreationDate), modificationDate: clean(info.ModDate)};
      const identifiers = [value('dc:identifier'), value('prism:isbn'), value('pdfx:isbn'), info.ISBN, info.Custom?.ISBN]
        .flatMap((entry) => Array.isArray(entry) ? entry : entry ? [entry] : []).map(clean).filter(Boolean);
      if (identifiers.length) book.metadata.identifiers = [...new Set(identifiers)];
      book.metadata.publicationDate = clean(value('dc:date', value('prism:publicationdate')));
      book.metadata.edition = clean(value('prism:edition', value('pdfx:edition', info.Custom?.Edition)));
    } catch (error) {
      if (error instanceof ExtractionError) throw error;
      book.warnings.push('PDF descriptive metadata could not be read.');
      retainSourceMetadata(book, {format: 'pdf', captureError: 'PDF_METADATA_UNAVAILABLE'}, limits);
    }
    if (!metadataOnly) {
    try {
      const pending = (await document.getOutline() ?? []).map((item) => ({item, depth: 0})).reverse();
      while (pending.length) {
        requireValue(book.toc.length < limits.maxSections, 'LIMIT_EXCEEDED', 'PDF outline exceeds its limit.');
        const {item, depth} = pending.pop();
        requireValue(depth < 128, 'LIMIT_EXCEEDED', 'PDF outline nesting exceeds its limit.');
        const destination = typeof item.dest === 'string' ? await document.getDestination(item.dest) : item.dest;
        let page;
        if (Array.isArray(destination) && destination.length) {
          page = Number.isInteger(destination[0]) ? destination[0] + 1 : (await document.getPageIndex(destination[0])) + 1;
        }
        if (page >= 1 && page <= document.numPages) book.toc.push({title: clean(item.title), depth, locator: {format: 'pdf', page}});
        else book.warnings.push('A PDF outline destination could not be resolved to a physical page.');
        for (const child of [...(item.items ?? [])].reverse()) pending.push({item: child, depth: depth + 1});
      }
    } catch (error) {
      if (error instanceof ExtractionError) throw error;
      book.warnings.push('PDF outline is incomplete because a destination could not be read.');
    }
    }
    const frontPages = [];
    for (let number = 1; number <= (metadataOnly ? Math.min(frontMatter ? FRONT_MATTER_LIMITS.pages : 1, document.numPages) : document.numPages); number += 1) {
      const page = await document.getPage(number);
      try {
        if (!metadataOnly || (frontMatter && number <= FRONT_MATTER_LIMITS.pages)) {
          const content = await page.getTextContent({disableNormalization: true});
          const text = pdfPageText(content.items);
          if (frontMatter && number <= FRONT_MATTER_LIMITS.pages) frontPages.push(captureFrontMatterPage(number, text, content.items));
          if (!metadataOnly) {
            addSection(book, {title: `Page ${number}`, text, locator: {format: 'pdf', page: number}}, limits);
            if (!text) book.metadata.pagesWithoutText.push(number);
          }
        }
        if (number === 1) {
          let surface;
          try {
            const original = page.getViewport({scale: 1});
            requireValue(Number.isFinite(original.width) && Number.isFinite(original.height)
              && original.width > 0 && original.height > 0, 'INVALID_COVER', 'Invalid PDF page geometry.');
            const viewport = page.getViewport({scale: Math.min(1, 700 / Math.max(original.width, original.height))});
            surface = document.canvasFactory.create(Math.max(1, Math.ceil(viewport.width)), Math.max(1, Math.ceil(viewport.height)));
            await page.render({canvasContext: surface.context, viewport}).promise;
            await retainCover(book, surface.canvas.toBuffer('image/png'), {outputDir, limits,
              kind: 'first_page', provenance: {format: 'pdf', page: 1, declaration: 'first_physical_page_preview'}});
          } catch (error) { book.warnings.push(`PDF cover unavailable: ${error.code ?? 'RENDER_FAILED'}`); }
          finally { if (surface) document.canvasFactory.destroy(surface); }
        }
      } finally { page.cleanup(); }
    }
    if (book.metadata.pagesWithoutText.length) {
      book.warnings.push(`${book.metadata.pagesWithoutText.length} PDF pages yielded no text (physical pages ${book.metadata.pagesWithoutText.join(', ')}). They may be blank or image-only. OCR was not run.`);
    }
    if (frontMatter) retainSourceMetadata(book, {...book.metadata.sourceMetadata, frontMatter: frontMatterEvidence(originalSourceSha256, frontPages, document.numPages)}, limits);
    if (!metadataOnly) book.warnings.push('PDF text follows physical page order with line and spacing reconstruction from text geometry. Columns, tables and code may need comparison with the original page; extraction completeness is not assessed.');
    return book;
  } finally { await task.destroy(); }
}

export async function extractBook(filename, {format, limits: overrides, outputDir, converter, derivedDir, signal, metadataOnly = false, frontMatter = false} = {}) {
  const limits = extractionLimits(overrides);
  requireValue(typeof metadataOnly === 'boolean' && typeof frontMatter === 'boolean', 'INVALID_REQUEST', 'metadataOnly and frontMatter must be booleans.');
  const detected = String(format ?? path.extname(filename).slice(1)).toLowerCase();
  requireValue(['pdf', 'epub', 'zip'].includes(detected) || (['mobi', 'prc'].includes(detected) && converter !== undefined), 'UNSUPPORTED_FORMAT',
    `Format ${detected || 'unknown'} is not supported by this extractor. MOBI/PRC require a separately reviewed converter; torrents are never fetched.`);
  let book;
  try {
    if (detected === 'pdf') book = await extractPdf(filename, {limits, outputDir, metadataOnly, frontMatter});
    else if (detected === 'epub') book = await extractEpub(filename, {limits, outputDir, metadataOnly});
    else if (detected === 'mobi' || detected === 'prc') {
      await sourceInfo(filename, limits);
      const converted = await convertBook(filename, {format: detected, converter, derivedDir, signal});
      book = await extractEpub(converted.path, {limits, outputDir, metadataOnly});
      book.metadata.format = detected;
      book.metadata.conversion = converted.provenance;
      retainSourceMetadata(book, {...book.metadata.sourceMetadata, format: detected,
        derivedFormat: 'epub', conversion: converted.provenance}, limits);
      if (book.thumbnail) book.thumbnail.provenance.conversion = converted.provenance;
      const convertedLocator = (locator) => ({format: detected,
        sourceSha256: converted.provenance.original.sha256,
        derived: {...locator, sha256: converted.provenance.derived.sha256}});
      book.sections = book.sections.map((section) => ({...section, locator: convertedLocator(section.locator)}));
      book.toc = book.toc.map((entry) => ({...entry, locator: convertedLocator(entry.locator)}));
      book.warnings.push(...converted.provenance.limitations);
    } else {
      book = blankBook(filename, 'zip');
      book.metadata.archive = await inventoryArchive(filename, {limits});
      book.warnings.push(...book.metadata.archive.warnings);
    }
    const metadata = {...book.metadata};
    delete metadata.archive;
    delete metadata.sourceMetadata;
    requireValue(JSON.stringify({title: book.title, authors: book.authors, description: book.description,
      language: book.language, publisher: book.publisher, metadata}).length <= limits.maxMetadataChars,
    'LIMIT_EXCEEDED', 'Book metadata exceeds its limit.');
    requireValue(book.toc.length <= limits.maxSections, 'LIMIT_EXCEEDED', 'Book navigation exceeds its limit.');
    if (metadataOnly) book.metadata.metadataOnly = true;
    else book.metadata.searchable = book.sections.some((section) => section.text.trim());
    if (!metadataOnly && !book.metadata.searchable && detected !== 'zip') book.warnings.push('No searchable text was extracted; the original source remains the reference.');
    requireValue(Buffer.byteLength(JSON.stringify(book)) <= limits.maxOutputBytes, 'LIMIT_EXCEEDED', 'Extraction result exceeds its response limit.');
    return book;
  } catch (error) {
    if (error instanceof ExtractionError || error instanceof ConversionError) throw error;
    if (error.name === 'PasswordException') fail('ENCRYPTED_BOOK', 'Password-protected PDF files are not supported.');
    throw new ExtractionError('EXTRACT_FAILED', `Book extraction failed: ${String(error.message).slice(0, 500)}`);
  }
}
