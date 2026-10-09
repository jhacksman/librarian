import {format as formatMessage} from 'node:util';
import {extractBook, inventoryArchive, extractionLimits, ExtractionError} from './extract.mjs';

// Exactly one bounded JSON request and one JSON response. The importer starts a
// fresh process per source and owns its hard deadline and exit/abort handling.
const MAX_REQUEST_BYTES = 16 * 1024;
const parent = process.ppid;
const parentWatch = setInterval(() => {
  if (process.ppid !== parent || parent === 1) process.exit(1);
}, 1000);
parentWatch.unref();
let diagnosticBytes = 0;
for (const method of ['log', 'info', 'warn', 'error', 'debug']) {
  console[method] = (...args) => {
    if (diagnosticBytes >= 16 * 1024) return;
    const raw = Buffer.from(`${formatMessage(...args)}\n`);
    const bounded = raw.subarray(0, 16 * 1024 - diagnosticBytes);
    diagnosticBytes += bounded.length;
    process.stderr.write(bounded);
  };
}

let response;
let outputLimit = 64 * 1024 ** 2;
try {
  const parts = [];
  let size = 0;
  for await (const part of process.stdin) {
    size += part.length;
    if (size > MAX_REQUEST_BYTES) throw new ExtractionError('INVALID_REQUEST', 'Extractor request exceeds 16 KiB.');
    parts.push(part);
  }
  const raw = Buffer.concat(parts).toString('utf8').trim();
  if (!raw || raw.includes('\n') || raw.includes('\r')) throw new ExtractionError('INVALID_REQUEST', 'Expected one JSON request line.');
  let request;
  try { request = JSON.parse(raw); }
  catch { throw new ExtractionError('INVALID_REQUEST', 'Extractor request is not valid JSON.'); }
  if (!request || typeof request !== 'object' || Array.isArray(request)
    || typeof request.path !== 'string' || !request.path || request.path.length > 4096
    || (request.outputDir !== undefined && typeof request.outputDir !== 'string')) {
    throw new ExtractionError('INVALID_REQUEST', 'Extractor request requires a source path and optional output directory.');
  }
  const limits = extractionLimits(request.limits);
  outputLimit = limits.maxOutputBytes;
  if (request.operation === 'inventoryArchive') response = {ok: true, archive: await inventoryArchive(request.path, {limits})};
  else if (request.operation === undefined || ['extractBook', 'extractMetadata'].includes(request.operation)) {
    // Converter configuration is supplied only by the trusted importer/operator,
    // never forwarded from an HTTP request body.
    response = {ok: true, book: await extractBook(request.path, {format: request.format, limits, outputDir: request.outputDir,
      converter: request.converter, derivedDir: request.derivedDir, metadataOnly: request.operation === 'extractMetadata', frontMatter: request.frontMatter ?? false})};
  } else throw new ExtractionError('INVALID_REQUEST', 'Unknown extractor operation.');
  if (Buffer.byteLength(JSON.stringify(response)) + 1 > outputLimit) throw new ExtractionError('LIMIT_EXCEEDED', 'Extractor response exceeds its byte limit.');
} catch (error) {
  response = {ok: false, error: {code: error.code ?? 'EXTRACT_FAILED', message: String(error.message).slice(0, 500)}};
  process.exitCode = 1;
} finally { clearInterval(parentWatch); }

await new Promise((resolve, reject) => process.stdout.write(`${JSON.stringify(response)}\n`, (error) => error ? reject(error) : resolve()));
