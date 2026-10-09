import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {constants, createReadStream} from 'node:fs';
import {access, chmod, link, lstat, mkdir, mkdtemp, open, realpath, rm} from 'node:fs/promises';
import path from 'node:path';

// Operator configuration only: never populate converter options from an HTTP
// request. Container mode requires reviewed network:none + owned-cgroup cleanup
// admission; Node itself cannot impose those controls on a native executable.
export const CONVERSION_LIMITS = Object.freeze({
  timeoutMs: 120000, maxSourceBytes: 2 * 1024 ** 3,
  maxOutputBytes: 512 * 1024 ** 2, maxLogBytes: 65536,
});
export class ConversionError extends Error {
  constructor(code, message) { super(message); this.name = 'ConversionError'; this.code = code; }
}
const requireValue = (condition, code, message) => {
  if (!condition) throw new ConversionError(code, message);
};
const absolute = (value) => typeof value === 'string' && value.length <= 4096
  && path.isAbsolute(value) && !/[\u0000-\u001f]/.test(value);

async function fileIdentity(filename, maximum, code = 'INVALID_SOURCE') {
  const info = await lstat(filename);
  requireValue(info.isFile() && !info.isSymbolicLink(), code, 'Expected a regular file without a symbolic link.');
  requireValue(info.size > 0 && info.size <= maximum, 'LIMIT_EXCEEDED', 'Conversion file exceeds its byte limit.');
  const digest = createHash('sha256');
  let bytes = 0;
  const stream = createReadStream(filename, {flags: constants.O_RDONLY | constants.O_NOFOLLOW});
  try {
    for await (const chunk of stream) {
      bytes += chunk.length;
      requireValue(bytes <= maximum, 'LIMIT_EXCEEDED', 'Conversion file exceeds its byte limit.');
      digest.update(chunk);
    }
  } finally { stream.destroy(); }
  requireValue(bytes === info.size, 'SOURCE_CHANGED', 'File changed while its identity was being read.');
  return {sha256: digest.digest('hex'), bytes};
}

async function executable(filename) {
  requireValue(absolute(filename), 'INVALID_CONVERTER', 'Converter executables must have explicit absolute paths.');
  try {
    const resolved = await realpath(filename);
    await access(resolved, constants.X_OK);
    const identity = await fileIdentity(resolved, 512 * 1024 ** 2, 'INVALID_CONVERTER');
    return {path: resolved, sha256: identity.sha256};
  } catch (error) {
    if (error instanceof ConversionError) throw error;
    throw new ConversionError('CONVERTER_UNAVAILABLE', 'Configured converter or sandbox executable is unavailable.');
  }
}

function settings(converter) {
  requireValue(converter && typeof converter === 'object' && !Array.isArray(converter),
    'CONVERTER_UNAVAILABLE', 'MOBI/PRC conversion requires an explicitly configured converter.');
  const keys = new Set(['executable', 'sandboxExecutable', 'installationRoot', 'isolation', ...Object.keys(CONVERSION_LIMITS)]);
  requireValue(Object.keys(converter).every((key) => keys.has(key)), 'INVALID_CONVERTER', 'Unknown converter configuration option.');
  const result = {...CONVERSION_LIMITS, ...converter, isolation: converter.isolation ?? 'bubblewrap'};
  requireValue(['bubblewrap', 'container'].includes(result.isolation), 'INVALID_CONVERTER', 'Unknown converter isolation mode.');
  for (const [key, maximum] of Object.entries(CONVERSION_LIMITS)) {
    requireValue(Number.isSafeInteger(result[key]) && result[key] > 0 && result[key] <= maximum,
      'INVALID_CONVERTER', `Invalid converter bound: ${key}`);
  }
  return result;
}

async function managedDirectory(directory) {
  requireValue(absolute(directory), 'INVALID_OUTPUT', 'A managed absolute derived directory is required.');
  await mkdir(directory, {recursive: true, mode: 0o700});
  const info = await lstat(directory);
  requireValue(info.isDirectory() && !info.isSymbolicLink(), 'INVALID_OUTPUT', 'Derived directory must not be a symbolic link.');
  // The operator owns ancestors; realpath removes their aliases before any use.
  return realpath(directory);
}

function cleanEnvironment(workspace) {
  return {PATH: '/usr/local/bin:/usr/bin:/bin', LANG: 'C.UTF-8', LC_ALL: 'C.UTF-8',
    HOME: workspace, TMPDIR: workspace, XDG_CONFIG_HOME: path.join(workspace, 'config'),
    XDG_CACHE_HOME: path.join(workspace, 'cache'), CALIBRE_CONFIG_DIRECTORY: path.join(workspace, 'calibre'),
    CALIBRE_TEMP_DIR: workspace, QT_QPA_PLATFORM: 'offscreen', PYTHONNOUSERSITE: '1'};
}

async function sandboxCommand(config, binary, sandbox, source, output, workspace, format) {
  const roots = ['/usr', '/bin', '/lib', '/lib64'];
  if (config.installationRoot !== undefined) {
    requireValue(absolute(config.installationRoot), 'INVALID_CONVERTER', 'Calibre installation root must be absolute.');
    const root = await realpath(config.installationRoot);
    requireValue(!['/', '/home', '/root', '/tmp', '/var', '/etc', '/run', '/proc', '/sys'].includes(root)
      && (await lstat(root)).isDirectory() && binary.path.startsWith(`${root}/`),
    'INVALID_CONVERTER', 'Calibre installation root must narrowly contain the configured executable.');
    roots.push(root);
  }
  const mounted = [];
  const args = ['--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL', '--clearenv'];
  for (const root of [...new Set(roots)]) {
    try {
      if ((await lstat(root)).isDirectory() || (await lstat(root)).isSymbolicLink()) {
        args.push('--ro-bind', root, root);
        mounted.push(root);
      }
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  requireValue(mounted.some((root) => binary.path.startsWith(`${root}/`)), 'INVALID_CONVERTER',
    'Converter outside system runtime roots requires a narrow installationRoot.');
  for (const filename of ['/etc/ld.so.cache', '/etc/fonts', '/etc/mime.types']) {
    try { await lstat(filename); args.push('--ro-bind', filename, filename); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  args.push('--proc', '/proc', '--dev', '/dev', '--dir', '/input', '--dir', '/output',
    '--ro-bind', source, `/input/source.${format}`, '--bind', path.dirname(output), '/output',
    '--size', String(config.maxOutputBytes), '--tmpfs', '/tmp', '--chdir', '/tmp');
  for (const [key, value] of Object.entries(cleanEnvironment('/tmp'))) args.push('--setenv', key, value);
  args.push('--', binary.path, `/input/source.${format}`, '/output/converted.epub');
  return {command: sandbox.path, args, env: cleanEnvironment(workspace)};
}

async function runConverter(command, args, {env, workspace, output, config, signal}) {
  if (signal?.aborted) throw new ConversionError('CONVERSION_ABORTED', 'Conversion was cancelled.');
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {cwd: workspace, env, shell: false, detached: true,
      stdio: ['ignore', 'pipe', 'pipe']});
    let failure;
    let diagnosticBytes = 0;
    const diagnostics = [];
    let checking = false;
    let finished = false;
    const kill = () => {
      if (child.pid) {
        try { process.kill(-child.pid, 'SIGKILL'); }
        catch (error) { if (error.code !== 'ESRCH') child.kill('SIGKILL'); }
      }
    };
    const stop = (code, message) => { if (!finished) { failure ??= new ConversionError(code, message); kill(); } };
    const aborted = () => stop('CONVERSION_ABORTED', 'Conversion was cancelled.');
    const timeout = setTimeout(() => stop('CONVERSION_TIMEOUT', 'Conversion exceeded its time limit.'), config.timeoutMs);
    const poll = setInterval(async () => {
      if (checking || finished) return;
      checking = true;
      try {
        const info = await lstat(output);
        if (!info.isFile() || info.isSymbolicLink()) stop('INVALID_CONVERSION', 'Converter output is not a regular file.');
        else if (info.size > config.maxOutputBytes) stop('LIMIT_EXCEEDED', 'Converted EPUB exceeds its byte limit.');
      } catch (error) { if (error.code !== 'ENOENT') stop('INVALID_CONVERSION', 'Cannot inspect converter output.'); }
      finally { checking = false; }
    }, 100);
    signal?.addEventListener('abort', aborted, {once: true});
    if (signal?.aborted) aborted();
    const collect = (part) => {
      diagnosticBytes += part.length;
      if (diagnosticBytes > config.maxLogBytes) stop('LIMIT_EXCEEDED', 'Converter diagnostics exceed their byte limit.');
      else diagnostics.push(part);
    };
    child.stdout.on('data', collect);
    child.stderr.on('data', collect);
    child.on('error', () => { failure ??= new ConversionError('CONVERTER_UNAVAILABLE', 'Converter process could not start.'); });
    child.on('close', (code) => {
      finished = true;
      clearTimeout(timeout);
      clearInterval(poll);
      signal?.removeEventListener('abort', aborted);
      // Also remove inherited descendants after an apparently successful exit.
      kill();
      if (failure) { reject(failure); return; }
      const log = Buffer.concat(diagnostics).toString('utf8');
      if (code !== 0) {
        reject(new ConversionError(/\bDRM\b|DRMError|encrypted|encryption|password.protect/i.test(log)
          ? 'ENCRYPTED_BOOK' : 'CONVERSION_FAILED',
        /\bDRM\b|DRMError|encrypted|encryption|password.protect/i.test(log)
          ? 'Encrypted or DRM-protected books cannot be converted; no removal was attempted.'
          : `Converter failed with exit code ${code ?? 'signal'}.`));
        return;
      }
      resolve({diagnosticBytes});
    });
  });
}

export async function convertBook(filename, {format, converter, derivedDir, signal} = {}) {
  const config = settings(converter);
  requireValue(process.platform === 'linux', 'CONVERTER_UNAVAILABLE', 'Native conversion is supported only in the reviewed Linux runtime.');
  requireValue(['mobi', 'prc'].includes(format), 'INVALID_SOURCE', 'Only MOBI and PRC sources use this converter.');
  requireValue(absolute(filename), 'INVALID_SOURCE', 'Converter source must have an absolute managed path.');
  const original = await fileIdentity(filename, config.maxSourceBytes);
  const binary = await executable(config.executable);
  const sandbox = config.isolation === 'bubblewrap' ? await executable(config.sandboxExecutable) : null;
  const directory = await managedDirectory(derivedDir);
  const workspace = await mkdtemp(path.join(directory, '.convert-'));
  const outputDirectory = path.join(workspace, 'output');
  const output = path.join(outputDirectory, 'converted.epub');
  try {
    await mkdir(outputDirectory, {mode: 0o700});
    const invocation = sandbox
      ? await sandboxCommand(config, binary, sandbox, filename, output, workspace, format)
      : {command: binary.path, args: [filename, output], env: cleanEnvironment(workspace)};
    const result = await runConverter(invocation.command, invocation.args,
      {env: invocation.env, workspace, output, config, signal});
    let derived;
    try {
      derived = await fileIdentity(output, config.maxOutputBytes, 'INVALID_CONVERSION');
      const handle = await open(output, constants.O_RDONLY | constants.O_NOFOLLOW);
      try {
        const header = Buffer.alloc(4);
        await handle.read(header, 0, header.length, 0);
        requireValue(header.equals(Buffer.from([0x50, 0x4b, 3, 4])), 'INVALID_CONVERSION', 'Converter did not create an EPUB ZIP file.');
      } finally { await handle.close(); }
    } catch (error) {
      if (error instanceof ConversionError) throw error;
      throw new ConversionError('INVALID_CONVERSION', 'Converter did not produce a readable EPUB.');
    }
    const after = await fileIdentity(filename, config.maxSourceBytes);
    requireValue(after.sha256 === original.sha256 && after.bytes === original.bytes,
      'SOURCE_CHANGED', 'Original source changed during conversion.');
    const destination = path.join(directory, `${original.sha256}-${derived.sha256}.epub`);
    await chmod(output, 0o400);
    try { await link(output, destination); }
    catch (error) {
      if (error.code !== 'EEXIST') throw error;
      let existing;
      try { existing = await fileIdentity(destination, config.maxOutputBytes, 'OUTPUT_CONFLICT'); }
      catch { throw new ConversionError('OUTPUT_CONFLICT', 'Retained derived artifact conflicts with an existing path.'); }
      requireValue(existing.sha256 === derived.sha256 && existing.bytes === derived.bytes,
        'OUTPUT_CONFLICT', 'Retained derived artifact conflicts with existing bytes.');
    }
    return {path: destination, provenance: {schemaVersion: 1,
      original: {format, ...original}, derived: {format: 'epub', ...derived, path: destination},
      converter: {name: 'calibre ebook-convert', executableSha256: binary.sha256,
        sandboxSha256: sandbox?.sha256 ?? null, isolation: config.isolation,
        isolationEnforcement: sandbox ? 'bubblewrap namespaces and die-with-parent'
          : 'external reviewed container: network:none and owned-cgroup cleanup'},
      diagnosticBytes: result.diagnosticBytes,
      limitations: ['Text, metadata and navigation were obtained through MOBI/PRC to EPUB conversion. Layout and section boundaries may change.',
        'Citations address retained derived EPUB sections, not original MOBI byte positions or printed pages. The unchanged original remains the reference.',
        'DRM removal, OCR and extraction-completeness verification were not performed.']}};
  } finally { await rm(workspace, {recursive: true, force: true}); }
}
