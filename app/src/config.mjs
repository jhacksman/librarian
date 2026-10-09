import { resolve, delimiter, isAbsolute } from 'node:path';
import { homedir } from 'node:os';

export function readConfig(env = process.env) {
  const maintenance = env.LIBRARIAN_MAINTENANCE === undefined ? '' : env.LIBRARIAN_MAINTENANCE;
  if (!['', 'enabled', 'disabled'].includes(maintenance)) {
    throw new Error('LIBRARIAN_MAINTENANCE must be enabled or disabled');
  }
  const dataDir = resolve(env.LIBRARIAN_DATA_DIR || `${homedir()}/.local/share/librarian`);
  const host = env.LIBRARIAN_HOST || '127.0.0.1';
  const port = Number(env.LIBRARIAN_PORT || 3474);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Invalid LIBRARIAN_PORT');
  const origin = env.LIBRARIAN_ORIGIN || `http://${host.includes(':') ? `[${host}]` : host}:${port}`;
  const parsed = new URL(origin);
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password || parsed.pathname !== '/') {
    throw new Error('LIBRARIAN_ORIGIN must be one HTTP(S) origin');
  }
  let converter;
  if (env.LIBRARIAN_EBOOK_CONVERT) {
    const isolation = env.LIBRARIAN_CONVERTER_ISOLATION || 'bubblewrap';
    if (!['bubblewrap', 'container'].includes(isolation)) throw new Error('Unknown converter isolation mode');
    for (const key of ['LIBRARIAN_EBOOK_CONVERT', 'LIBRARIAN_BWRAP', 'LIBRARIAN_CALIBRE_ROOT']) {
      if (env[key] && !isAbsolute(env[key])) throw new Error(`${key} must be an absolute installed path`);
    }
    if (isolation === 'bubblewrap' && !env.LIBRARIAN_BWRAP) throw new Error('Set LIBRARIAN_BWRAP for converter isolation');
    converter = { executable: env.LIBRARIAN_EBOOK_CONVERT, isolation,
      ...(env.LIBRARIAN_BWRAP ? { sandboxExecutable: env.LIBRARIAN_BWRAP } : {}),
      ...(env.LIBRARIAN_CALIBRE_ROOT ? { installationRoot: env.LIBRARIAN_CALIBRE_ROOT } : {}) };
  }
  return {
    dataDir, dbPath: resolve(dataDir, 'library.sqlite'), host, port, origin: parsed.origin,
    maintenanceEnabled: maintenance !== 'disabled',
    importRoots: (env.LIBRARIAN_IMPORT_ROOTS || `${dataDir}-inbox`).split(delimiter).filter(Boolean).map(p => resolve(p)),
    importOptions: converter ? { converter } : {},
    retrieval: {
      endpoint: env.LIBRARIAN_MODEL_ENDPOINT || 'http://127.0.0.1:11434',
      provider: env.LIBRARIAN_MODEL_PROVIDER || 'ollama',
      embeddingModel: env.LIBRARIAN_EMBEDDING_MODEL || null,
      chatModel: env.LIBRARIAN_CHAT_MODEL || null,
      embeddingRevision: env.LIBRARIAN_EMBEDDING_REVISION || '',
      chatRevision: env.LIBRARIAN_CHAT_REVISION || '',
      queryPrefix: env.LIBRARIAN_EMBEDDING_QUERY_PREFIX || '',
      documentPrefix: env.LIBRARIAN_EMBEDDING_DOCUMENT_PREFIX || '',
      contextTokens: Number(env.LIBRARIAN_CONTEXT_TOKENS || 8192),
      timeoutMs: Number(env.LIBRARIAN_MODEL_TIMEOUT_MS || 45000),
      maxContextChars: Number(env.LIBRARIAN_MAX_CONTEXT_CHARS || 24000),
      allowedHosts: (env.LIBRARIAN_ALLOWED_MODEL_HOSTS || '').split(',').filter(Boolean),
      embedding: {
        ...(env.LIBRARIAN_EMBEDDING_ENDPOINT ? { endpoint: env.LIBRARIAN_EMBEDDING_ENDPOINT } : {}),
        ...(env.LIBRARIAN_EMBEDDING_PROVIDER ? { provider: env.LIBRARIAN_EMBEDDING_PROVIDER } : {}),
      },
      chat: {
        ...(env.LIBRARIAN_CHAT_ENDPOINT ? { endpoint: env.LIBRARIAN_CHAT_ENDPOINT } : {}),
        ...(env.LIBRARIAN_CHAT_PROVIDER ? { provider: env.LIBRARIAN_CHAT_PROVIDER } : {}),
      },
    },
  };
}
