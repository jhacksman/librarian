import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readConfig } from '../src/config.mjs';

test('maintenance stays enabled when the setting is absent', () => {
  assert.equal(readConfig({}).maintenanceEnabled, true);
  assert.equal(readConfig({ LIBRARIAN_MAINTENANCE: undefined }).maintenanceEnabled, true);
});

test('an empty maintenance setting preserves the enabled default', () => {
  assert.equal(readConfig({ LIBRARIAN_MAINTENANCE: '' }).maintenanceEnabled, true);
});

test('maintenance can be explicitly enabled', () => {
  assert.equal(readConfig({ LIBRARIAN_MAINTENANCE: 'enabled' }).maintenanceEnabled, true);
});

test('maintenance can be explicitly disabled', () => {
  assert.equal(readConfig({ LIBRARIAN_MAINTENANCE: 'disabled' }).maintenanceEnabled, false);
});

test('maintenance rejects unknown values without case or whitespace coercion', () => {
  for (const value of ['true', 'false', '0', '1', 'on', 'off', 'ENABLED', 'Disabled',
    ' enabled', 'disabled ', ' ', 'disabled\n']) {
    assert.throws(() => readConfig({ LIBRARIAN_MAINTENANCE: value }),
      /LIBRARIAN_MAINTENANCE must be enabled or disabled/, JSON.stringify(value));
  }
});

test('local Ask settings preserve explicit adapters and never choose a model implicitly', () => {
  const config = readConfig({ LIBRARIAN_CHAT_PROVIDER: 'openai', LIBRARIAN_CHAT_ENDPOINT: 'http://127.0.0.1:18094/v1',
    LIBRARIAN_CHAT_MODEL: 'installed-chat', LIBRARIAN_CHAT_REVISION: 'approved-sha',
    LIBRARIAN_CONTEXT_TOKENS: '8192', LIBRARIAN_MODEL_TIMEOUT_MS: '60000', LIBRARIAN_MAX_CONTEXT_CHARS: '24000',
    LIBRARIAN_ALLOWED_MODEL_HOSTS: '192.168.1.10,10.0.0.4' }).retrieval;
  assert.equal(config.chatModel, 'installed-chat');
  assert.equal(config.chatRevision, 'approved-sha');
  assert.equal(config.chat.provider, 'openai');
  assert.equal(config.chat.endpoint, 'http://127.0.0.1:18094/v1');
  assert.equal(config.timeoutMs, 60000);
  assert.deepEqual(config.allowedHosts, ['192.168.1.10', '10.0.0.4']);
  assert.equal(readConfig({}).retrieval.chatModel, null);
});
