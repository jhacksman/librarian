import { readFileSync } from 'node:fs';

// PID alone is not an identity after a restart. Linux start ticks plus boot ID
// distinguish an importer from an unrelated process that reused its PID.
export function processIdentity(pid = process.pid) {
  if (!Number.isSafeInteger(pid) || pid < 1) return null;
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8');
    const fields = stat.slice(stat.lastIndexOf(')') + 2).trim().split(/\s+/);
    const startTicks = fields[19];
    if (!/^\d+$/.test(startTicks || '')) return null;
    return `${readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim()}:${startTicks}`;
  } catch { return null; }
}

export function lockOwnerAlive(lock) {
  if (!lock || !Number.isSafeInteger(lock.pid) || lock.pid < 1) return false;
  try { process.kill(lock.pid, 0); }
  catch (error) { if (error.code === 'ESRCH') return false; }
  const identity = processIdentity(lock.pid);
  // Legacy or unreadable identities remain busy; inability to prove ownership
  // is not permission to interrupt another importer.
  return !(identity && lock.process_identity && identity !== lock.process_identity);
}
