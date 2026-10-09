import { lockOwnerAlive } from './process-lock.mjs';
import { TERMINAL_STATUSES } from './store.mjs';

export function importOwner(store) {
  const lock = store.db.prepare('SELECT * FROM import_lock WHERE id=1').get();
  return lockOwnerAlive(lock) ? lock : null;
}

export function reconcileInterruptedImports(store) {
  return store.transaction(() => {
    const owner = importOwner(store);
    if (!owner) store.db.exec('DELETE FROM import_lock WHERE id=1');
    const interrupted = [];
    const rows = store.db.prepare("SELECT id,state FROM import_jobs WHERE state IN ('running','queued','paused')").all();
    for (const row of rows) {
      if (owner?.job_id === row.id) continue;
      const job = store.getJob(row.id);
      const resumable = job.summary.inventoryComplete === true;
      const dispositions = store.db.prepare('SELECT status,count(*) AS count FROM import_job_files WHERE job_id=? GROUP BY status').all(row.id);
      const terminalOnly = dispositions.every(item => TERMINAL_STATUSES.includes(item.status));
      // Heal a finalization window, including a previous restart that marked it
      // paused. Deliberately paused work and incomplete inventories stay distinct.
      if (row.state === 'paused' && !(job.summary.interrupted === true && resumable && terminalOnly)) continue;
      if (resumable && terminalOnly) {
        const failed = dispositions.some(item => item.status === 'failed');
        store.refreshJob(row.id, { state: failed ? 'completed_with_errors' : 'completed', current: null,
          completed: true, interrupted: true, error: null,
          recoveryNotice: 'The previous process stopped after all files received dispositions. Import status was finalized from the saved results.' });
        interrupted.push(row.id);
        continue;
      }
      store.refreshJob(row.id, { state: resumable ? 'paused' : 'failed', current: null,
        completed: false, interrupted: true,
        error: resumable ? null : 'Inventory was interrupted. Import the source folder again to create a complete inventory.',
        recoveryNotice: resumable ? 'The previous import stopped. Resume to continue from verified managed copies.' : 'The source inventory did not finish; scan the folder again.' });
      interrupted.push(row.id);
    }
    return { owner, interrupted };
  });
}
