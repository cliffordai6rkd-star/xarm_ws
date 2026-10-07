"""Attach Latent training without restarting the active Contact worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import fcntl
import psutil

root = Path(__file__).resolve().parent
python = str(Path('.conda-env/bin/python').resolve())
helper = 'scripts/xarm_wm_flow_sweep.py'
contact = psutil.Process(2748131)
launcher = psutil.Process(2748123)
guard = (root/'parallel_supervisor.lock').open('a')
fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert '--family' in contact.cmdline() and 'contact_wm' in contact.cmdline()
assert 'scripts/train_xarm_wm_flow_sweep.sh' in launcher.cmdline()
# Retire only the sequential shell. The existing training process continues.
launcher.terminate()
launcher.wait(timeout=5)
assert contact.is_running() and contact.status() != psutil.STATUS_ZOMBIE
(root/'launcher.pid').write_text(str(os.getpid())+'\n')
env = os.environ.copy()
env['WM_EVALUATION_CONTEXT'] = 'isolated_after_parallel_training'
with (root/'latent_wm/train.log').open('ab', buffering=0) as log:
    latent = subprocess.Popen([python, '-u', helper, 'run', str(root), '--family',
                               'latent_wm', '--defer-evaluation'],
                              stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
metadata = dict(mode='parallel_on_shared_gpu', supervisor_pid=os.getpid(),
                contact_pid=contact.pid, latent_pid=latent.pid, started_unix=time.time(),
                final_evaluations='sequential after both training processes exit')
(root/'parallel_execution.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps(metadata), flush=True)
low_memory_samples = 0
fallback = False
while True:
    active_contact = contact.is_running() and contact.status() != psutil.STATUS_ZOMBIE
    active_latent = latent.poll() is None
    if not active_contact and not active_latent:
        break
    available = psutil.virtual_memory().available
    low_memory_samples = low_memory_samples + 1 if available < 768*1024**2 else 0
    if active_contact and active_latent and low_memory_samples >= 3 and not fallback:
        print('Low available RAM; checkpointing Latent and deferring it until Contact finishes.', flush=True)
        latent.send_signal(signal.SIGTERM)
        fallback = True
    time.sleep(5)
metadata.update(training_workers_finished_unix=time.time(), memory_fallback=fallback,
                latent_exit_code=latent.returncode)
(root/'parallel_execution.json').write_text(json.dumps(metadata, indent=2)+'\n')
contact_record = json.loads((root/'contact_wm/timing.json').read_text())
if not contact_record.get('training_completed_at'):
    raise SystemExit('Contact training did not finish; inspect contact_wm/train.log')
if latent.returncode and not fallback:
    raise SystemExit('Latent training failed; inspect latent_wm/train.log')
if fallback:
    with (root/'latent_wm/train.log').open('ab', buffering=0) as log:
        subprocess.run([python,'-u',helper,'run',str(root),'--family','latent_wm','--defer-evaluation'],
                       stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
# Contact's original worker may have evaluated while Latent was training.
# Refresh it with no other training on the GPU; no optimizer updates are made.
for family in ('contact_wm','latent_wm'):
    with (root/family/'train.log').open('ab', buffering=0) as log:
        command = [python,'-u',helper,'run',str(root),'--family',family]
        if family == 'contact_wm':
            command.append('--refresh-evaluation')
        subprocess.run(command, stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
subprocess.run([python,helper,'report',str(root)],check=True)
print('Both training jobs and all six isolated evaluations completed.', flush=True)
