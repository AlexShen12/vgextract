"""Live console and machine-readable progress for long experiment operations."""
from contextlib import contextmanager
import logging
from pathlib import Path
import threading
import time

from .core import atomic_json, now

LOG = logging.getLogger(__name__)


def progress(root, event, **details):
    record = {'event': event, 'updated_at': now(), **details}
    atomic_json(Path(root)/'progress'/'current.json', record)
    LOG.info('Qwen3.5 %s %s', event, ' '.join(f'{k}={v}' for k, v in details.items()))


@contextmanager
def activity(root, event, *, heartbeat_seconds=60, **details):
    """Report long work while blocked in model startup or an HTTP request."""
    started = time.monotonic()
    stopped = threading.Event()
    progress(root, event+'-start', **details)

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            try:
                progress(root, event+'-waiting', elapsed_seconds=round(time.monotonic()-started, 1), **details)
            except OSError:
                LOG.exception('Could not write Qwen3.5 heartbeat')

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2)
