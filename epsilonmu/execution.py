"""Bounded subprocess execution. This is not a security sandbox."""

import os
import subprocess
import sys
import tempfile
import time
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    stdout: str
    stderr: str
    exit_code: int | None
    runtime: float
    timed_out: bool


def run_script(path: str | Path, timeout: float = 10, instrument: bool = False) -> RunResult:
    path = Path(path).resolve()
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='epsilonmu-') as directory:
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        copy = Path(directory) / path.name
        shutil.copy2(path, copy)
        command = [sys.executable, str(copy)]
        if instrument:
            command = [sys.executable, str(Path(__file__).with_name('_probe.py')), str(copy)]
        try:
            completed = subprocess.run(command, cwd=directory,
                                       env=env, capture_output=True, text=True,
                                       timeout=timeout, errors='replace')
            return RunResult(completed.stdout, completed.stderr, completed.returncode,
                             time.monotonic() - start, False)
        except subprocess.TimeoutExpired as exc:
            def decode(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
            return RunResult(decode(exc.stdout), decode(exc.stderr), None,
                             time.monotonic() - start, True)
