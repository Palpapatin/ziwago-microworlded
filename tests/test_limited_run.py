from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.limited_run as limited_run


def test_job_assignment_failure_terminates_suspended_child_without_resuming():
    process = mock.Mock(pid=123, _handle=456)
    command = ["python", "-c", "print('must not run')"]
    with mock.patch.object(limited_run.subprocess, "Popen", return_value=process) as popen, \
            mock.patch.object(limited_run, "assign", return_value=(False, 5)), \
            mock.patch.object(limited_run.ctypes, "windll", create=True) as windll:
        with pytest.raises(RuntimeError, match="suspended workload was terminated"):
            limited_run.start_in_job(
                command, 789, stdout=None, stderr=None)

    popen.assert_called_once_with(
        command, stdout=None, stderr=None, creationflags=limited_run.CREATE_SUSPENDED)
    process.kill.assert_called_once_with()
    process.wait.assert_called_once_with()
    windll.ntdll.NtResumeProcess.assert_not_called()
