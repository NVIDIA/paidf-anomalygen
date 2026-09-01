# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for scripts/skill_utility/train_control.sh.

Two things here have broken before and are worth pinning: the detach chain (the PID file must hold the
*real* torchrun PID, in its own session, or a stop kills the wrapper and leaves training orphaned) and
``wait``'s three-way verdict (treating "still running" as "finished" is what makes a caller skip
Steps 4-7). Training itself is stood in for by a script that prints the markers the poller greps for,
so no GPU or model is involved.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "skill_utility" / "train_control.sh"

FINISHED, FAILED, RUNNING, USAGE = 0, 1, 2, 64

# The marker the trainer prints at the end of a clean run; wait() gates Step 4 on it.
_DONE = "Done with training"

# Stand-in trainer bodies. LINGERS outlives the test so stop/liveness can be observed; ECHO_ARGV
# reports the argv train_control composed, then finishes cleanly.
_LINGERS = "import time; time.sleep(30)\n"
_ECHO_ARGV = f"import sys; print(sys.argv[1:], flush=True); print({_DONE!r}, flush=True)\n"

needs_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="torchrun not on PATH")


@pytest.fixture
def sandbox(tmp_path):
    """A working dir laid out like the repo root: a recipe, and a --scripts dir to hold train.py.

    Teardown stops any run still alive, so a test that only cares about the launch does not have to
    remember to — a leaked trainer would otherwise sleep on past the run.
    """
    (tmp_path / "ag_config").mkdir()
    (tmp_path / "ag_config" / "r.yaml").write_text("task_type: texture_ft\n")
    (tmp_path / "fake").mkdir()
    yield tmp_path
    for pidfile in (tmp_path / "results").glob("train_*.pid"):
        _run(tmp_path, "stop", "--name", pidfile.stem.removeprefix("train_"))


def _train_py(sandbox, body):
    (sandbox / "fake" / "train.py").write_text(body)


def _run(sandbox, *args, timeout=120):
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=sandbox,
        timeout=timeout,
        env={**os.environ, "PATH": f"{pathlib.Path(shutil.which('python') or 'python').parent}:{os.environ['PATH']}"},
    )


def _start(sandbox, *extra, name="demo", **kw):
    args = ("start", "--name", name, "--recipe", "ag_config/r.yaml", "--scripts", "fake", *extra)
    return _run(sandbox, *args, **kw)


def _wait(sandbox, name="demo", timeout_s="60", interval="1"):
    return _run(sandbox, "wait", "--name", name, "--timeout", timeout_s, "--interval", interval)


def _pid(sandbox, name="demo"):
    return int((sandbox / "results" / f"train_{name}.pid").read_text().strip())


def _alive(pid):
    return subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0


def _session_of(pid):
    out = subprocess.run(["ps", "-o", "sid=", "-p", str(pid)], capture_output=True, text=True)
    return out.stdout.strip()


@needs_torchrun
def test_pid_file_holds_the_real_trainer_not_the_setsid_wrapper(sandbox):
    """'$!' would name the setsid wrapper, which exits at once — 'stop' would then kill nothing and
    leave training running with no way to find it."""
    _train_py(sandbox, _LINGERS)
    assert _start(sandbox).returncode == 0
    cmdline = pathlib.Path(f"/proc/{_pid(sandbox)}/cmdline").read_bytes().decode().replace("\0", " ")
    assert "train.py" in cmdline and "torchrun" in cmdline


@needs_torchrun
def test_training_runs_in_its_own_session(sandbox):
    """The point of setsid: a harness cleaning up with 'pkill -s <sid>' must miss the run."""
    _train_py(sandbox, _LINGERS)
    _start(sandbox)
    pid = _pid(sandbox)
    assert _session_of(pid) == str(pid), "trainer should lead its own session"
    assert _session_of(pid) != _session_of(os.getpid())


@needs_torchrun
@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        # The bare '--' separates launcher args from the experiment override; dropping it silently
        # trains the default experiment.
        ((), ["'--', 'experiment=anomalygen_texture_ft'", "'--recipe=ag_config/r.yaml'"]),
        # The documented smoke test ('-- trainer.max_iter=20 ...') depends on these arriving after
        # experiment=, where hydra reads them.
        (
            ("--", "trainer.max_iter=20", "checkpoint.save_iter=10"),
            ["'experiment=anomalygen_texture_ft', 'trainer.max_iter=20', 'checkpoint.save_iter=10'"],
        ),
    ],
    ids=["bare-separator", "hydra-overrides"],
)
def test_the_composed_argv_reaches_the_trainer(sandbox, extra, expected):
    _train_py(sandbox, _ECHO_ARGV)
    _start(sandbox, *extra)
    assert _wait(sandbox).returncode == FINISHED
    log = (sandbox / "results" / "train_demo.log").read_text()
    for fragment in expected:
        assert fragment in log
    # An unquoted empty array would add a stray '' that hydra rejects — true with or without overrides.
    assert "''" not in log, f"empty argument leaked into argv: {log}"


@needs_torchrun
def test_wait_reports_finished_only_on_the_marker(sandbox):
    _train_py(sandbox, f"import time; time.sleep(1); print({_DONE!r}, flush=True)\n")
    _start(sandbox)
    out = _wait(sandbox)
    assert out.returncode == FINISHED
    assert "FINISHED" in out.stdout


@needs_torchrun
def test_wait_reports_failed_on_a_traceback(sandbox):
    _train_py(sandbox, "raise RuntimeError('boom')\n")
    _start(sandbox)
    out = _wait(sandbox)
    assert out.returncode == FAILED
    assert "FAILED" in out.stderr


@needs_torchrun
def test_wait_reports_failed_when_the_run_dies_without_the_marker(sandbox):
    """A killed run exits 0-ish from the poller's view but never printed the marker — the case that
    must not be mistaken for success."""
    _train_py(sandbox, _LINGERS)
    _start(sandbox)
    _run(sandbox, "stop", "--name", "demo")  # kill it mid-run: the marker is never printed
    out = _wait(sandbox, timeout_s="20")
    assert out.returncode == FAILED
    assert _DONE not in (sandbox / "results" / "train_demo.log").read_text()


@needs_torchrun
def test_wait_reports_running_when_the_slice_elapses(sandbox):
    """The normal case for most of a real run: a healthy trainer and an expired budget. It must be
    distinguishable from both finishing and failing, or the caller stops early."""
    _train_py(sandbox, _LINGERS)
    _start(sandbox)
    out = _wait(sandbox, timeout_s="2")
    assert out.returncode == RUNNING
    assert "RUNNING" in out.stdout
    assert out.returncode not in (FINISHED, FAILED)


@needs_torchrun
def test_wait_is_repeatable_until_it_finishes(sandbox):
    """The skill calls wait in a loop; an early slice returning 2 must not disturb a later 0."""
    _train_py(sandbox, f"import time; time.sleep(4); print({_DONE!r}, flush=True)\n")
    _start(sandbox)
    assert _wait(sandbox, timeout_s="1").returncode == RUNNING
    assert _wait(sandbox, timeout_s="60").returncode == FINISHED


@needs_torchrun
def test_refuses_to_start_a_second_run_over_a_live_one(sandbox):
    """Both would write the same log and PID file, and the second start would orphan the first."""
    _train_py(sandbox, _LINGERS)
    assert _start(sandbox).returncode == 0
    first = _pid(sandbox)
    out = _start(sandbox)
    assert out.returncode == USAGE
    assert "already alive" in out.stderr
    assert _pid(sandbox) == first, "the refused start must not clobber the PID file"


@needs_torchrun
def test_finished_means_the_trainer_has_actually_exited(sandbox):
    """The marker is printed *before* torchrun tears down. Returning FINISHED while the process still
    holds the GPUs sends Step 4 to generate on memory that has not been released yet."""
    _train_py(sandbox, f"print({_DONE!r}, flush=True)\nimport time; time.sleep(3)\n")
    _start(sandbox)
    pid = _pid(sandbox)
    assert _wait(sandbox).returncode == FINISHED
    assert not _alive(pid), "wait returned FINISHED while the trainer was still running"


@needs_torchrun
@pytest.mark.parametrize(
    ("first_body", "first_rc"),
    [(f"print({_DONE!r}, flush=True)\n", FINISHED), ("raise RuntimeError('boom')\n", FAILED)],
    ids=["previous-run-finished", "previous-run-crashed"],
)
def test_wait_judges_the_current_run_not_the_previous_one(sandbox, first_body, first_rc):
    """The log is keyed on --name alone and wait greps it whole, so a second run under the same name
    would be judged by the first one's markers. False-FINISHED is the damaging direction: gate train
    clears on the old artifacts (best_checkpoint.txt is only rewritten at on_train_end) and Step 4
    generates the whole batch from the previous checkpoint — no error, wrong model."""
    _train_py(sandbox, first_body)
    _start(sandbox)
    assert _wait(sandbox).returncode == first_rc, "precondition: the first run reaches its verdict"

    _train_py(sandbox, "import time; time.sleep(30)\n")
    assert _start(sandbox).returncode == 0
    try:
        out = _wait(sandbox, timeout_s="2")
        assert out.returncode == RUNNING, f"verdict leaked from the previous run: rc={out.returncode}"
    finally:
        _run(sandbox, "stop", "--name", "demo")


@needs_torchrun
def test_start_preserves_the_previous_run_log_instead_of_discarding_it(sandbox):
    """Rotation must not lose the evidence — a failed run's log is what you read to fix it."""
    _train_py(sandbox, "raise RuntimeError('boom')\n")
    _start(sandbox)
    assert _wait(sandbox).returncode == FAILED
    _train_py(sandbox, f"print({_DONE!r}, flush=True)\n")
    assert _start(sandbox).returncode == 0
    rotated = list((sandbox / "results").glob("train_demo.log.*"))
    assert len(rotated) == 1, f"expected one rotated log, got {rotated}"
    assert "boom" in rotated[0].read_text(), "the previous run's traceback must survive rotation"
    assert "boom" not in (sandbox / "results" / "train_demo.log").read_text()


@needs_torchrun
def test_a_finished_run_can_be_restarted(sandbox):
    """The stale-PID guard must key off liveness, not the file existing."""
    _train_py(sandbox, f"print({_DONE!r}, flush=True)\n")
    _start(sandbox)
    assert _wait(sandbox).returncode == FINISHED
    assert _start(sandbox).returncode == 0, "a dead PID file should not block a rerun"


@pytest.mark.parametrize("name", ["a;b", "a b", "../x", "-flag", "a$(id)"])
def test_rejects_names_that_would_reach_a_shell(sandbox, name):
    out = _start(sandbox, name=name)
    assert out.returncode == USAGE, f"accepted {name!r}"
    assert "invalid --name" in out.stderr


def test_rejects_a_missing_recipe_before_launching(sandbox):
    """Catching it here beats a detached run that dies seconds later in a log nobody is reading."""
    _train_py(sandbox, "pass\n")
    out = _run(sandbox, "start", "--name", "demo", "--recipe", "nope.yaml", "--scripts", "fake")
    assert out.returncode == USAGE and "recipe not found" in out.stderr
    assert not (sandbox / "results").exists(), "must not leave a results dir behind"


def test_rejects_a_scripts_dir_with_no_train_py(sandbox):
    out = _run(sandbox, "start", "--name", "demo", "--recipe", "ag_config/r.yaml", "--scripts", "fake")
    assert out.returncode == USAGE and "no train.py" in out.stderr


def test_wait_before_start_is_an_error_not_a_verdict(sandbox):
    """Must not look like FINISHED or RUNNING — there is nothing to wait for."""
    out = _wait(sandbox, name="never")
    assert out.returncode == USAGE
    assert out.returncode not in (FINISHED, RUNNING)


def test_unknown_command_and_option_are_refused(sandbox):
    assert _run(sandbox, "resume", "--name", "demo").returncode == USAGE
    assert _run(sandbox, "wait", "--name", "demo", "--nam", "x").returncode == USAGE


@pytest.mark.parametrize("argv", [[], ["--help"], ["-h"], ["wait", "--help"]])
def test_help_is_reachable_from_every_position(sandbox, argv):
    """'--help' in command position was being read as the command name and failing name validation
    instead of printing usage."""
    out = _run(sandbox, *argv)
    assert out.returncode == USAGE
    assert "launch an AnomalyGen fine-tune" in out.stderr, out.stderr
    assert "SPDX" not in out.stderr and "#!/" not in out.stderr, "help leaked the file preamble"
    assert "set -uo pipefail" not in out.stderr, "help ran past the header into the code"


def test_documented_exit_codes_match_the_implementation():
    """The skill branches on these numbers; the header is what a reader trusts."""
    header = _SCRIPT.read_text().split("set -uo pipefail")[0]
    assert "0  finished" in header
    assert "1  failed" in header
    assert "2  running" in header
