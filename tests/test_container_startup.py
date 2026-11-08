"""Execute the Docker startup command with fake bootstrap programs and a real POSIX shell."""
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("existing-index", True, 23, 37, 0, ["uvicorn"]),
    ("fetch-failure", False, 23, 0, 23, ["fetch"]),
    ("build-failure", False, 0, 37, 37, ["fetch", "build"]),
    ("bootstrap-success", False, 0, 0, 0, ["fetch", "build", "uvicorn"]),
]


def shell_command(command):
    """Prefer a POSIX sh on PATH on every OS (Git for Windows provides one).

    WSL is used only when FMKA_TEST_ALLOW_WSL=1, because starting it can boot a virtual machine as a side
    effect of a plain pytest run.
    """
    shell = shutil.which("sh")
    if shell is not None:
        return [shell, "-c", command]
    if os.name == "nt" and os.environ.get("FMKA_TEST_ALLOW_WSL") == "1":
        wsl = shutil.which("wsl")
        if wsl is None:
            pytest.skip("POSIX startup test: no sh on PATH and no WSL launcher")
        # Windows may ship a wsl.exe launcher without a usable distribution.
        # Probe bytes: its failure diagnostics can be UTF-16, not the locale codec.
        try:
            available = subprocess.run([wsl, "--exec", "sh", "-c", "exit 0"], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pytest.skip("POSIX startup test requires a usable WSL shell")
        if available.returncode:
            pytest.skip("POSIX startup test requires a usable WSL shell")
        return [wsl, "--exec", "sh", "-c", command]
    pytest.skip("POSIX startup test requires sh on PATH (on Windows: Git for Windows, or set FMKA_TEST_ALLOW_WSL=1)")


@pytest.fixture(scope="module")
def startup_runs(tmp_path_factory):
    """Use one shell invocation for all scenarios, also keeping mutation runs inexpensive."""
    tmp_path = tmp_path_factory.mktemp("container-startup")
    line = next(line for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
                if line.startswith("CMD "))
    startup = json.loads(line.removeprefix("CMD "))
    assert startup[:2] == ["sh", "-c"] and len(startup) == 3
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    programs = {
        "python": '''#!/bin/sh
case "$*" in
  '-m fmka.cli fetch') printf 'fetch\\n' >> events; exit "$FETCH_EXIT" ;;
  '-m fmka.cli build') printf 'build\\n' >> events; exit "$BUILD_EXIT" ;;
  *) printf 'unexpected python arguments: %s\\n' "$*" >&2; exit 99 ;;
esac
''',
        "uvicorn": '''#!/bin/sh
printf 'uvicorn\\n' >> events
printf '%s\\n' "$$" > server.pid
''',
    }
    for name, script in programs.items():
        path = bin_dir / name
        path.write_text(script, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    commands = []
    for name, existing, fetch_exit, build_exit, _, _ in CASES:
        case = tmp_path / name
        case.mkdir()
        if existing:
            db = case / "data" / "processed" / "fmka.duckdb"
            db.parent.mkdir(parents=True)
            db.write_bytes(b"test index sentinel")
        command = 'printf "%s\\n" "$$" > shell.pid; ' + startup[2]
        commands.append(f'(cd {shlex.quote(name)} && '
                        f'export PATH="$PWD/../bin:$PATH" FETCH_EXIT={fetch_exit} BUILD_EXIT={build_exit}; '
                        'sh -c ' + shlex.quote(command) + ')\n'
                        f'printf "%s\\n" "$?" > {shlex.quote(name)}/exit.status')
    result = subprocess.run(shell_command("\n".join(commands)), cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path


@pytest.mark.parametrize("name,existing,fetch_exit,build_exit,expected_exit,expected_events", CASES,
                         ids=[case[0] for case in CASES])
def test_container_command_fails_closed_and_execs_server(
        startup_runs, name, existing, fetch_exit, build_exit, expected_exit, expected_events):
    tmp_path = startup_runs / name
    assert int((tmp_path / "exit.status").read_text()) == expected_exit
    assert (tmp_path / "events").read_text(encoding="utf-8").splitlines() == expected_events
    if "uvicorn" in expected_events:
        assert (tmp_path / "server.pid").read_bytes() == (tmp_path / "shell.pid").read_bytes()
    else:
        assert not (tmp_path / "server.pid").exists()
