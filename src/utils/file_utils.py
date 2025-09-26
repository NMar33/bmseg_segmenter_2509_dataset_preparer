#src/utils/file_utils.py
"""
Utility functions for file system operations and running external processes.
"""
import subprocess
import sys
from pathlib import Path

def create_dir_if_not_exists(path: Path) -> None:
    """Creates a directory including parent directories if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def run_subprocess(command: list[str], cwd: Path | str | None = None) -> None:
    """
    Runs an external command safely, streaming its output and checking for errors.

    Args:
        command: A list of strings representing the command and its arguments.
        cwd: The working directory from which to run the command.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.
    """
    # DEV: Используем subprocess.Popen вместо os.system или subprocess.run(shell=True)
    # Это намного безопаснее, так как предотвращает shell injection.
    # Мы стримим stdout/stderr в реальном времени, что полезно для долгих
    # процессов вроде gdown.
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        cwd=cwd
    )

    # Read and print output line by line
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)

    # Wait for the process to complete and get the exit code
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command '{' '.join(command)}' failed with exit code {return_code}")