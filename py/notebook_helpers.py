import os
import pty
import subprocess
import sys


def run_subprocess(command: list[str]) -> None:
    """Run a subprocess command with real-time output"""

    # Use pty for better terminal emulation
    master, slave = pty.openpty()
    process = subprocess.Popen(command, stdout=slave, stderr=slave, stdin=slave, close_fds=True)
    os.close(slave)
    try:
        while True:
            try:
                data = os.read(master, 1024)
                if not data:
                    break
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            except OSError:
                break
    finally:
        os.close(master)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Command failed with return code: {process.returncode}")
