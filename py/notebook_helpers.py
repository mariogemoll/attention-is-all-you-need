import fcntl
import os
import pty
import struct
import subprocess
import sys
import termios


def run_subprocess(command: list[str], width: int = 100) -> None:
    """Run a subprocess command with real-time output"""

    # Use pty for better terminal emulation
    master, slave = pty.openpty()

    # Set the terminal size (width x height)
    # TIOCSWINSZ is the ioctl command to set window size
    winsize = struct.pack("HHHH", 24, width, 0, 0)  # height=24, width=width
    fcntl.ioctl(slave, termios.TIOCSWINSZ, winsize)
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
