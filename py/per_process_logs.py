# redirect_io.py
import atexit
import io
import os
import pathlib
import sys
import threading
import traceback
from typing import Any, Protocol


class _WritableFile(Protocol):
    def write(self, data: bytes, /) -> Any: ...
    def flush(self) -> None: ...


class _TeeForwarder(threading.Thread):
    def __init__(self, rfd: int, log_f: _WritableFile, mirror_fd: int | None) -> None:
        super().__init__(daemon=True)
        self.rfd = rfd
        self.log_f = log_f
        self.mirror_fd = mirror_fd

    def run(self) -> None:
        try:
            while True:
                chunk = os.read(self.rfd, 8192)
                if not chunk:
                    break
                self.log_f.write(chunk)
                self.log_f.flush()
                if self.mirror_fd is not None:
                    os.write(self.mirror_fd, chunk)
        finally:
            try:
                os.close(self.rfd)
            except Exception:
                pass


def _fd_to_textio(fd: int) -> io.TextIOWrapper:
    # Wrap a dup'ed fd as a line-buffered text stream
    return io.TextIOWrapper(
        os.fdopen(fd, "wb", buffering=0), encoding="utf-8", line_buffering=True, errors="replace"
    )


def redirect_stdio(
    out_path: str, err_path: str | None = None, append: bool = True, also_console: bool = False
) -> tuple[io.TextIOWrapper, io.TextIOWrapper]:
    """
    Redirect this process's stdout/stderr to files.
    Returns (console_stdout, console_stderr) as text streams you can still write to.
    """
    mode_txt = "a" if append else "w"
    mode_bin = "ab" if append else "wb"

    out_p = pathlib.Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if err_path is None:
        if out_p.suffix in {".out", ".log"}:
            err_p = out_p.with_suffix(".err")
        else:
            err_p = out_p.parent / (out_p.name + ".err")
    else:
        err_p = pathlib.Path(err_path)
        err_p.parent.mkdir(parents=True, exist_ok=True)

    out_fb = open(out_p, mode_bin, buffering=0)
    err_fb = open(err_p, mode_bin, buffering=0)

    # Save original console fds
    orig_out_fd = os.dup(1)
    orig_err_fd = os.dup(2)
    console_stdout = _fd_to_textio(orig_out_fd)
    console_stderr = _fd_to_textio(orig_err_fd)

    if also_console:
        # Tee both streams
        r1, w1 = os.pipe()
        os.dup2(w1, 1)
        os.close(w1)
        _TeeForwarder(r1, out_fb, orig_out_fd).start()

        r2, w2 = os.pipe()
        os.dup2(w2, 2)
        os.close(w2)
        _TeeForwarder(r2, err_fb, orig_err_fd).start()
    else:
        os.dup2(out_fb.fileno(), 1)
        os.dup2(err_fb.fileno(), 2)

    sys.stdout = os.fdopen(1, mode_txt, buffering=1, encoding="utf-8", errors="replace")
    sys.stderr = os.fdopen(2, mode_txt, buffering=1, encoding="utf-8", errors="replace")

    try:
        import faulthandler

        faulthandler.enable(file=sys.stderr)
    except Exception:
        pass

    def _excepthook(t: type[BaseException], e: BaseException, tb: Any) -> None:
        traceback.print_exception(t, e, tb, file=sys.stderr)

    sys.excepthook = _excepthook

    @atexit.register
    def _cleanup() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        for f in (out_fb, err_fb):
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    print(f"[pid={os.getpid()}] redirecting to {out_p} / {err_p}", flush=True)
    return console_stdout, console_stderr
