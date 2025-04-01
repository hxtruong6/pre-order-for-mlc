import os
import sys
import contextlib


@contextlib.contextmanager
def suppress_stderr():
    """
    Suppress C-level stderr (e.g., from GLPK or other C libraries)
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = sys.stderr.fileno()

    # Duplicate original stderr so we can restore later
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


import os
import sys
import contextlib


@contextlib.contextmanager
def suppress_output():
    """
    Suppress both C-level stdout and stderr (e.g., from GLPK or other C libs)
    """
    with open(os.devnull, "w") as devnull:
        # Save original file descriptors
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)

        # Redirect stdout and stderr to /dev/null
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        try:
            yield
        finally:
            # Restore original stdout and stderr
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
