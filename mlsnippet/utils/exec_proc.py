import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from threading import Thread

__all__ = ['timed_wait_proc', 'exec_proc']


if sys.version_info[:2] >= (3, 3):
    def timed_wait_proc(proc, timeout):
        try:
            return proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return None
else:
    def timed_wait_proc(proc, timeout):
        itv = min(timeout * .1, .5)
        tot = 0.
        exit_code = None
        while tot + 1e-7 < timeout and exit_code is None:
            exit_code = proc.poll()
            if exit_code is None:
                time.sleep(itv)
                tot += itv
        return exit_code


@contextmanager
def exec_proc(args, on_stdout=None, on_stderr=None, stderr_to_stdout=False,
              buffer_size=16*1024, ctrl_c_timeout=3, kill_timeout=60, **kwargs):
    """
    Execute an external program within a context.

    Args:
        args: Arguments of the program.
        on_stdout ((bytes) -> None): Callback for capturing stdout.
        on_stderr ((bytes) -> None): Callback for capturing stderr.
        stderr_to_stdout (bool): Whether or not to redirect stderr to
            stdout?  If specified, `on_stderr` will be ignored.
            (default :obj:`False`)
        buffer_size (int): Size of buffers for reading from stdout and stderr.
        ctrl_c_timeout (int): Seconds to wait for the program to
            respond to CTRL+C signal. (default 3)
        kill_timeout (int): Seconds to wait for the program to terminate after
            being killed. (default 60)
        **kwargs: Other named arguments passed to :func:`subprocess.Popen`.

    Yields:
        subprocess.Popen: The process object.
    """
    # check the arguments
    if stderr_to_stdout:
        kwargs['stderr'] = subprocess.STDOUT
        on_stderr = None
    if on_stdout is not None:
        kwargs['stdout'] = subprocess.PIPE
    if on_stderr is not None:
        kwargs['stderr'] = subprocess.PIPE

    # output reader
    def reader_func(fd, action):
        while not giveup_waiting[0]:
            buf = os.read(fd, buffer_size)
            if not buf:
                break
            action(buf)

    def make_reader_thread(fd, action):
        th = Thread(target=reader_func, args=(fd, action))
        th.daemon = True
        th.start()
        return th

    # internal flags
    giveup_waiting = [False]

    # launch the process
    stdout_thread = None  # type: Thread
    stderr_thread = None  # type: Thread
    proc = subprocess.Popen(args, **kwargs)

    try:
        if on_stdout is not None:
            stdout_thread = make_reader_thread(proc.stdout.fileno(), on_stdout)
        if on_stderr is not None:
            stderr_thread = make_reader_thread(proc.stderr.fileno(), on_stderr)

        try:
            yield proc
        except KeyboardInterrupt:  # pragma: no cover
            if proc.poll() is None:
                # Wait for a while to ensure the program has properly dealt
                # with the interruption signal.  This will help to capture
                # the final output of the program.
                # TODO: use signal.signal instead for better treatment
                _ = timed_wait_proc(proc, 1)

    finally:
        if proc.poll() is None:
            # First, try to interrupt the process with Ctrl+C signal
            ctrl_c_signal = (signal.SIGINT if sys.platform != 'win32'
                             else signal.CTRL_C_EVENT)
            os.kill(proc.pid, ctrl_c_signal)
            if timed_wait_proc(proc, ctrl_c_timeout) is None:
                # If the Ctrl+C signal does not work, terminate it.
                proc.kill()
            # Finally, wait for at most 60 seconds
            if timed_wait_proc(proc, kill_timeout) is None:  # pragma: no cover
                giveup_waiting[0] = True

        # Close the pipes such that the reader threads will ensure to exit,
        # if we decide to give up waiting.
        def close_pipes():
            for f in (proc.stdout, proc.stderr, proc.stdin):
                if f is not None:
                    f.close()

        if giveup_waiting[0]:  # pragma: no cover
            close_pipes()

        # Wait for the reader threads to exit
        for th in (stdout_thread, stderr_thread):
            if th is not None:
                th.join()

        # Ensure all the pipes are closed.
        if not giveup_waiting[0]:
            close_pipes()
