import ctypes
import os
import signal
import sys

PR_SET_PDEATHSIG = 1


def main():
    parent, *command = sys.argv[1:]
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() == int(parent):
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
