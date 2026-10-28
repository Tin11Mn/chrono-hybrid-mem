"""Run regression tests with inherited model credentials removed and WAN blocked.

Usage: python scripts/test_offline.py -q -m "not local_research"
Tests may configure SDK doubles; real non-loopback sockets remain unavailable.
"""

import os
from pathlib import Path
import socket
import sys


def main():
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "MEMORY_")):
            del os.environ[name]
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise OSError("Outbound network disabled by offline test runner")
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise OSError("Outbound network disabled by offline test runner")
        return original_connect_ex(sock, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    try:
        import pytest
        return pytest.main(sys.argv[1:] or ["-q", "-m", "not local_research"])
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


if __name__ == "__main__":
    raise SystemExit(main())
