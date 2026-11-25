import socket

import pytest


def test_a_test_cannot_open_a_connection() -> None:
    with pytest.raises(AssertionError, match="network connection"):
        socket.create_connection(("127.0.0.1", 9))


def test_a_test_cannot_resolve_a_name() -> None:
    with pytest.raises(AssertionError, match="network connection"):
        socket.getaddrinfo("example.com", 443)
