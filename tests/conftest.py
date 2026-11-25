"""Shared test setup. Tests never open a socket (AGENTS.md rule 10)."""

import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "sendto", refuse)  # UDP needs no connect()
    monkeypatch.setattr(socket, "getaddrinfo", refuse)  # a DNS lookup is a network call too
