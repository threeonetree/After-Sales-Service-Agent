"""Shared pytest setup: automated regression tests must stay offline."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def fail_connect(*args, **kwargs):
        raise AssertionError("Tests must not access the network; use a fixture or mock.")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)
