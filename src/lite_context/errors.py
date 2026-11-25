"""Errors. Every refusal is an exception; nothing is dropped or guessed silently."""

from __future__ import annotations

from .report import Report


class LiteContextError(Exception):
    """Base class of every lite-context error."""


class DoesNotFit(LiteContextError):
    """The content does not fit the budget after every listed step ran (design §5.1)."""

    def __init__(self, message: str, report: Report) -> None:
        super().__init__(message)
        self.report = report


class UnsupportedInput(LiteContextError):
    """Input outside an adapter's documented range (design §3.2): refused, never converted
    on a best-effort basis."""
