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


class BrokenPairing(LiteContextError):
    """A tool call without its result or a result without its call (design §3.1)."""


class StaleState(LiteContextError):
    """A `ContextState` that no longer matches the history it is given (design §5.4, §6.2)."""


class UnreportedChange(LiteContextError):
    """A step changed, removed or added a message it did not report (design §5.2 rule 5)."""


class ProtectedChange(LiteContextError):
    """A step touched a pinned message, the last human turn or a thinking block (§5.2)."""
