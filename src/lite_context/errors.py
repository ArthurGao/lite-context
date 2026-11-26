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


class EmptySummary(LiteContextError):
    """The summariser replied with nothing usable (design §6.1)."""


class TruncatedSummary(LiteContextError):
    """The summary was cut off by the output limit: half a summary cannot stand for the history."""


class SeqConflict(LiteContextError):
    """A `Transcript` append whose `expected_seq` is not the transcript's tail: someone else
    appended first (design §8). Nothing was written; read again and retry."""

    def __init__(self, transcript: str, expected: int, actual: int) -> None:
        super().__init__(
            f"transcript {transcript!r}: expected seq {expected}, the tail is {actual}"
        )
        self.expected = expected
        self.actual = actual


class UnknownTranscript(LiteContextError):
    """A transcript id the store has never created (design §8)."""


class StoreError(LiteContextError):
    """The store could not do what was asked (a connection, a missing migration); never a
    conflict or an unknown transcript, which have their own errors."""
