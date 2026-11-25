"""What a call to lite-context did, in numbers (design §5.2 rule 5). C1 adds the actions and the
before / after counts once `fit()` produces them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Report:
    """Token numbers are estimates from the counter, never vendor-billed counts (design §4)."""

    limit: int
    reserve_output: int
    fixed_tokens: int
