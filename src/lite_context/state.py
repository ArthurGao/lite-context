"""`ContextState`: what the caller stores next to its history so the next call cuts in the same
places (design §5.4)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

VERSION = 1


@dataclass(frozen=True)
class Cut:
    """A cutting step has cut every unit up to the message `through`; `digest` covers the input's
    refs from the start through that message, so a history changed before it is detected."""

    through: str
    digest: str


def digest(refs: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(refs).encode()).hexdigest()


@dataclass(frozen=True)
class ContextState:
    cuts: Mapping[str, Cut] = field(default_factory=dict)  # step name -> its cut

    def to_json(self) -> str:
        cuts = {n: {"through": c.through, "digest": c.digest} for n, c in self.cuts.items()}
        return json.dumps({"version": VERSION, "cuts": cuts}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ContextState:
        try:
            data = json.loads(text)
            if data.get("version") != VERSION:
                raise ValueError(f"unknown ContextState version {data.get('version')!r}")
            cuts = {
                str(n): Cut(_str(c["through"]), _str(c["digest"])) for n, c in data["cuts"].items()
            }
        except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"not a ContextState: {exc}") from exc
        return cls(cuts)


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value
