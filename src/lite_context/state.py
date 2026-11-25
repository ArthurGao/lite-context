"""`ContextState`: what the caller stores next to its history so the next call cuts in the same
places (design §5.4)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields

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
class Compaction:
    """A summary standing for the messages after the pinned prefix, through `covers_through`
    (design §6). `text` is exactly what the model is shown."""

    covers_through: str
    covered_digest: str  # over the keys from after the pinned prefix through covers_through
    text: str
    provenance: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""  # ISO 8601


@dataclass(frozen=True)
class ContextState:
    cuts: Mapping[str, Cut] = field(default_factory=dict)  # step name -> its cut
    compaction: Compaction | None = None

    def to_json(self) -> str:
        cuts = {n: {"through": c.through, "digest": c.digest} for n, c in self.cuts.items()}
        summary = None if self.compaction is None else asdict(self.compaction)
        data = {"version": VERSION, "cuts": cuts, "compaction": summary}
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ContextState:
        try:
            data = json.loads(text)
            if data.get("version") != VERSION:
                raise ValueError(f"unknown ContextState version {data.get('version')!r}")
            cuts = {
                str(n): Cut(_str(c["through"]), _str(c["digest"])) for n, c in data["cuts"].items()
            }
            raw = data.get("compaction")
            compaction = None if raw is None else _compaction(raw)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a ContextState: {exc}") from exc
        return cls(cuts, compaction)


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value


def _compaction(raw: Mapping[str, object]) -> Compaction:
    if set(raw) != {f.name for f in fields(Compaction)}:
        raise KeyError(f"compaction fields {sorted(raw)}")
    provenance = raw["provenance"]
    if not isinstance(provenance, dict):
        raise TypeError("provenance must be an object")
    return Compaction(
        covers_through=_str(raw["covers_through"]),
        covered_digest=_str(raw["covered_digest"]),
        text=_str(raw["text"]),
        provenance=provenance,
        created_at=_str(raw["created_at"]),
    )
