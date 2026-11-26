"""Request bodies (design §9). Unknown fields are refused, so a typo is a 422, not ignored."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..compaction import SummaryPolicy

Format = Literal["anthropic", "openai", "neutral"]


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BudgetIn(Body):
    limit: int = Field(gt=0)
    reserve_output: int = Field(ge=0)


class TruncateSpec(Body):
    step: Literal["truncate_tool_results"]
    max_tokens: int = Field(gt=0)


class MaskResultsSpec(Body):
    step: Literal["mask_tool_results"]
    keep_last: int = Field(default=3, ge=0)


class MaskOpaqueSpec(Body):
    step: Literal["mask_opaque"]
    keep_last: int = Field(default=3, ge=0)


class DropOldestSpec(Body):
    step: Literal["drop_oldest"]
    keep_last_turns: int = Field(default=4, ge=0)


StepSpec = Annotated[
    TruncateSpec | MaskResultsSpec | MaskOpaqueSpec | DropOldestSpec, Field(discriminator="step")
]


class Fitting(Body):
    """What `fit()` takes besides the messages."""

    budget: BudgetIn
    fixed: list[str] = Field(default_factory=list)
    steps: list[StepSpec] = Field(default_factory=list)
    high: float = 0.9
    low: float = 0.7


class PolicyIn(Body):
    """How much a summary keeps and the header mark (a random one when not given)."""

    keep_last_turns: int = Field(default=4, ge=1)
    mark: str | None = None

    def policy(self) -> SummaryPolicy:
        return SummaryPolicy(keep_last_turns=self.keep_last_turns, mark=self.mark)


class Conversation(Body):
    format: Format
    messages: list[Any]
    keys: list[str] | None = None  # one per neutral message; needed for compaction


class FitIn(Conversation, Fitting):
    state: dict[str, Any] | None = None


class PlanIn(Conversation, PolicyIn):
    state: dict[str, Any] | None = None


class AcceptIn(PolicyIn):
    """The plan as `.../compaction/plan` returned it, with the summary written from it. A client
    sends `{**plan, "reply": ...}`: `system` and `prompt` come back too and are not read."""

    system: str = ""
    prompt: list[Any] = Field(default_factory=list)
    covered: list[str] = Field(min_length=1)
    mark: str
    reply: str
    truncated: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)
    now: datetime | None = None


class SplitIn(Body):
    text: str
    max_tokens: int = Field(gt=0)
    overlap: float = Field(default=0.06, ge=0, lt=1)
    prefer: str = "\n"
    # A regular expression from the client. Python's `re` has no time limit; the shell serves
    # only this machine (design §9), and the length cap keeps patterns small.
    heading: str | None = Field(default=None, max_length=200)


class Unit(Body):
    text: str | None = None
    tokens: int | None = Field(default=None, ge=0)
    key: str | None = None

    @model_validator(mode="after")
    def _sized(self) -> Unit:
        if self.text is None and self.tokens is None:
            raise ValueError("a unit needs text or tokens")
        return self


class PackIn(Body):
    units: list[Unit]
    budget: int = Field(gt=0)
    overhead: int = Field(default=0, ge=0)


class SelectIn(Body):
    units: list[Unit]
    budget: int = Field(gt=0)


class TranscriptIn(Body):
    id: str  # checked by Transcript, the same for every store


class ItemIn(Body):
    message: Any = None
    state: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _one(self) -> ItemIn:
        if (self.message is None) == (self.state is None):
            raise ValueError("an item is a message or a state")
        return self


class EntriesIn(Body):
    format: Format
    items: list[ItemIn] = Field(min_length=1)
    expected_seq: int = Field(ge=0)


class ProjectIn(Fitting):
    format: Format


class TranscriptPlanIn(PolicyIn):
    format: Format


class TranscriptAcceptIn(AcceptIn):
    format: Format | None = None  # echoed back with the plan; not read
    expected_seq: int = Field(ge=0)
