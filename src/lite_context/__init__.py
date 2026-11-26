"""lite-context: what each LLM call should see (docs/lite-context-design.md)."""

from .compaction import (
    CompactionPlan,
    Summary,
    SummaryPolicy,
    accept_summary,
    acompact,
    plan_compaction,
)
from .count import Budget, CharEstimate, Counter, count_message, count_messages
from .errors import (
    BrokenPairing,
    DoesNotFit,
    EmptySummary,
    LiteContextError,
    ProtectedChange,
    SeqConflict,
    StaleState,
    StoreError,
    TruncatedSummary,
    UnknownTranscript,
    UnreportedChange,
    UnsupportedInput,
)
from .fit import Fitted, fit
from .report import Action, Report
from .state import Compaction, ContextState, Cut
from .store import InMemoryStore, Store
from .transcript import Entry, Transcript
from .types import (
    ContentForm,
    Message,
    Opaque,
    OpaqueKind,
    Part,
    Role,
    Text,
    ToolCall,
    ToolResult,
    with_keys,
)

__all__ = [
    "Action",
    "BrokenPairing",
    "Budget",
    "CharEstimate",
    "Compaction",
    "CompactionPlan",
    "ContentForm",
    "ContextState",
    "Counter",
    "Cut",
    "DoesNotFit",
    "EmptySummary",
    "Entry",
    "Fitted",
    "InMemoryStore",
    "LiteContextError",
    "Message",
    "Opaque",
    "OpaqueKind",
    "Part",
    "ProtectedChange",
    "Report",
    "Role",
    "SeqConflict",
    "StaleState",
    "Store",
    "StoreError",
    "Summary",
    "SummaryPolicy",
    "Text",
    "ToolCall",
    "ToolResult",
    "Transcript",
    "TruncatedSummary",
    "UnknownTranscript",
    "UnreportedChange",
    "UnsupportedInput",
    "accept_summary",
    "acompact",
    "count_message",
    "count_messages",
    "fit",
    "plan_compaction",
    "with_keys",
]
