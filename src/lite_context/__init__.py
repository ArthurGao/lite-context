"""lite-context: what each LLM call should see (docs/lite-context-design.md)."""

from .compaction import CompactionPlan, Summary, SummaryPolicy, acompact, plan_compaction
from .count import Budget, CharEstimate, Counter, count_message, count_messages
from .errors import (
    BrokenPairing,
    DoesNotFit,
    EmptySummary,
    LiteContextError,
    ProtectedChange,
    StaleState,
    TruncatedSummary,
    UnreportedChange,
    UnsupportedInput,
)
from .fit import Fitted, fit
from .report import Action, Report
from .state import Compaction, ContextState, Cut
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
    "Fitted",
    "LiteContextError",
    "Message",
    "Opaque",
    "OpaqueKind",
    "Part",
    "ProtectedChange",
    "Report",
    "Role",
    "StaleState",
    "Summary",
    "SummaryPolicy",
    "Text",
    "ToolCall",
    "ToolResult",
    "TruncatedSummary",
    "UnreportedChange",
    "UnsupportedInput",
    "acompact",
    "count_message",
    "count_messages",
    "fit",
    "plan_compaction",
    "with_keys",
]
