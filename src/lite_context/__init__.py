"""lite-context: what each LLM call should see (docs/lite-context-design.md)."""

from .count import Budget, CharEstimate, Counter, count_message, count_messages
from .errors import (
    BrokenPairing,
    DoesNotFit,
    LiteContextError,
    ProtectedChange,
    StaleState,
    UnreportedChange,
    UnsupportedInput,
)
from .fit import Fitted, fit
from .report import Action, Report
from .state import ContextState, Cut
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
    "ContentForm",
    "ContextState",
    "Counter",
    "Cut",
    "DoesNotFit",
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
    "Text",
    "ToolCall",
    "ToolResult",
    "UnreportedChange",
    "UnsupportedInput",
    "count_message",
    "count_messages",
    "fit",
    "with_keys",
]
