"""lite-context: what each LLM call should see (docs/lite-context-design.md)."""

from .count import Budget, CharEstimate, Counter, count_message, count_messages
from .errors import DoesNotFit, LiteContextError, UnsupportedInput
from .report import Report
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
    "Budget",
    "CharEstimate",
    "ContentForm",
    "Counter",
    "DoesNotFit",
    "LiteContextError",
    "Message",
    "Opaque",
    "OpaqueKind",
    "Part",
    "Report",
    "Role",
    "Text",
    "ToolCall",
    "ToolResult",
    "UnsupportedInput",
    "count_message",
    "count_messages",
    "with_keys",
]
