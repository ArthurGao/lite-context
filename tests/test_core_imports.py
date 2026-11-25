"""The core uses the standard library only (design §2, AGENTS.md rule 6)."""

import subprocess
import sys

# Design §2: the core modules. Extras (lite_model adapter, tiktoken, pg, server) are not here.
CORE = [
    "lite_context",
    "lite_context.types",
    "lite_context.count",
    "lite_context.report",
    "lite_context.errors",
    "lite_context.adapters",
    "lite_context.adapters.anthropic",
    "lite_context.adapters.openai",
]

PROBE = """
import importlib, sys
before = set(sys.modules)
for name in {modules!r}:
    importlib.import_module(name)
loaded = set(sys.modules) - before
third_party = sorted(
    m for m in loaded
    if m.split(".")[0] not in sys.stdlib_module_names and m.split(".")[0] != "lite_context"
)
print(",".join(third_party))
"""


def test_importing_the_core_loads_no_third_party_module() -> None:
    out = subprocess.run(
        [sys.executable, "-I", "-c", PROBE.format(modules=CORE)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == ""
