@AGENTS.md

# Claude Code notes

- At the start of a session: read the status line, §14 (the current stage and its acceptance) and §15
  (open questions) of `docs/lite-context-design.md`, and skim the project KB at
  `/Users/arthurgao/Docs/Knowledgebase/projects/lite-context/`.
- Load the `llm-app-architecture` skill before touching `fit()`, the steps, compaction or `Transcript`: design
  §5–§8 put its §9 and §11 into practice, and its warnings describe what a model writes by default (a mutable
  `messages` list compacted by overwriting it).
- Stage close-out is AGENTS.md Workflow step 4: `/simplify` through the Skill tool, then the Codex
  review command there (foreground, it waits). Do not start the next stage before it is done.
- Plan mode before: any change to the types or adapters (design §3), the rules of §5.2 or §6, or work in a
  host repository (C4) or lite-agent (C7).
- Other relevant user-level skills: `testing-and-ci` (property tests, fakes), `rag-design` (§7: long documents,
  map-reduce, chunk selection), `postgres-rls-migrations` (C5), `llm-model-eval` (measuring masking vs summary),
  `dev-machine` (ports, local PostgreSQL), `kb-maintain`.
- Never start long-lived services unasked; never call llm-gateway subscription profiles.
