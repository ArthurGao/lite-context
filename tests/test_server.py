"""The REST shell (design §9, C6): every endpoint gives what the library gives for the same input
(each case runs through both), it serves only this machine, and it refuses oversized bodies."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import asdict
from datetime import datetime
from typing import Any, TypeVar

import httpx
import pytest
from builders import keyless_turn

from lite_context import (
    Budget,
    ContextState,
    DoesNotFit,
    Message,
    SummaryPolicy,
    Text,
    fit,
    plan_compaction,
    with_keys,
)
from lite_context.adapters import anthropic, openai
from lite_context.codec import message_to_dict
from lite_context.count import CharEstimate, count_messages
from lite_context.docs import pack, select, split
from lite_context.server.app import check_host, create_app
from lite_context.steps import DropOldest, MaskOpaque, MaskToolResults, Step, TruncateToolResults
from lite_context.store import InMemoryStore
from lite_context.testing import sample_conversations
from lite_context.transcript import Transcript

T = TypeVar("T")
BASE = "http://127.0.0.1:8750"
NOW = "2026-11-26T00:00:00+00:00"
EST = CharEstimate()


def run(body: Callable[[], Awaitable[T]]) -> T:
    return asyncio.run(body())  # type: ignore[arg-type]


def client(app: Any) -> httpx.AsyncClient:
    """The app in process, as a client on this machine sees it."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)


def call(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async def go() -> httpx.Response:
        async with client(app) as c:
            return await c.request(method, path, **kwargs)

    return run(go)


def jsonish(value: Any) -> Any:
    """What a value looks like after JSON: tuples become lists."""
    return json.loads(json.dumps(value))


def state_json(state: ContextState) -> Any:
    return json.loads(state.to_json())


def report_json(fitted_report: Any) -> Any:
    return jsonish(asdict(fitted_report))


# --- /v1/fit: the same answer as fit() ---

STEP_CASES: list[tuple[list[dict[str, Any]], list[Step]]] = [
    ([], []),
    ([{"step": "mask_tool_results", "keep_last": 1}], [MaskToolResults(keep_last=1)]),
    (
        [
            {"step": "truncate_tool_results", "max_tokens": 20},
            {"step": "mask_opaque", "keep_last": 0},
            {"step": "drop_oldest", "keep_last_turns": 1},
        ],
        [
            TruncateToolResults(max_tokens=20),
            MaskOpaque(keep_last=0),
            DropOldest(keep_last_turns=1),
        ],
    ),
]


IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0K"},
}


def conversations() -> list[list[Message]]:
    """Keyed conversations that every adapter round-trips: tool batches, images, several turns."""
    plain = [m for t in range(1, 4) for m in keyless_turn(t, size=300)]
    with_image = [
        Message(
            "user",
            (
                Text("look"),
                anthropic.to_messages([{"role": "user", "content": [IMAGE]}])[0].parts[0],
            ),
        ),
        Message("assistant", (Text("a cat"),)),
        *plain,
    ]
    long = [m for t in range(1, 7) for m in keyless_turn(t, size=120)]
    return [with_keys(c, [f"k{i}" for i in range(len(c))]) for c in (plain, with_image, long)]


def budget_for(messages: list[Message], share: float) -> Budget:
    return Budget(int(count_messages(messages, EST) * share) + 20, 10)


@pytest.mark.parametrize("conv", range(3))
@pytest.mark.parametrize("steps", range(len(STEP_CASES)))
def test_fit_in_anthropic_form_gives_what_the_library_gives(conv: int, steps: int) -> None:
    neutral = conversations()[conv]
    native, keys = anthropic.from_messages(neutral), [m.key or "" for m in neutral]
    specs, objects = STEP_CASES[steps]
    budget = budget_for(neutral, 0.8)
    request = {
        "format": "anthropic",
        "messages": native,
        "keys": keys,
        "budget": {"limit": budget.limit, "reserve_output": budget.reserve_output},
        "fixed": ["be brief"],
        "steps": specs,
    }
    app = create_app()
    first = call(app, "POST", "/v1/fit", json=request)
    try:
        fitted = fit(
            with_keys(anthropic.to_messages(native), keys),
            budget=budget,
            fixed=["be brief"],
            steps=objects,
        )
    except DoesNotFit as exc:
        assert first.status_code == 422
        assert first.json() == {
            "error": "does_not_fit",
            "detail": str(exc),
            "report": report_json(exc.report),
        }
        return
    assert first.status_code == 200, first.text
    assert first.json() == {
        "messages": jsonish(anthropic.from_messages(fitted.messages)),
        "state": state_json(fitted.state),
        "report": report_json(fitted.report),
    }
    # The state handed back is the one the next call carries, as a library caller would.
    again = call(app, "POST", "/v1/fit", json={**request, "state": first.json()["state"]})
    refit = fit(
        with_keys(anthropic.to_messages(native), keys),
        budget=budget,
        fixed=["be brief"],
        steps=objects,
        state=fitted.state,
    )
    assert again.json()["messages"] == jsonish(anthropic.from_messages(refit.messages))


def test_fit_in_openai_form_keeps_the_system_message_and_counts_it_as_fixed() -> None:
    neutral = conversations()[0]
    system = {"role": "system", "content": "s" * 400}
    native = [system, *openai.from_messages(neutral)]
    keys = [m.key or "" for m in neutral]
    converted = openai.to_messages(native)
    budget = budget_for(neutral, 1.2)
    fitted = fit(
        with_keys(converted.messages, keys),
        budget=budget,
        fixed=list(converted.fixed_texts()),
        steps=[MaskToolResults(keep_last=1)],
    )
    response = call(
        create_app(),
        "POST",
        "/v1/fit",
        json={
            "format": "openai",
            "messages": native,
            "keys": keys,
            "budget": {"limit": budget.limit, "reserve_output": budget.reserve_output},
            "steps": [{"step": "mask_tool_results", "keep_last": 1}],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["messages"] == jsonish(openai.from_messages(fitted.messages, converted.fixed))
    assert body["messages"][0] == system
    assert body["report"] == report_json(fitted.report)
    assert body["report"]["fixed_tokens"] >= EST.text(system["content"])


def test_fit_in_neutral_form_takes_and_gives_codec_records() -> None:
    neutral = sample_conversations()[1]  # the neutral form takes any message, thinking included
    budget = budget_for(neutral, 0.9)
    fitted = fit(
        neutral, budget=budget, steps=[MaskOpaque(keep_last=0), MaskToolResults(keep_last=0)]
    )
    assert fitted.report.actions
    response = call(
        create_app(),
        "POST",
        "/v1/fit",
        json={
            "format": "neutral",
            "messages": [message_to_dict(m) for m in neutral],
            "budget": {"limit": budget.limit, "reserve_output": budget.reserve_output},
            "steps": [
                {"step": "mask_opaque", "keep_last": 0},
                {"step": "mask_tool_results", "keep_last": 0},
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["messages"] == jsonish([message_to_dict(m) for m in fitted.messages])
    assert response.json()["report"] == report_json(fitted.report)


# --- compaction: plan, then accept ---


def history(turns: int) -> list[Message]:
    return with_keys(
        [m for t in range(1, turns + 1) for m in keyless_turn(t, size=50)],
        [f"k{i}" for i in range(turns * 4)],
    )


def test_compaction_plan_and_accept_give_what_the_library_gives() -> None:
    msgs = history(3)
    native, keys = anthropic.from_messages(msgs), [m.key or "" for m in msgs]
    app = create_app()
    planned = call(
        app,
        "POST",
        "/v1/compaction/plan",
        json={
            "format": "anthropic",
            "messages": native,
            "keys": keys,
            "keep_last_turns": 1,
            "mark": "m0",
        },
    )
    plan = plan_compaction(msgs, policy=SummaryPolicy(keep_last_turns=1, mark="m0"))
    assert plan is not None
    assert planned.status_code == 200, planned.text
    assert planned.json() == {
        "system": plan.system,
        "prompt": jsonish(anthropic.from_messages(plan.prompt)),
        "covered": list(plan.covered),
        "mark": "m0",
        "keep_last_turns": 1,
    }
    accepted = call(
        app,
        "POST",
        "/v1/compaction/accept",
        json={
            **planned.json(),
            "reply": "they asked twice",
            "provenance": {"model": "m"},
            "now": NOW,
        },
    )
    expected = plan.accept(
        "they asked twice", provenance={"model": "m"}, now=datetime.fromisoformat(NOW)
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json() == {"state": state_json(expected)}


def test_nothing_to_compact_is_null() -> None:
    msgs = history(1)
    response = call(
        create_app(),
        "POST",
        "/v1/compaction/plan",
        json={
            "format": "anthropic",
            "messages": anthropic.from_messages(msgs),
            "keys": [m.key for m in msgs],
            "keep_last_turns": 1,
        },
    )
    assert (response.status_code, response.json()) == (200, None)


def test_a_truncated_or_empty_summary_is_refused() -> None:
    plan = {"system": "x", "prompt": [], "covered": ["k0"], "mark": "m0", "keep_last_turns": 1}
    for extra in ({"reply": "half", "truncated": True}, {"reply": "  "}):
        response = call(create_app(), "POST", "/v1/compaction/accept", json={**plan, **extra})
        assert response.status_code == 422
        assert response.json()["error"] in ("truncated_summary", "empty_summary")


# --- documents ---

DOC = "SECTION 1\n" + "alpha line\n" * 40 + "SECTION 2\n" + "beta line\n" * 40


def test_split_gives_what_the_library_gives() -> None:
    pattern = r"^SECTION \d+"
    pieces = split(DOC, max_tokens=60, heading=re.compile(pattern, re.M))
    response = call(
        create_app(),
        "POST",
        "/v1/docs/split",
        json={"text": DOC, "max_tokens": 60, "heading": pattern},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"pieces": [asdict(p) for p in pieces]}
    assert len(pieces) > 2


UNITS: list[dict[str, Any]] = [
    {"text": "a" * 70},
    {"tokens": 30},
    {"text": "b" * 35, "key": "dup"},
    {"text": "c" * 35, "key": "dup"},
    {"text": "d" * 200},
    {"tokens": 5},
]


def cost(unit: dict[str, Any]) -> int:
    return unit["tokens"] if "tokens" in unit else EST.text(unit["text"])


def test_pack_gives_what_the_library_gives_as_indices() -> None:
    units = [u for u in UNITS if cost(u) + 5 <= 60]
    indexed = list(range(len(units)))
    groups = pack(indexed, budget=60, cost=lambda i: cost(units[i]), overhead=5)
    response = call(
        create_app(), "POST", "/v1/docs/pack", json={"units": units, "budget": 60, "overhead": 5}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"groups": groups}
    too_big = call(create_app(), "POST", "/v1/docs/pack", json={"units": UNITS, "budget": 50})
    assert too_big.status_code == 422 and too_big.json()["error"] == "does_not_fit"


def test_select_gives_what_the_library_gives_as_indices() -> None:
    indexed = list(range(len(UNITS)))
    chosen = select(
        indexed,
        budget=60,
        cost=lambda i: cost(UNITS[i]),
        key=lambda i: UNITS[i].get("key", UNITS[i].get("text", i)),
    )
    response = call(create_app(), "POST", "/v1/docs/select", json={"units": UNITS, "budget": 60})
    assert response.status_code == 200, response.text
    assert response.json() == asdict(chosen)
    assert response.json()["duplicates"] == [3]


# --- transcripts: the same script through HTTP and through Transcript ---

BUDGET = {"limit": 700, "reserve_output": 0}
STEPS = [{"step": "mask_tool_results", "keep_last": 1}]


async def by_library(store: InMemoryStore) -> list[Any]:
    t = Transcript(store, "talk")
    await t.create()
    seq, seen = 0, []
    for n in range(1, 5):
        seq = await t.append(keyless_turn(n, size=150), expected_seq=seq)
        fitted = await t.project(
            budget=Budget(700, 0), steps=[MaskToolResults(keep_last=1)], counter=EST
        )
        seen.append(jsonish(anthropic.from_messages(fitted.messages)))
        seq = await t.append([fitted.state], expected_seq=seq)
    plan_t = await t.entries()
    messages = [e.item for e in plan_t if isinstance(e.item, Message)]
    state = [e.item for e in plan_t if isinstance(e.item, ContextState)][-1]
    plan = plan_compaction(
        messages, state=state, policy=SummaryPolicy(keep_last_turns=1, mark="m0")
    )
    assert plan is not None
    new = plan.accept("so far", now=datetime.fromisoformat(NOW))
    seq = await t.append([new], expected_seq=seq)
    fitted = await t.project(
        budget=Budget(700, 0), steps=[MaskToolResults(keep_last=1)], counter=EST
    )
    seen.append(jsonish(anthropic.from_messages(fitted.messages)))
    return seen


async def by_http(app: Any) -> list[Any]:
    async with client(app) as c:
        return await script(c)


async def script(c: httpx.AsyncClient) -> list[Any]:
    async def post(path: str, body: dict[str, Any]) -> Any:
        r = await c.post(path, json=body)
        assert r.status_code == 200, r.text
        return r.json()

    assert await post("/v1/transcripts", {"id": "talk"}) == {"id": "talk", "last_seq": 0}
    project = {"format": "anthropic", "budget": BUDGET, "steps": STEPS}
    seq, seen = 0, []
    for n in range(1, 5):
        items = [{"message": m} for m in anthropic.from_messages(keyless_turn(n, size=150))]
        entries = {"format": "anthropic", "items": items, "expected_seq": seq}
        seq = (await post("/v1/transcripts/talk/entries", entries))["last_seq"]
        projected = await post("/v1/transcripts/talk/project", project)
        seen.append(projected["messages"])
        state = {
            "format": "anthropic",
            "items": [{"state": projected["state"]}],
            "expected_seq": seq,
        }
        seq = (await post("/v1/transcripts/talk/entries", state))["last_seq"]
    plan_body = {"format": "anthropic", "keep_last_turns": 1, "mark": "m0"}
    plan = await post("/v1/transcripts/talk/compaction/plan", plan_body)
    assert plan["expected_seq"] == seq
    accepted = await post(
        "/v1/transcripts/talk/compaction/accept", {**plan, "reply": "so far", "now": NOW}
    )
    assert accepted == {"last_seq": seq + 1}
    seen.append((await post("/v1/transcripts/talk/project", project))["messages"])
    return seen


def test_a_transcript_over_http_projects_what_the_library_projects() -> None:
    expected = run(lambda: by_library(InMemoryStore()))
    assert run(lambda: by_http(create_app(store=InMemoryStore()))) == expected
    assert len({json.dumps(p) for p in expected}) == len(expected)  # each step changed something


def test_a_transcript_over_http_on_postgresql_projects_the_same(
    pg_store_factory: Callable[[], Any],
) -> None:
    expected = run(lambda: by_library(InMemoryStore()))

    async def over_pg() -> list[Any]:
        store = pg_store_factory()
        try:
            return await by_http(create_app(store=store))
        finally:
            await store.aclose()

    assert run(over_pg) == expected


def test_entries_read_back_in_the_format_asked_for() -> None:
    app = create_app(store=InMemoryStore())
    call(app, "POST", "/v1/transcripts", json={"id": "t"})
    native = anthropic.from_messages(keyless_turn(1, size=5))
    call(
        app,
        "POST",
        "/v1/transcripts/t/entries",
        json={"format": "anthropic", "items": [{"message": m} for m in native], "expected_seq": 0},
    )
    body = call(app, "GET", "/v1/transcripts/t/entries?after=2&format=anthropic").json()
    assert body == {
        "entries": [
            {"seq": 3, "kind": "message", "item": native[2]},
            {"seq": 4, "kind": "message", "item": native[3]},
        ]
    }


# --- errors: one status per kind ---


def test_conflicts_unknown_transcripts_and_bad_input_have_their_status() -> None:
    app = create_app(store=InMemoryStore())
    call(app, "POST", "/v1/transcripts", json={"id": "t"})
    item = {"message": {"role": "user", "content": "hi"}}
    ok = {"format": "anthropic", "items": [item], "expected_seq": 0}
    assert call(app, "POST", "/v1/transcripts/t/entries", json=ok).status_code == 200
    conflict = call(app, "POST", "/v1/transcripts/t/entries", json=ok)
    assert conflict.status_code == 409
    assert conflict.json() == {
        "error": "seq_conflict",
        "detail": "transcript 't': expected seq 0, the tail is 1",
        "expected": 0,
        "actual": 1,
    }
    missing = call(app, "POST", "/v1/transcripts/nope/entries", json=ok)
    assert (missing.status_code, missing.json()["error"]) == (404, "unknown_transcript")
    unknown_step = call(
        app,
        "POST",
        "/v1/fit",
        json={"format": "anthropic", "messages": [], "budget": BUDGET, "steps": [{"step": "fold"}]},
    )
    assert unknown_step.status_code == 422
    extra_field = call(
        app,
        "POST",
        "/v1/fit",
        json={"format": "anthropic", "messages": [], "budget": BUDGET, "x": 1},
    )
    assert extra_field.status_code == 422
    unsupported = call(
        app,
        "POST",
        "/v1/fit",
        json={
            "format": "anthropic",
            "messages": [{"role": "system", "content": "x"}],
            "budget": BUDGET,
        },
    )
    assert (unsupported.status_code, unsupported.json()["error"]) == (422, "unsupported_input")


def test_a_stale_state_is_a_conflict() -> None:
    msgs = conversations()[0]
    native, keys = anthropic.from_messages(msgs), [m.key or "" for m in msgs]
    request = {
        "format": "anthropic",
        "messages": native,
        "keys": keys,
        "budget": {"limit": int(count_messages(msgs, EST) * 0.6), "reserve_output": 0},
        "steps": [{"step": "mask_tool_results", "keep_last": 0}],
    }
    first = call(create_app(), "POST", "/v1/fit", json=request).json()
    assert first["state"]["cuts"]
    renamed = ["x" + k for k in keys]
    stale = call(
        create_app(), "POST", "/v1/fit", json={**request, "keys": renamed, "state": first["state"]}
    )
    assert (stale.status_code, stale.json()["error"]) == (409, "stale_state")


def test_transcripts_without_a_database_are_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITECONTEXT_DATABASE_URL", raising=False)
    response = call(create_app(), "POST", "/v1/transcripts", json={"id": "t"})
    assert response.status_code == 503
    assert "LITECONTEXT_DATABASE_URL" in response.json()["detail"]


# --- only this machine; bounded bodies ---


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "127.0.0.2"])
def test_loopback_addresses_may_be_bound(host: str) -> None:
    check_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "localhost", "::", "example.com"])
def test_any_other_address_is_refused(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        check_host(host)


def test_foreign_hosts_pages_and_forms_are_refused() -> None:
    app = create_app()
    body = {"text": "x", "max_tokens": 10}
    foreign_host = call(app, "POST", "/v1/docs/split", json=body, headers={"host": "evil.example"})
    assert foreign_host.status_code == 400
    page = call(app, "POST", "/v1/docs/split", json=body, headers={"origin": "http://evil.example"})
    assert page.status_code == 403
    form = call(
        app, "POST", "/v1/docs/split", content=b"text=x", headers={"content-type": "text/plain"}
    )
    assert form.status_code == 415
    assert call(app, "POST", "/v1/docs/split", json=body).status_code == 200


def test_an_oversized_body_is_refused_whether_or_not_it_says_its_length() -> None:
    app = create_app(max_body=1_000)
    big = {"text": "x" * 2_000, "max_tokens": 10}
    declared = call(app, "POST", "/v1/docs/split", json=big)
    assert declared.status_code == 413

    async def chunks() -> Any:
        payload = json.dumps(big).encode()
        for i in range(0, len(payload), 300):
            yield payload[i : i + 300]

    chunked = call(
        app,
        "POST",
        "/v1/docs/split",
        content=chunks(),
        headers={"content-type": "application/json"},
    )
    assert chunked.status_code == 413
    small = call(app, "POST", "/v1/docs/split", json={"text": "x", "max_tokens": 10})
    assert small.status_code == 200


def test_a_summary_with_a_message_built_by_hand_round_trips_through_neutral() -> None:
    # A guard for the codec path the shell uses: a neutral message with every field set.
    m = Message("user", (Text("x", extra={"a": 1}),), key="k", pinned=True)
    response = call(
        create_app(),
        "POST",
        "/v1/fit",
        json={"format": "neutral", "messages": [message_to_dict(m)], "budget": BUDGET},
    )
    assert response.json()["messages"] == [message_to_dict(m)]


def test_a_declared_length_over_the_limit_is_refused_before_any_body_is_read() -> None:
    app = create_app(max_body=1_000)
    sent: list[dict[str, Any]] = []
    reads = 0

    async def receive() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(dict(message))

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/docs/split",
        "headers": [(b"content-length", b"5000000"), (b"content-type", b"application/json")],
        "query_string": b"",
    }
    run(lambda: app(scope, receive, send))
    assert sent[0]["status"] == 413 and reads == 0


def test_a_system_message_is_not_stored_in_a_transcript() -> None:
    app = create_app(store=InMemoryStore())
    call(app, "POST", "/v1/transcripts", json={"id": "t"})
    response = call(
        app,
        "POST",
        "/v1/transcripts/t/entries",
        json={
            "format": "openai",
            "items": [{"message": {"role": "system", "content": "be brief"}}],
            "expected_seq": 0,
        },
    )
    assert response.status_code == 422
    assert "fixed text" in response.json()["detail"]
    assert call(app, "GET", "/v1/transcripts/t/entries").json() == {"entries": []}


def test_a_heading_that_is_not_a_regular_expression_is_a_422() -> None:
    response = call(
        create_app(),
        "POST",
        "/v1/docs/split",
        json={"text": "a\nb", "max_tokens": 5, "heading": "("},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid"


def test_a_transcript_id_no_store_can_keep_is_a_422_not_a_503() -> None:
    app = create_app(store=InMemoryStore())
    response = call(app, "POST", "/v1/transcripts", json={"id": "a\u0000b"})
    assert (response.status_code, response.json()["error"]) == (422, "invalid")
