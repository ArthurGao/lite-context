"""The REST shell's routes (design §9). Each route parses, calls the library once and serialises:
the same input gives what the library gives. Loopback only, no authentication (design §9).

`check_host`, the Host / Origin / content-type checks and their reasons are copied from
lite-agent `src/lite_agent/server.py` @ v0.2.4 (lite-context imports no lite module, rule 6).
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import errors
from ..compaction import plan_compaction
from ..count import CharEstimate
from ..docs import pack, select, split
from ..fit import fit
from ..state import ContextState
from ..store import Store
from ..transcript import Transcript
from ..types import Message as NeutralMessage
from . import wire
from .schemas import (
    AcceptIn,
    EntriesIn,
    FitIn,
    Format,
    PackIn,
    PlanIn,
    ProjectIn,
    SelectIn,
    SplitIn,
    TranscriptAcceptIn,
    TranscriptIn,
    TranscriptPlanIn,
    Unit,
)

DEFAULT_MAX_BODY = 32 * 1024 * 1024
NO_DATABASE = "transcripts need PostgreSQL: set LITECONTEXT_DATABASE_URL (extra `pg`)"


def check_host(host: str) -> None:
    """Refuse any address that is not loopback: the shell has no authentication. Only a numeric
    address is taken: a name, even `localhost`, can resolve to anything."""
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(f"litecontext serve binds to loopback only (no authentication), not {host}")


def _loopback_name(name: str | None) -> bool:
    if not name:
        return False
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name.strip("[]")).is_loopback
    except ValueError:
        return False


def _host_name(host: str) -> str:
    """`127.0.0.1:8750` -> `127.0.0.1`, `[::1]:8750` -> `::1`, `localhost` -> `localhost`."""
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class _BoundedBody:
    """Reads a request body only up to `limit` bytes: a declared length over it, or a body that
    grows past it, is a 413 before the route sees anything."""

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.limit:
            await self._too_large(scope, receive, send)
            return
        body, more = bytearray(), True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            more = message.get("more_body", False)
            if len(body) > self.limit:
                await self._too_large(scope, receive, send)
                return
        sent = False

        async def replay() -> Message:
            nonlocal sent
            if sent:
                return await receive()  # the client's disconnect, if it comes
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        detail = f"the request body is over {self.limit} bytes (LITECONTEXT_MAX_BODY)"
        await JSONResponse({"error": "too_large", "detail": detail}, status_code=413)(
            scope, receive, send
        )


def create_app(store: Store | None = None, *, max_body: int = DEFAULT_MAX_BODY) -> ASGIApp:
    """The shell. Transcripts use `store`, or a `PgStore` on LITECONTEXT_DATABASE_URL (opened on
    first use, closed on shutdown); without either they answer 503."""
    owned = store is None
    if store is None and os.environ.get("LITECONTEXT_DATABASE_URL"):
        from ..pg import PgStore

        store = PgStore(os.environ["LITECONTEXT_DATABASE_URL"])

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owned and store is not None:
            await store.aclose()

    app = FastAPI(
        title="lite-context", summary="What each LLM call should see (design §9)", lifespan=lifespan
    )

    @app.middleware("http")
    async def only_from_this_machine(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Bound to loopback, but a web page can still reach it: by rebinding its own name to
        # 127.0.0.1 (its Host is then that name) or by sending from the browser (its Origin is
        # the page's). With no authentication, only loopback names are served.
        host = request.headers.get("host", "")
        if not _loopback_name(_host_name(host)):
            return JSONResponse({"detail": "only loopback host names are served"}, status_code=400)
        origin = request.headers.get("origin")
        if origin is not None and origin != f"http://{host}":
            return JSONResponse({"detail": "requests from web pages are refused"}, status_code=403)
        # a JSON body needs a CORS preflight a foreign page never passes; a form does not
        kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if request.method == "POST" and kind != "application/json":
            return JSONResponse({"detail": "send application/json"}, status_code=415)
        return await call_next(request)

    @app.exception_handler(wire.Refused)
    async def on_refused(_: Request, exc: wire.Refused) -> JSONResponse:
        return JSONResponse(exc.body, status_code=exc.status)

    # What the library refuses, from any route: its error kind and message (design §9).
    @app.exception_handler(errors.LiteContextError)
    @app.exception_handler(ValueError)
    @app.exception_handler(re.error)
    async def on_library_error(_: Request, exc: Exception) -> JSONResponse:
        answer = wire.refused(exc)
        return JSONResponse(answer.body, status_code=answer.status)

    def transcript(tid: str) -> Transcript:
        if store is None:
            raise wire.Refused(503, "no_database", NO_DATABASE)
        return Transcript(store, tid)

    # --- stateless (plain `def`: CPU work runs in the thread pool, off the event loop) ---

    @app.post("/v1/fit")
    def fit_(body: FitIn) -> Any:
        messages, fixed, texts = wire.neutral(body.format, body.messages, body.keys)
        f = fit(
            messages,
            budget=wire.budget(body.budget),
            fixed=[*texts, *body.fixed],
            steps=[wire.step(s) for s in body.steps],
            state=wire.state_in(body.state),
            high=body.high,
            low=body.low,
        )
        return wire.fitted(body.format, f, fixed)

    @app.post("/v1/compaction/plan")
    def plan_(body: PlanIn) -> Any:
        messages, _, _ = wire.neutral(body.format, body.messages, body.keys)
        plan = plan_compaction(messages, state=wire.state_in(body.state), policy=body.policy())
        return None if plan is None else wire.plan_out(body.format, plan)

    @app.post("/v1/compaction/accept")
    def accept_(body: AcceptIn) -> Any:
        return {"state": wire.state_out(wire.accepted(body))}

    @app.post("/v1/docs/split")
    def split_(body: SplitIn) -> Any:
        heading = re.compile(body.heading, re.M) if body.heading else None
        pieces = split(
            body.text,
            max_tokens=body.max_tokens,
            overlap=body.overlap,
            prefer=body.prefer,
            heading=heading,
        )
        return {"pieces": [wire.report(p) for p in pieces]}

    @app.post("/v1/docs/pack")
    def pack_(body: PackIn) -> Any:
        indices = list(range(len(body.units)))
        groups = pack(indices, budget=body.budget, cost=_cost(body.units), overhead=body.overhead)
        return {"groups": groups}

    @app.post("/v1/docs/select")
    def select_(body: SelectIn) -> Any:
        units = body.units

        def key(i: int) -> Any:
            u = units[i]
            return u.key if u.key is not None else u.text if u.text is not None else ("#", i)

        chosen = select(list(range(len(units))), budget=body.budget, cost=_cost(units), key=key)
        return wire.report(chosen)

    # --- transcripts (stateful) ---

    @app.post("/v1/transcripts")
    async def create_(body: TranscriptIn) -> Any:
        t = transcript(body.id)
        await t.create()
        return {"id": body.id, "last_seq": await t.last_seq()}

    @app.post("/v1/transcripts/{tid}/entries")
    async def append_(tid: str, body: EntriesIn) -> Any:
        t = transcript(tid)
        items: list[NeutralMessage | ContextState] = []
        for item in body.items:
            state = wire.state_in(item.state)
            items.extend(
                [state] if state is not None else wire.stored_message(body.format, item.message)
            )
        return {"last_seq": await t.append(items, expected_seq=body.expected_seq)}

    @app.get("/v1/transcripts/{tid}/entries")
    async def entries_(tid: str, after: int = Query(0, ge=0), format: Format = "neutral") -> Any:
        out = []
        for e in await transcript(tid).entries(after=after):
            if isinstance(e.item, ContextState):
                out.append({"seq": e.seq, "kind": "state", "item": wire.state_out(e.item)})
            else:
                # OpenAI: the list of chat messages it becomes (several tool results, several).
                natives = wire.native(format, [e.item])
                item = natives if format == "openai" else natives[0]
                out.append({"seq": e.seq, "kind": "message", "item": item})
        return {"entries": out}

    @app.post("/v1/transcripts/{tid}/project")
    async def project_(tid: str, body: ProjectIn) -> Any:
        f = await transcript(tid).project(
            budget=wire.budget(body.budget),
            steps=[wire.step(s) for s in body.steps],
            fixed=body.fixed,
            high=body.high,
            low=body.low,
        )
        return wire.fitted(body.format, f)

    @app.post("/v1/transcripts/{tid}/compaction/plan")
    async def transcript_plan_(tid: str, body: TranscriptPlanIn) -> Any:
        plan, tail = await transcript(tid).plan_compaction(policy=body.policy())
        if plan is None:
            return None
        return {**wire.plan_out(body.format, plan), "format": body.format, "expected_seq": tail}

    @app.post("/v1/transcripts/{tid}/compaction/accept")
    async def transcript_accept_(tid: str, body: TranscriptAcceptIn) -> Any:
        state = wire.accepted(body)
        return {"last_seq": await transcript(tid).append([state], expected_seq=body.expected_seq)}

    return _BoundedBody(app, max_body)


def _cost(units: list[Unit]) -> Callable[[int], int]:
    """A unit's tokens when the client gave them, otherwise the estimate of its text."""
    estimate = CharEstimate()

    def cost(i: int) -> int:
        tokens = units[i].tokens
        return tokens if tokens is not None else estimate.text(units[i].text or "")

    return cost
