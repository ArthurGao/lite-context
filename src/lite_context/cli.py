"""`litecontext`: the command line (design §8, §9). No logic: it parses, calls the library and
prints. `migrate` and `pending` need the extra `pg` (URL from LITECONTEXT_DATABASE_URL); `serve`
the extra `server` (LITECONTEXT_HOST, LITECONTEXT_PORT, LITECONTEXT_MAX_BODY, design §10)."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

URL_VARIABLE = "LITECONTEXT_DATABASE_URL"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="litecontext", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("migrate", "apply the transcript tables' pending migrations"),
        ("pending", "list the migrations not yet applied"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--schema", default="lite_context")
    serve = commands.add_parser("serve", help="the REST shell, loopback only (design §9)")
    serve.add_argument("--host", default=None, help="default LITECONTEXT_HOST, else 127.0.0.1")
    serve.add_argument("--port", default=None, help="default LITECONTEXT_PORT, else 8750")
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    return _database(args.command, args.schema)


def _setting(value: str | None, name: str, default: int, *, top: int | None = None) -> int:
    """A positive whole number from the command line or LITECONTEXT_<name>; ValueError naming
    the setting otherwise."""
    raw = value if value is not None else os.environ.get(f"LITECONTEXT_{name}", str(default))
    try:
        number = int(raw)
    except ValueError:
        number = 0
    if number < 1 or (top is not None and number > top):
        limit = f"1 to {top}" if top else "a positive whole number"
        flag = f" (or --{name.lower()})" if name == "PORT" else ""
        raise ValueError(f"LITECONTEXT_{name}{flag} must be {limit}, not {raw!r}")
    return number


def _serve(host_arg: str | None, port_arg: str | None) -> int:
    try:
        import uvicorn

        from .server import check_host, create_app
        from .server.app import DEFAULT_MAX_BODY
    except ImportError:
        return _fail("serve needs the extra: pip install 'lite-context[server]'", 2)
    host = host_arg or os.environ.get("LITECONTEXT_HOST", "127.0.0.1")
    try:
        check_host(host)
        port = _setting(port_arg, "PORT", 8750, top=65535)
        max_body = _setting(None, "MAX_BODY", DEFAULT_MAX_BODY)
    except ValueError as exc:
        return _fail(str(exc), 2)
    uvicorn.run(create_app(max_body=max_body), host=host, port=port)
    return 0


def _database(command: str, schema: str) -> int:
    try:
        import psycopg

        from .pg import db
    except ImportError:
        return _fail("PostgreSQL needs the extra: pip install 'lite-context[pg]'", 2)
    url = os.environ.get(URL_VARIABLE)
    if not url:
        return _fail(f"{URL_VARIABLE} is not set", 2)
    try:
        db.check_schema(schema)
    except ValueError as exc:
        return _fail(str(exc), 2)
    try:
        with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
            versions = (
                db.migrate(conn, schema) if command == "migrate" else db.pending(conn, schema)
            )
        done = "applied {} to {}" if command == "migrate" else "pending in {1}: {0}"
        print(done.format(", ".join(versions), schema) if versions else f"{schema} is up to date")
    except psycopg.Error as exc:  # any of them may quote the connection string
        return _fail(str(db.connection_error(exc, url)), 1)
    return 0


def _fail(message: str, code: int) -> int:
    print(f"litecontext: {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
