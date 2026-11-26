"""`litecontext`: the command line (design §8). No logic: it parses, calls the library and
prints. `migrate` and `pending` need the extra `pg`; the URL comes from LITECONTEXT_DATABASE_URL."""

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
    args = parser.parse_args(argv)
    return _database(args.command, args.schema)


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
