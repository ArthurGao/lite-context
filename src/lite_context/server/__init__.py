"""The REST shell (design §9, extra `server`): lite-context over HTTP for hosts in other
languages. No logic of its own (rule 9): `schemas` parses, `wire` converts, `app` calls."""

from .app import check_host, create_app

__all__ = ["check_host", "create_app"]
