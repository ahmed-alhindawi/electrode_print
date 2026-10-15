"""The web app as a static site: the same slicer, run in the browser.

GitHub Pages serves files and nothing else, so there is no server to answer
the page's ``/api/...`` requests.  Instead the page loads Pyodide -- CPython
compiled to WebAssembly -- in a Web Worker, installs this package into it, and
hands every request to :func:`handle`, which answers it with the very
:class:`~electrode_slicer.web.server.SliceService` the local server uses.
Nothing is reimplemented in JavaScript, and an STL never leaves the browser.

Build the site with::

    python -m electrode_slicer.web.pages build/pages

and serve that directory from anywhere that serves static files.
"""

from __future__ import annotations

import json

from ..server import SliceService, _json_reply

__all__ = ["handle"]

_service: SliceService | None = None


def handle(method: str, route: str, body: str) -> tuple[int, bytes, str, dict]:
    """Answer one request from the page: ``(status, body, content type, headers)``."""
    global _service
    if _service is None:
        _service = SliceService()
    route = route.split("?", 1)[0]
    if method == "GET":
        if route == "/api/patterns":
            reply = _json_reply(200, _service.patterns())
        elif route == "/api/health":
            reply = _json_reply(200, {"ok": True})
        else:
            reply = _json_reply(404, {"error": "not found"})
    else:
        try:
            payload = json.loads(body or "")
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            reply = _json_reply(400, {"error": "request body must be a JSON object"})
        else:
            reply = _service.respond(route, payload)
    if reply.refusal:
        print(f"refused {route}: {reply.refusal}")
    return reply.status, reply.body, reply.content_type, reply.headers
