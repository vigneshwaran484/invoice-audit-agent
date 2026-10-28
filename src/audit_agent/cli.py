"""Command-line runner.

  uv run python -m audit_agent.cli audit INV-2026-0005          # runs, pauses for approval
  uv run python -m audit_agent.cli status INV-2026-0005
  uv run python -m audit_agent.cli decide INV-2026-0005 --approve [--explanation "..."]
  uv run python -m audit_agent.cli decide INV-2026-0005 --reject

Because state lives in the SQLite checkpointer, `audit` and `decide` can run in different processes
(kill the first one, restart, and the thread resumes where it paused).
"""

from __future__ import annotations

import argparse
import asyncio
import json

from . import service
from .graph import open_agent


async def _main(args) -> None:
    async with open_agent() as app:
        if args.cmd == "audit":
            out = await service.start(app, args.invoice_id, restart=args.restart)
        elif args.cmd == "status":
            out = await service.current(app, args.invoice_id) or {"status": "not started"}
        else:
            edits = {"explanation": args.explanation} if args.explanation else {}
            out = await service.decide(app, args.invoice_id, approved=args.approve, edits=edits)
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(prog="audit_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("invoice_id")
    a.add_argument("--restart", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("invoice_id")
    d = sub.add_parser("decide")
    d.add_argument("invoice_id")
    g = d.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    d.add_argument("--explanation")
    asyncio.run(_main(ap.parse_args()))


if __name__ == "__main__":
    main()
