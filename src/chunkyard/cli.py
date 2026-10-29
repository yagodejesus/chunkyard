"""Command line: ``chunkyard chunks.jsonl``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .lint import fix, lint, load_chunks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chunkyard", description="Lint the chunks of a RAG corpus before embedding them.")
    p.add_argument("file", help="chunks as .jsonl, .json or .csv")
    p.add_argument("--min-tokens", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--near", type=float, default=0.85, help="near-duplicate Jaccard threshold")
    p.add_argument("--fix", metavar="OUT", help="write cleaned chunks (.jsonl)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    a = p.parse_args(argv)
    report = lint(load_chunks(a.file), min_tokens=a.min_tokens, max_tokens=a.max_tokens, near=a.near)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str) if a.json else report)
    if a.fix:
        rows = fix(report)
        Path(a.fix).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        if not a.json:
            print(f"\n✓ wrote {len(rows)} of {len(report.chunks)} chunks to {a.fix}")
    bad = report.errors if a.fail_on == "error" else [f for f in report.findings if f.severity != "info"]
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
