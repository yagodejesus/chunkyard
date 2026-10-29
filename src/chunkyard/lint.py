"""Lint a RAG corpus (the chunks that go into your vector database)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Chunk", "Finding", "Report", "load_chunks", "lint", "fix", "CODES", "approx_tokens"]

CODES = {
    "C001": ("error", "empty chunk"),
    "C002": ("warning", "chunk too short to be useful"),
    "C003": ("warning", "chunk longer than max_tokens"),
    "C004": ("warning", "exact duplicate"),
    "C005": ("warning", "near-duplicate"),
    "C006": ("warning", "heavy boilerplate (≥30% lines repeated across documents)"),
    "C007": ("warning", "broken encoding"),
    "C008": ("warning", "markup leftovers (HTML tags or entities)"),
    "C009": ("info", "cut mid-sentence"),
    "C010": ("warning", "no source/document metadata"),
    "C011": ("warning", "low-information text"),
}
TEXT_KEYS = ("text", "content", "page_content", "chunk", "body")
ID_KEYS = ("id", "chunk_id", "_id", "uuid")
DOC_KEYS = ("doc_id", "document_id", "source", "url", "file", "path", "title")
_MOJIBAKE = re.compile("Ã[\u0080-\u00bf]|â€|\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MARKUP = re.compile(r"</?(?:div|span|p|br|td|tr|li|ul|a|img|script|style|table|nav|footer)\b[^>]*>|&(?:nbsp|amp|lt|gt|quot|#\d+);", re.I)


def approx_tokens(text: str) -> int:
    return max(math.ceil(len(text) / 4), len(re.findall(r"\w+|[^\w\s]", text))) if text else 0


@dataclass
class Chunk:
    index: int
    id: Any
    text: str
    doc: str | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    text_key: str = "text"


@dataclass(frozen=True)
class Finding:
    code: str
    chunk: int
    id: Any
    message: str

    @property
    def severity(self) -> str:
        return CODES[self.code][0]


def _pick(d: dict, keys: Iterable[str]) -> tuple[str | None, Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return k, d[k]
    meta = d.get("metadata") or {}
    if isinstance(meta, dict):
        for k in keys:
            if k in meta and meta[k] not in (None, ""):
                return f"metadata.{k}", meta[k]
    return None, None


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read chunks from ``.jsonl``, ``.json`` (a list) or ``.csv``; text/id/source keys are detected."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
    elif text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return from_records(rows)


def from_records(rows: Iterable[dict[str, Any] | str]) -> list[Chunk]:
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, str):
            out.append(Chunk(i, i, r, None, {"text": r}))
            continue
        tk, txt = _pick(r, TEXT_KEYS)
        _, cid = _pick(r, ID_KEYS)
        _, doc = _pick(r, DOC_KEYS)
        out.append(Chunk(i, cid if cid is not None else i, str(txt or ""), str(doc) if doc is not None else None, r, tk or "text"))
    return out


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", t))


def _shingles(text: str, k: int = 3) -> set[str]:
    words = _norm(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


_PRIME = (1 << 61) - 1


def _near_duplicates(chunks: list[Chunk], threshold: float, perms: int = 64, bands: int = 16) -> list[tuple[int, int, float]]:
    rng = random.Random(1234)
    coeffs = [(rng.randrange(1, _PRIME), rng.randrange(0, _PRIME)) for _ in range(perms)]
    rows = perms // bands
    sets: dict[int, set[str]] = {}
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for c in chunks:
        sh = _shingles(c.text)
        if not sh:
            continue
        sets[c.index] = sh
        hashes = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh]
        sig = [min((a * h + b) % _PRIME for h in hashes) for a, b in coeffs]
        for band in range(bands):
            buckets[(band, tuple(sig[band * rows : (band + 1) * rows]))].append(c.index)
    pairs: dict[tuple[int, int], float] = {}
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                key = (min(a, b), max(a, b))
                if key in pairs:
                    continue
                sa, sb = sets[key[0]], sets[key[1]]
                j = len(sa & sb) / len(sa | sb)
                if j >= threshold:
                    pairs[key] = j
    return sorted((a, b, round(j, 3)) for (a, b), j in pairs.items())


def _boilerplate_lines(chunks: list[Chunk], min_share: float) -> set[str]:
    owners: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        owner = c.doc if c.doc is not None else f"#{c.index}"
        for line in {ln.strip() for ln in c.text.splitlines()}:
            if len(line) >= 8:
                owners[_norm(line)].add(owner)
    n_owners = len({c.doc if c.doc is not None else f"#{c.index}" for c in chunks}) or 1
    need = max(3, math.ceil(min_share * n_owners))
    return {line for line, who in owners.items() if line and len(who) >= need}


@dataclass
class Report:
    findings: list[Finding]
    stats: dict[str, Any]
    chunks: list[Chunk] = field(default_factory=list, repr=False)
    near_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    boilerplate: set[str] = field(default_factory=set)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(f.code for f in self.findings).items()))

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "stats": self.stats, "counts": self.counts(), "findings": [asdict(f) | {"severity": f.severity} for f in self.findings]}

    def __str__(self) -> str:
        s = self.stats
        lines = [
            f"{'✓' if self.ok else '✗'} {s['chunks']:,} chunks from {s['documents']:,} documents — {len(self.errors)} errors, {len(self.warnings)} warnings",
            f"  tokens/chunk: median {s['tokens_median']}, p95 {s['tokens_p95']}, max {s['tokens_max']} · {s['tokens_total']:,} tokens to embed",
            f"  duplicates {s['duplicate_pct']:.1f}% · near-duplicates {s['near_duplicate_pct']:.1f}% · boilerplate {s['boilerplate_token_pct']:.1f}% of tokens",
        ]
        groups: dict[str, list[Finding]] = defaultdict(list)
        for f in self.findings:
            groups[f.code].append(f)
        for code, items in sorted(groups.items()):
            ids = ", ".join(str(f.id) for f in items[:6]) + (" …" if len(items) > 6 else "")
            lines.append(f"  {code} {CODES[code][1]} ×{len(items)}  (chunks {ids})")
            lines.append(f"       e.g. {items[0].message}")
        return "\n".join(lines)


def lint(
    chunks: list[Chunk],
    *,
    min_tokens: int = 20,
    max_tokens: int = 1000,
    near: float = 0.85,
    boilerplate_share: float = 0.2,
    count: Callable[[str], int] = approx_tokens,
) -> Report:
    findings: list[Finding] = []
    add = lambda code, c, msg: findings.append(Finding(code, c.index, c.id, msg))  # noqa: E731
    sizes: list[int] = []
    seen: dict[str, Chunk] = {}
    dup = 0
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.doc is not None:
            by_doc[c.doc].append(c)
    boiler = _boilerplate_lines(chunks, boilerplate_share)
    boiler_tokens = 0
    for c in chunks:
        t = c.text
        n = count(t)
        sizes.append(n)
        if not t.strip():
            add("C001", c, f"chunk {c.id} has no text")
            continue
        if n < min_tokens:
            add("C002", c, f"chunk {c.id} has {n} tokens: {t.strip()[:60]!r}")
        if n > max_tokens:
            add("C003", c, f"chunk {c.id} has {n:,} tokens (> {max_tokens:,})")
        key = _norm(t)
        if key in seen:
            dup += 1
            add("C004", c, f"chunk {c.id} repeats chunk {seen[key].id}")
        else:
            seen[key] = c
        bl = [ln for ln in t.splitlines() if _norm(ln) in boiler]
        if bl:
            bt = sum(count(ln) for ln in bl)
            boiler_tokens += bt
            if bt / max(n, 1) >= 0.3:
                add("C006", c, f"chunk {c.id} is {bt / max(n, 1):.0%} boilerplate, e.g. {bl[0].strip()[:60]!r}")
        if m := _MOJIBAKE.search(t):
            add("C007", c, f"chunk {c.id} contains {m.group()!r}")
        if len(_MARKUP.findall(t)) >= 2:
            add("C008", c, f"chunk {c.id} contains {_MARKUP.search(t).group()!r}")
        stripped = t.rstrip()
        if stripped and stripped[-1].isalnum() and n >= 0.8 * max_tokens:
            add("C009", c, f"chunk {c.id} ends mid-sentence: …{stripped[-40:]!r}")
        if c.doc is None:
            add("C010", c, f"chunk {c.id} has no source/document id")
        words = re.findall(r"\w+", t.casefold())
        letters = sum(ch.isalpha() for ch in t)
        if (len(words) >= 30 and len(set(words)) / len(words) < 0.3) or (len(t) >= 40 and letters / len(t) < 0.2):
            add("C011", c, f"chunk {c.id} looks repetitive or non-textual: {t.strip()[:50]!r}")
    pairs = _near_duplicates([c for c in chunks if c.text.strip()], near)
    flagged = set()
    ids = {c.index: c for c in chunks}
    exact = {f.chunk for f in findings if f.code == "C004"}
    for a, b, j in pairs:
        if b in exact or b in flagged:
            continue
        flagged.add(b)
        add("C005", ids[b], f"chunk {ids[b].id} is {j:.0%} similar to chunk {ids[a].id}")
    findings.sort(key=lambda f: (f.chunk, f.code))
    total = sum(sizes)
    stats = {
        "chunks": len(chunks),
        "documents": len(by_doc),
        "tokens_total": total,
        "tokens_median": int(statistics.median(sizes)) if sizes else 0,
        "tokens_p95": sorted(sizes)[max(0, math.ceil(0.95 * len(sizes)) - 1)] if sizes else 0,
        "tokens_max": max(sizes, default=0),
        "duplicate_pct": 100 * dup / len(chunks) if chunks else 0.0,
        "near_duplicate_pct": 100 * len(flagged) / len(chunks) if chunks else 0.0,
        "boilerplate_token_pct": 100 * boiler_tokens / total if total else 0.0,
    }
    return Report(findings, stats, chunks, pairs, boiler)


def fix(report: Report, strip_boilerplate: bool = True, drop: Iterable[str] = ("C001", "C004", "C005")) -> list[dict[str, Any]]:
    """Records without empty chunks and (near-)duplicates, with boilerplate lines removed."""
    bad = {f.chunk for f in report.findings if f.code in set(drop)}
    out = []
    for c in report.chunks:
        if c.index in bad:
            continue
        rec = dict(c.raw)
        if strip_boilerplate and report.boilerplate:
            text = "\n".join(ln for ln in c.text.splitlines() if _norm(ln) not in report.boilerplate).strip()
            if not text:
                continue
            if c.text_key.startswith("metadata."):
                rec["metadata"] = {**rec.get("metadata", {}), c.text_key[9:]: text}
            else:
                rec[c.text_key] = text
        out.append(rec)
    return out
