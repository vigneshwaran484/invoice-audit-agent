"""Day 5 · Retrieval over contract clauses, used as claim evidence.

Chunking: one chunk per clause. Each clause is a self-contained legal statement (price, validity,
volume discount), so splitting smaller would separate a price from its discount, and merging would
pull in other SKUs' prices as noise. Chunk size is therefore "one clause", not a token count.

Retrieval: BM25 over clause text (pure Python, deterministic, no model download), then a rerank
that boosts chunks whose supplier/SKU metadata exactly match the discrepancy. Embeddings would help
with paraphrased contracts; here SKU codes are exact tokens, which lexical search handles best.
Quality is measured by ``evidence_hit_rate`` / recall@k in tests and evals.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

CLAUSE_RE = re.compile(r"^Clause (?P<id>(?P<sup>SUP-[A-Z]+)-\d+)\. (?P<body>.*\((?P<sku>SKU-[A-Z0-9-]+)\).*)$", re.M)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

KIND_HINTS = {
    "price_above_contract": "supplied at INR per unit price",
    "missed_volume_discount": "orders units or more single invoice receive discount",
    "duplicate_invoice": "supplied at INR per unit",
    "no_contract": "supplied",
}


@dataclass(frozen=True)
class Chunk:
    clause_id: str
    supplier_id: str
    sku: str
    text: str


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


def chunk_clauses(md: str) -> list[Chunk]:
    return [Chunk(m["id"], m["sup"], m["sku"], f"Clause {m['id']}. {m['body']}") for m in CLAUSE_RE.finditer(md)]


class ClauseRetriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _bm25(self, q: list[str], i: int) -> float:
        d = self.docs[i]
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
        return s

    def search(self, query: str, k: int = 3, supplier_id: str | None = None, sku: str | None = None) -> list[Chunk]:
        q = tokenize(query)
        scored = []
        for i, c in enumerate(self.chunks):
            s = self._bm25(q, i)
            if supplier_id and c.supplier_id == supplier_id:  # rerank on exact metadata
                s += 5.0
            if sku and c.sku == sku:
                s += 5.0
            scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.chunks[i] for s, i in scored[:k] if s > 0]

    def evidence_for(self, supplier_id: str, sku: str | None, kind: str, k: int = 1) -> list[Chunk]:
        query = f"{supplier_id} {sku or ''} {KIND_HINTS.get(kind, '')}"
        return self.search(query, k=k, supplier_id=supplier_id, sku=sku)


_retriever: ClauseRetriever | None = None


def get_retriever() -> ClauseRetriever:
    global _retriever
    if _retriever is None:
        from .store import get_store

        _retriever = ClauseRetriever(chunk_clauses(get_store().clauses_md))
    return _retriever
