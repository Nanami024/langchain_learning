from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from retrieval_core import tokenize

if TYPE_CHECKING:
    from analytics import DataAnalyst
    from compliance import ComplianceRetriever

_LEXICON_CACHE: CorpusLexicon | None = None


@dataclass(frozen=True)
class CorpusLexicon:
    data_terms: frozenset[str]
    policy_terms: frozenset[str]
    data_top: frozenset[str]
    policy_top: frozenset[str]
    db_value_terms: frozenset[str]
    risk_terms: frozenset[str]


def _freq_terms(docs_texts: list[str], *, min_df: int = 2, top_k: int = 400) -> set[str]:
    df: Counter[str] = Counter()
    for text in docs_texts:
        for tok in set(tokenize(text)):
            if len(tok) >= 2 or (len(tok) == 2 and tok[0].isascii()):
                df[tok] += 1
    return {t for t, n in df.most_common(top_k) if n >= min_df}


def _db_distinct_terms(analyst: DataAnalyst, columns: tuple[str, ...] = ("category", "region")) -> set[str]:
    terms: set[str] = set()
    tbl = analyst.settings.sales_table
    cols = [c for c in columns if c in analyst.sales_columns]
    if not cols:
        return terms
    try:
        from sqlalchemy import text

        from database import get_readonly_engine

        with get_readonly_engine().connect() as conn:
            for col in cols:
                rows = conn.execute(
                    text(f"SELECT DISTINCT `{col}` FROM `{tbl}` WHERE `{col}` IS NOT NULL LIMIT 80")
                ).all()
                for row in rows:
                    val = str(row[0] or "").strip()
                    if val:
                        terms.add(val.lower())
                        terms.update(tokenize(val))
    except Exception:
        pass
    return {t for t in terms if t}


def build_corpus_lexicon(analyst: DataAnalyst, retriever: ComplianceRetriever) -> CorpusLexicon:
    data_texts = [d.page_content for d in analyst.data_docs]
    policy_texts = [d.page_content for d in retriever.docs]
    data_terms = _freq_terms(data_texts, min_df=1, top_k=500)
    policy_terms = _freq_terms(policy_texts, min_df=2, top_k=800)
    db_values = _db_distinct_terms(analyst)
    data_terms |= db_values
    risk_seed_docs = [t for t in policy_texts if len(set(tokenize(t)) & policy_terms) >= 12]
    risk_terms = _freq_terms(risk_seed_docs or policy_texts, min_df=2, top_k=150)

    return CorpusLexicon(
        data_terms=frozenset(data_terms),
        policy_terms=frozenset(policy_terms),
        data_top=frozenset(list(data_terms)[:200]),
        policy_top=frozenset(list(policy_terms)[:300]),
        db_value_terms=frozenset(db_values),
        risk_terms=frozenset(risk_terms),
    )


def get_lexicon(analyst: DataAnalyst, retriever: ComplianceRetriever) -> CorpusLexicon:
    global _LEXICON_CACHE
    if _LEXICON_CACHE is None:
        _LEXICON_CACHE = build_corpus_lexicon(analyst, retriever)
    return _LEXICON_CACHE
