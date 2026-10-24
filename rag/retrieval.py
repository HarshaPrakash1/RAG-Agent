"""The five retrieval strategies, each one thing different from the last.

Every strategy has the same signature:

    search(question, k) -> [(source_file, passage_text), ...]   best first

They are written to be read top to bottom. baseline is four lines; each one
after it adds exactly one idea.
"""
import csv
import functools
import os
from typing import Literal, Optional
import cohere

from pydantic import BaseModel, Field

from . import config
from .util import retry


# ---------------------------------------------------------------- the stores

@functools.lru_cache(maxsize=1)
def vector_store():
    from langchain_pinecone import PineconeVectorStore

    emb = config.embeddings()
    dim = len(emb.embed_query("probe"))
    return PineconeVectorStore(index_name=config.index_name(dim), embedding=emb)


@functools.lru_cache(maxsize=1)
def local_chunks():
    """The same chunks that went into Pinecone, kept in memory as text.

    BM25 counts words, so it needs the words. There is no way to ask a vector
    database for them, which is the first practical thing you learn about
    hybrid search: you end up maintaining two indexes.
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    with open(config.MANIFEST) as f:
        names = sorted(row["file"] for row in csv.DictReader(f))

    chunks = []
    for name in names:
        page = TextLoader(os.path.join(config.SECTIONS, name)).load()[0]
        for piece in splitter.split_documents([page]):
            chunks.append((name, " ".join(piece.page_content.split())))
    return chunks


@functools.lru_cache(maxsize=1)
def bm25():
    from rank_bm25 import BM25Okapi

    return BM25Okapi([text.lower().split() for _, text in local_chunks()])


# ------------------------------------------------------- reading the question
#
# TODO 2.1 (2 pts) — read the company and year out of the question.
#
# A question like "What did Micron say about HBM demand in fiscal 2025?"
# carries three separate things: a company, a year, and a thing to look for.
# The embedding has no idea they are different. Pull them apart, filter the
# index on the first two, and search for the third.

class Filter(BaseModel):
    """Which filings the question is about, and what to look for in them."""
    # TODO 2.1 — three fields. Two of them are optional, and it matters that
    # they are, because plenty of questions name no company at all.
    ticker: Optional[Literal["NVDA", "AMD", "MU"]] = Field(
        default=None,
        description="Ticker of the company, only if the question names exactly one of them. NVIDIA=NVDA, AMD=AMD, Micron=MU"
    )
    fiscal_year: Optional[int] = Field(
        default=None,
        description="Fiscal year of the filing, only if the question names exactly one."
    )
    search: str = Field(
        description="What to look for in the filings, whith the company name and year removed"
    )


def extract_filter(question):
    # TODO 2.1 — with_structured_output, as in W05b.1. Tell the model to
    # leave company and year empty unless the question names exactly one.
    model = config.chat().with_structured_output(Filter)
    prompt = (
        "From the question, pull out the company, the fiscal year and what to search for."
        "leave the company and year empty unless the question names exactly one."
        f"Questions: {question}"
    )
    return retry(model.invoke, prompt)


def where_of(f):
    """Turn a Filter into a Pinecone filter dict.

    Pinecone wants operators, so ticker NVDA becomes {"ticker": {"$eq": "NVDA"}}.
    Watch the empty case: {} is not the same as no filter, and passing {} will
    not do what you want. Return None when there is nothing to filter on.
    """
    # TODO 2.1
    where = {}
    if f.ticker:
        where["ticker"] = {"$eq": f.ticker}
    if f.fiscal_year:
        where["fiscal_year"] = {"$eq": f.fiscal_year}
    return where if where else None


# --------------------------------------------------------- many queries, fused
#
# TODO 2.2 (2 pts) — multi-query and reciprocal rank fusion.
#
# One phrasing of a search finds what that phrasing happens to be close to.
# Ask four ways and you get four ranked lists, and then you have a new
# problem: how do you merge ranked lists whose scores are not comparable?
#
# RRF is the answer, and it is three lines. Each list votes for an item with
#
#     1 / (RRF_K + rank)
#
# and you sum the votes. Rank is the position in that list, counting from 0.
# RRF_K (60, in config) flattens the top of each list so that rank 1 and rank
# 2 are not wildly far apart, which stops one confident list from dominating.
#
# What RRF deliberately ignores: the similarity scores. Only positions. A
# cosine score and a BM25 score are on different scales, so averaging them is
# meaningless, and ranks are the one thing every retriever agrees on.

class Queries(BaseModel):
    """Several ways of asking the same thing."""
    # TODO 2.2
    quries: list[str] = Field(
        description="Different search queries with the same meaning, varied in wording and vocabulary."
    )


def write_queries(text, n=4):
    # TODO 2.2 — ask the model for n searches that cover the same intent.
    # Return the original text as a single-item list if it gives you nothing.
    model = config.chat().with_structured_output(Queries)
    prompt = (
        f"Write {n} different search queries that look for the same thing as the text below."
        "keep the meaning the same, vary the wording and the vocabulary.\n\n"
        f"Text: {text}"
    )
    out = retry(model.invoke, prompt)
    if not out or not out.quries:
        return [text]
    return out.quries[:n]


def rrf(rankings, k=config.RRF_K):
    """Fuse ranked lists. Takes a list of lists, returns one list, best first.

    Every item must be hashable, which is why the strategies pass
    (source, text) tuples around rather than Document objects.
    """
    # TODO 2.2
    scores = {}
    for ranking in rankings:
        for position, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + 1 / (k + position + 1)
    return sorted(scores, key=scores.get, reverse=True)


# --------------------------------------------------------------- the retrieval
# dense() is given. It is the one you already know.

def dense(query, k, where=None):
    hits = vector_store().similarity_search(query, k=k, filter=where)
    return [(h.metadata["source"], h.page_content) for h in hits]


def lexical(query, k):
    """BM25. Scores every chunk once, then takes the top k."""
    scores = bm25().get_scores(query.lower().split())
    best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [local_chunks()[i] for i in best]


# ---------------------------------------------------------------- reranking
#
# TODO 2.3 (2 pts) — reorder a pool with a cross-encoder.
#
# Everything above ranks by comparing two vectors that were computed apart
# from each other. The question never met the passage. A cross-encoder reads
# both at once and scores the pair, which is far better and far too slow to
# run over the whole index. So it goes last, over a pool of ten.



def rerank_cohere(question, passages, top_n):
    # TODO 2.3 — cohere.ClientV2(...).rerank(model="rerank-v3.5", ...).
    # It returns results carrying .index into the list you gave it.
    if not passages:
        return []
    co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    response = co.rerank(
        model="rerank-v3.5",
        query=question,
        documents=[text for _, text in passages],
        top_n=top_n
    )
    return [passages[r.index] for r in response.results]






def rerank(question, passages, top_n=config.K):
    """Cohere reranks the pool. Needs COHERE_API_KEY in .env."""
    return rerank_cohere(question, passages, top_n)


# -------------------------------------------------------------- the strategies
#
# Five strategies, and each one differs from the one above it in exactly one
# way. That is what makes the table in notebook 04 readable: when a number
# moves, there is only one thing it can be.
#
# All five return the same shape:  [(source_file, passage_text), ...]

def baseline(question, k=config.K):
    """Given. Embed the question as typed, one search, no filter."""
    return dense(question, k)


def with_filter(question, k=config.K):
    # TODO 2.1 — extract, then search the filtered index for f.search only.
    f = extract_filter(question)
    return dense(f.search, k=k, where=where_of(f))


def filter_rrf(question, k=config.K):
    # TODO 2.2 — one dense search per rewrite, same filter on each, fused.
    f = extract_filter(question)
    where = where_of(f)
    rankings = [dense(q, k, where=where) for q in write_queries(f.search)]
    return rrf(rankings)[:k]


def filter_rrf_rerank(question, k=config.K):
    # TODO 2.3 — fuse to config.POOL candidates, then rerank down to k.
    # Note it is the original question that goes to the reranker, not
    # f.search. The cross-encoder reads language, so give it the real one.
    f = extract_filter(question)
    where = where_of(f)
    rankings = [dense(q, k, where=where) for q in write_queries(f.search)]
    pool = rrf(rankings)[:config.POOL]
    return rerank(question, pool, top_n=k)


def hybrid_rrf(question, k=config.K):
    """Dense and BM25, fused by rank. Catches the exact-term questions that
    embeddings paraphrase away."""
    f = extract_filter(question)
    dense_hits = dense(f.search, k, where=where_of(f))
    lexical_hits = lexical(f.search, k)
    return rrf([dense_hits, lexical_hits])[:k]


STRATEGIES = {
    "baseline": baseline,
    "filter": with_filter,
    "filter+rrf": filter_rrf,
    "filter+rrf+rerank": filter_rrf_rerank,
    "hybrid+rrf": hybrid_rrf,
}
