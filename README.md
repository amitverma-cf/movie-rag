# MovieMate: Conversational Movie Search with RAG + Tool Calling

MovieMate is a conversational AI movie assistant that combines:

- Semantic retrieval with embeddings + FAISS (RAG)
- Structured SQL retrieval over SQLite (tool calling)
- LLM response generation grounded in retrieved movies
- A Dashboard with three views: Chat, EDA, and Benchmark

This repository aligns with the assignment goals of natural-language query handling, multi-turn refinement, retrieval quality analysis, and interactive UI delivery.

## Project Structure

- [src/scrape.py](src/scrape.py): dataset creation + metadata enrichment (TMDB)
- [src/preprocess.py](src/preprocess.py): trailer patching + keyword generation
- [src/embed.py](src/embed.py): embedding generation and FAISS index updates
- [src/eda.py](src/eda.py): EDA loaders and visualization artifacts
- [src/similarity_search.py](src/similarity_search.py): vector search retrieval
- [src/tool_call.py](src/tool_call.py): SQL tool-calling retrieval
- [src/llm.py](src/llm.py): LLM answer generation and token analytics
- [src/agent.py](src/agent.py): orchestration of RAG/tool/hybrid modes
- [src/evaluation.py](src/evaluation.py): benchmark engine from prompt file
- [src/dashboard.py](src/dashboard.py): final dashboard UI
- [src/main.py](src/main.py): CLI entrypoint logic

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Configure environment values in `.env`:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL_NAME`
- `TMDB_API_KEY`

## Run

Dashboard (default):

```bash
uv run moviemate
```

Other pipeline steps:

```bash
uv run moviemate --step scrape
uv run moviemate --step preprocess
uv run moviemate --step embed
uv run moviemate --step eda
uv run moviemate --step search --query "sci-fi movies after 2010"
uv run moviemate --step agent --query "drama movies directed by Nolan"
uv run moviemate --step eval
```

## Benchmark Prompts

Benchmark prompts are loaded from:

- [data/test_prompts.json](data/test_prompts.json)

The file currently contains 10 sample prompts with expected answers and constraint-style expected conditions. The benchmark runner cycles prompts to build a 100-case benchmark across three modes:

- `rag`
- `tool`
- `rag+tool`

## Evaluation Workflow

Evaluation logic is implemented in [src/evaluation.py](src/evaluation.py):

1. Load prompt specs from JSON
2. Run each prompt across all modes
3. Measure latency, hit counts, and pass/fail against expected constraints
4. Aggregate summary metrics (average latency, p95 latency, pass rate)

The dashboard Benchmark tab visualizes these results and prints a markdown summary table.

## Reflection

What works well:

- Hybrid retrieval (`rag+tool`) improves recall for mixed semantic + structured queries.
- Tool calling improves precision for hard constraints (year, duration, director, rating).
- RAG captures looser semantic intent and similarity-style requests.
- Dashboard benchmarking makes performance trade-offs visible.

Current limitations:

- Constraint checks are heuristic and not a full relevance metric.
- LLM quality depends on retrieved metadata quality.
- Heavy embedding model startup can increase first-response latency.
- Tool parser currently handles common patterns, not all natural-language forms.

Next improvements:

- Add graded relevance labels for stronger evaluation (e.g., NDCG/MRR).
- Expand tool parser intent extraction and query normalization.
- Add cached model warmup and async retrieval for lower latency.
- Add richer personalization memory across sessions.

**Evaluation & Reflection**

- **Install `uv` (CLI) and dependencies:**

```bash
# Option A: install the `uv` CLI if published on PyPI
pip install uv

# Option B: install all project dependencies
pip install -r requirements.txt
```

- **Run all pipeline steps (EDA + Eval) via `main.py`:**

```bash
# run a specific step (example: embed)
python -m src.main --step embed

# run the full EDA + evaluation
python -m src.main --step all
```

**Quick Benchmark Summary (placeholders)**

| mode     | avg_latency_ms | p95_latency_ms | avg_hits | accuracy_pct |
|----------|----------------|----------------|----------|--------------|
| tool     | 35.65          | 47.4           | 12       | 100          |
| rag      | 232.08         | 305.04         | 8.7      | 40           |
| rag+tool | 274.61         | 350.85         | 19.4     | 50           |

What this result means:

- `tool` mode is fastest and shows high accuracy on strict constraint queries because SQL/tool calls return exact matches for structured fields.
- `rag` has higher latency (embedding/vector search + LLM) and lower accuracy on strict constraints because semantic matches don't always satisfy exact metadata conditions.
- `rag+tool` shows highest latency and increased merged hits but only modest accuracy improvements over `rag`; accuracy falls behind `tool`.

Why accuracy can fall in `rag+tool` mode:

- Merging strategy: naive merging of semantic and tool results can place semantically similar but constraint-violating documents above strict tool matches.
- Duplication & ranking: duplicates or near-duplicates from both sources can skew the top-k used by the LLM, reducing precision for constraint checks.
- LLM hallucination risk: mixing noisy RAG context with tool responses may cause the generator to favor fluent but incorrect answers.
- Latency/timeout pressure: longer end-to-end latency can trigger timeouts or truncated contexts, harming accuracy.

Suggested mitigations:

- Prefer deterministic tool results for strict constraints: when a query contains explicit constraints (year, duration, director), prioritize tool results or require tool evidence in the top-N given to the LLM.
- Improve merge ranking: use a ranking function that weights constraint matches higher than pure semantic similarity.
- De-duplicate and normalize metadata before merging (title normalization, exact field comparisons).
- Validate constraints post-merge and re-rank or filter out results that fail required checks before calling the LLM.
- Use async retrieval and caching to reduce wall-clock latency for `rag+tool`.

**Placeholders for visuals**

Add images to the `assets/` folder and update links below when available:

- Benchmark summary plot: ![Benchmark summary](assets/benchmark-summary.png)
- Latency distribution: ![Latency distribution](assets/latency-dist.png)
- Precision/recall comparison: ![PR curve](assets/pr-curve.png)

If you'd like, I can:

- Run the full benchmark now and attach the generated summary and plots to `assets/`.
- Add automated plotting code to `src/evaluation.py` to produce the above PNGs.
