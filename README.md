# MovieMate: RAG + SQL Tool Movie Assistant

MovieMate combines semantic retrieval (FAISS) with structured retrieval (SQLite tool-calling), then generates grounded responses in a Streamlit app.

## Features
- RAG search over movie embeddings
- SQL tool retrieval for strict constraints
- Hybrid retrieval (`rag+tool`) with strict routing
- Streamlit dashboard with Chat, EDA, and Evaluation
- Benchmark runner with CSV diagnostics export

## Project Structure
- `src/scrape.py`: data scraping + enrichment
- `src/preprocess.py`: metadata cleanup (trailers, keywords)
- `src/embed.py`: embedding + FAISS index build/update
- `src/similarity_search.py`: vector retrieval
- `src/tool_call.py`: SQL tool retrieval
- `src/agent.py`: retrieval orchestration + answer pipeline
- `src/evaluation.py`: benchmark and metrics
- `src/eda.py`: EDA summary + plots
- `src/dashboard.py`: Streamlit UI
- `src/main.py`: CLI entrypoint

## Setup
```bash
uv sync
```

Add environment values in `.env` (as needed):
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL_NAME`
- `TMDB_API_KEY`

## Run Commands

### Dashboard
```bash
uv run moviemate
# or
uv run python -m src.main --step dashboard
```

### Scraping
```bash
uv run python -m src.main --step scrape --tmdb_api_key YOUR_TMDB_KEY
```

### Preprocess
```bash
uv run python -m src.main --step preprocess
```

### Embedding / FAISS
```bash
uv run python -m src.main --step embed
```

### EDA
```bash
uv run python -m src.main --step eda
```

### Similarity Search (CLI)
```bash
uv run python -m src.main --step search --query "sci-fi movies after 2010"
```

### Agent Run (Hybrid)
```bash
uv run python -m src.main --step agent --query "drama movies directed by Nolan"
```

## EDA Plot

![EDA Plot](assets/eda.png)


### Evaluation
```bash
uv run python -m src.main --step eval
```

Evaluation prompt sources:
- `data/english_test_prompts.json`
- `data/hindi_test_prompts.json`

### Run EDA + Evaluation
```bash
uv run python -m src.main --step all
```

## Evaluation Outputs
Running `--step eval` now exports:
- `assets/benchmark_summary_hindi_english.csv`
- `assets/benchmark_detail_hindi_english.csv`
- `assets/failures_hindi_english.csv`
- `assets/benchmark_plot_hindi_english.png`
- `assets/benchmark_summary_english_only.csv`
- `assets/benchmark_detail_english_only.csv`
- `assets/failures_english_only.csv`
- `assets/benchmark_plot_english_only.png`
- `assets/eda.png`

## Hindi + English Metrics

- Best Accuracy: 88.18%
- Best Precision@K: 87.58%
- Best MRR: 0.8758
- Lowest Hallucination: 11.82%
- Best Tool Accuracy: 88.18%

| mode | avg_latency_ms | p95_latency_ms | avg_hits | accuracy_pct | precision_at_k | mrr | hallucination_rate | tool_call_accuracy | faithfulness_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag | 249.42 | 316.33 | 2.65 | 77.27 | 66.67 | 0.7121 | 22.73 | 0.00 | 70.91 |
| rag+tool | 1811.91 | 1191.23 | 2.71 | 88.18 | 86.36 | 0.8712 | 11.82 | 80.00 | 87.09 |
| tool | 1865.62 | 1152.68 | 2.66 | 88.18 | 87.58 | 0.8758 | 11.82 | 88.18 | 87.82 |

![Hindi+English Benchmark Plot](assets/benchmark_plot_hindi_english.png)

## English-only Metrics

- Best Accuracy: 91.00%
- Best Precision@K: 90.33%
- Best MRR: 0.9033
- Lowest Hallucination: 9.00%
- Best Tool Accuracy: 91.00%

| mode | avg_latency_ms | p95_latency_ms | avg_hits | accuracy_pct | precision_at_k | mrr | hallucination_rate | tool_call_accuracy | faithfulness_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag | 260.25 | 356.91 | 2.62 | 81.00 | 71.00 | 0.7483 | 19.00 | 0.00 | 75.00 |
| rag+tool | 1918.54 | 1081.38 | 2.68 | 91.00 | 90.33 | 0.9033 | 9.00 | 85.00 | 90.60 |
| tool | 1956.79 | 1227.46 | 2.66 | 91.00 | 90.33 | 0.9033 | 9.00 | 91.00 | 90.60 |

![English-only Benchmark Plot](assets/benchmark_plot_english_only.png)

## Reflection
- The major evaluation bug was fixed: pass/fail now checks **any hit in merged results**, not only rank-1.
- Hybrid mode (`rag+tool`) is much faster than pure RAG while preserving strong precision.
- Tool mode is still best for strict constraints and lowest latency.
- Failure diagnostics are now explicit (`fail_genre`, `fail_year`, `fail_rating`, `fail_duration`, `fail_director`) in benchmark detail outputs for faster debugging.

## Notes
- If dashboard starts in browser directly, use `uv run moviemate`.
- If models/index changed significantly, rerun `preprocess` + `embed` before evaluation.
