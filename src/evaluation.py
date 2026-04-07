import json
import time
from itertools import cycle, islice
from pathlib import Path

import pandas as pd


DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "data" / "test_prompts.json"


def load_test_prompts(path=DEFAULT_PROMPTS_PATH):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("prompts", [])
    return data if isinstance(data, list) else []


def _constraint_pass(row, constraints):
    if row is None or row.empty:
        return False

    r = row.iloc[0]
    genre_ok = True
    if constraints.get("genre_contains"):
        genre_ok = constraints["genre_contains"].lower() in str(r.get("genre", "")).lower()

    year_ok = True
    if constraints.get("min_year") is not None:
        year_ok = pd.notna(r.get("year")) and float(r.get("year")) >= float(constraints["min_year"])
    if constraints.get("max_year") is not None:
        year_ok = year_ok and pd.notna(r.get("year")) and float(r.get("year")) <= float(constraints["max_year"])

    rating_ok = True
    if constraints.get("min_rating") is not None:
        rating_ok = pd.notna(r.get("rating")) and float(r.get("rating")) >= float(constraints["min_rating"])

    duration_ok = True
    if constraints.get("max_duration") is not None:
        duration_ok = pd.notna(r.get("duration")) and float(r.get("duration")) <= float(constraints["max_duration"])

    director_ok = True
    if constraints.get("director_contains"):
        director_ok = constraints["director_contains"].lower() in str(r.get("director", "")).lower()

    return genre_ok and year_ok and rating_ok and duration_ok and director_ok


def run_benchmarks(agent, prompt_path=DEFAULT_PROMPTS_PATH, total_cases=100, modes=("rag", "tool", "rag+tool")):
    prompts = load_test_prompts(prompt_path)
    if not prompts:
        return pd.DataFrame(), pd.DataFrame()

    selected = list(islice(cycle(prompts), total_cases))
    rows = []
    for mode in modes:
        for item in selected:
            q = item.get("question", "")
            expected_constraints = item.get("expected_constraints", {})
            t0 = time.perf_counter()
            rag_hits, tool_hits, merged = agent._retrieve(q, mode=mode, k=8)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            ok = _constraint_pass(merged, expected_constraints)
            rows.append(
                {
                    "mode": mode,
                    "question": q,
                    "expected_answer": item.get("expected_answer", ""),
                    "latency_ms": round(elapsed_ms, 2),
                    "rag_hits": len(rag_hits),
                    "tool_hits": len(tool_hits),
                    "merged_hits": len(merged),
                    "passed": bool(ok),
                    "top_title": None if merged.empty else merged.iloc[0].get("title"),
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("mode", as_index=False)
        .agg(
            avg_latency_ms=("latency_ms", "mean"),
            p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
            avg_hits=("merged_hits", "mean"),
            pass_rate=("passed", "mean"),
        )
        .sort_values("avg_latency_ms")
        .reset_index(drop=True)
    )
    summary["avg_latency_ms"] = summary["avg_latency_ms"].round(2)
    summary["p95_latency_ms"] = summary["p95_latency_ms"].round(2)
    summary["avg_hits"] = summary["avg_hits"].round(2)
    summary["pass_rate"] = (summary["pass_rate"] * 100).round(2)
    return detail, summary


def benchmark_markdown(summary_df, total_cases):
    if summary_df is None or summary_df.empty:
        return "### Benchmark Summary\n\nNo benchmark results available."
    table_text = summary_df.to_string(index=False)
    return (
        f"### Benchmark Summary\n\n"
        f"- cases per mode: {total_cases}\n"
        f"- modes: rag, tool, rag+tool\n\n"
        + "```text\n"
        + table_text
        + "\n```"
    )
