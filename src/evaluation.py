import json
import time
from itertools import cycle, islice
from typing import Optional

import pandas as pd

from src.config import DEFAULT_TOP_K, ENGLISH_TEST_PROMPTS_PATH, HINDI_TEST_PROMPTS_PATH


def _read_prompt_file(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("prompts", [])
    return data if isinstance(data, list) else []


def load_test_prompts(path=None, prompt_language="hindi+english"):
    if path is not None:
        return _read_prompt_file(path)

    mode = (prompt_language or "hindi+english").strip().lower()
    english = _read_prompt_file(ENGLISH_TEST_PROMPTS_PATH)
    hindi = _read_prompt_file(HINDI_TEST_PROMPTS_PATH)
    if mode == "english_only":
        return english
    if mode == "hindi_only":
        return hindi
    return english + hindi


def _constraint_pass(rows, constraints):
    if rows is None or rows.empty:
        return False
    for _, row in rows.iterrows():
        if _constraint_breakdown(row.to_dict(), constraints)["all"]:
            return True
    return False


def _constraint_breakdown(row_dict, constraints):
    checks = {
        "genre": True,
        "year": True,
        "rating": True,
        "duration": True,
        "director": True,
    }

    if constraints.get("genre_contains"):
        checks["genre"] = constraints["genre_contains"].lower() in str(row_dict.get("genre", "")).lower()

    if constraints.get("min_year") is not None:
        y = row_dict.get("year")
        checks["year"] = pd.notna(y) and float(y) >= float(constraints["min_year"])
    if constraints.get("max_year") is not None:
        y = row_dict.get("year")
        checks["year"] = checks["year"] and pd.notna(y) and float(y) <= float(constraints["max_year"])

    if constraints.get("min_rating") is not None:
        r = row_dict.get("rating")
        checks["rating"] = pd.notna(r) and float(r) >= float(constraints["min_rating"])

    if constraints.get("max_duration") is not None:
        d = row_dict.get("duration")
        checks["duration"] = pd.notna(d) and float(d) <= float(constraints["max_duration"])

    if constraints.get("director_contains"):
        checks["director"] = constraints["director_contains"].lower() in str(row_dict.get("director", "")).lower()

    checks["all"] = all(checks.values())
    return checks


def _retrieval_metrics(merged_df, constraints, k=8):
    if merged_df is None or merged_df.empty:
        return {
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "hallucination_rate": 1.0,
            "faithfulness_score": 0.0,
        }

    top = merged_df.head(k)
    flags = [
        _constraint_breakdown(row._asdict() if hasattr(row, "_asdict") else row.to_dict(), constraints)["all"]
        for _, row in top.iterrows()
    ]
    k_used = max(len(flags), 1)
    relevant = sum(1 for f in flags if f)
    precision_at_k = relevant / k_used

    first_rel_rank = next((idx + 1 for idx, f in enumerate(flags) if f), None)
    mrr = 1.0 / first_rel_rank if first_rel_rank else 0.0

    hallucination_rate = 0.0 if relevant > 0 else 1.0
    faithfulness_score = 0.6 * precision_at_k + 0.4 * (1.0 - hallucination_rate)

    return {
        "precision_at_k": round(precision_at_k, 4),
        "mrr": round(mrr, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "faithfulness_score": round(faithfulness_score, 4),
    }


def _tool_call_accuracy(tool_hits_df, constraints, k=DEFAULT_TOP_K):
    if tool_hits_df is None or tool_hits_df.empty:
        return 0.0
    top = tool_hits_df.head(int(k))
    if top.empty:
        return 0.0
    for _, row in top.iterrows():
        if _constraint_breakdown(row.to_dict(), constraints)["all"]:
            return 1.0
    return 0.0


def run_benchmarks(
    agent,
    prompt_path=None,
    total_cases: Optional[int] = None,
    modes=("rag", "tool", "rag+tool"),
    prompt_language="hindi+english",
    k: int = DEFAULT_TOP_K,
):
    prompts = load_test_prompts(path=prompt_path, prompt_language=prompt_language)
    if not prompts:
        return pd.DataFrame(), pd.DataFrame()

    # If total_cases is None, run once over every prompt in the file (no cycling)
    if total_cases is None:
        selected = list(prompts)
        total_cases = len(selected)
    else:
        selected = list(islice(cycle(prompts), total_cases))
    rows = []
    for mode in modes:
        for item in selected:
            q = item.get("question", "")
            expected_constraints = item.get("expected_constraints", {})
            t0 = time.perf_counter()
            rag_hits, tool_hits, merged = agent._retrieve(q, mode=mode, k=int(k))
            elapsed_ms = (time.perf_counter() - t0) * 1000

            ok = _constraint_pass(merged, expected_constraints)
            rmetrics = _retrieval_metrics(merged, expected_constraints, k=int(k))
            tool_acc = _tool_call_accuracy(tool_hits, expected_constraints, k=int(k))
            top_row = {} if merged.empty else merged.iloc[0].to_dict()
            fail_breakdown = _constraint_breakdown(top_row, expected_constraints)
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
                    "precision_at_k": rmetrics["precision_at_k"],
                    "mrr": rmetrics["mrr"],
                    "hallucination_rate": rmetrics["hallucination_rate"],
                    "tool_call_accuracy": tool_acc,
                    "faithfulness_score": rmetrics["faithfulness_score"],
                    "top_title": None if merged.empty else merged.iloc[0].get("title"),
                    "fail_genre": bool(not fail_breakdown["genre"]),
                    "fail_year": bool(not fail_breakdown["year"]),
                    "fail_rating": bool(not fail_breakdown["rating"]),
                    "fail_duration": bool(not fail_breakdown["duration"]),
                    "fail_director": bool(not fail_breakdown["director"]),
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
            precision_at_k=("precision_at_k", "mean"),
            mrr=("mrr", "mean"),
            hallucination_rate=("hallucination_rate", "mean"),
            tool_call_accuracy=("tool_call_accuracy", "mean"),
            faithfulness_score=("faithfulness_score", "mean"),
        )
        .sort_values("avg_latency_ms")
        .reset_index(drop=True)
    )
    summary["avg_latency_ms"] = summary["avg_latency_ms"].round(2)
    summary["p95_latency_ms"] = summary["p95_latency_ms"].round(2)
    summary["avg_hits"] = summary["avg_hits"].round(2)
    summary["pass_rate"] = (summary["pass_rate"] * 100).round(2)
    summary["precision_at_k"] = (summary["precision_at_k"] * 100).round(2)
    summary["mrr"] = summary["mrr"].round(4)
    summary["hallucination_rate"] = (summary["hallucination_rate"] * 100).round(2)
    summary["tool_call_accuracy"] = (summary["tool_call_accuracy"] * 100).round(2)
    summary["faithfulness_score"] = (summary["faithfulness_score"] * 100).round(2)
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
