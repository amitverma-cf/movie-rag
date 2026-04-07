import html
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.agent import MovieAgent
from src.eda import build_eda_artifacts
from src.evaluation import run_benchmarks


def _youtube_embed_url(url):
    if not url:
        return ""
    parsed = urlparse(str(url).strip())
    if "youtube.com" in parsed.netloc:
        vid = parse_qs(parsed.query).get("v", [""])[0]
        return f"https://www.youtube.com/embed/{vid}" if vid else ""
    if "youtu.be" in parsed.netloc:
        vid = parsed.path.strip("/")
        return f"https://www.youtube.com/embed/{vid}" if vid else ""
    return ""


def _movie_card(row):
    title = html.escape(str(row.get("title", "Unknown")))
    year = html.escape(str(row.get("year", "NA")))
    rating = html.escape(str(row.get("rating", "NA")))
    genre = html.escape(str(row.get("genre", "NA")))
    director = html.escape(str(row.get("director", "NA")))
    cast = html.escape(str(row.get("cast", "NA")))
    description = html.escape(str(row.get("description", "No description.")))
    source = html.escape(str(row.get("source", "N/A")))
    poster_url = str(row.get("poster_url", "") or "").strip()
    trailer_url = str(row.get("trailer_url", "") or "").strip()
    embed = _youtube_embed_url(trailer_url)

    poster = (
        f"<img src='{html.escape(poster_url)}' alt='poster' style='width:130px;height:195px;object-fit:cover;border-radius:12px;border:1px solid #264653;'>"
        if poster_url
        else "<div style='width:130px;height:195px;border-radius:12px;border:1px dashed #264653;display:flex;align-items:center;justify-content:center;'>No Poster</div>"
    )
    trailer = (
        f"<iframe src='{html.escape(embed)}' title='Trailer' width='100%' height='220' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture' allowfullscreen></iframe>"
        if embed
        else (f"<a href='{html.escape(trailer_url)}' target='_blank'>Open Trailer</a>" if trailer_url else "No trailer available")
    )

    return f"""
<div style='background:linear-gradient(135deg,#8e8ae0,#e9358b);border:1px solid #90be6d;border-radius:16px;padding:14px;margin:10px 0;'>
  <div style='display:flex;gap:14px;flex-wrap:wrap;'>
    <div>{poster}</div>
    <div style='flex:1;min-width:260px;'>
      <h3 style='margin:0 0 8px 0;color:#1d3557;'>{title} ({year})</h3>
      <div style='font-size:14px;line-height:1.6;'>
        <b>Rating:</b> {rating} | <b>Genre:</b> {genre} | <b>Source:</b> {source}<br>
        <b>Director:</b> {director}<br>
        <b>Cast:</b> {cast}<br>
        <b>Description:</b> {description}
      </div>
    </div>
  </div>
  <div style='margin-top:12px;'>{trailer}</div>
</div>
"""


def _cards_html(df):
    if df is None or df.empty:
        return "<div style='padding:10px;border:1px dashed #90be6d;border-radius:10px;'>No movie cards available.</div>"
    return "".join(_movie_card(row) for _, row in df.iterrows())


def _benchmark_plot(summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if summary_df is None or summary_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No evaluation data", ha="center", va="center")
            ax.axis("off")
        plt.tight_layout()
        return fig

    axes[0].bar(summary_df["mode"], summary_df["avg_latency_ms"], color=["#457b9d", "#e76f51", "#2a9d8f"])
    axes[0].set_title("Avg Latency (ms)")
    axes[0].grid(axis="y", alpha=0.25)

    if "accuracy_pct" in summary_df.columns:
        acc_series = summary_df["accuracy_pct"]
    elif "pass_rate" in summary_df.columns:
        acc_series = summary_df["pass_rate"]
    else:
        acc_series = pd.Series([0] * len(summary_df))
    axes[1].bar(summary_df["mode"], acc_series, color=["#264653", "#f4a261", "#2a9d8f"])
    axes[1].set_title("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    return fig


def _ensure_agent():
    if "moviemate_agent" not in st.session_state:
        st.session_state.moviemate_agent = MovieAgent()
    return st.session_state.moviemate_agent


def _render_chat(agent):
    st.subheader("Chat")
    mode = st.radio("Retrieval Mode", ["rag", "tool", "rag+tool"], horizontal=True, index=2)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_cards" not in st.session_state:
        st.session_state.last_cards = ""
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {}

    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    prompt = st.chat_input("Ask about movies")
    if prompt:
        result = agent.run(prompt, mode=mode, history=st.session_state.chat_history)
        st.session_state.chat_history.append((prompt, result["answer"]))
        st.session_state.last_cards = _cards_html(result["movies"])
        st.session_state.last_metrics = {
            "mode": mode,
            "rag_hits": result.get("rag_count", 0),
            "tool_hits": result.get("tool_count", 0),
            **(result.get("analytics") or {}),
        }
        st.rerun()

    metrics = st.session_state.last_metrics
    if metrics:
        st.markdown("### Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mode", str(metrics.get("mode", "n/a")))
        c2.metric("Latency (s)", str(metrics.get("latency_sec", "n/a")))
        c3.metric("Total Tokens", str(metrics.get("total_tokens", "n/a")))
        c4.metric("Accuracy Proxy (hits)", str(int(metrics.get("rag_hits", 0)) + int(metrics.get("tool_hits", 0))))

    st.markdown("### Movie Cards")
    st.markdown(st.session_state.last_cards or _cards_html(pd.DataFrame()), unsafe_allow_html=True)


def _render_eda(agent):
    st.subheader("EDA")
    if st.button("Refresh EDA"):
        summary, fig = build_eda_artifacts(agent.movies)
        st.session_state.eda_summary = summary.reset_index(names=["feature"])
        st.session_state.eda_plot = fig

    if "eda_summary" not in st.session_state or "eda_plot" not in st.session_state:
        summary, fig = build_eda_artifacts(agent.movies)
        st.session_state.eda_summary = summary.reset_index(names=["feature"])
        st.session_state.eda_plot = fig

    st.dataframe(st.session_state.eda_summary, use_container_width=True)
    st.pyplot(st.session_state.eda_plot, clear_figure=False)


def _render_evaluation(agent):
    st.subheader("Evaluation")
    cases = st.number_input("Cases per mode", min_value=10, max_value=300, value=100, step=10)

    if st.button("Run Evaluation"):
        detail, summary = run_benchmarks(agent, total_cases=int(cases))
        summary = summary.copy()
        if "pass_rate" in summary.columns:
            summary = summary.rename(columns={"pass_rate": "accuracy_pct"})
        st.session_state.eval_detail = detail
        st.session_state.eval_summary = summary
        st.session_state.eval_plot = _benchmark_plot(summary)

    if "eval_summary" in st.session_state:
        st.markdown("### Comparison: RAG vs Tool Call vs RAG+Tool Call")
        st.dataframe(st.session_state.eval_summary, use_container_width=True)
        st.pyplot(st.session_state.eval_plot, clear_figure=False)
        with st.expander("Detailed Results"):
            st.dataframe(st.session_state.eval_detail, use_container_width=True)


def render_dashboard():
    st.set_page_config(page_title="MovieMate Dashboard", page_icon=":movie_camera:", layout="wide")

    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #8481de 0%, #5e8ae0 100%); }
        .sidebar-card { background:#1d3557;color:#f88aee;border-radius:14px;padding:12px;margin-bottom:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## MovieMate")
        st.markdown("<div class='sidebar-card'>Navigate views and compare retrieval quality.</div>", unsafe_allow_html=True)
        view = st.radio("Sidebar Tabs", ["Chat", "EDA", "Evaluation"], index=0)

    agent = _ensure_agent()

    if view == "Chat":
        _render_chat(agent)
    elif view == "EDA":
        _render_eda(agent)
    else:
        _render_evaluation(agent)


def launch_dashboard():
    # If not running inside streamlit runtime, spawn streamlit process.
    if not any("streamlit" in arg.lower() for arg in sys.argv):
        subprocess.run(
            [
                "streamlit",
                "run",
                str(Path(__file__).resolve()),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            check=False,
        )
        return
    render_dashboard()


if __name__ == "__main__":
    render_dashboard()
