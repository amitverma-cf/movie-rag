import html
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.eda import build_eda_artifacts
from src.evaluation import run_benchmarks


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _chat_artifact_paths():
    return {
        "history": ASSETS_DIR / "chat_history.csv",
        "suggestions": ASSETS_DIR / "chat_suggestions.csv",
    }


def _load_chat_history_from_assets():
    path = _chat_artifact_paths()["history"]
    if not path.exists():
        return []
    df = pd.read_csv(path, keep_default_na=False)
    history = []
    for _, row in df.iterrows():
        history.append(
            (
                str(row.get("user_message", "")),
                {
                    "text": str(row.get("assistant_text", "")),
                    "cards_html": str(row.get("cards_html", "")),
                },
            )
        )
    return history


def _save_chat_history_to_assets(chat_history):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for user_msg, bot_msg in chat_history:
        if isinstance(bot_msg, dict):
            assistant_text = str(bot_msg.get("text", ""))
            cards_html = str(bot_msg.get("cards_html", ""))
        else:
            assistant_text = str(bot_msg)
            cards_html = ""
        rows.append(
            {
                "user_message": str(user_msg),
                "assistant_text": assistant_text,
                "cards_html": cards_html,
            }
        )
    pd.DataFrame(rows).to_csv(_chat_artifact_paths()["history"], index=False)


def _load_suggestions_from_assets():
    path = _chat_artifact_paths()["suggestions"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def _save_suggestions_to_assets(df):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy() if df is not None else pd.DataFrame()
    out.to_csv(_chat_artifact_paths()["suggestions"], index=False)


def _clear_chat_artifacts():
    for path in _chat_artifact_paths().values():
        if path.exists():
            path.unlink()


def _compact_agent_history(chat_history):
    compact = []
    for user_msg, bot_msg in chat_history:
        if isinstance(bot_msg, dict):
            compact.append((user_msg, str(bot_msg.get("text", ""))))
        else:
            compact.append((user_msg, str(bot_msg)))
    return compact


def _suggestions_from_history(agent, chat_history, k=6):
    user_queries = [str(u).strip() for u, _ in chat_history if str(u).strip()]
    if not user_queries:
        return pd.DataFrame()
    query = " ".join(user_queries[-6:])
    _, _, merged = agent._retrieve(query, mode="rag+tool", k=int(k))
    return merged.head(int(k)).reset_index(drop=True)


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
    description = html.escape(str(row.get("description", "No description available.")))
    poster_url = str(row.get("poster_url", "") or "").strip()
    trailer_url = str(row.get("trailer_url", "") or "").strip()
    embed = _youtube_embed_url(trailer_url)

    poster_html = (
        f"<img src='{html.escape(poster_url)}' style='aspect-ratio: 3/4;height:240px;object-fit:cover;border-radius:8px;'>"
        if poster_url else "<div style='width:100%;height:240px;background:#1e293b;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#64748b;'>No Poster</div>"
    )
    
    trailer_html = (
        f"<iframe src='{html.escape(embed)}' style='aspect-ratio: 4/3;height:240px;border-radius:8px;border:none;' allowfullscreen></iframe>"
        if embed else f"<div style='margin-top:10px;text-align:center;'><a href='{html.escape(trailer_url)}' target='_blank' style='color:#38bdf8;text-decoration:none;font-size:13px;'>View Trailer</a></div>" if trailer_url else ""
    )

    return f"""
<div style='
    background:#0f172a;
    border:1px solid #334155;
    border-radius:12px;
    padding:12px;
    color:#f1f5f9;
    display:flex;
    flex-direction:column;
    gap:12px;
'>
    <div style='
        display:flex;
        flex-direction:row;
        gap:16px;
    '>
        <div style="height:240px; aspect-ratio: 3/4; overflow:hidden; border-radius:8px;">
            {poster_html}
        </div>
        <div style="height:240px; aspect-ratio: 4/3; border-radius:8px; overflow:hidden;">
            {trailer_html}
        </div>
    </div>
    <div style='
        border-top:1px solid #1e293b;
        padding-top:10px;
    '>
        <h4 style='
            margin:0 0 6px 0;
            font-size:16px;
            color:#f8fafc;
        '>
            {title} ({year})
        </h4>
        <div style='
            font-size:12px;
            color:#94a3b8;
            line-height:1.5;
        '>
            <b>Rating:</b> {rating} &nbsp; | &nbsp;
            <b>Genre:</b> {genre}<br>
            <b>Director:</b> {director}<br>
            <b>Cast:</b> {cast}  <br>
            <p style='margin-top:8px;'>{description}</p>
        </div>
    </div>
</div>
"""


def _cards_html(df):
    if df is None or df.empty:
        return ""
    cards = [_movie_card(row) for _, row in df.iterrows()]
    rows = []
    for i in range(0, len(cards), 2):
        left = cards[i]
        right = cards[i + 1] if i + 1 < len(cards) else ""
        rows.append(
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;'>"
            f"<div>{left}</div><div>{right}</div></div>"
        )
    return f"<div style='margin-top:20px;'>{''.join(rows)}</div>"


def _benchmark_plot(summary_df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    if summary_df is None or summary_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No evaluation data", ha="center", va="center")
            ax.axis("off")
        plt.tight_layout()
        return fig

    def _bar(ax, col, title, pct=False):
        if col in summary_df.columns:
            vals = summary_df[col]
        else:
            vals = pd.Series([0] * len(summary_df))
        ax.bar(summary_df["mode"], vals, color=["#457b9d", "#e76f51", "#2a9d8f"])
        ax.set_title(title)
        if pct:
            ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)

    _bar(axes[0], "avg_latency_ms", "Avg Latency (ms)")
    _bar(axes[1], "accuracy_pct", "Accuracy (%)", pct=True)
    _bar(axes[2], "precision_at_k", "Precision@K (%)", pct=True)
    _bar(axes[3], "mrr", "MRR")
    _bar(axes[4], "hallucination_rate", "Hallucination Rate (%)", pct=True)
    _bar(axes[5], "faithfulness_score", "Faithfulness (%)", pct=True)

    plt.tight_layout()
    return fig


def _eda_artifact_paths():
    return {
        "summary": ASSETS_DIR / "eda_summary.csv",
        "plot": ASSETS_DIR / "eda.png",
    }


def _load_eda_from_assets():
    paths = _eda_artifact_paths()
    if not paths["summary"].exists() or not paths["plot"].exists():
        return None
    return pd.read_csv(paths["summary"])


def _save_eda_to_assets(summary_df, fig):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    paths = _eda_artifact_paths()
    summary_df.to_csv(paths["summary"], index=False)
    fig.savefig(paths["plot"], dpi=160, bbox_inches="tight")


def _eval_artifact_paths(prompt_language):
    suffix = "english_only" if prompt_language == "english_only" else "hindi_english"
    return {
        "summary": ASSETS_DIR / f"benchmark_summary_{suffix}.csv",
        "detail": ASSETS_DIR / f"benchmark_detail_{suffix}.csv",
        "failures": ASSETS_DIR / f"failures_{suffix}.csv",
        "plot": ASSETS_DIR / f"benchmark_plot_{suffix}.png",
    }


def _load_eval_from_assets(prompt_language):
    paths = _eval_artifact_paths(prompt_language)
    if not paths["summary"].exists() or not paths["detail"].exists():
        return None, None

    summary = pd.read_csv(paths["summary"])
    detail = pd.read_csv(paths["detail"])
    if "pass_rate" in summary.columns and "accuracy_pct" not in summary.columns:
        summary = summary.rename(columns={"pass_rate": "accuracy_pct"})
    return detail, summary


def _save_eval_to_assets(detail, summary, prompt_language, fig):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    paths = _eval_artifact_paths(prompt_language)
    detail.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    detail[detail["passed"] == False].to_csv(paths["failures"], index=False)
    fig.savefig(paths["plot"], dpi=160, bbox_inches="tight")


@st.cache_resource(show_spinner=False)
def _get_cached_agent():
    from src.agent import MovieAgent

    return MovieAgent()


@st.cache_data(show_spinner=False)
def _cached_eda(_movies_df):
    summary, fig = build_eda_artifacts(_movies_df)
    return summary.reset_index(names=["feature"]), fig


def _ensure_agent():
    return _get_cached_agent()


def _render_chat(agent):
    st.subheader("Chat")
    mode_col, clear_col = st.columns([5, 1])
    with mode_col:
        mode = st.radio("", ["rag", "tool", "rag+tool"], horizontal=True, index=2)
    with clear_col:
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_suggestions = pd.DataFrame()
            _clear_chat_artifacts()
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = _load_chat_history_from_assets()
    if "chat_suggestions" not in st.session_state:
        st.session_state.chat_suggestions = _load_suggestions_from_assets()
        if st.session_state.chat_suggestions.empty and st.session_state.chat_history:
            st.session_state.chat_suggestions = _suggestions_from_history(agent, st.session_state.chat_history)
            _save_suggestions_to_assets(st.session_state.chat_suggestions)
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {}

    if not st.session_state.chat_suggestions.empty:
        st.markdown("### Suggested For You")
        st.markdown(_cards_html(st.session_state.chat_suggestions), unsafe_allow_html=True)

    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            if isinstance(bot_msg, dict):
                st.markdown(str(bot_msg.get("text", "")))
                cards_html = str(bot_msg.get("cards_html", "") or "").strip()
                if cards_html:
                    st.markdown(cards_html, unsafe_allow_html=True)
            else:
                bot_text = str(bot_msg)
                split_token = "\n\n<div style='margin-top:20px;'"
                if split_token in bot_text:
                    text_part, html_part = bot_text.split(split_token, 1)
                    st.markdown(text_part)
                    st.markdown("<div style='margin-top:20px;'" + html_part, unsafe_allow_html=True)
                else:
                    st.markdown(bot_text)

    prompt = st.chat_input("Message MovieMate")
    if prompt:
        result = agent.run(prompt, mode=mode, history=_compact_agent_history(st.session_state.chat_history))
        
        # Merge text and cards into one assistant bubble
        full_response = result["answer"]
        cards_html = _cards_html(result["movies"])
        st.session_state.chat_history.append(
            (
                prompt,
                {
                    "text": full_response,
                    "cards_html": cards_html,
                },
            )
        )
        _save_chat_history_to_assets(st.session_state.chat_history)
        st.session_state.chat_suggestions = _suggestions_from_history(agent, st.session_state.chat_history)
        _save_suggestions_to_assets(st.session_state.chat_suggestions)
        st.session_state.last_metrics = {
            "mode": mode,
            "rag_hits": result.get("rag_count", 0),
            "tool_hits": result.get("tool_count", 0),
            **(result.get("analytics") or {}),
        }
        st.rerun()

    metrics = st.session_state.last_metrics
    if metrics:
        with st.expander("Session Metrics", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode", str(metrics.get("mode", "n/a")))
            c2.metric("Latency (s)", str(metrics.get("latency_sec", "n/a")))
            c3.metric("Total Tokens", str(metrics.get("total_tokens", "n/a")))
            c4.metric("Accuracy Proxy", str(int(metrics.get("rag_hits", 0)) + int(metrics.get("tool_hits", 0))))

def _render_eda(agent):
    st.subheader("EDA")
    if st.button("Refresh EDA"):
        _cached_eda.clear()
        summary, fig = _cached_eda(agent.movies)
        st.session_state.eda_summary = summary
        _save_eda_to_assets(summary, fig)

    if "eda_summary" not in st.session_state:
        loaded_summary = _load_eda_from_assets()
        if loaded_summary is not None:
            st.session_state.eda_summary = loaded_summary
        else:
            summary, fig = _cached_eda(agent.movies)
            st.session_state.eda_summary = summary
            _save_eda_to_assets(summary, fig)

    st.dataframe(st.session_state.eda_summary, use_container_width=True)
    eda_plot_path = _eda_artifact_paths()["plot"]
    if eda_plot_path.exists():
        st.image(str(eda_plot_path), use_container_width=True)


def _render_evaluation(agent):
    st.subheader("Evaluation")
    st.caption("Uses prompts from data/english_test_prompts.json and data/hindi_test_prompts.json.")
    lang_filter = st.radio(
        "Prompt Set",
        ["Hindi+English", "English only"],
        horizontal=True,
        index=0,
    )

    prompt_language = "english_only" if lang_filter == "English only" else "hindi+english"
    state_key = f"eval::{prompt_language}"
    paths = _eval_artifact_paths(prompt_language)

    if state_key not in st.session_state:
        detail, summary = _load_eval_from_assets(prompt_language)
        if detail is not None and summary is not None:
            st.session_state[state_key] = (detail, summary)
        else:
            detail, summary = run_benchmarks(agent, total_cases=None, prompt_language=prompt_language)
            summary = summary.copy()
            if "pass_rate" in summary.columns:
                summary = summary.rename(columns={"pass_rate": "accuracy_pct"})
            plot_fig = _benchmark_plot(summary)
            _save_eval_to_assets(detail, summary, prompt_language, plot_fig)
            st.session_state[state_key] = (detail, summary)

    if st.button("Run Evaluation"):
        detail, summary = run_benchmarks(agent, total_cases=None, prompt_language=prompt_language)
        summary = summary.copy()
        if "pass_rate" in summary.columns:
            summary = summary.rename(columns={"pass_rate": "accuracy_pct"})
        plot_fig = _benchmark_plot(summary)
        _save_eval_to_assets(detail, summary, prompt_language, plot_fig)
        st.session_state[state_key] = (detail, summary)

    if state_key in st.session_state:
        st.session_state.eval_detail, st.session_state.eval_summary = st.session_state[state_key]
        st.session_state.eval_plot = _benchmark_plot(st.session_state.eval_summary)

    if "eval_summary" in st.session_state:
        st.markdown("### Comparison: RAG vs Tool Call vs RAG+Tool Call")
        s = st.session_state.eval_summary
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Accuracy", f"{s['accuracy_pct'].max():.2f}%" if "accuracy_pct" in s else "n/a")
        c2.metric("Best Precision@K", f"{s['precision_at_k'].max():.2f}%" if "precision_at_k" in s else "n/a")
        c3.metric("Best MRR", f"{s['mrr'].max():.4f}" if "mrr" in s else "n/a")
        c4.metric("Lowest Hallucination", f"{s['hallucination_rate'].min():.2f}%" if "hallucination_rate" in s else "n/a")
        c5.metric("Best Tool Accuracy", f"{s['tool_call_accuracy'].max():.2f}%" if "tool_call_accuracy" in s else "n/a")
        st.dataframe(st.session_state.eval_summary, use_container_width=True)
        if paths["plot"].exists():
            st.image(str(paths["plot"]), use_container_width=True)
        else:
            st.pyplot(st.session_state.eval_plot, clear_figure=False)
        with st.expander("Detailed Results"):
            st.dataframe(st.session_state.eval_detail, use_container_width=True)


def render_dashboard():
    st.set_page_config(page_title="MovieMate", page_icon="movie_camera", layout="wide")

    st.markdown(
        """
        <style>
        .stApp { background: #020617; color: #f1f5f9; }
        .sidebar-card { background:#0f172a; color:#94a3b8; border-radius:8px; padding:12px; margin-bottom:12px; border:1px solid #1e293b; font-size:13px; }
        .logo-block { background:#0f172a; color:#f1f5f9; border-radius:8px; padding:16px; margin-bottom:16px; text-align:center; }
        .stButton>button { background:#0f172a; border:1px solid #334155; color:#94a3b8; border-radius:6px; transition:0.2s; width: 100%; }
        .stButton>button:hover { border-color:#38bdf8; color:#38bdf8; background:#0f172a; }
        div[data-testid="stRadio"] label { color:#94a3b8; }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { background:#1e293b; border:1px solid #38bdf8; border-radius:8px; padding:6px 8px; }
        .stChatInputContainer { background: #0f172a !important; }
        div[data-testid="stExpander"] { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<div class='logo-block'><h1 style='margin:0; letter-spacing:1px;'>MOVIEMATE</h1><div style='font-size:11px; margin-top:8px; color:#64748b; font-weight:600;'>Amit Verma | SE23UCSE020</div></div>", unsafe_allow_html=True)
        
        if "active_view" not in st.session_state:
            st.session_state.active_view = "EDA"

        view = st.radio(
            "Sections",
            options=["EDA", "Chat", "Evaluation"],
            index=["EDA", "Chat", "Evaluation"].index(st.session_state.active_view),
            horizontal=False,
        )
        st.session_state.active_view = view

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
                "--server.fileWatcherType",
                "none",
            ],
            check=False,
        )
        return
    render_dashboard()


if __name__ == "__main__":
    render_dashboard()
