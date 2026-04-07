import html
from urllib.parse import parse_qs, urlparse

import gradio as gr
import matplotlib.pyplot as plt

from src.agent import MovieAgent
from src.eda import build_eda_artifacts
from src.evaluation import benchmark_markdown, run_benchmarks


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


def _metrics_md(analytics, mode, rag_count, tool_count):
    a = analytics or {}
    return "\n".join(
        [
            "### Metrics",
            f"- mode: {mode}",
            f"- model: {a.get('model', 'n/a')}",
            f"- latency_sec: {a.get('latency_sec', 'n/a')}",
            f"- prompt_tokens: {a.get('prompt_tokens', 'n/a')}",
            f"- completion_tokens: {a.get('completion_tokens', 'n/a')}",
            f"- total_tokens: {a.get('total_tokens', 'n/a')}",
            f"- tokens_per_sec: {a.get('tokens_per_sec', 'n/a')}",
            f"- rag_hits: {rag_count}",
            f"- tool_hits: {tool_count}",
        ]
    )


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
    fig, ax = plt.subplots(figsize=(9, 4))
    if summary_df is None or summary_df.empty:
        ax.text(0.5, 0.5, "No benchmark data", ha="center", va="center")
        ax.axis("off")
        plt.tight_layout()
        return fig
    ax.bar(summary_df["mode"], summary_df["avg_latency_ms"], color=["#457b9d", "#e76f51", "#2a9d8f"])
    ax.set_title("Average Latency by Mode (100 queries)")
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def launch_dashboard():
    agent = MovieAgent()

    def switch_view(view):
        return (
            gr.update(visible=view == "chat"),
            gr.update(visible=view == "eda"),
            gr.update(visible=view == "benchmark"),
        )

    def load_eda_view():
        summary, fig = build_eda_artifacts(agent.movies)
        return summary.reset_index(names=["feature"]), fig

    def run_benchmark():
        detail, summary = run_benchmarks(agent, total_cases=100)
        md = benchmark_markdown(summary, total_cases=100)
        fig = _benchmark_plot(summary)
        return md, summary, detail, fig

    def run_chat(message, history, mode):
        history = history or []
        result = agent.run(message, mode=mode, history=history)
        answer = result["answer"]
        history = history + [(message, answer)]
        metrics_md = _metrics_md(result["analytics"], mode, result["rag_count"], result["tool_count"])
        cards = _cards_html(result["movies"])
        return history, metrics_md, cards, ""

    css = """
    .app-shell {background: linear-gradient(135deg, #8481de, #5e8ae0);}
    .sidebar-card {background:#1d3557;color:#f88aee;border-radius:14px;padding:12px;}
    """

    with gr.Blocks(title="MovieMate Dashboard", css=css) as demo:
        with gr.Row(elem_classes="app-shell"):
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("## MovieMate")
                gr.Markdown("<div class='sidebar-card'>Navigate views and compare retrieval quality.</div>")
                view = gr.Radio(["chat", "eda", "benchmark"], value="chat", label="Sidebar Tabs")

            with gr.Column(scale=4):
                with gr.Column(visible=True) as chat_view:
                    gr.Markdown("## Chat")
                    mode = gr.Radio(["rag", "tool", "rag+tool"], value="rag+tool", label="Retrieval Mode")
                    chat = gr.Chatbot(label="Assistant", height=340)
                    msg = gr.Textbox(label="Ask about movies")
                    send = gr.Button("Send", variant="primary")
                    metrics = gr.Markdown("### Metrics")
                    cards = gr.HTML("", label="Movie Cards")
                    send.click(run_chat, inputs=[msg, chat, mode], outputs=[chat, metrics, cards, msg])
                    msg.submit(run_chat, inputs=[msg, chat, mode], outputs=[chat, metrics, cards, msg])

                with gr.Column(visible=False) as eda_view:
                    gr.Markdown("## EDA")
                    eda_btn = gr.Button("Refresh EDA")
                    eda_summary = gr.Dataframe(label="Summary Statistics")
                    eda_plot = gr.Plot(label="EDA Plots")
                    eda_btn.click(load_eda_view, outputs=[eda_summary, eda_plot])

                with gr.Column(visible=False) as benchmark_view:
                    gr.Markdown("## Benchmark")
                    bench_btn = gr.Button("Run Evaluation")
                    bench_md = gr.Markdown("")
                    bench_summary = gr.Dataframe(label="Summary by Mode")
                    bench_detail = gr.Dataframe(label="Detailed Results")
                    bench_plot = gr.Plot(label="Latency Comparison")
                    bench_btn.click(run_benchmark, outputs=[bench_md, bench_summary, bench_detail, bench_plot])

        view.change(switch_view, inputs=[view], outputs=[chat_view, eda_view, benchmark_view])

    demo.launch(share=False, inline=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch_dashboard()
