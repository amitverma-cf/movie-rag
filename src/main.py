import argparse

from src.agent import MovieAgent
from src.dashboard import launch_dashboard
from src.eda import load_movies, run_eda
from src.embed import build_or_update_faiss
from src.evaluation import benchmark_markdown, run_benchmarks
from src.llm import MovieLLM
from src.preprocess import generate_missing_keywords_with_keybert, patch_missing_trailers
from src.scrape import scrape_movies
from src.similarity_search import Retriever


def run_eval():
    agent = MovieAgent()
    detail, summary = run_benchmarks(agent, total_cases=100)
    print(benchmark_markdown(summary, total_cases=100))
    print(detail.head(20))


def run_llm_demo(movies, retriever):
    llm_client = MovieLLM()
    query = "highly rated sci-fi movies after 2010"
    results = retriever.retrieve(query, movies, 8)
    payload = llm_client.llm_answer(query, rag_results=results, tool_results=None, merged_results=results, history_text="")
    print(payload["text"])
    print(payload["analytics"])


def run_agent_demo(query):
    agent = MovieAgent()
    payload = agent.run(query, mode="rag+tool", history=[])
    print(payload["answer"])
    print(payload["analytics"])
    print(payload["movies"][["title", "year", "rating", "genre", "source"]].head(8))


def main():
    parser = argparse.ArgumentParser(description="MovieMate modules runner")
    parser.add_argument(
        "--step",
        choices=["scrape", "preprocess", "eda", "embed", "search", "llm", "agent", "dashboard", "eval", "all"],
        default="dashboard",
    )
    parser.add_argument("--tmdb_api_key", default="", help="Required for scrape step")
    parser.add_argument("--query", default="highly rated sci-fi movies after 2010")
    args = parser.parse_args()

    if args.step == "scrape":
        scrape_movies(args.tmdb_api_key)
        return

    if args.step == "preprocess":
        patch_missing_trailers()
        generate_missing_keywords_with_keybert()
        return

    if args.step == "eda":
        movies = load_movies()
        run_eda(movies)
        return

    if args.step == "embed":
        build_or_update_faiss()
        return

    if args.step == "dashboard":
        launch_dashboard()
        return

    if args.step == "agent":
        run_agent_demo(args.query)
        return

    movies = load_movies()
    retriever = Retriever()

    if args.step == "search":
        print(retriever.retrieve(args.query, movies, 8)[["title", "year", "rating", "genre", "score"]])
    elif args.step == "llm":
        run_llm_demo(movies, retriever)
    elif args.step == "eval":
        run_eval()
    elif args.step == "all":
        run_eda(movies)
        run_eval()


if __name__ == "__main__":
    main()
