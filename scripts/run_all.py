import os
import sys
import time
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.run_subgoal as run_subgoal
import scripts.run_fd as run_fd
import scripts.run_llm as run_llm


def run_all(domains=None, planners=None, sizes=None, fd_path=None, backend="ollama", model=None):
    if domains is None:
        domains = ["blocksworld", "gripper", "barman"]
    if planners is None:
        planners = ["symbolic-llm", "mcts-llm", "fast-downward", "cot-llm"]
    if sizes is None:
        sizes = [3]
    if fd_path is None:
        fd_path = r"C:\Users\dell\downward\fast-downward.py"

    overall_results = []

    print("\n" + "=" * 75)
    print("🚀 STARTING BENCHMARK ACROSS ALL SELECTED PLANNERS & DOMAINS")
    print("=" * 75)
    print(f"Domains:  {domains}")
    print(f"Planners: {planners}")
    print(f"Sizes:    {sizes}")
    print(f"Backend:  {backend}")
    print(f"FD Path:  {fd_path}")
    print("=" * 75 + "\n")

    for domain in domains:
        print(f"\n📂 DOMAIN: {domain.upper()}")
        print("-" * 50)

        for planner in planners:
            print(f"\n▶ Running Planner: [{planner}] on [{domain}]...")
            start_t = time.perf_counter()
            try:
                if planner == "symbolic-llm":
                    run_subgoal.run(
                        model=model,
                        max_tokens=512,
                        domain=domain,
                        algorithm="symbolic-llm",
                        temperature=0.0,
                        fd_planner="seq-opt-fdss-1",
                        p_num=[1],
                        p_size=sizes,
                        p_idx=[1],
                        path=fd_path,
                        backend=backend
                    )
                elif planner == "mcts-llm":
                    run_subgoal.run(
                        model=model,
                        max_tokens=512,
                        domain=domain,
                        algorithm="mcts-llm",
                        temperature=0.0,
                        fd_planner="seq-opt-fdss-1",
                        p_num=[2],
                        p_size=sizes,
                        p_idx=[1],
                        path=fd_path,
                        backend=backend
                    )
                elif planner in ["fast-downward", "fd"]:
                    run_fd.run_fd(
                        domain=domain,
                        planner="seq-opt-fdss-1",
                        p_size=sizes,
                        p_idx=[1],
                        path=fd_path
                    )
                elif planner in ["cot-llm", "llm", "cot"]:
                    run_llm.run(
                        model=model,
                        max_tokens=4096,
                        domain=domain,
                        algorithm="cot",
                        temperature=0.0,
                        p_num=[1],
                        p_size=sizes,
                        p_idx=[1],
                        backend=backend
                    )
                else:
                    print(f"Unknown planner: {planner}")
                    continue

                elapsed = time.perf_counter() - start_t
                overall_results.append((domain, planner, True, elapsed))
                print(f"✔ [{planner}] on [{domain}] completed in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.perf_counter() - start_t
                overall_results.append((domain, planner, False, elapsed))
                print(f"✖ [{planner}] on [{domain}] failed after {elapsed:.2f}s: {e}")

    print("\n" + "=" * 75)
    print("📊 BENCHMARK SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Domain':<15} | {'Planner':<18} | {'Status':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for domain, planner, success, elapsed in overall_results:
        status_str = "SUCCESS" if success else "FAILED"
        print(f"{domain:<15} | {planner:<18} | {status_str:<10} | {elapsed:<10.2f}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run All Planners & Benchmarks")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["blocksworld", "gripper", "barman"],
        choices=["blocksworld", "gripper", "barman"],
        help="List of domains to run"
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["symbolic-llm", "mcts-llm", "fast-downward", "cot-llm"],
        choices=["symbolic-llm", "mcts-llm", "fast-downward", "cot-llm", "fd", "cot"],
        help="List of planners to run"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[3],
        help="Problem sizes (e.g. --sizes 3 4 5)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "vllm", "openai"],
        help="LLM backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (optional)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=r"C:\Users\dell\downward\fast-downward.py",
        help="Path to fast-downward.py"
    )

    args = parser.parse_args()
    run_all(
        domains=args.domains,
        planners=args.planners,
        sizes=args.sizes,
        fd_path=args.path,
        backend=args.backend,
        model=args.model
    )


if __name__ == "__main__":
    main()
