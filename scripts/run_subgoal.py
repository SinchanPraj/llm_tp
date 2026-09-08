import time
import os
import sys
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.symbolic as sym
import scripts.subgoal as subgoal
from utils import llm_functions as llm
import pddlpy


def run_subgoal_experiment(domain, prob_size, prob_idx, plan_number, model, max_tokens, algorithm, temperature, fd_planner, path, backend=None):
    """
    Run one problem instance with either symbolic-llm (Ollama + Fast Downward) or mcts-llm (vLLM + MCTS).
    """
    domain_f = os.path.join(PROJECT_ROOT, "domains", f"domain_{domain}.pddl")
    problem_f = os.path.join(PROJECT_ROOT, "experiments", domain, "problem", f"{domain}{prob_size}_{prob_idx}.pddl")

    if algorithm == "symbolic-llm":
        plan_dir = os.path.join(PROJECT_ROOT, "experiments", domain, "plan", algorithm)
        plan_f = os.path.join(plan_dir, f"{domain}{prob_size}_{prob_idx}.pddl")
    else:
        plan_dir = os.path.join(PROJECT_ROOT, "experiments", domain, "plan", algorithm)
        plan_f = os.path.join(plan_dir, f"{domain}{prob_size}_{prob_idx}_N{plan_number}.pddl")
    os.makedirs(plan_dir, exist_ok=True)

    if not os.path.exists(domain_f):
        raise FileNotFoundError(f"Domain file not found: {domain_f}")
    if not os.path.exists(problem_f):
        raise FileNotFoundError(f"Problem file not found: {problem_f}")

    domprob = pddlpy.DomainProblem(domain_f, problem_f)

    with open(problem_f, "r", encoding="utf-8") as f:
        problem = f.read()

    # Subgoals are decomposed via LLM (Ollama or configured backend)
    subgoal_backend = "ollama" if (backend is None or backend == "auto") else backend
    subgoals = subgoal.generate_subgoals(
        model, domain, problem, max_tokens, temperature=0.0, backend=subgoal_backend
    )

    start_time = time.perf_counter()
    if algorithm == "symbolic-llm":
        final_plan = subgoal.solve_symbolic_llm(
            domprob, domain, model, prob_size, prob_idx, subgoals, fd_planner, plan_f, path
        )
    else:
        # MCTS uses vLLM to sample candidate plans with token logprobs
        mcts_backend = "vllm" if (backend is None or backend == "auto") else backend
        final_tree, final_plan, states_path, actions_path = subgoal.solve_mcts_llm(
            domprob, domain, model, plan_f, plan_number, prob_size, prob_idx,
            subgoals, max_tokens, temperature, backend=mcts_backend
        )
    end_time = time.perf_counter()
    planning_time = end_time - start_time

    print(f"Planning time: {planning_time:.4f}s")

    success, log = sym.validate(domain_f=domain_f, problem_f=problem_f, plan_f=plan_f)
    print(f"Problem size: {prob_size}, Problem idx: {prob_idx}, Valid: {success}")

    return planning_time, success


def run(model, max_tokens, domain, algorithm, temperature, fd_planner, p_num, p_size, p_idx, path, backend=None):
    results = defaultdict(list)

    results_dir = os.path.join(PROJECT_ROOT, "experiments", domain, "plan", algorithm)
    os.makedirs(results_dir, exist_ok=True)

    subgoal_prob_dir = os.path.join(PROJECT_ROOT, "experiments", domain, "subgoal_problems", algorithm)
    os.makedirs(subgoal_prob_dir, exist_ok=True)

    print(f"\n=======================================================", flush=True)
    print(f"Running Experiment: domain={domain}, algorithm={algorithm}", flush=True)
    print(f"Backend: {backend or 'auto'}, Model: {model}", flush=True)
    print(f"Sizes: {list(p_size)}, Plan counts: {list(p_num)}", flush=True)
    print(f"=======================================================\n", flush=True)

    for plan_number in p_num:
        for prob_size in p_size:
            for prob_idx in p_idx:
                try:
                    planning_time, success = run_subgoal_experiment(
                        domain, prob_size, prob_idx, plan_number, model, max_tokens,
                        algorithm, temperature, fd_planner, path, backend=backend
                    )
                    results[(prob_size, plan_number)].append((prob_idx, planning_time, success))
                except Exception as e:
                    print(f"Experiment failed for prob_size={prob_size}, prob_idx={prob_idx}, plan_number={plan_number}. Error: {e}")
                    results[(prob_size, plan_number)].append((prob_idx, None, False))

        # Calculate and print summary statistics
        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            for (prob_size, plan_number), result_list in results.items():
                successful_results = [t for _, t, success in result_list if success and t is not None]
                if successful_results:
                    total_time = sum(successful_results)
                    avg_time = total_time / len(successful_results)
                    success_count = len(successful_results)
                    success_rate = success_count / len(result_list)
                else:
                    avg_time = 0
                    success_rate = 0

                summary_str = (
                    f"Problem size: {prob_size}, Plan number: {plan_number}\n"
                    f"Average planning time: {avg_time:.4f} seconds\n"
                    f"Success rate: {success_rate:.4%}\n"
                )
                summary_file.write(summary_str)
                print("\n" + summary_str)

                for prob_idx, _, success in result_list:
                    status = "Success" if success else "Failure"
                    problem_str = f"Problem {prob_idx}: {status}\n"
                    summary_file.write(problem_str)
                    print(problem_str.strip())
                summary_file.write("\n")
                print()


def main():
    parser = argparse.ArgumentParser(description="Run Neuro-Symbolic Task Planning Experiments")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. Qwen/Qwen3-32B, qwen2.5:1.5b)")
    parser.add_argument("--domain", type=str, required=True, choices=["barman", "blocksworld", "gripper"], help="PDDL domain")
    parser.add_argument("--subgoal_planner", "--subgoalplanner", dest="subgoal_planner", type=str, required=True, choices=["mcts-llm", "symbolic-llm"], help="Subgoal planner algorithm")
    parser.add_argument("--backend", type=str, choices=["auto", "vllm", "ollama", "openai"], default="auto", help="LLM serving backend")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--planner", type=str, choices=["seq-opt-fdss-1", "seq-sat-lama-2011"], default="seq-opt-fdss-1", help="Fast Downward planner configuration")
    parser.add_argument("--ns", type=int, nargs="+", default=[3], help="Number of sampled candidate plans (e.g. --ns 3 4 5)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3], help="Problem sizes (e.g. --sizes 3 4 5)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per LLM completion (default: 512)")
    parser.add_argument("--path", type=str, default="", help="Path to fast-downward.py")

    args = parser.parse_args()

    # Intelligent default model selection if unspecified
    model = args.model
    if not model:
        if args.backend == "ollama":
            model = llm.DEFAULT_OLLAMA_MODEL
        else:
            model = llm.DEFAULT_VLLM_MODEL

    run(
        model=model,
        max_tokens=args.max_tokens,
        domain=args.domain,
        algorithm=args.subgoal_planner,
        temperature=args.temperature,
        fd_planner=args.planner,
        p_num=args.ns,
        p_size=args.sizes,
        p_idx=[1],
        path=args.path,
        backend=args.backend
    )


if __name__ == "__main__":
    main()
