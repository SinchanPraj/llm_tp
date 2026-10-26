import sys
import os
import time
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.symbolic as sym
import utils.prompts as prompts
import utils.llm_functions as llm


# chain-of-thought
def cot(model, domain, problem, temperature, backend="vllm"):
    if domain == "barman":
        args = prompts.get_barman_args(1)
    elif domain == "blocksworld":
        args = prompts.get_blocksworld_args(1)
    elif domain == "gripper":
        args = prompts.get_gripper_args(1)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    system_prompt = args.cot_prompt

    chat_history = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Question:
Problem PDDL: \n{problem}\n
Plan PDDL:
"""
        }
    ]

    completion = llm.get_session_completion(
        chat_history,
        model=model,
        max_tokens=4096,
        temperature=temperature,
        logprobs=False,
        backend=backend
    )
    response_text = completion["choices"][0]['message']["content"].strip()
    if "Plan PDDL:" in response_text:
        plan_part = response_text.split("Plan PDDL:")[-1].strip()
    else:
        plan_part = response_text

    clean_lines = []
    for line in plan_part.split('\n'):
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("(") or cleaned.startswith(";"):
            clean_lines.append(cleaned)

    if clean_lines:
        return "\n".join(clean_lines)
    return plan_part


def run_cot_experiment(domain, prob_size, prob_idx, model, max_tokens, algorithm, temperature, backend="vllm"):
    domain_f = os.path.join(PROJECT_ROOT, "domains", f"domain_{domain}.pddl")
    problem_f = os.path.join(PROJECT_ROOT, "experiments", domain, "problem", f"{domain}{prob_size}_{prob_idx}.pddl")
    plan_dir = os.path.join(PROJECT_ROOT, "experiments", domain, "plan", algorithm)
    os.makedirs(plan_dir, exist_ok=True)
    llm_plan_f = os.path.join(plan_dir, f"{domain}{prob_size}_{prob_idx}.pddl")

    with open(problem_f, "r", encoding="utf-8") as f:
        problem = f.read()

    start_time = time.perf_counter()
    plan = cot(model, domain, problem, temperature, backend=backend)
    end_time = time.perf_counter()
    planning_time = end_time - start_time

    with open(llm_plan_f, "w", encoding="utf-8") as f:
        f.write(plan)

    success, log = sym.validate(domain_f=domain_f, problem_f=problem_f, plan_f=llm_plan_f)
    print(f"Problem size: {prob_size}, Problem idx: {prob_idx}, Valid: {success}")

    return planning_time, success


def run(model, max_tokens, domain, algorithm, temperature, p_num, p_size, p_idx, backend="vllm"):
    results = defaultdict(list)
    results_dir = os.path.join(PROJECT_ROOT, "experiments", domain, "plan", algorithm)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Running Direct CoT Planning: domain={domain}")
    print(f"Backend: {backend}, Model: {model}")
    print(f"Sizes: {list(p_size)}")
    print(f"=======================================================\n")

    for plan_number in p_num:
        for prob_size in p_size:
            for prob_idx in p_idx:
                try:
                    planning_time, success = run_cot_experiment(
                        domain, prob_size, prob_idx, model, max_tokens, algorithm, temperature, backend=backend
                    )
                    results[(prob_size, plan_number)].append((prob_idx, planning_time, success))
                except Exception as e:
                    print(f"Experiment failed for prob_size={prob_size}, prob_idx={prob_idx}. Error: {e}")
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
    parser = argparse.ArgumentParser(description="Run Chain-of-Thought (CoT) Direct LLM Planning")
    parser.add_argument("--model", type=str, default=None, help="Name of model (e.g. Qwen/Qwen3-32B, qwen2.5:1.5b)")
    parser.add_argument("--domain", type=str, required=True, choices=["barman", "blocksworld", "gripper"], help="Name of PDDL domain")
    parser.add_argument("--backend", type=str, choices=["vllm", "ollama", "openai", "auto"], default="auto", help="LLM backend")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3], help="Problem sizes (e.g. --sizes 3 4 5)")
    args = parser.parse_args()

    model = args.model

    run(
        model=model,
        max_tokens=4096,
        domain=args.domain,
        algorithm="cot",
        temperature=args.temperature,
        p_num=[1],
        p_size=args.sizes,
        p_idx=[1],
        backend=args.backend
    )


if __name__ == "__main__":
    main()
