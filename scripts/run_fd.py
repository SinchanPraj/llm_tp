import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.symbolic as sym


# Main function to run planner and calculate average planning time for each problem size
def run_fd(domain, planner, p_size, p_idx, path):
    total_times = {size: [] for size in p_size}
    success_counts = {size: 0 for size in p_size}

    print(f"\n=======================================================")
    print(f"Running Fast Downward: domain={domain}, planner={planner}")
    print(f"Sizes: {list(p_size)}")
    print(f"=======================================================\n")

    results_dir = os.path.join(PROJECT_ROOT, "experiments", domain, "plan", planner)
    os.makedirs(results_dir, exist_ok=True)

    summary_file_path = os.path.join(results_dir, "summary.txt")
    with open(summary_file_path, "a", encoding="utf-8") as summary_file:
        for prob_size in p_size:
            for prob_idx in p_idx:
                domain_f = os.path.join(PROJECT_ROOT, "domains", f"domain_{domain}.pddl")
                problem_f = os.path.join(PROJECT_ROOT, "experiments", domain, "problem", f"{domain}{prob_size}_{prob_idx}.pddl")
                fd_sas_f = os.path.join(results_dir, f"{domain}{prob_size}_{prob_idx}.pddl.sas")
                fd_plan_f = os.path.join(results_dir, f"{domain}{prob_size}_{prob_idx}.plan")
                
                success, plan, output, planning_time = sym.fd_planner(fd_plan_f, fd_sas_f, domain_f, problem_f, planner, path)

                if success:
                    total_times[prob_size].append(planning_time)
                    success_counts[prob_size] += 1
                    summary_str = f"Problem Size: {prob_size}, Problem Index: {prob_idx} - Success\n"
                    summary_file.write(summary_str)
                    print(summary_str.strip())

                    summary_str = f"Planning Time: {planning_time:.4f} seconds\n"
                    summary_file.write(summary_str)
                    print(summary_str.strip())

                else:
                    summary_str = f"Problem Size: {prob_size}, Problem Index: {prob_idx} - Failed\n"
                    summary_file.write(summary_str)
                    print(summary_str.strip())

            # After all problem indices for a specific size are completed
            if success_counts[prob_size] > 0:
                average_time = sum(total_times[prob_size]) / success_counts[prob_size]
                summary_str = f"\nAverage planning time for problem size {prob_size} (successful only): {average_time:.4f} seconds\n"
                summary_file.write(summary_str)
                print(summary_str)
            else:
                summary_str = f"No successful plans for problem size {prob_size}\n\n"
                summary_file.write(summary_str)
                print(summary_str)


def main():
    parser = argparse.ArgumentParser(description="Run Fast Downward Classical Baseline")
    parser.add_argument("--domain", type=str, required=True, choices=["barman", "blocksworld", "gripper"], help="Name of PDDL domain")
    parser.add_argument("--planner", type=str, choices=["seq-opt-fdss-1", "seq-sat-lama-2011"], default="seq-opt-fdss-1", help="Fast Downward configuration")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3], help="Problem sizes (e.g. --sizes 3 4 5)")
    parser.add_argument("--path", type=str, default=r"C:\Users\dell\downward\fast-downward.py", help="Path to fast-downward.py")
    args = parser.parse_args()

    run_fd(args.domain, args.planner, args.sizes, [1], args.path)


if __name__ == "__main__":
    main()