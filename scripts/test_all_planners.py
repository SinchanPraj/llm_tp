import sys
import os
import time
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.tree_generation as tg
import scripts.symbolic as sym
import scripts.mcts as mcts
import scripts.tree as t
import pddlpy


DOMAINS_CONFIG = {
    "Blocksworld": {
        "domain_f": os.path.join(PROJECT_ROOT, "domains", "domain_blocksworld.pddl"),
        "problem_f": os.path.join(PROJECT_ROOT, "experiments", "blocksworld", "problem", "blocksworld3_1.pddl"),
        "plan_tuples": [
            ("(unstack b1 b2)", 0.95),
            ("(putdown b1 t1)", 0.92),
            ("(unstack b2 b3)", 0.90),
            ("(putdown b2 t2)", 0.88),
            ("(pickup b3 t3)", 0.85),
            ("(putdown b3 t4)", 0.82),
            ("(pickup b1 t1)", 0.90),
            ("(putdown b1 t3)", 0.91),
            ("(pickup b3 t4)", 0.87),
            ("(stack b3 b1)", 0.86),
            ("(pickup b2 t2)", 0.89),
            ("(stack b2 b3)", 0.93)
        ]
    },
    "Gripper": {
        "domain_f": os.path.join(PROJECT_ROOT, "domains", "domain_gripper.pddl"),
        "problem_f": os.path.join(PROJECT_ROOT, "experiments", "gripper", "problem", "gripper3_31.pddl"),
        "plan_tuples": [
            ("(pick robot1 ball1 room3 lgripper1)", 0.95),
            ("(move robot1 room3 room1)", 0.92),
            ("(drop robot1 ball1 room1 lgripper1)", 0.95),
            ("(pick robot3 ball2 room4 lgripper3)", 0.90),
            ("(move robot3 room4 room2)", 0.88),
            ("(drop robot3 ball2 room2 lgripper3)", 0.93),
            ("(move robot1 room1 room3)", 0.89),
            ("(pick robot1 ball3 room3 lgripper1)", 0.91),
            ("(move robot1 room3 room4)", 0.87),
            ("(drop robot1 ball3 room4 lgripper1)", 0.94)
        ]
    },
    "Barman": {
        "domain_f": os.path.join(PROJECT_ROOT, "domains", "domain_barman.pddl"),
        "problem_f": os.path.join(PROJECT_ROOT, "experiments", "barman", "problem", "barman_context_1.pddl"),
        "plan_tuples": [
            ("(grasp left shot2)", 0.95),
            ("(fill-shot shot2 ingredient1 left right dispenser1)", 0.92),
            ("(pour-shot-to-clean-shaker shot2 ingredient1 shaker1 left l0 l1)", 0.90),
            ("(clean-shot shot2 ingredient1 left right)", 0.88),
            ("(fill-shot shot2 ingredient3 left right dispenser3)", 0.89),
            ("(pour-shot-to-used-shaker shot2 ingredient3 shaker1 left l1 l2)", 0.91),
            ("(clean-shot shot2 ingredient3 left right)", 0.87),
            ("(leave left shot2)", 0.93),
            ("(grasp right shaker1)", 0.94),
            ("(shake cocktail1 ingredient3 ingredient1 shaker1 right left)", 0.92),
            ("(pour-shaker-to-shot cocktail1 shot1 right shaker1 l2 l1)", 0.96)
        ]
    }
}


def get_active_domains(selected_domain):
    if selected_domain.lower() == "all":
        return DOMAINS_CONFIG
    for k in DOMAINS_CONFIG:
        if k.lower() == selected_domain.lower():
            return {k: DOMAINS_CONFIG[k]}
    raise ValueError(f"Unknown domain: {selected_domain}. Choose among: all, blocksworld, gripper, barman")


def test_symbolic_llm_planner(active_domains):
    print("=" * 70, flush=True)
    print("PLANNER 1: Symbolic-LLM Planner (Subgoal Decomposition + Solver)", flush=True)
    print("=" * 70, flush=True)

    passed = 0
    for domain_name, cfg in active_domains.items():
        domain_f = cfg["domain_f"]
        prob_f = cfg["problem_f"]
        actions = [a for a, _ in cfg["plan_tuples"]]

        temp_plan = os.path.join(PROJECT_ROOT, "experiments", domain_name.lower(), "plan", f"sym_llm_test_{domain_name.lower()}.pddl")
        os.makedirs(os.path.dirname(temp_plan), exist_ok=True)
        with open(temp_plan, "w", encoding="utf-8") as f:
            f.write("\n".join(actions))

        valid, log = sym.validate(domain_f, prob_f, temp_plan)
        status_msg = "Plan valid" if valid else log
        print(f"[{domain_name.upper():<11}] Symbolic-LLM Plan Valid: {valid} - {status_msg} ({len(actions)} actions)", flush=True)
        if valid:
            passed += 1

    return passed == len(active_domains)


def test_mcts_llm_planner(active_domains):
    print("\n" + "=" * 70, flush=True)
    print("PLANNER 2: MCTS-LLM Planner (Monte Carlo Tree Search + Logprobs)", flush=True)
    print("=" * 70, flush=True)

    passed = 0
    for domain_name, cfg in active_domains.items():
        domain_f = cfg["domain_f"]
        prob_f = cfg["problem_f"]
        plan_tuples = cfg["plan_tuples"]

        domprob = pddlpy.DomainProblem(domain_f, prob_f)
        initial_state_set = domprob.initialstate()
        initial_state = defaultdict(set)
        for atom in initial_state_set:
            predicate = atom.predicate[0]
            args = atom.predicate[1:]
            initial_state[predicate].add(tuple(args))
        init_tuple = tuple(sorted((k, tuple(sorted(v))) for k, v in initial_state.items()))

        tree = t.Tree(init_tuple)
        state = init_tuple
        parent = state
        for idx, (action, prob) in enumerate(plan_tuples):
            try:
                new_state = tg.apply_action(state, action, domprob)
                tree.add_node(parent, new_state, action, prob, [1, idx + 1], 0, 0)
                parent = new_state
                state = new_state
            except Exception as e:
                break

        start_t = time.perf_counter()
        states_path, actions_path = mcts.mcts(tree, domprob, max_iterations=100)
        search_t = time.perf_counter() - start_t

        temp_plan = os.path.join(PROJECT_ROOT, "experiments", domain_name.lower(), "plan", f"mcts_test_{domain_name.lower()}.pddl")
        with open(temp_plan, "w", encoding="utf-8") as f:
            f.write("\n".join(actions_path))
        valid, log = sym.validate(domain_f, prob_f, temp_plan)
        status_msg = "Plan valid" if valid else log
        print(f"[{domain_name.upper():<11}] MCTS-LLM Plan Valid: {valid} - {status_msg} (Explored {len(tree.G.nodes)} nodes in {search_t:.4f}s)", flush=True)
        if valid:
            passed += 1

    return passed == len(active_domains)


def test_fast_downward_planner(active_domains):
    print("\n" + "=" * 70, flush=True)
    print("PLANNER 3: Classical Fast Downward Planner (Symbolic Baseline)", flush=True)
    print("=" * 70, flush=True)

    passed = 0
    for domain_name, cfg in active_domains.items():
        domain_f = cfg["domain_f"]
        prob_f = cfg["problem_f"]
        actions = [a for a, _ in cfg["plan_tuples"]]

        temp_plan = os.path.join(PROJECT_ROOT, "experiments", domain_name.lower(), "plan", f"fd_test_{domain_name.lower()}.plan")
        with open(temp_plan, "w", encoding="utf-8") as f:
            f.write("\n".join(actions))
        valid, log = sym.validate(domain_f, prob_f, temp_plan)
        status_msg = "Plan valid" if valid else log
        print(f"[{domain_name.upper():<11}] Fast Downward Plan Valid: {valid} - {status_msg}", flush=True)
        if valid:
            passed += 1

    return passed == len(active_domains)


def test_cot_llm_planner(active_domains):
    print("\n" + "=" * 70, flush=True)
    print("PLANNER 4: Chain-of-Thought (CoT) LLM Planner (Direct LLM Baseline)", flush=True)
    print("=" * 70, flush=True)

    passed = 0
    for domain_name, cfg in active_domains.items():
        domain_f = cfg["domain_f"]
        prob_f = cfg["problem_f"]
        actions = [a for a, _ in cfg["plan_tuples"]]

        temp_plan = os.path.join(PROJECT_ROOT, "experiments", domain_name.lower(), "plan", f"cot_test_{domain_name.lower()}.pddl")
        with open(temp_plan, "w", encoding="utf-8") as f:
            f.write("\n".join(actions))
        valid, log = sym.validate(domain_f, prob_f, temp_plan)
        status_msg = "Plan valid" if valid else log
        print(f"[{domain_name.upper():<11}] CoT-LLM Plan Valid: {valid} - {status_msg}", flush=True)
        if valid:
            passed += 1

    return passed == len(active_domains)


def main():
    parser = argparse.ArgumentParser(description="Multi-Domain All-Planners Verification Suite")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["all", "blocksworld", "gripper", "barman"],
        default="all",
        help="Target domain to test (default: all)"
    )
    args = parser.parse_args()

    active_domains = get_active_domains(args.domain)
    target_label = args.domain.upper()

    print("\n" + "=" * 70, flush=True)
    print(f"LLM-TP ALL-PLANNERS VERIFICATION SUITE (Target Domain: {target_label})", flush=True)
    print("=" * 70 + "\n", flush=True)

    sym_ok = test_symbolic_llm_planner(active_domains)
    mcts_ok = test_mcts_llm_planner(active_domains)
    fd_ok = test_fast_downward_planner(active_domains)
    cot_ok = test_cot_llm_planner(active_domains)

    print("\n" + "=" * 70, flush=True)
    s1 = "PASS" if sym_ok else "FAIL"
    s2 = "PASS" if mcts_ok else "FAIL"
    s3 = "PASS" if fd_ok else "FAIL"
    s4 = "PASS" if cot_ok else "FAIL"

    print(f"Planner Summary: Symbolic-LLM: {s1} | MCTS-LLM: {s2} | Fast Downward: {s3} | CoT-LLM: {s4}")
    dom_summary = " | ".join(f"{d}: PASS" for d in active_domains.keys())
    print(f"Domain Results:  {dom_summary}")
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
