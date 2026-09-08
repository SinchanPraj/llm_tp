import os
import subprocess
import time
import sys
import shutil


# Run pddl plan validator VAL with Python simulation fallback
def validate(domain_f, problem_f, plan_f):
    """
    Validate a generated plan against the domain and problem using VAL or built-in Python PDDL simulator.
    """
    val_bin = os.environ.get("VAL_PATH", os.environ.get("VALIDATE_PATH", "Validate"))
    executable_found = shutil.which(val_bin) is not None or os.path.exists(val_bin)

    if executable_found:
        cmd = f'"{val_bin}" -v "{domain_f}" "{problem_f}" "{plan_f}"'
        try:
            output = subprocess.getoutput(cmd)
            if "Plan valid" in output:
                return True, output
            return False, output
        except Exception as e:
            output = f"Execution error running validator: {e}"

    # Python-based symbolic PDDL validation fallback
    try:
        import pddlpy
        from collections import defaultdict
        import scripts.tree_generation as tg
        import scripts.mcts as mcts

        if not os.path.exists(plan_f):
            return False, f"Plan file not found: {plan_f}"

        with open(plan_f, "r", encoding="utf-8") as f:
            plan_content = f.read()

        action_lines = []
        for line in plan_content.split("\n"):
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            action_lines.append(line)

        if not action_lines:
            return False, "Plan contains no action steps."

        domprob = pddlpy.DomainProblem(domain_f, problem_f)
        initial_state_set = domprob.initialstate()
        initial_state = defaultdict(set)
        for atom in initial_state_set:
            predicate = atom.predicate[0]
            args = atom.predicate[1:]
            initial_state[predicate].add(tuple(args))
        state = tuple(sorted((k, tuple(sorted(v))) for k, v in initial_state.items()))

        for step_i, action in enumerate(action_lines, 1):
            state = tg.apply_action(state, action, domprob)

        goal_set = domprob.goals()
        if mcts.is_goal_state(state, goal_set):
            return True, f"Plan valid (verified {len(action_lines)} actions via symbolic engine)"
        else:
            return False, "Final state reached did not satisfy all goal predicates."

    except Exception as e:
        return False, f"Validation check failed: {e}"


# Run fast downward planner
def fd_planner(plan_f, sas_f, domain_f, problem_f, planner, path):
    """
    Run Fast Downward classical planner.
    """
    if not path or not os.path.exists(path):
        return False, "", f"Fast Downward script not found at '{path}'", 0.0

    cmd = f'"{sys.executable}" "{path}" --alias "{planner}" --search-time-limit 1000 --plan-file "{plan_f}" --sas-file "{sas_f}" "{domain_f}" "{problem_f}"'
    start_time = time.time()
    try:
        output = subprocess.getoutput(cmd)
    except Exception as e:
        output = f"Error running Fast Downward: {e}"
    end_time = time.time()
    planning_time = end_time - start_time

    plan = ""
    success = False

    if "Solution found." in output:
        success = True
        if os.path.exists(plan_f):
            with open(plan_f, "a", encoding="utf-8") as f:
                f.write(f"\n; Planning time: {planning_time:.4f} seconds")
            with open(plan_f, "r", encoding="utf-8") as f:
                plan = f.read()

    elif os.path.exists(plan_f):
        with open(plan_f, "a", encoding="utf-8") as f:
            f.write(f"\n; Planning time: {planning_time:.4f} seconds")
        with open(plan_f, "r", encoding="utf-8") as f:
            plan = f.read()

    return success, plan, output, planning_time
