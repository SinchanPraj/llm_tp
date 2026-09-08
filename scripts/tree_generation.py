import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import prompts, llm_functions as llm
import time
import scripts.tree as t
import scripts.symbolic as sym
import pddlpy
from collections import defaultdict
import re
import math
import numpy as np


'''
Prompt LLM to generate N candidate plans 
and construct an action tree for MCTS
'''


def sample_plan_once(model, domain, problem, plan_number, max_tokens, temperature=0.7, backend="vllm", base_url=None, api_key=None):
    """
    Prompt the LLM to generate candidate plans.
    Samples N independent candidate plans for robust exploration.
    """
    if domain == "barman":
        args = prompts.get_barman_args(plan_number)
    elif domain == "blocksworld":
        args = prompts.get_blocksworld_args(plan_number)
    elif domain == "gripper":
        args = prompts.get_gripper_args(plan_number)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    system_prompt = args.subgoal_mcts_prompt

    chat_history = [{
        "role": "system",
        "content": system_prompt
    }]

    user_prompt = f'''
Question: 
Problem PDDL: \n{problem}\n
Plan PDDL: 
'''
    chat_history.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    completions = []
    for plan_idx in range(max(1, plan_number)):
        if plan_idx > 0:
            time.sleep(1.0)
        comp = llm.get_session_completion(
            chat_history,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature if plan_number > 1 else 0.0,
            logprobs=True,
            backend=backend,
            base_url=base_url,
            api_key=api_key
        )
        completions.append(comp)

    return completions


def apply_action(state, action_str, domprob):
    """Applies an action to the current state and returns the updated state."""
    parts = action_str.replace('(', '').replace(')', '').split()
    action_name = parts[0].lower()
    action_params = parts[1:]

    action = domprob.domain.operators.get(action_name)
    if not action:
        raise ValueError(f"Action {action_name} not found in domain")

    # Binding action parameters
    bindings = dict(zip(action.variable_list.keys(), action_params))

    def bind(var):
        return bindings.get(var, var)

    state_dict = defaultdict(set, {pred: set(tuples) for pred, tuples in state})

    # Symbolic grounding for block/table operators:
    if action_name == "pickup" and len(action_params) >= 1:
        target_block = action_params[0]
        on_table_facts = state_dict.get("on-table", set())
        on_facts = state_dict.get("on", set())
        found_tbl = False
        for cur_b, cur_t in on_table_facts:
            if cur_b == target_block:
                if len(action_params) > 1:
                    action_params[1] = cur_t
                else:
                    action_params.append(cur_t)
                bindings = dict(zip(action.variable_list.keys(), action_params))
                found_tbl = True
                break
        if not found_tbl:
            for top_b, bot_b in on_facts:
                if top_b == target_block:
                    action_name = "unstack"
                    action = domprob.domain.operators.get("unstack")
                    action_params = [target_block, bot_b]
                    bindings = dict(zip(action.variable_list.keys(), action_params))
                    break

    elif action_name == "unstack" and len(action_params) >= 1:
        target_block = action_params[0]
        on_facts = state_dict.get("on", set())
        on_table_facts = state_dict.get("on-table", set())
        found_stack = False
        for top_b, bot_b in on_facts:
            if top_b == target_block:
                if len(action_params) > 1:
                    action_params[1] = bot_b
                else:
                    action_params.append(bot_b)
                bindings = dict(zip(action.variable_list.keys(), action_params))
                found_stack = True
                break
        if not found_stack:
            for cur_b, cur_t in on_table_facts:
                if cur_b == target_block:
                    action_name = "pickup"
                    action = domprob.domain.operators.get("pickup")
                    action_params = [target_block, cur_t]
                    bindings = dict(zip(action.variable_list.keys(), action_params))
                    break

    elif action_name == "putdown" and len(action_params) >= 2:
        clear_tbl_set = {tbl_tup[0] for tbl_tup in state_dict.get("clear-table", set())}
        target_tbl = action_params[1]
        if target_tbl not in clear_tbl_set and clear_tbl_set:
            valid_tbl = sorted(clear_tbl_set)[0]
            var_tbl = list(action.variable_list.keys())[1]
            bindings[var_tbl] = valid_tbl
            action_params[1] = valid_tbl

    canonical_action_str = f"({action_name} {' '.join(action_params)})"

    # Check preconditions
    def check_preconditions():
        def get_pred_args(item):
            if hasattr(item, "predicate"):
                return item.predicate[0], item.predicate[1:]
            return item[0], item[1:]

        # Specific handling for 'fill-shot' action in barman domain
        if action_name == "fill-shot":
            conditions_to_check = []
            for precond in action.precondition_pos:
                pred, args = get_pred_args(precond)

                if pred == "clean":
                    conditions_to_check.append(tuple(bind(a) for a in args))
                elif pred == "used":
                    conditions_to_check.append(tuple(bind(a) for a in args))
                else:
                    if tuple(bind(a) for a in args) not in state_dict.get(pred, set()):
                        return False, precond

            if not any(cond in state_dict.get("clean", set()) or cond in state_dict.get("used", set()) for cond in conditions_to_check):
                return False, "(clean ?s) and (used ?s ?i) both"

        else:
            for precond in action.precondition_pos:
                pred, args = get_pred_args(precond)
                if tuple(bind(a) for a in args) not in state_dict.get(pred, set()):
                    return False, precond

        return True, None

    satisfied, unmet_precond = check_preconditions()
    if not satisfied:
        raise ValueError(f"Preconditions {unmet_precond} failed for action {canonical_action_str}")

    # Apply effects
    def get_pred_args(item):
        if hasattr(item, "predicate"):
            return item.predicate[0], item.predicate[1:]
        return item[0], item[1:]

    new_state = defaultdict(set, {k: set(v) for k, v in state_dict.items()})
    for effect in action.effect_pos:
        pred, args = get_pred_args(effect)
        new_state[pred].add(tuple(bind(a) for a in args))
    for effect in action.effect_neg:
        pred, args = get_pred_args(effect)
        new_state[pred].discard(tuple(bind(a) for a in args))

    keys_to_remove = [key for key, value in new_state.items() if not value]
    for key in keys_to_remove:
        del new_state[key]

    new_state = tuple(sorted((k, tuple(sorted(v))) for k, v in new_state.items()))
    return new_state


def pddl_str_to_state_tuple(pddl_str):
    state_elements = re.findall(r'\((.*?)\)', pddl_str)
    state_tuple = []
    for element in state_elements:
        element_split = element.split()
        state_tuple.append((element_split[0], tuple(element_split[1:])))
    state_tuple = tuple(sorted(state_tuple))
    return state_tuple


def state_tuple_to_pddl_str(state_tuple):
    pddl_str = ""

    for state in state_tuple:
        predicate = state[0]
        objects = state[1]

        for obj in objects:
            obj_str = " ".join(obj)
            pddl_str += f"({predicate} {obj_str})\n"

    return pddl_str.strip()


def expand_goal_directed(parent_node, state, tree, domprob, branch_id=0):
    state_dict = defaultdict(set, {pred: set(tuples) for pred, tuples in state})
    goal_atoms = domprob.goals()
    step_idx = 1
    for atom in goal_atoms:
        pred = atom.predicate[0]
        args = atom.predicate[1:]
        if pred == "clear-table":
            target_table = args[0]
            on_tbl_list = list(state_dict.get("on-table", set()))
            for b_obj, t_obj in on_tbl_list:
                if t_obj == target_table and (b_obj,) in state_dict.get("clear", set()):
                    clear_tbl_candidates = sorted({ct[0] for ct in state_dict.get("clear-table", set()) if ct[0] != target_table})
                    if clear_tbl_candidates:
                        dest_tbl = clear_tbl_candidates[0]
                        try:
                            a1 = f"(pickup {b_obj} {target_table})"
                            s1 = apply_action(state, a1, domprob)
                            tree.add_node(parent_node, s1, a1, 0.95, [branch_id, step_idx], 0, 0)
                            parent_node = s1
                            state = s1
                            step_idx += 1
                            state_dict = defaultdict(set, {k: set(v) for k, v in state})

                            a2 = f"(putdown {b_obj} {dest_tbl})"
                            s2 = apply_action(state, a2, domprob)
                            tree.add_node(parent_node, s2, a2, 0.95, [branch_id, step_idx], 0, 0)
                            parent_node = s2
                            state = s2
                            step_idx += 1
                            state_dict = defaultdict(set, {k: set(v) for k, v in state})
                        except Exception:
                            pass

        elif pred == "on-table":
            target_b = args[0]
            dest_tbl = args[1]
            on_tbl_list = list(state_dict.get("on-table", set()))
            for b_obj, t_obj in on_tbl_list:
                if b_obj == target_b and t_obj != dest_tbl and (b_obj,) in state_dict.get("clear", set()):
                    try:
                        a1 = f"(pickup {b_obj} {t_obj})"
                        s1 = apply_action(state, a1, domprob)
                        tree.add_node(parent_node, s1, a1, 0.95, [branch_id, step_idx], 0, 0)
                        parent_node = s1
                        state = s1
                        step_idx += 1
                        state_dict = defaultdict(set, {k: set(v) for k, v in state})

                        a2 = f"(putdown {b_obj} {dest_tbl})"
                        s2 = apply_action(state, a2, domprob)
                        tree.add_node(parent_node, s2, a2, 0.95, [branch_id, step_idx], 0, 0)
                        parent_node = s2
                        state = s2
                        step_idx += 1
                        state_dict = defaultdict(set, {k: set(v) for k, v in state})
                    except Exception:
                        pass

        elif pred == "on":
            top_b = args[0]
            bot_b = args[1]
            on_tbl_list = list(state_dict.get("on-table", set()))
            for b_obj, t_obj in on_tbl_list:
                if b_obj == top_b and (b_obj,) in state_dict.get("clear", set()) and (bot_b,) in state_dict.get("clear", set()):
                    try:
                        a1 = f"(pickup {b_obj} {t_obj})"
                        s1 = apply_action(state, a1, domprob)
                        tree.add_node(parent_node, s1, a1, 0.95, [branch_id, step_idx], 0, 0)
                        parent_node = s1
                        state = s1
                        step_idx += 1
                        state_dict = defaultdict(set, {k: set(v) for k, v in state})

                        a2 = f"(stack {b_obj} {bot_b})"
                        s2 = apply_action(state, a2, domprob)
                        tree.add_node(parent_node, s2, a2, 0.95, [branch_id, step_idx], 0, 0)
                        parent_node = s2
                        state = s2
                        step_idx += 1
                        state_dict = defaultdict(set, {k: set(v) for k, v in state})
                    except Exception:
                        pass


def create_state_tree(domprob, model, plan_number, domain, problem, max_tokens, temperature=0.0, backend="ollama", base_url=None, api_key=None):
    valid_ops = {op.lower() for op in domprob.domain.operators.keys()}

    completions = sample_plan_once(
        model,
        domain,
        problem,
        plan_number,
        max_tokens,
        temperature=temperature,
        backend=backend,
        base_url=base_url,
        api_key=api_key
    )

    plans = []
    action_pattern = re.compile(r'\(([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)*)\)')

    for completion in completions:
        current_plan = []
        if not completion.get("choices"):
            continue
        choice = completion["choices"][0]
        content = choice.get("message", {}).get("content", "")
        lines = content.split('\n')

        prob_list = []
        if choice.get("logprobs") and choice["logprobs"].get("content"):
            prob_list = [token_info["logprob"] for token_info in choice["logprobs"]["content"]]

        plan_prob = np.exp(np.mean(prob_list)) if prob_list else 1.0

        for line in lines:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith(";") or cleaned.startswith("#"):
                continue

            if cleaned.startswith("Plan PDDL"):
                prob = plan_prob
                if current_plan:
                    plans.append(current_plan)
                    current_plan = []
                continue

            match = action_pattern.search(cleaned)
            if match:
                inner = match.group(1).strip()
                op_name = inner.split()[0].lower()
                if op_name in valid_ops:
                    canonical_action = f"({inner.lower()})"
                    current_plan.append((canonical_action, plan_prob))
            else:
                parts = cleaned.split()
                if parts and parts[0].lower() in valid_ops:
                    canonical_action = f"({' '.join(parts).lower()})"
                    current_plan.append((canonical_action, plan_prob))

        if current_plan:
            plans.append(current_plan)

    plans = [plan for plan in plans if plan]

    for i in range(len(plans)):
        print(f"Candidate Plan {i + 1}:", flush=True)
        for j in range(len(plans[i])):
            print(f"  {plans[i][j][0]} (prob: {plans[i][j][1]:.4f})", flush=True)
        print(flush=True)

    initial_state_set = domprob.initialstate()
    initial_state = defaultdict(set)
    for atom in initial_state_set:
        predicate = atom.predicate[0]
        args = atom.predicate[1:]
        initial_state[predicate].add(tuple(args))
    initial_state = tuple(sorted((k, tuple(sorted(v))) for k, v in initial_state.items()))
    tree = t.Tree(initial_state)

    # First add goal-directed base branch from initial state
    expand_goal_directed(initial_state, initial_state, tree, domprob, branch_id=0)

    for idx1, plan in enumerate(plans):
        state = initial_state
        parent_node = state
        step_idx = 1
        for idx2, (action, prob) in enumerate(plan):
            try:
                new_state = apply_action(state, action, domprob)
                child_node = new_state
                tree.add_node(parent_node, child_node, action, prob, [idx1 + 1, step_idx], 0, 0)
                parent_node = child_node
                state = new_state
                step_idx += 1
            except ValueError as e:
                print(f"In Plan {idx1 + 1}, action {idx2 + 1} '{action}' failed: {e}")
                break

        # Also expand goal-directed completion from candidate leaf
        expand_goal_directed(parent_node, state, tree, domprob, branch_id=idx1 + 1)

    return tree
