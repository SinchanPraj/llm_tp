import sys
import os
import time
import json
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import llm_functions as llm
import scripts.tree_generation as tg
import scripts.subgoal as subgoal
import scripts.symbolic as sym
import scripts.mcts as mcts
import pddlpy


def test_ollama_backend():
    print("=" * 65, flush=True)
    print("TEST 1: Testing Ollama Backend (Symbolic LLM Subgoals)", flush=True)
    print("=" * 65, flush=True)
    print(f"Ollama Endpoint: {llm.OLLAMA_BASE_URL}", flush=True)
    print(f"Ollama Model:    {llm.DEFAULT_OLLAMA_MODEL}", flush=True)

    test_prompt = [
        {"role": "system", "content": "You are a task planning assistant. Output intermediate goals in PDDL format like (:goal (on a b))."},
        {"role": "user", "content": "Problem PDDL:\n(define (problem prob)\n  (:domain blocksworld)\n  (:objects b1 b2 b3 - block t1 t2 t3 - table)\n  (:init (arm-empty) (on b1 b2) (on b2 b3) (on-table b3 t3) (clear b1) (clear-table t1) (clear-table t2))\n  (:goal (and (on-table b1 t3) (on b3 b1) (on b2 b3))))\nSubgoals:"}
    ]

    try:
        start_t = time.perf_counter()
        resp = llm.get_session_completion(
            test_prompt,
            model=llm.DEFAULT_OLLAMA_MODEL,
            max_tokens=256,
            temperature=0.0,
            logprobs=False,
            backend="ollama",
            timeout=60
        )
        elapsed = time.perf_counter() - start_t
        content = resp["choices"][0]["message"]["content"]
        print(f"[SUCCESS] Ollama responded in {elapsed:.2f}s:", flush=True)
        print(f"Response:\n{content.strip()}\n", flush=True)
        return True
    except Exception as e:
        print(f"[FAIL] Ollama connection failed: {e}\n", flush=True)
        return False


def is_server_reachable(base_url, timeout=1.5):
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Authorization": "Bearer local-key"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def test_vllm_logprobs_seqprob():
    print("=" * 65, flush=True)
    print("TEST 2: Testing vLLM Backend (MCTS LLM Seqprob Generation)", flush=True)
    print("=" * 65, flush=True)
    print(f"vLLM Endpoint: {llm.VLLM_BASE_URL}", flush=True)
    print(f"vLLM Model:    {llm.DEFAULT_VLLM_MODEL}", flush=True)

    server_up = is_server_reachable(llm.VLLM_BASE_URL, timeout=1.5)

    if server_up:
        test_prompt = [
            {"role": "system", "content": "You are a PDDL planner. Output only candidate plans."},
            {"role": "user", "content": "Question:\nProblem PDDL:\n(:init (ontable a) (ontable b) (clear a) (clear b) (handempty))\n(:goal (on a b))\n\n2 different plan PDDL:\n"}
        ]
        try:
            start_t = time.perf_counter()
            resp = llm.get_session_completion(
                test_prompt,
                model=llm.DEFAULT_VLLM_MODEL,
                max_tokens=256,
                temperature=0.7,
                logprobs=True,
                backend="vllm",
                timeout=30
            )
            elapsed = time.perf_counter() - start_t
            print(f"[SUCCESS] vLLM live endpoint responded in {elapsed:.2f}s with token logprobs!", flush=True)
            
            actions_with_prob = llm.seq_prob(resp)
            print("\nCalculated Action Sequence Probabilities via seq_prob():", flush=True)
            for action, prob in actions_with_prob:
                print(f"  Action: {action:<30} -> Prob: {prob:.6f}", flush=True)
            print(flush=True)
            return True
        except Exception as e:
            print(f"[NOTICE] Live request to vLLM failed: {e}", flush=True)

    print(f"[STATUS] Live vLLM server is currently not running on {llm.VLLM_BASE_URL}.", flush=True)
    print("To launch vLLM serving Qwen3-32B or Qwen2.5 for MCTS, execute:", flush=True)
    print("  vllm serve Qwen/Qwen3-32B --host 127.0.0.1 --port 8000 --dtype auto --api-key local-key --max-model-len 16384\n", flush=True)

    print("Verifying seq_prob() token-logprob calculation engine on candidate plans:", flush=True)
    simulated_completion = {
        "choices": [
            {
                "message": {
                    "content": "Plan PDDL 1\n(unstack b1 b2)\n(putdown b1 t1)\n\nPlan PDDL 2\n(unstack b1 b2)\n(putdown b1 t2)\n(unstack b2 b3)\n"
                },
                "logprobs": {
                    "content": [
                        {"token": "Plan", "logprob": -0.05},
                        {"token": " P", "logprob": -0.01},
                        {"token": "DDL", "logprob": -0.001},
                        {"token": " 1\n", "logprob": -0.02},
                        {"token": "(", "logprob": -0.01},
                        {"token": "un", "logprob": -0.05},
                        {"token": "stack", "logprob": -0.10},
                        {"token": " b", "logprob": -0.02},
                        {"token": "1", "logprob": -0.01},
                        {"token": " b", "logprob": -0.02},
                        {"token": "2", "logprob": -0.01},
                        {"token": ")\n", "logprob": -0.01},
                        {"token": "(", "logprob": -0.01},
                        {"token": "put", "logprob": -0.15},
                        {"token": "down", "logprob": -0.01},
                        {"token": " b", "logprob": -0.02},
                        {"token": "1", "logprob": -0.01},
                        {"token": " t", "logprob": -0.03},
                        {"token": "1", "logprob": -0.01},
                        {"token": ")\n", "logprob": -0.01},
                        {"token": "\n", "logprob": -0.05},
                        {"token": "Plan", "logprob": -0.1},
                        {"token": " P", "logprob": -0.01},
                        {"token": "DDL", "logprob": -0.001},
                        {"token": " 2\n", "logprob": -0.02},
                        {"token": "(", "logprob": -0.01},
                        {"token": "un", "logprob": -0.05},
                        {"token": "stack", "logprob": -0.10},
                        {"token": " b", "logprob": -0.02},
                        {"token": "1", "logprob": -0.01},
                        {"token": " b", "logprob": -0.02},
                        {"token": "2", "logprob": -0.01},
                        {"token": ")\n", "logprob": -0.01},
                        {"token": "(", "logprob": -0.01},
                        {"token": "put", "logprob": -0.15},
                        {"token": "down", "logprob": -0.01},
                        {"token": " b", "logprob": -0.02},
                        {"token": "1", "logprob": -0.01},
                        {"token": " t", "logprob": -0.03},
                        {"token": "2", "logprob": -0.01},
                        {"token": ")\n", "logprob": -0.01},
                        {"token": "(", "logprob": -0.01},
                        {"token": "un", "logprob": -0.05},
                        {"token": "stack", "logprob": -0.10},
                        {"token": " b", "logprob": -0.02},
                        {"token": "2", "logprob": -0.01},
                        {"token": " b", "logprob": -0.02},
                        {"token": "3", "logprob": -0.01},
                        {"token": ")\n", "logprob": -0.01},
                    ]
                }
            }
        ]
    }
    res = llm.seq_prob(simulated_completion)
    for act, pr in res:
        print(f"  [Verified seq_prob] {act:<30} -> Prob: {pr:.6f}", flush=True)
    print(flush=True)
    return True


def test_mcts_tree_search():
    print("=" * 65, flush=True)
    print("TEST 3: Testing PDDL Simulation & MCTS Tree Search", flush=True)
    print("=" * 65, flush=True)
    domain_f = os.path.join(PROJECT_ROOT, "domains", "domain_blocksworld.pddl")
    problem_f = os.path.join(PROJECT_ROOT, "experiments", "blocksworld", "problem", "blocksworld3_1.pddl")

    if not os.path.exists(domain_f) or not os.path.exists(problem_f):
        print("[SKIP] PDDL test files not found.", flush=True)
        return

    domprob = pddlpy.DomainProblem(domain_f, problem_f)
    print(f"Domain:  {domain_f}", flush=True)
    print(f"Problem: {problem_f}", flush=True)

    initial_state_set = domprob.initialstate()
    from collections import defaultdict
    import scripts.tree as t

    initial_state = defaultdict(set)
    for atom in initial_state_set:
        predicate = atom.predicate[0]
        args = atom.predicate[1:]
        initial_state[predicate].add(tuple(args))
    initial_state = tuple(sorted((k, tuple(sorted(v))) for k, v in initial_state.items()))

    tree = t.Tree(initial_state)

    # Valid candidate plan that solves the goal
    valid_plan = [
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

    # Alternative candidate plan with suboptimal / different branch
    alt_plan = [
        ("(unstack b1 b2)", 0.95),
        ("(putdown b1 t5)", 0.70),
        ("(unstack b2 b3)", 0.85),
        ("(putdown b2 t6)", 0.65)
    ]

    candidate_plans = [valid_plan, alt_plan]

    for idx1, plan in enumerate(candidate_plans):
        state = initial_state
        parent_node = state
        for idx2, (action, prob) in enumerate(plan):
            try:
                new_state = tg.apply_action(state, action, domprob)
                child_node = new_state
                tree.add_node(parent_node, child_node, action, prob, [idx1 + 1, idx2 + 1], 0, 0)
                parent_node = child_node
                state = new_state
            except ValueError as e:
                print(f"Action failed in simulation: {action}: {e}")
                break

    print(f"Action tree constructed with {len(tree.G.nodes)} state nodes and {len(tree.G.edges)} transitions.", flush=True)

    start_t = time.perf_counter()
    states_path, actions_path = mcts.mcts(tree, domprob, max_iterations=100)
    mcts_t = time.perf_counter() - start_t

    print(f"MCTS Search completed in {mcts_t:.4f}s.", flush=True)
    print(f"Selected Optimal Action Path ({len(actions_path)} steps):", flush=True)
    for step, act in enumerate(actions_path, 1):
        print(f"  {step}. {act}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    print("\n" + "=" * 65, flush=True)
    print("LLM-TP Backend & MCTS / Symbolic Exploration Runner", flush=True)
    print("=" * 65 + "\n", flush=True)
    test_ollama_backend()
    test_vllm_logprobs_seqprob()
    test_mcts_tree_search()
    print("=" * 65, flush=True)
    print("Exploration & Verification Complete.")
    print("=" * 65 + "\n", flush=True)
