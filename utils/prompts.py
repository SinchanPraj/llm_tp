import argparse
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_file(rel_path):
    # Normalize relative path
    clean_path = rel_path.replace("\\", "/")
    if clean_path.startswith("../"):
        clean_path = clean_path[3:]
    elif clean_path.startswith("./"):
        clean_path = clean_path[2:]

    full_path = os.path.join(PROJECT_ROOT, clean_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    if os.path.exists(rel_path):
        with open(rel_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_barman_args(K):
    parser = argparse.ArgumentParser()

    domain = _read_file("domains/domain_barman.pddl")
    problem_context_1 = _read_file("experiments/barman/problem/barman_context_1.pddl")
    problem_context_2 = _read_file("experiments/barman/problem/barman_context_2.pddl")
    plan_context_1 = _read_file("experiments/barman/plan/barman_context_1.pddl")
    plan_context_2 = _read_file("experiments/barman/plan/barman_context_2.pddl")
    plan_context_1_1 = _read_file("experiments/barman/plan/barman_context_1_1.pddl")
    plan_context_2_2 = _read_file("experiments/barman/plan/barman_context_2_2.pddl")
    plan_context_2_3 = _read_file("experiments/barman/plan/barman_context_2_3.pddl")
    problem_subgoal_3_1 = _read_file("experiments/barman/problem/barman_subgoal_3_1.pddl")
    problem_subgoal_3_2 = _read_file("experiments/barman/problem/barman_subgoal_3_2.pddl")
    problem_subgoal_3_3 = _read_file("experiments/barman/problem/barman_subgoal_3_3.pddl")
    plan_subgoal_3_1 = _read_file("experiments/barman/plan/barman_subgoal_3_1.pddl")
    plan_subgoal_3_2 = _read_file("experiments/barman/plan/barman_subgoal_3_2.pddl")
    plan_subgoal_3_3 = _read_file("experiments/barman/plan/barman_subgoal_3_3.pddl")
    plan_subgoal_3_2_1 = _read_file("experiments/barman/plan/barman_subgoal_3_2_1.pddl")
    plan_subgoal_3_3_1 = _read_file("experiments/barman/plan/barman_subgoal_3_3_1.pddl")


    parser.add_argument("--domain", type=str, default=domain)
    parser.add_argument("--problem_context_1", type=str, default=problem_context_1)
    parser.add_argument("--problem_context_2", type=str, default=problem_context_2)
    parser.add_argument("--plan_context_1", type=str, default=plan_context_1)
    parser.add_argument("--plan_context_2", type=str, default=plan_context_2)

    cot_prompt = f''' 
    You are a helpful assistant confident in PDDL planning.
    Given a domain PDDL, a problem PDDL and several planning examples, generate a plan PDDL to achieve the goal state.
    Employ chain-of-thought reasoning to ensure each step logically follows from the previous one. 
    Let's think step by step.
    Do not use quotation marks for code blocks, and do not start with expressions like 'plan'.
    Return ONLY the plan PDDL, NO other words or explanation. Return only the pddl, excluding comments.
       
    Follow the pattern of each examples below. 

    Domain PDDL:\n{domain}\n
    
    Rules:
    When preparing cocktail, grasp a shot with largest index. 
    For example, when there are 9 shots (shot1 to shot9), then always use shot9. When there are 10 shots (shot1 to shot10), then always use shot10. When there are 11 shots (shot1 to shot11), then always use shot11. 

    Domain PDDL:
    {domain}

    Example 1
    Problem PDDL: 
    \n{problem_context_1}\n

    Thoughts:
    1. To prepare cocktail, grasp a shot with largest index.
    2. Prepare first ingredient
    3. Clean the shot
    4. Prepare second ingredient
    5. Leave the shot and grasp shaker
    6. Shake the ingredients and pour the cocktail
 
    Plan PDDL:
    {plan_context_1_1}


    Example 2
    Problem PDDL: 
    {problem_context_2}

    Thoughts:
    1. Prepare the first cocktail as in Example 1.
    2. Before preparing the second cocktail, empty, clean, and leave the shaker.
    3. Prepare the second cocktail.
    1) If the shot already contains the first ingredient, prepare the first ingredient immediately.
    2) Else, clean the second ingredient of previous cocktail from the shot, and then prepare the first ingredient of current cocktail. 

    Plan PDDL:
    {plan_context_2_2}
    '''

    subgoal_prompt = f'''
    You are a helpful assistant generating PDDL subgoals.  
    Given a domain PDDL and a problem PDDL, generate subgoals in the from of PDDL goal state. 
    Decide each subgoal by looking at how the problem is solved based on the plan example provided for another task, and breaking down the original goal state into multiple subgoals.
    Create subgoals in the order they need to be achieved first.
    Distinguish each subgoal as 'Subgoal 1:', 'Subgoal 2:', and so on.
    ONLY return PDDL goal. Return nothing else.
    
    Domain PDDL:
    {domain}
        
    Example Problem PDDL 1:
    {plan_context_2}
    
    The following steps are needed to achieve the given goal state:
    {plan_context_2_3}
    
    Example subgoals
    Subgoal 1:
    (:goal (and (contains shot2 cocktail1)))
    
    Subgoal 2:
    (:goal (and (contains shot1 cocktail2)))
    '''

    subgoal_mcts_prompt = f'''
    You are a helpful assistant confident in PDDL planning.
    Given a domain PDDL, a problem PDDL and several planning examples, generate {K} different plan PDDLs each starting with "Plan PDDL:"
    Do not use quotation marks for code blocks, and do not start with expressions like 'plan'.
    Return ONLY the plan PDDLs starting with "Plan PDDL:", NO other words or explanation. Return only the pddl, excluding comments.
            
    Each plan should be different with every previous plans. To generate various plans, differentiate the order of prepared ingredients.
        
    Follow the pattern of each examples below. 
    Let's think step by step.

    Domain PDDL:\n{domain}\n
    
    Rules:
    When preparing cocktail, grasp a shot with largest index. 
    For example, when there are 9 shots (shot1 to shot9), then always use shot9. When there are 10 shots (shot1 to shot10), then always use shot10. When there are 11 shots (shot1 to shot11), then always use shot11. 

    Example 1
    Problem PDDL: 
    \n{problem_subgoal_3_1}\n

    Thoughts:
    1. To prepare cocktail, grasp a shot with largest index.
    2. Prepare first ingredient
    3. Clean the shot
    4. Prepare second ingredient
    5. Leave the shot and grasp shaker
    6. Shake the ingredients and pour the cocktail
    
    Plan PDDL:
    {plan_subgoal_3_1}
    

    Example 2
    Problem PDDL: 
    {problem_subgoal_3_2}
    
    Thoughts:
    1. The shaker is shaked(not empty, not clean), so empty, clean and leave the shaker.
    2. To prepare cocktail, grasp a shot with largest index.
    3. Since shot4 is used with ingredient2, let's clean the shot first.
    4. Prepare first ingredient
    5. Clean the shot
    6. Prepare second ingredient
    7. Leave the shot and grasp shaker
    8. Shake the ingredients and pour the cocktail
    
    Plan PDDL 1:
    {plan_subgoal_3_2}
    
    Plan PDDL 2:
    {plan_subgoal_3_2_1}
    
    
    Example 3
    Problem PDDL: 
    {problem_subgoal_3_3}
    
    Thoughts:
    1. The shaker is shaked(not empty, not clean), so empty, clean and leave the shaker.
    2. To prepare cocktail, grasp a shot with largest index. 
    3. Since shot4 is used with ingredient2, let's clean the shot first.
    4. Prepare first ingredient
    5. Clean the shot
    6. Prepare second ingredient
    7. Leave the shot and grasp shaker
    8. Shake the ingredients and pour the cocktail
    
    Plan PDDL 1:
    {plan_subgoal_3_3}
    
    Plan PDDL 2:
    {plan_subgoal_3_3_1}
    
    '''

    mcts_prompt = f'''
    You are a helpful assistant confident in PDDL planning.
    Given a domain PDDL, a problem PDDL and several planning examples, generate {K} different plan PDDLs each starting with "Plan PDDL:"
    Do not use quotation marks for code blocks, and do not start with expressions like 'plan'.
    Return ONLY the plan PDDLs starting with "Plan PDDL:", NO other words or explanation. Return only the pddl, excluding comments.

    Each plan should be different with every previous plans. To generate various plans, differentiate the order of prepared ingredients.

    Follow the pattern of each examples below. 
    Let's think step by step.

    Domain PDDL:\n{domain}\n

    Rules:
    When preparing cocktail, grasp a shot with largest index. 
    For example, when there are 9 shots (shot1 to shot9), then always use shot9. When there are 10 shots (shot1 to shot10), then always use shot10. When there are 11 shots (shot1 to shot11), then always use shot11. 

    Example 1
    Problem PDDL: 
    \n{problem_context_1}\n

    Thoughts:
    1. To prepare cocktail, grasp a shot with largest index.
    2. Prepare first ingredient
    3. Clean the shot
    4. Prepare second ingredient
    5. Leave the shot and grasp shaker
    6. Shake the ingredients and pour the cocktail
 
    Plan PDDL:
    {plan_context_1_1}


    Example 2
    Problem PDDL: 
    {problem_context_2}

    Thoughts:
    1. Prepare the first cocktail as in Example 1.
    2. Before preparing the second cocktail, empty, clean, and leave the shaker.
    3. Prepare the second cocktail.
    1) If the shot already contains the first ingredient, prepare the first ingredient immediately.
    2) Else, clean the second ingredient of previous cocktail from the shot, and then prepare the first ingredient of current cocktail. 

    Plan PDDL:
    {plan_context_2_2}
    '''

    parser.add_argument("--cot_prompt", type=str, default=cot_prompt)
    parser.add_argument("--subgoal_prompt", type=str, default=subgoal_prompt)
    parser.add_argument("--subgoal_mcts_prompt", type=str, default=subgoal_mcts_prompt)
    parser.add_argument("--mcts_prompt", type=str, default=mcts_prompt)

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


def get_blocksworld_args(K):
    parser = argparse.ArgumentParser()

    # open files
    domain = _read_file("domains/domain_blocksworld.pddl")
    problem_3_1 = _read_file("experiments/blocksworld/problem/blocksworld3_1.pddl")
    plan_3_1 = _read_file("experiments/blocksworld/plan/blocksworld_3_31.pddl")
    with open("experiments/blocksworld/problem/blocksworld_context_8_1.pddl", "r") as f:
        problem_context_8_1 = f.read()
    with open("experiments/blocksworld/plan/blocksworld_context_8_1.pddl", "r") as f:
        plan_context_8_1 = f.read()
    with open("experiments/blocksworld/problem/blocksworld_context_9_1.pddl", "r") as f:
        problem_context_9_1 = f.read()
    with open("experiments/blocksworld/problem/blocksworld_context_9_2.pddl", "r") as f:
        problem_context_9_2 = f.read()
    with open("experiments/blocksworld/problem/blocksworld_context_10_1.pddl", "r") as f:
        problem_context_10_1 = f.read()
    with open("experiments/blocksworld/problem/blocksworld_context_10_2.pddl", "r") as f:
        problem_context_10_2 = f.read()
    with open("experiments/blocksworld/plan/blocksworld_context_9_1.pddl", "r") as f:
        plan_context_9_1 = f.read()
    with open("experiments/blocksworld/plan/blocksworld_context_9_2.pddl", "r") as f:
        plan_context_9_2 = f.read()
    with open("experiments/blocksworld/plan/blocksworld_context_10_1.pddl", "r") as f:
        plan_context_10_1 = f.read()
    with open("experiments/blocksworld/plan/blocksworld_context_10_2.pddl", "r") as f:
        plan_context_10_2 = f.read()

    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_0.pddl", "r") as f:
        plan_8_0 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_1.pddl", "r") as f:
        plan_8_1 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_2.pddl", "r") as f:
        plan_8_2 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_3.pddl", "r") as f:
        plan_8_3 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_4.pddl", "r") as f:
        plan_8_4 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_5.pddl", "r") as f:
        plan_8_5 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_6.pddl", "r") as f:
        plan_8_6 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_7.pddl", "r") as f:
        plan_8_7 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_8.pddl", "r") as f:
        plan_8_8 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_9.pddl", "r") as f:
        plan_8_9 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld8_31_K0_10.pddl", "r") as f:
        plan_8_10 = f.read()

    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_0.pddl", "r") as f:
        plan_9_0 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_1.pddl", "r") as f:
        plan_9_1 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_2.pddl", "r") as f:
        plan_9_2 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_3.pddl", "r") as f:
        plan_9_3 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_4.pddl", "r") as f:
        plan_9_4 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_5.pddl", "r") as f:
        plan_9_5 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_6.pddl", "r") as f:
        plan_9_6 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_7.pddl", "r") as f:
        plan_9_7 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_8.pddl", "r") as f:
        plan_9_8 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_9.pddl", "r") as f:
        plan_9_9 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_10.pddl", "r") as f:
        plan_9_10 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld9_31_K0_11.pddl", "r") as f:
        plan_9_11 = f.read()

    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_0.pddl", "r") as f:
        plan_10_0 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_1.pddl", "r") as f:
        plan_10_1 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_2.pddl", "r") as f:
        plan_10_2 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_3.pddl", "r") as f:
        plan_10_3 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_4.pddl", "r") as f:
        plan_10_4 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_5.pddl", "r") as f:
        plan_10_5 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_6.pddl", "r") as f:
        plan_10_6 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_7.pddl", "r") as f:
        plan_10_7 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_8.pddl", "r") as f:
        plan_10_8 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_9.pddl", "r") as f:
        plan_10_9 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_10.pddl", "r") as f:
        plan_10_10 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_11.pddl", "r") as f:
        plan_10_11 = f.read()
    with open("experiments/blocksworld/plan/context/blocksworld10_31_K0_12.pddl", "r") as f:
        plan_10_12 = f.read()

    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_0.pddl", "r") as f:
        problem_8_0 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_1.pddl", "r") as f:
        problem_8_1 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_2.pddl", "r") as f:
        problem_8_2 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_3.pddl", "r") as f:
        problem_8_3 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_4.pddl", "r") as f:
        problem_8_4 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_5.pddl", "r") as f:
        problem_8_5 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_6.pddl", "r") as f:
        problem_8_6 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_7.pddl", "r") as f:
        problem_8_7 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_8.pddl", "r") as f:
        problem_8_8 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_9.pddl", "r") as f:
        problem_8_9 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld8_31_K0_10.pddl", "r") as f:
        problem_8_10 = f.read()

    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_0.pddl", "r") as f:
        problem_9_0 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_1.pddl", "r") as f:
        problem_9_1 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_2.pddl", "r") as f:
        problem_9_2 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_3.pddl", "r") as f:
        problem_9_3 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_4.pddl", "r") as f:
        problem_9_4 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_5.pddl", "r") as f:
        problem_9_5 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_6.pddl", "r") as f:
        problem_9_6 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_7.pddl", "r") as f:
        problem_9_7 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_8.pddl", "r") as f:
        problem_9_8 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_9.pddl", "r") as f:
        problem_9_9 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_10.pddl", "r") as f:
        problem_9_10 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld9_31_K0_11.pddl", "r") as f:
        problem_9_11 = f.read()

    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_0.pddl", "r") as f:
        problem_10_0 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_1.pddl", "r") as f:
        problem_10_1 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_2.pddl", "r") as f:
        problem_10_2 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_3.pddl", "r") as f:
        problem_10_3 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_4.pddl", "r") as f:
        problem_10_4 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_5.pddl", "r") as f:
        problem_10_5 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_6.pddl", "r") as f:
        problem_10_6 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_7.pddl", "r") as f:
        problem_10_7 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_8.pddl", "r") as f:
        problem_10_8 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_9.pddl", "r") as f:
        problem_10_9 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_10.pddl", "r") as f:
        problem_10_10 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_11.pddl", "r") as f:
        problem_10_11 = f.read()
    with open("experiments/blocksworld/problem/context/blocksworld10_31_K0_12.pddl", "r") as f:
        problem_10_12 = f.read()


    cot_prompt = f'''
    You are a rule-following assistant confident in PDDL planning.
    Domain PDDL, problem PDDL and several planning examples are given. Generate a plan PDDL to achieve the goal state.
    Employ chain-of-thought reasoning to ensure each step logically follows from the previous one. 
    Return ONLY one plan PDDL, no other words or explanation. Don't include 'Plan PDDL:' when starting.
    Let's think step by step.
    
    Follow the pattern of each examples below.

    Domain PDDL:\n{domain}\n

    Example 1: 8 blocks
    Problem PDDL 1:
    {problem_context_8_1}
    Thoughts:
    1. (on b6 b7) (on b7 b8) (on-table b8 t3) : change the order of blocks on t3 (b6 to b8).
    2. (on b4 b3) (on b3 b5) (on-table b5 t2) : change the order of blocks on t2 (b3 to b5).
    3. (on b2 b1) (on-table b1 t1) : change the order of blocks on t1 (b1 to b2).
    Plan PDDL 1:
    {plan_context_8_1}

    Example 2: 9 blocks
    Problem PDDL 1: 
    {problem_context_9_1}
    Thoughts:
    1. (on b7 b8) (on b8 b9) (on-table b9 t3) : change the order of blocks on t3 (b7 to b9).
    2. (on b5 b6) (on b6 b4) (on-table b4 t2) : change the order of blocks on t2 (b4 to b6).
    3. (on b3 b1) (on b1 b2) (on-table b2 t1) : change the order of blocks on t1 (b1 to b3).
    Plan PDDL 1:
    {plan_context_9_1}
    
    Problem PDDL 2: 
    {problem_context_9_2}
    Thoughts:
    1. (on b8 b9) (on b9 b7) (on-table b7 t3) : change the order of blocks on t3 (b7 to b9).
    2. (on b4 b6) (on b6 b5) (on-table b5 t2) : change the order of blocks on t2 (b4 to b6). 
    3. (on b3 b2) (on b2 b1) (on-table b1 t1) : change the order of blocks on t1 (b1 to b3).
    Plan PDDL 2:
    {plan_context_9_2}
    
    
    Example 3: 10 blocks
    Problem PDDL 1: 
    {problem_context_10_1}   
    Thoughts:
    1. (on b9 b10) (on b10 b8) (on b8 b7) (on-table b7 t3) : change the order of blocks on t3 (b7 to b10).
    2. (on b6 b4) (on b4 b5) (on-table b5 t2) : change the order of blocks on t2 (b4 to b6).
    3. (on b3 b2) (on b2 b1) (on-table b1 t1) : change the order of blocks on t1 (b1 to b3).
    Plan PDDL 1:
    {plan_context_10_1}
    
    Problem PDDL 2: 
    {problem_context_10_2}
    Thoughts:
    1. (on b7 b9) (on b9 b10) (on b10 b8) (on-table b8 t3)
    2. (on b5 b6) (on b6 b4) (on-table b4 t2)
    3. (on b2 b1) (on b1 b3) (on-table b3 t1)
    Plan PDDL 2:
    {plan_context_10_2}  
    '''

    subgoal_prompt = f'''
    You are a helpful assistant generating PDDL subgoals.  
    Given a domain PDDL and a problem PDDL, decompose the final goal into sequential intermediate subgoals.
    For a 3-block problem, ALWAYS output exactly 4 separate subgoals:
    1. First subgoal: clear the destination table (e.g. (:goal (and (clear b1) (clear-table t3))))
    2. Second subgoal: place bottom block on destination table (e.g. (:goal (and (on-table b1 t3))))
    3. Third subgoal: stack middle block (e.g. (:goal (and (on b3 b1))))
    4. Fourth subgoal: stack top block (e.g. (:goal (and (on b2 b3))))

    ONLY return the PDDL goal expressions like (:goal (and ...)). Return nothing else.

    Domain PDDL:
    {domain}

    Example Problem PDDL:
    (define (problem prob-example)
      (:domain blocksworld)
      (:objects b1 b2 b3 - block t1 t2 t3 - table)
      (:init (on b1 b2) (on b2 b3) (on-table b3 t3) (clear b1) (clear-table t1) (clear-table t2))
      (:goal (and (on-table b1 t3) (on b3 b1) (on b2 b3))))

    Example Subgoals:
    Subgoal 1:
    (:goal (and (clear b1) (clear-table t3)))
    Subgoal 2:
    (:goal (and (on-table b1 t3)))
    Subgoal 3:
    (:goal (and (on b3 b1)))
    Subgoal 4:
    (:goal (and (on b2 b3)))
    '''

    subgoal_mcts_prompt = f'''
    You are a helpful assistant confident in PDDL planning.
    Given a domain PDDL, a problem PDDL and planning examples, generate a concise plan PDDL to achieve the goal state.
    
    Rules:
    - Output ONLY valid PDDL actions starting with "Plan PDDL:", one action per line.
    - When putting a block on a table using (putdown ?b ?t), ?t MUST be an unoccupied table declared with (clear-table ?t) in the initial state.
    
    Domain PDDL:\n{domain}\n
    
    Example 1: Clearing an occupied table
    Problem PDDL:
    (define (problem prob-sub1)
      (:domain blocksworld)
      (:objects b1 b2 b3 - block t1 t2 t3 t4 t5 t6 - table)
      (:init (arm-empty) (on b1 b2) (on b2 b3) (on-table b3 t3) (clear b1) (clear-table t1) (clear-table t2) (clear-table t4) (clear-table t5) (clear-table t6))
      (:goal (and (clear b1) (clear-table t3))))
    Plan PDDL:
    (unstack b1 b2)
    (putdown b1 t1)
    (unstack b2 b3)
    (putdown b2 t2)
    (pickup b3 t3)
    (putdown b3 t4)

    Example 2: Putting a block on a clear table
    Problem PDDL:
    (define (problem prob-sub2)
      (:domain blocksworld)
      (:objects b1 b2 b3 - block t1 t2 t3 t4 t5 t6 - table)
      (:init (arm-empty) (clear b1) (clear b2) (clear b3) (on-table b1 t1) (on-table b2 t2) (on-table b3 t4) (clear-table t3) (clear-table t5) (clear-table t6))
      (:goal (and (on-table b1 t3))))
    Plan PDDL:
    (pickup b1 t1)
    (putdown b1 t3)

    Example 3: Stacking a block
    Problem PDDL:
    (define (problem prob-sub3)
      (:domain blocksworld)
      (:objects b1 b2 b3 - block t1 t2 t3 t4 t5 t6 - table)
      (:init (arm-empty) (clear b1) (clear b2) (clear b3) (on-table b1 t3) (on-table b2 t2) (on-table b3 t4) (clear-table t1) (clear-table t5) (clear-table t6))
      (:goal (and (on b3 b1))))
    Plan PDDL:
    (pickup b3 t4)
    (stack b3 b1)

    Example 4: Stacking another block
    Problem PDDL:
    (define (problem prob-sub4)
      (:domain blocksworld)
      (:objects b1 b2 b3 - block t1 t2 t3 t4 t5 t6 - table)
      (:init (arm-empty) (clear b2) (clear b3) (on-table b1 t3) (on b3 b1) (on-table b2 t2) (clear-table t1) (clear-table t4) (clear-table t5) (clear-table t6))
      (:goal (and (on b2 b3))))
    Plan PDDL:
    (pickup b2 t2)
    (stack b2 b3)
    '''

    mcts_prompt = f'''
    You are a helpful assistant confident in PDDL planning.
    Given a domain PDDL, a problem PDDL and several planning examples, generate {K} different plan PDDLs each starting with "Plan PDDL:"
    Do not use quotation marks for code blocks, and do not start with expressions like 'plan'.
    Return ONLY the plan PDDLs starting with "Plan PDDL:", NO other words or explanation. Return only the pddl, excluding comments.
    
    Each plan should be different with every previous plans.
    
    Follow the pattern of each examples below. 
    Let's think step by step.

    Domain PDDL:\n{domain}\n
    
    Example 1: 8 blocks
    Problem PDDL 1:
    {problem_context_8_1}
    Thoughts:
    1. (on b6 b7) (on b7 b8) (on-table b8 t3) : change the order of blocks on t3 (b6 to b8).
    2. (on b4 b3) (on b3 b5) (on-table b5 t2) : change the order of blocks on t2 (b3 to b5).
    3. (on b2 b1) (on-table b1 t1) : change the order of blocks on t1 (b1 to b2).
    Plan PDDL 1:
    {plan_context_8_1}

    Example 2: 9 blocks
    Problem PDDL 1: 
    {problem_context_9_1}
    Thoughts:
    1. (on b7 b8) (on b8 b9) (on-table b9 t3) : change the order of blocks on t3 (b7 to b9).
    2. (on b5 b6) (on b6 b4) (on-table b4 t2) : change the order of blocks on t2 (b4 to b6).
    3. (on b3 b1) (on b1 b2) (on-table b2 t1) : change the order of blocks on t1 (b1 to b3).
    Plan PDDL 1:
    {plan_context_9_1}
    
    Problem PDDL 2: 
    {problem_context_9_2}
    Thoughts:
    1. (on b8 b9) (on b9 b7) (on-table b7 t3) : change the order of blocks on t3 (b7 to b9).
    2. (on b4 b6) (on b6 b5) (on-table b5 t2) : change the order of blocks on t2 (b4 to b6). 
    3. (on b3 b2) (on b2 b1) (on-table b1 t1) : change the order of blocks on t1 (b1 to b3).
    Plan PDDL 2:
    {plan_context_9_2}
    
    
    Example 3: 10 blocks
    
    Rules:
    - to achieve goal (clear b7)(clear b8)(clear b9)(clear b10), the last action should be ALWAYS (stack b10 b4). You CANNOT put b10 on a table.
    - before putting b10 in new place, unstack b10 from b4. PLEASE don't use (pickup b10 t4). ALWAYS (unstack b10 b4)
    - When generating {K} different plans, try to change actions about b10 when stacking it or unstacking it.
    
    Problem PDDL 1: 
    {problem_context_10_1}   
    Thoughts:
    1. (on b9 b10) (on b10 b8) (on b8 b7) (on-table b7 t3) : change the order of blocks on t3 (b7 to b10).
    2. (on b6 b4) (on b4 b5) (on-table b5 t2) : change the order of blocks on t2 (b4 to b6).
    3. (on b3 b2) (on b2 b1) (on-table b1 t1) : change the order of blocks on t1 (b1 to b3).
    Plan PDDL 1:
    {plan_context_10_1}
    
    Problem PDDL 2: 
    {problem_context_10_2}
    Thoughts:
    1. (on b7 b9) (on b9 b10) (on b10 b8) (on-table b8 t3)
    2. (on b5 b6) (on b6 b4) (on-table b4 t2)
    3. (on b2 b1) (on b1 b3) (on-table b3 t1)
    Plan PDDL 2:
    {plan_context_10_2}
    
    '''

    # parse argements
    parser.add_argument("--cot_prompt", type=str, default=cot_prompt)
    parser.add_argument("--subgoal_prompt", type=str, default=subgoal_prompt)
    parser.add_argument("--subgoal_mcts_prompt", type=str, default=subgoal_mcts_prompt)
    parser.add_argument("--mcts_prompt", type=str, default=mcts_prompt)

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    return args


def get_gripper_args(plan_number):
    parser = argparse.ArgumentParser()

    # open files
    domain = _read_file("domains/domain_gripper.pddl")

    with open("experiments/gripper/problem/gripper2_32.pddl", "r") as f:
        problem_2_32 = f.read()
    with open("experiments/gripper/problem/gripper3_31.pddl", "r") as f:
        problem_3_31 = f.read()
    with open("experiments/gripper/problem/gripper4_31.pddl", "r") as f:
        problem_4_31 = f.read()
    with open("experiments/gripper/plan/gripper2_32.pddl", "r") as f:
        plan_2_32 = f.read()
    with open("experiments/gripper/plan/gripper3_31.pddl", "r") as f:
        plan_3_31 = f.read()
    with open("experiments/gripper/plan/gripper4_31.pddl", "r") as f:
        plan_4_31 = f.read()

    with open("experiments/gripper/problem/gripper3_31_K0_0.pddl", "r") as f:
        problem_subgoal_3_0 = f.read()
    with open("experiments/gripper/problem/gripper3_31_K0_1.pddl", "r") as f:
        problem_subgoal_3_1 = f.read()
    with open("experiments/gripper/problem/gripper3_31_K0_2.pddl", "r") as f:
        problem_subgoal_3_2 = f.read()
    with open("experiments/gripper/plan/gripper3_31_K0_0.pddl", "r") as f:
        plan_subgoal_3_0 = f.read()
    with open("experiments/gripper/plan/gripper3_31_K0_1.pddl", "r") as f:
        plan_subgoal_3_1 = f.read()
    with open("experiments/gripper/plan/gripper3_31_K0_2.pddl", "r") as f:
        plan_subgoal_3_2 = f.read()

    cot_prompt = f'''
    You are a helpful assistant confident in PDDL planning.
    Given a domain PDDL, a problem PDDL and several planning examples, generate a plan PDDL to achieve the goal state.
    Employ chain-of-thought reasoning to ensure each step logically follows from the previous one. 
    Let's think step by step.
    Do not use quotation marks for code blocks, and do not start with expressions like 'plan'.
    Return ONLY the plan PDDL, NO other words or explanation. Return only the pddl, excluding comments.
       
    Follow the pattern of each examples below. 
    
    Domain PDDL:
    {domain}
    
    Example 1
    Problem PDDL:
    {problem_2_32}
    
    Thoughts
    Subgoal 1: (at ball1 room2)
    1. ball1 is in room4, but currently there is no robot in room4. So move robot4 from room3 to room4.
    2. Pick up ball1 with free gripper of robot4.
    3. Move the robot to the room where ball1 should be placed.
    4. Drop ball1 in the room.
    
    Subgoal 2: (at ball2 room3)
    1. ball2 is in room4, but currently there is no robot in room4. So move robot4 from room2 to room4.
    2. Pick up ball2 with free gripper of robot4.
    3. Move the robot to the room where ball2 should be placed.
    4. Drop ball2 in the room.
    
    Plan PDDL:
    {plan_2_32}
    
    
    Example 2
    Problem PDDL: 
    {problem_3_31}
    
    Thoughts
    Subgoal 1: (at ball1 room1)
    1. ball1 is in room3, and robot1 is also in room3. Pick up ball1 with a free gripper of robot1.
    2. Move the robot to the room where ball1 should be placed.
    3. Drop ball1 in the room.
    
    Subgoal 2: (at ball2 room2)
    1. ball2 is in room4, and robot4 is also in room4. Pick up ball2 with a free gripper of robot4.
    2. Move the robot to the room where ball2 should be placed.
    3. Drop ball2 in the room.
    
    Subgoal 3: (at ball3 room4)
    1. ball3 is in room3, but currently there is no robot in room3. So move robot1 from room2 to room3.
    2. Pick up ball3 with free gripper of robot1.
    3. Move the robot to the room where ball3 should be placed.
    4. Drop ball3 in the room.
    
    Plan PDDL:
    {plan_3_31}

    '''

    subgoal_prompt = f'''
    You are a helpful assistant generating PDDL subgoals.  
    Given a domain PDDL and a problem PDDL, generate subgoals in the from of PDDL goal state. 
    Decide each subgoal by looking at how the problem is solved based on the plan example provided for another task, and breaking down the original goal state into multiple subgoals.
    Create subgoals in the order they need to be achieved first.
    Distinguish each subgoal as 'Subgoal 1:', 'Subgoal 2:', and so on.
    ONLY return PDDL goal. Return nothing else.
    
    Domain PDDL:
    {domain}
        
    Example Problem PDDL 1:
    {problem_3_31}
    
    The following steps are needed to achieve the given goal state:
    {plan_3_31}
    
    Example subgoals
    Subgoal 1:
    (:goal (and (at ball1 room1)))
    
    Subgoal 2:
    (:goal (and (at ball2 room2)))
    
    Subgoal 3:
    (:goal (and (at ball3 room4)))
    
    Rules: Just repeat the original goal when there is 1 ball.

    '''

    subgoal_mcts_prompt = f'''
    You are a helpful assistant confident in PDDL planning.
    Given a domain PDDL, a problem PDDL and several planning examples, generate {plan_number} different plan PDDLs each starting with "Plan PDDL:"
    Do not use quotation marks for code blocks, and do not start with expressions like 'plan'.
    Return ONLY the plan PDDLs starting with "Plan PDDL:", NO other words or explanation. Return only the pddl, excluding comments.
    
    Each plan should be different with every previous plans.
    
    Follow the pattern of each examples below. 
    Let's think step by step.

    Domain PDDL:
    {domain}
    
    Rules:
    Only interact with the ball in the goal state. 
    
    Example 1
    Problem PDDL: 
    {problem_subgoal_3_0}

    Plan PDDL:
    {plan_subgoal_3_0}
    
    Example 2
    Problem PDDL: 
    {problem_subgoal_3_1}

    Plan PDDL:
    {plan_subgoal_3_1}
    
    Example 3
    Problem PDDL: 
    {problem_subgoal_3_2}
    
    Plan PDDL:
    {plan_subgoal_3_2}
    
    '''

    mcts_prompt = f'''
    You are a helpful assistant confident in PDDL planning.
    Given a domain PDDL, a problem PDDL and several planning examples, generate {plan_number} different plan PDDLs each starting with "Plan PDDL:"
    Do not use quotation marks for code blocks, and do not start with expressions like 'plan'.
    Return ONLY the plan PDDLs starting with "Plan PDDL:", NO other words or explanation. Return only the pddl, excluding comments.
    
    Each plan should be different with every previous plans.
    
    Follow the pattern of each examples below. 
    Let's think step by step.

    Domain PDDL:
    {domain}
    
    Rules:
    Only interact with the ball in the goal state. 
    
    Example 1
    Problem PDDL:
    {problem_3_31}
    
    Plan PDDL:
    {plan_3_31}
    
    Example 2
    Problem PDDL:
    {problem_4_31}

    Plan PDDL:
    {plan_4_31}
    '''

    parser.add_argument("--cot_prompt", type=str, default=cot_prompt)
    parser.add_argument("--subgoal_prompt", type=str, default=subgoal_prompt)
    parser.add_argument("--subgoal_mcts_prompt", type=str, default=subgoal_mcts_prompt)
    parser.add_argument("--mcts_prompt", type=str, default=mcts_prompt)

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    return args
