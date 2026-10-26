import time
import pddlpy
from collections import defaultdict
import math


def ucb1(tree, node, exploration=1):
    parent_node = list(tree.G.predecessors(node))[0]
    if tree.G.nodes[node]['visits'] == 0:
        return float('inf')
    else:
        return (tree.G.nodes[node]['win'] / tree.G.nodes[node]['visits']) + \
            exploration * math.sqrt((math.log(tree.G.nodes[parent_node]['visits']) / tree.G.nodes[node]['visits']))


# select among visited nodes
def selection_expansion(tree, node):
    visited_nodes = set()
    root_node = tree.root

    while True:
        if node in visited_nodes:
            return node

        visited_nodes.add(node)
        successors = list(tree.G.successors(node))

        successors = [child for child in successors if child != root_node]

        if not successors:
            return node
        elif len(successors) == 1:
            node = successors[0]
        else:
            node = max(successors, key=lambda n: ucb1(tree, n))

        if tree.G.nodes[node]['visits'] == 0:
            return node


# Check whether the current node satisfies the goal state or not
def is_goal_state(node, goal_state_set):
    state_dict = defaultdict(set, {pred: set(tuples) for pred, tuples in node})
    for atom in goal_state_set:
        pred = atom.predicate[0]
        args = atom.predicate[1:]
        if tuple(args) not in state_dict.get(pred, set()):
            return False
    return True


# Traverse all reachable leaf nodes from current_node
def simulation(tree, node, domprob):
    goal_state_set = domprob.goals()
    current_node = node
    depth = 0
    visited_nodes = set()

    def get_action_weight(n, child):
        for u, v, key, data in tree.G.edges(n, data=True, keys=True):
            if v == child:
                return data['weight']
        return 0  # Default weight if action not found

    while True:
        if current_node in visited_nodes:
            return 0

        visited_nodes.add(current_node)

        if is_goal_state(current_node, goal_state_set):
            return 1 / (1 + depth)  # Goal state found, return reward
        successors = list(tree.G.successors(current_node))
        if not successors:
            return 0  # No further nodes to explore
        best_child = max(successors, key=lambda child: get_action_weight(current_node, child))
        current_node = best_child
        depth += 1  # Increase depth as we move closer to goal


def backpropagation(tree, node, reward):
    visited_nodes = set()
    root_node = tree.root

    while node:
        if node in visited_nodes and node != root_node:
            break

        visited_nodes.add(node)

        tree.G.nodes[node]['visits'] += 1
        tree.G.nodes[node]['win'] += reward

        parents = list(tree.G.predecessors(node))
        node = parents[0] if parents else None

        if node == root_node:
            root_node = None


def mcts(tree, domprob, max_iterations=100):
    iteration = 0
    goal_found = False
    goal_node = None

    goal_state_set = domprob.goals()
    if is_goal_state(tree.root, goal_state_set):
        return ["; cost = 0 (unit cost)"], ["; cost = 0 (unit cost)"]

    while iteration < max_iterations and not goal_found:
        node = selection_expansion(tree, tree.root)
        reward = simulation(tree, node, domprob)
        backpropagation(tree, node, reward)
        if reward == 1:
            goal_found = True
            goal_node = node  # Save the goal node
        iteration += 1

    # Extract path from root to goal node
    if goal_found:
        states_path = []
        actions_path = []
        current_node = goal_node
        while current_node != tree.root:
            states_path.append(current_node)
            parent_node = list(tree.G.predecessors(current_node))[0]

            for u, v, key, data in tree.G.edges(parent_node, data=True, keys=True):
                if v == current_node:
                    actions_path.append(key)

            current_node = parent_node
        states_path.append(tree.root)
        states_path.reverse()
        actions_path.reverse()
        return states_path, actions_path
    else:
        return [], []

