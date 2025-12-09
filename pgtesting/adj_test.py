import itertools
from collections import deque

# script use to define all  the permuations of possible play maps
REGION_ADJ_NA = {
    1: {2},
    2: {1, 3, 4},
    3: {2, 4, 7},
    4: {2, 3, 5, 6, 7},
    5: {4, 6},
    6: {4, 5, 7},
    7: {3, 4, 6},
}

def is_connected(subset, graph):
    """
    subset: a tuple/list/set of region IDs (e.g. (1,2,3,4,5))
    graph:  adjacency dict like REGION_ADJ_NA
    returns True if the induced subgraph on 'subset' is connected
    """
    subset = set(subset)
    # start BFS from any node in the subset
    start = next(iter(subset))
    visited = set([start])
    q = deque([start])

    while q:
        node = q.popleft()
        for nbr in graph[node]:
            if nbr in subset and nbr not in visited:
                visited.add(nbr)
                q.append(nbr)

    return visited == subset  

def count_connected_combos(k, graph):
    """
    k: size of the group (e.g. 5)
    graph: adjacency dict
    returns (count, combos) where combos is a list of the valid subsets as sorted tuples
    """
    nodes = sorted(graph.keys())
    valid_combos = []

    for combo in itertools.combinations(nodes, k):
        if is_connected(combo, graph):
            valid_combos.append(combo)

    return len(valid_combos), valid_combos



if __name__ == "__main__":
    k = 5 #number of regions required this is changed for what player count we are testing

    count, combos = count_connected_combos(k, REGION_ADJ_NA)

    print(f"For k = {k}, number of connected combos = {count}")
    print("Combos:")
    for c in combos:
        print(c)
