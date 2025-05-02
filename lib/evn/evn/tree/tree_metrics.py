import evn

evn.dispatch(dict)


def inspect(dct):
    return tree_metrics(dct)


def tree_metrics(tree, subtree_pattern_threshold=2.0):
    """
    Analyze a nested dictionary (tree) and compute a wide range of structural,
    semantic, and integrity metrics.

    The function safely handles cycles (self-references), gathers statistics
    on depth, width, branching, and key usage, and detects repeated subtree
    patterns if runtime permits.

    Parameters
    ----------
    tree : dict
        The nested dictionary to analyze.
    subtree_pattern_threshold : float, optional
        Maximum allowed multiple of baseline runtime to spend on
        repeated subtree pattern analysis (default is 2.0).

    Returns
    -------
    metrics : dict
        A nested dictionary with the following structure:

        {
            'structure': {
                'max_depth': int,
                'min_leaf_depth': int,
                'avg_leaf_depth': float,
                'leaf_depth_stddev': float,
                'max_width': int,
                'avg_branching_factor': float,
                'num_internal_nodes': int,
                'num_leaves': int,
                'total_elements': int,
                'total_leaf_size': int
            },
            'keys': {
                'all_keys': set,
                'key_reuse_ratio': float
            },
            'integrity': {
                'num_cycles': int,
                'cycle_paths': list of tuples
            },
            'subtrees': {
                'repeated_subtrees': dict or str
            },
            'timing': {
                'runtime_sec': float
            }
        }
    """
    from collections import deque, Counter
    from math import sqrt
    import time

    visited_ids = set()
    key_counter = Counter()
    depth_list = []
    subtree_counter = Counter()
    # seen_nodes = {}

    total_nodes = 0
    num_internal_nodes = 0
    num_leaves = 0
    total_leaf_size = 0
    max_depth = 0
    max_width = 0
    branch_counts = []
    cycle_paths = []

    all_keys = set()
    cycle_ids = set()

    queue = deque([(tree, 1, 'root', id(tree), ())])
    start_time = time.time()

    while queue:
        level_size = len(queue)
        max_width = max(max_width, level_size)

        for _ in range(level_size):
            node, depth, path_label, node_id, path = queue.popleft()
            current_path = path + (path_label, )

            if node_id in visited_ids:
                if node_id not in cycle_ids:
                    cycle_ids.add(node_id)
                    cycle_paths.append(current_path)
                continue
            visited_ids.add(node_id)

            if isinstance(node, dict):
                num_internal_nodes += 1
                keys = list(node.keys())
                children = list(node.values())
                key_counter.update(keys)
                all_keys.update(keys)
                branch_counts.append(len(children))

                subtree_repr = tuple(
                    sorted((k, id(v)) for k, v in node.items()))
                subtree_counter[subtree_repr] += 1

                for k, v in node.items():
                    queue.append((v, depth + 1, k, id(v), current_path))
            else:
                num_leaves += 1
                total_nodes += 1
                depth_list.append(depth)
                total_leaf_size += len(node) if hasattr(
                    node, '__len__') and not isinstance(node, str) else 1

            max_depth = max(max_depth, depth)

    elapsed = time.time() - start_time

    total_elements = total_nodes + num_internal_nodes
    avg_leaf_depth = sum(depth_list) / len(depth_list) if depth_list else 0
    min_leaf_depth = min(depth_list) if depth_list else 0
    leaf_depth_stddev = (sqrt(
        sum((d - avg_leaf_depth)**2
            for d in depth_list) / len(depth_list)) if depth_list else 0)
    avg_branching = sum(branch_counts) / len(
        branch_counts) if branch_counts else 0
    key_reuse_ratio = sum(
        key_counter.values()) / len(key_counter) if key_counter else 0

    repeated_subtrees = ({
        k: v
        for k, v in subtree_counter.items() if v > 1
    } if elapsed < subtree_pattern_threshold else
                         'Not computed (runtime threshold exceeded)')

    return {
        'max_depth': max_depth,
        'min_leaf_depth': min_leaf_depth,
        'avg_leaf_depth': round(avg_leaf_depth, 2),
        'leaf_depth_stddev': round(leaf_depth_stddev, 2),
        'max_width': max_width,
        'avg_branching_factor': round(avg_branching, 2),
        'num_internal_nodes': num_internal_nodes,
        'num_leaves': num_leaves,
        'total_elements': total_elements,
        'total_leaf_size': total_leaf_size,
        'all_keys': all_keys,
        'key_reuse_ratio': round(key_reuse_ratio, 2),
        'num_cycles': len(cycle_paths),
        'cycle_paths': cycle_paths,
        'repeated_subtrees': repeated_subtrees,
        'runtime': round(elapsed, 3),
    }
