from collections import deque
from typing import Dict, List, Optional, Set

from smarttvleakage.utils.constants import Direction


def breadth_first_search(start_key: str,
                         distance: int,
                         adj_list: Dict[str, Dict[str, str]],
                         wraparound: Optional[Dict[str, Dict[str, str]]],
                         shortcuts: Optional[Dict[str, Dict[str, str]]],
                         directions: List[Direction]) -> List[str]:
    frontier = deque()
    frontier.append((0, start_key))
    result: List[str] = []
    visited: Set[str] = { start_key }

    while len(frontier) > 0:
        dist, key = frontier.pop() 

        if dist == distance:
            result.append(key)
            continue

        neighbors_dict = {d: k for d, k in adj_list[key].items()}

        if wraparound is not None:
            neighbors_dict.update(wraparound.get(key, dict()))

        if shortcuts is not None:
            neighbors_dict.update(shortcuts.get(key, dict()))

        direction = directions[dist]

        if direction == Direction.HORIZONTAL:
            neighbors = [neighbor for d, neighbor in neighbors_dict.items() if d in ('left', 'right')]
        elif direction == Direction.VERTICAL:
            neighbors = [neighbor for d, neighbor in neighbors_dict.items() if d in ('down', 'up')]
        else:
            neighbors = list(neighbors_dict.values())

        for neighbor in neighbors:
            if neighbor not in visited:
                frontier.append((dist + 1, neighbor))
                visited.add(neighbor)

        for neighbor in neighbors_dict.values():
            visited.add(neighbor)

    return result
