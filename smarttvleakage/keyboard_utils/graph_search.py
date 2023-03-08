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

        if (wraparound is not None) and (key in wraparound):
            neighbors_dict.update(wraparound[key])

        if (shortcuts is not None) and (key in shortcuts):
            neighbors_dict.update(shortcuts[key])

        direction = directions[dist]

        if direction == Direction.HORIZONTAL:
            neighbors = [neighbor for d, neighbor in neighbors_dict.items() if d in ('left', 'right')]
        elif direction == Direction.VERTICAL:
            neighbors = [neighbor for d, neighbor in neighbors_dict.items() if d in ('down', 'up')]
        elif direction != Direction.ANY:
            neighbors = [neighbor for d, neighbor in neighbors_dict.items() if (d == direction.name.lower())]
        else:
            neighbors = list(neighbors_dict.values())

        for neighbor in neighbors:
            if neighbor not in visited:
                frontier.append((dist + 1, neighbor))
                visited.add(neighbor)

        for neighbor in neighbors_dict.values():
            visited.add(neighbor)

    return result


def bfs(start_key: str,
        distance: int,
        adj_list: Dict[str, Dict[str, str]],
        wraparound: Optional[Dict[str, Dict[str, str]]],
        shortcuts: Optional[Dict[str, Dict[str, str]]]):
    frontier = deque()
    frontier.append((0, start_key))
    result: List[str] = []
    visited: Set[str] = { start_key }

    while len(frontier) > 0:
        dist, key = frontier.pop() 

        if dist == distance:
            result.append(key)
            continue

        neighbors = [k for k in adj_list[key]]

        if '9' in neighbors:
            print('{}: {}'.format(key, dist))

        if (wraparound is not None) and (key in wraparound):
            neighbors.extend(wraparound[key])

        if (shortcuts is not None) and (key in shortcuts):
            neighbors.extend(shortcuts[key])

        for neighbor in neighbors:
            if neighbor not in visited:
                frontier.append((dist + 1, neighbor))
                visited.add(neighbor)

        for neighbor in neighbors:
            visited.add(neighbor)

    return result



def follow_path(start_key: str,
                adj_list: Dict[str, Dict[str, str]],
                wraparound: Optional[Dict[str, Dict[str, str]]],
                shortcuts: Optional[Dict[str, Dict[str, str]]],
                directions: List[Direction]) -> Optional[str]:
    current_key = start_key

    for direction in directions:
        neighbors_dict = {d: k for d, k in adj_list[current_key].items()}

        if (wraparound is not None) and (current_key in wraparound):
            neighbors_dict.update(wraparound[current_key])

        if (shortcuts is not None) and (current_key in shortcuts):
            neighbors_dict.update(shortcuts[current_key])

        assert direction in (Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN), 'Invalid direction: {}'.format(direction)
        
        neighbors = [neighbor for d, neighbor in neighbors_dict.items() if (d == direction.name.lower())]

        if len(neighbors) == 0:
            return None

        assert len(neighbors) == 1, 'Found {} neighbors of {} in direction {}'.format(len(neighbors), current_key, direction)
        current_key = neighbors[0]

    return current_key
