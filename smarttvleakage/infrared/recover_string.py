import os.path
from argparse import ArgumentParser
from collections import namedtuple, deque
from typing import List, Iterable, Dict

from smarttvleakage.audio.data_types import Move
from smarttvleakage.utils.constants import Direction
from smarttvleakage.utils.file_utils import read_json


SearchState = namedtuple('SearchState', ['current_string', 'current_key', 'move_idx'])
PathState = namedtuple('PathState', ['key', 'idx'])


def get_key_for_path(start_key: str, path: List[Direction], keyboard_graph: Dict[str, Dict[str, str]]) -> str:
    prev_direction = Direction.RIGHT

    trajectories: deque = deque()
    init_state = PathState(key=start_key, idx=0)
    trajectories.append(init_state)

    while len(trajectories) > 0:
        state = trajectories.popleft()

        if state.idx == len(path):
            yield state.key
        else:
            direction = path[state.idx]
            directions: List[Direction] = []

            if direction != Direction.ANY:
                directions.append(direction)
            else:
                # Find the previous known direction
                prev_idx = state.idx - 1
                prev_direction = Direction.ANY
                while (prev_idx >= 0) and (prev_direction != Direction.ANY):
                    prev_direction = path[prev_idx]
                    prev_idx -= 1

                if prev_direction != Direction.ANY:
                    directions.append(prev_direction)

                # Find the next known direction
                next_idx = state.idx + 1
                next_direction = Direction.ANY
                while (next_idx < len(path)) and (next_direction != Direction.ANY):
                    next_direction = path[next_idx]
                    next_idx += 1

                if (next_direction != Direction.ANY) and (next_direction not in directions):
                    directions.append(next_direction)
                
                # Include the remaining directions in an arbitrary order
                if Direction.RIGHT not in directions:
                    directions.append(Direction.RIGHT)

                if Direction.LEFT not in directions:
                    directions.append(Direction.LEFT)

                if Direction.UP not in directions:
                    directions.append(Direction.UP)

                if Direction.DOWN not in directions:
                    directions.append(Direction.DOWN)

            for direction in reversed(directions):
                next_key = keyboard_graph[state.key].get(direction.name.lower())

                if next_key is not None:
                    next_state = PathState(key=next_key, idx=(state.idx + 1))
                    trajectories.append(next_state)


def recover_string(move_seq: List[Move], keyboard_graph: Dict[str, Dict[str, str]]) -> Iterable[str]:
    search_frontier: deque = deque()

    init_state = SearchState(current_string='', current_key='q', move_idx=0)
    search_frontier.append(init_state)

    while len(search_frontier) > 0:
        current_state = search_frontier.pop()  # Get the first state on the queue

        if len(current_state.current_string) == (len(move_seq) - 1):
            yield current_state.current_string
            continue

        move = move_seq[current_state.move_idx]

        for next_key in get_key_for_path(start_key=current_state.current_key, path=move.directions, keyboard_graph=keyboard_graph):
            next_state = SearchState(current_string=current_state.current_string + next_key,
                                     current_key=next_key,
                                     move_idx=current_state.move_idx + 1)
            search_frontier.append(next_state)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--processed-path', type=str, required=True)
    parser.add_argument('--graph-path', type=str, required=True)
    args = parser.parse_args()

    # Get the target string using the processed path file name
    file_name = os.path.basename(args.processed_path)
    target = file_name.replace('.json', '')

    # Read in the adjacency list (without any special treatment for now)
    graph = read_json(args.graph_path)['adjacency_list']
    
    # Parse the move sequence
    move_seq_raw = read_json(args.processed_path)
    move_seq = list(map(lambda d: Move.from_dict(d), move_seq_raw))

    for idx, guess in enumerate(recover_string(move_seq, keyboard_graph=graph)):
        print('{}. {}'.format(idx, guess))

        if guess == target:
            break
