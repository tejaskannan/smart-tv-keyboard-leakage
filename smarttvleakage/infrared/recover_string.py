import os.path
from argparse import ArgumentParser
from collections import namedtuple, deque
from typing import List, Iterable, Dict, Set

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.data_types import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.constants import Direction, SmartTVType, KeyboardType
from smarttvleakage.utils.file_utils import read_json
from smarttvleakage.utils.transformations import get_keyboard_mode, get_string_from_keys


SearchState = namedtuple('SearchState', ['current_keys', 'current_key', 'move_idx', 'keyboard_mode'])
DirectionState = namedtuple('DirectionState', ['prev_direction', 'directions'])

SELECT_SOUNDS = frozenset(['<DONE>', '<CHANGE>', '<NEXT>', '<CANCEL>', '<LANGUAGE>', '<SETTINGS>', '<LEFT>', '<RIGHT>', '<UP>', '<DOWN>', '<CAPS>'])
DELETE_SOUNDS = frozenset(['<BACK>', '<DELETEALL>'])


def iterate_directions(directions: List[Direction]) -> Iterable[List[Direction]]:
    direction_frontier: deque = deque()
    init_state = DirectionState(prev_direction=Direction.ANY, directions=[])
    direction_frontier.append(init_state)

    while len(direction_frontier) > 0:
        current_state = direction_frontier.pop()

        if len(current_state.directions) == len(directions):
            yield current_state.directions
            continue
    
        direction_idx = len(current_state.directions)
        current_direction = directions[direction_idx]

        candidate_directions: List[Direction] = []

        if current_direction == Direction.ANY:
            if (direction_idx > 0) and (directions[direction_idx - 1] != Direction.ANY):
                candidate_directions.append(directions[direction_idx - 1])

            # Include the remaining directions in an arbitrary order
            all_directions = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]
            for candidate in all_directions:
                if candidate not in candidate_directions:
                    candidate_directions.append(candidate)
        else:
            candidate_directions.append(current_direction)

        for direction in candidate_directions:
            next_state = DirectionState(prev_direction=direction,
                                        directions=current_state.directions + [direction])
            direction_frontier.append(next_state)


def recover_string(move_seq: List[Move], keyboard: MultiKeyboardGraph) -> Iterable[str]:
    search_frontier: deque = deque()

    keyboard_mode = keyboard.get_start_keyboard_mode()
    init_state = SearchState(current_keys=[],
                             current_key='q',
                             move_idx=0,
                             keyboard_mode=keyboard_mode)
    search_frontier.append(init_state)

    while len(search_frontier) > 0:
        current_state = search_frontier.pop()  # Get the first state on the queue

        if len(current_state.current_keys) == (len(move_seq) - 1):
            current_string = get_string_from_keys(current_state.current_keys)

            if len(current_string) > 0:
                yield current_string

            continue

        move = move_seq[current_state.move_idx]
        seen_neighbors: Set[str] = set()

        for direction_choice in iterate_directions(move.directions):
            neighbor = keyboard.follow_path(start_key=current_state.current_key,
                                            use_shortcuts=True,
                                            use_wraparound=True,
                                            directions=direction_choice,
                                            mode=current_state.keyboard_mode)

            if neighbor is None:
                continue
            elif (move.end_sound == sounds.SAMSUNG_SELECT) and (neighbor not in SELECT_SOUNDS):
                continue
            elif (neighbor in seen_neighbors):
                continue

            next_keyboard_mode = get_keyboard_mode(key=neighbor,
                                                   mode=current_state.keyboard_mode,
                                                   keyboard_type=keyboard.keyboard_type)

            next_state = SearchState(current_keys=current_state.current_keys + [neighbor],
                                     current_key=neighbor,
                                     move_idx=current_state.move_idx + 1,
                                     keyboard_mode=next_keyboard_mode)

            search_frontier.append(next_state)
            seen_neighbors.add(neighbor)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--processed-path', type=str, required=True)
    args = parser.parse_args()

    # Fix the tv type to samsung
    tv_type = SmartTVType.SAMSUNG
    keyboard_type = KeyboardType.SAMSUNG

    # Make the keyboard graph
    keyboard = MultiKeyboardGraph(keyboard_type)

    # Parse the move sequence
    move_sequences = read_json(args.processed_path)

    # Get the labels
    folder, file_name = os.path.split(args.processed_path)
    labels_path = os.path.join(folder, file_name.replace('.json', '_labels.json'))
    labels = read_json(labels_path)

    assert len(labels) == len(move_sequences), 'Found {} labels and {} move sequences.'.format(len(labels), len(move_sequences))

    for seq_idx, move_seq_raw in enumerate(move_sequences):
        move_seq = list(map(lambda d: Move.from_dict(d), move_seq_raw))

        rank = 1
        for offset in range(0, len(move_seq)):
            if (offset > 0) and (move_seq[offset - 1].end_sound != sounds.SAMSUNG_SELECT):
                break

            for guess in recover_string(move_seq[offset:], keyboard=keyboard):
                print('{}. {}'.format(rank, guess))

                if guess == labels[seq_idx]:
                    break

                rank += 1

        print('==========')

