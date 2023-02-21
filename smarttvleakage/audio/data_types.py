from typing import Any, Dict, List, Union
from smarttvleakage.utils.constants import Direction


class Move:

    def __init__(self, num_moves: int, end_sound: str, directions: Union[Direction, List[Direction]], start_time: int = 0, end_time: int = 0, move_times: List[int] = []):
        self._num_moves = int(num_moves)
        self._end_sound = str(end_sound)
        self._directions = directions
        self._start_time = int(start_time)
        self._end_time = int(end_time)
        self._move_times = move_times

        if len(move_times) == 0:
            self._move_times = [0 for _ in range(num_moves)]

    @property
    def num_moves(self) -> int:
        return self._num_moves

    @property
    def end_sound(self) -> str:
        return self._end_sound

    @property
    def move_times(self) -> List[int]:
        return self._move_times

    @property
    def directions(self) -> Union[Direction, List[Direction]]:
        return self._directions

    @property
    def start_time(self) -> int:
        return self._start_time

    @property
    def end_time(self) -> int:
        return self._end_time

    def __eq__(self, other: Any):
        if isinstance(other, Move):
            if isinstance(other.directions, list) and isinstance(self.directions, list):
                directions_equal = len(other.directions) == len(self.directions)
                directions_equal = directions_equal and all([other_dir == self_dir for other_dir, self_dir in zip(other.directions, self.directions)])
            else:
                directions_equal = other.directions == self.directions

            return directions_equal and (other.num_moves == self.num_moves) and (other.end_sound == self.end_sound) and (other.end_time == self.end_time)
        else:
            return False

    def __repr__(self):
        return 'Move(num_moves={}, end_sound={}, directions={}, start_time={}, end_time={})'.format(self.num_moves, self.end_sound, self.directions, self.start_time, self.end_time)

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.directions, list):
            serialized_directions = [d.name.lower() for d in self.directions]
        else:
            serialized_directions = self.directions.name.lower()

        return {
            'num_moves': self.num_moves,
            'end_sound': self.end_sound,
            'directions': serialized_directions,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'move_times': self.move_times
        }

    @classmethod
    def from_dict(cls, serialized: Dict[str, Any]):
        serialized_directions = serialized['directions']
        if isinstance(serialized_directions, list):
            directions = [Direction[d.upper()] for d in serialized_directions]
        else:
            directions = Direction[serialized_directions.upper()]

        return Move(num_moves=int(serialized['num_moves']),
                    end_sound=str(serialized['end_sound']),
                    directions=directions,
                    start_time=int(serialized['start_time']),
                    end_time=int(serialized['end_time']))
