from typing import Any, Dict, List, Union
from smarttvleakage.utils.constants import Direction


class Move:

    def __init__(self, num_moves: int, end_sound: str, directions: Union[Direction, List[Direction]], end_time: int = 0):
        self._num_moves = int(num_moves)
        self._end_sound = str(end_sound)
        self._directions = directions
        self._end_time = int(end_time)

    @property
    def num_moves(self) -> int:
        return self._num_moves

    @property
    def end_sound(self) -> str:
        return self._end_sound

    @property
    def directions(self) -> Union[Direction, List[Direction]]:
        return self._directions

    @property
    def end_time(self) -> int:
        return self._end_time

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.directions, list):
            serialized_directions = [d.name.lower() for d in self.directions]
        else:
            serialized_directions = self.directions.name.lower()

        return {
            'num_moves': self.num_moves,
            'end_sound': self.end_sound,
            'directions': serialized_directions,
            'end_time': self.end_time
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
                    end_time=int(serialized['end_time']))

