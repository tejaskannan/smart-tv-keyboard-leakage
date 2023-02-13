#!/usr/bin/python3
import numpy as np
import sys
import re
import time
from argparse import ArgumentParser
from enum import Enum, auto
from collections import Counter
from typing import List, Optional, Dict


MODE2_REGEX = re.compile(r'([a-z]+) ([0-9]*)')


class IRAction(Enum):
    PULSE = auto()
    SPACE = auto()
    TIMEOUT = auto()

class RemoteKey(Enum):
    SELECT = auto()
    RIGHT = auto()
    LEFT = auto()
    DOWN = auto()
    UP = auto()
    BACK = auto()
    HOME = auto()
    EXIT = auto()
    UNKNOWN = auto()
    NONE = auto()


KEY_MAPPING: Dict[int, RemoteKey]  = {
    0xe0e016e9: RemoteKey.SELECT,
    0xe0e046b9: RemoteKey.RIGHT,
    0xe0e0a659: RemoteKey.LEFT,
    0xe0e006f9: RemoteKey.UP,
    0xe0e08679: RemoteKey.DOWN,
    0xe0e01ae5: RemoteKey.BACK,
    0xe0e0b44b: RemoteKey.EXIT,
    0xe0e09e61: RemoteKey.HOME
}


class Signal:

    def __init__(self, encoding: str):
        self._encoding = encoding
        self._history: List[int] = []

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def avg_time(self) -> float:
        return float(np.mean(self._history))

    @property
    def std_time(self) -> float:
        return float(np.std(self._history))

    def append_to_history(self, value: int):
        self._history.append(value)


class SignalEncoder:

    TOLERANCE = 0.25

    def __init__(self):
        self._signal_dict: Dict[IRAction, Dict[int, Signal]] = dict()
        self._current_char = 'a'

    def get_next_char(self) -> str:
        if self._current_char == 'z':
            next_char = 'A'
        elif self._current_char == 'Z':
            raise ValueError('Out of characters')
        else:
            next_char = chr(ord(self._current_char) + 1)

        encoding = self._current_char
        self._current_char = next_char

        return encoding

    def encode(self, action: IRAction, length: int) -> str:
        """
        Adds the reading for the given (action, length) pair to the
        running dataset and returns the encoding for this item.
        """
        if action not in self._signal_dict:
            self._signal_dict[action] = dict()  # Dictionary mapping length -> signal

        encoding = None

        for existing_length, existing_signal in self._signal_dict[action].items():
            # Try to match this length against what we have collected so far
            lower_bound, upper_bound = (1.0 - self.TOLERANCE) * existing_length, (1.0 + self.TOLERANCE) * existing_length

            if (lower_bound <= length) and (length <= upper_bound):
                existing_signal.append_to_history(length)
                encoding = existing_signal.encoding
                break

        # If we could not find a match, then add a new entry to the dataset
        if encoding is None:
            encoding = self.get_next_char()
            self._signal_dict[action][length] = Signal(encoding)

        return encoding


class SignalDecoder:

    def __init__(self):
        self._encoded_zero: Optional[str] = None
        self._encoded_one: Optional[str] = None

    def create_binary_decodings(self, encoding: str):
        pair_counter: Counter = Counter()

        for idx in range(0, len(encoding), 2):
            # Extract the pair of characters
            character_pair = encoding[idx:(idx + 2)]
            if len(character_pair) < 2:
                continue

            pair_counter[character_pair] += 1

        # Arbitrarily set the most frequent to 0 and next
        # most frequent to 1. The assignment of these
        # two does not matter, we just need to be consistent.
        top_two = pair_counter.most_common(2)
        self._encoded_zero = top_two[0][0]
        self._encoded_one = top_two[1][0]

    def decode(self, encoding: str) -> RemoteKey:
        if (self._encoded_zero is None) or (self._encoded_one is None):
            self.create_binary_decodings(encoding)
            print('Zero -> {}, One -> {}'.format(self._encoded_zero, self._encoded_one))

        bit_list: List[str] = []

        for idx in range(0, len(encoding), 2):
            # Extract the pair of characters
            character_pair = encoding[idx:(idx + 2)]
            if len(character_pair) < 2:
                continue

            if character_pair == self._encoded_zero:
                bit_list.append('0')
            elif character_pair == self._encoded_one:
                bit_list.append('1')

        if len(bit_list) < 30:
            return RemoteKey.NONE

        decoded = int(''.join(bit_list), 2)
        return KEY_MAPPING.get(decoded, RemoteKey.UNKNOWN)

    def run(self, encoder: SignalEncoder, output_path: str):

        encoded_characters: List[str] = []
        start_time = time.time()

        for line in sys.stdin:
            current_time = time.time() - start_time

            # Match the mode2 line
            match = MODE2_REGEX.match(line)
            if match is None:
                continue

            action_type, length = IRAction[match.group(1).upper()], int(match.group(2))
            encoding = encoder.encode(action_type, length=length)

            encoded_characters.append(encoding)

            #print('{} -> {} ({})'.format(action_type, length, encoding))

            if action_type == IRAction.TIMEOUT:
                encoding = ''.join(encoded_characters)
                keystroke = self.decode(encoding)

                with open(output_path, 'a') as fout:
                    fout.write('{:.6f} {}\n'.format(current_time, keystroke.name.lower()))

                #print('Detected: {}'.format(keystroke))
                encoded_characters = []


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True, help='The output file (.txt) to log results to.')
    args = parser.parse_args()

    assert args.output_path.endswith('.txt'), 'Must provide a `.txt` output file. Got: {}'.format(args.output_path)

    decoder = SignalDecoder()
    encoder = SignalEncoder()

    decoder.run(encoder, output_path=args.output_path)
