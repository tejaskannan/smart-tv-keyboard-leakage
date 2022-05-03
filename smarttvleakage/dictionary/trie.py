import os
from collections import deque, defaultdict
from typing import Optional, Dict, Any, List, DefaultDict

from smarttvleakage.utils.file_utils import read_jsonl_gz, append_jsonl_gz



class TrieNode:
    
    def __init__(self, character: str, count: int, is_end: bool):
        self._character = character
        self._count = count
        self._children = []
        self._is_end = is_end

    @property
    def count(self) -> int:
        return self._count

    @property
    def character(self) -> str:
        return self._character

    @property
    def children(self) -> List[Any]:
        return self._children

    @property
    def is_end(self) -> bool:
        return self._is_end

    def set_is_end(self):
        self._is_end = True

    def increase_count(self, count: count):
        self._count += count

    def set_children(self, children: List[Any]):
        self._children = children

    def add_child(self, character: str, count: int, is_end: bool):
        node = self.get_child(character)

        if node is None:
            node = TrieNode(character=character, count=count, is_end=is_end)
            self._children.append(node)
        else:
            node.increase_count(count=count)

            if (not node.is_end) and (is_end):
                node.set_is_end()

        return node

    def get_child(self, character: str):
        for child in self._children:
            if child.character == character:
                return child

        return None

    def get_child_characters(self, length: Optional[int]) -> Dict[str, int]:
        children: Dict[str, int] = dict()

        if length is None:
            for child in self._children:
                children[child.character] = child.count
        else:
            for child in self._children:
                count = get_count_for_depth(root=child, depth=(length - 1))

                if count > 0:
                    children[child.character] = count

        return children


class Trie:

    def __init__(self, max_depth: int):
        self._root = TrieNode(character='<ROOT>', count=0, is_end=False)
        self._max_depth = max_depth

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def add_string(self, string: str, count: int):
        node = self._root
        self._root.increase_count(count=count)

        for idx, character in enumerate(string):
            next_node = node.add_child(character, count=count, is_end=(idx == (len(string) - 1)))

            if idx < self.max_depth:
                node = next_node

    def get_score_for_string(self, string: str, should_aggregate: bool) -> float:
        total_count = self._root.count

        node = self._root
        for character in string:
            node = node.get_child(character=character)

            if node is None:
                return 0.0

        if should_aggregate:
            count = node.count
        else:
            count = node.count - sum(c for c in node.get_child_characters(length=None).values())

        return count / total_count

    def get_next_characters(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        node = self._root

        if prefix == '':
            return node.get_child_characters(length=length)

        child_dict = dict()

        for idx, character in enumerate(prefix):
            node = node.get_child(character=character)

            if not (length is None):
                length -= 1

            if node is None:
                return child_dict
            else:
                if idx == (len(prefix) - 1):
                    next_dict = node.get_child_characters(length=length)
                else:
                    next_dict = node.get_child_characters(length=None)

                if len(next_dict) == 0:
                    return child_dict
                else:
                    child_dict = next_dict

        return child_dict

    def get_num_nodes(self) -> int:

        def get_num_nodes_helper(root: TrieNode) -> int:
            if len(root.children) == 0:
                return 1

            return 1 + sum(get_num_nodes_helper(child) for child in root.children)

        return get_num_nodes_helper(root=self._root) - 1


def get_count_for_depth(root: TrieNode, depth: int) -> int:
    if depth == 0 and root.is_end:
        return root.count
    elif depth <= 0:
        return 0
    else:
        return sum(get_count_for_depth(child, depth=(depth - 1)) for child in root.children)
