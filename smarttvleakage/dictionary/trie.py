import os
import numpy as np
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Optional, Dict, Any, List, DefaultDict, Set, Iterable

from smarttvleakage.utils.file_utils import read_jsonl_gz, append_jsonl_gz
from smarttvleakage.utils.constants import BIG_NUMBER, END_CHAR


ROOT = '<ROOT>'


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


def get_min_priority(pq: PriorityQueue) -> float:
    if pq.empty():
        return BIG_NUMBER

    item = pq.get()
    pq.put(item)
    return item.priority


class TrieNode:
    
    def __init__(self, character: str, count: int):
        self._character = character
        self._count = count
        self._children: Dict[str, Any] = dict()  # Maps character to trie node

    @property
    def count(self) -> int:
        return self._count

    @property
    def character(self) -> str:
        return self._character

    @property
    def children(self) -> Dict[str, Any]:
        return self._children

    @property
    def is_end(self) -> bool:
        return END_CHAR in self.children

    def increase_count(self, count: count):
        self._count += count

    def set_children(self, children: List[Any]):
        self._children = {child.character: child for child in children}

    def add_child(self, character: str, count: int):
        node = self.get_child(character)

        if node is None:
            node = TrieNode(character=character, count=count)
            self._children[node.character] = node
        else:
            node.increase_count(count=count)

        return node

    def get_child(self, character: str) -> Any:
        return self._children.get(character)

    def get_child_characters(self, length: Optional[int]) -> Dict[str, int]:
        children: Dict[str, int] = dict()

        for child in self._children.values():
            if length is None:
                children[child.character] = child.count
            else:
                count = get_count_for_depth(root=child, depth=(length - 1))

                if count > 0:
                    children[child.character] = count

        return children

    def __str__(self) -> str:
        return 'Character: {}, Count: {}'.format(self.character, self.count)


class Trie:

    def __init__(self, max_depth: int):
        self._root = TrieNode(character=ROOT, count=0)
        self._max_depth = max_depth

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def add_string(self, string: str, count: int, should_index_prefixes: bool):
        node = self._root
        self._root.increase_count(count=count)

        for idx, character in enumerate(string):
            next_node = node.add_child(character, count=count)

            if should_index_prefixes or (idx == (len(string) - 1)):
                if idx >= self.max_depth:
                    node.add_child(END_CHAR, count=count)
                else:
                    next_node.add_child(END_CHAR, count=count)

            if idx < self.max_depth:
                node = next_node

        #node.add_child(END_CHAR, count=count)

    def get_next_characters(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        node = self._root

        if prefix == '':
            return node.get_child_characters(length=length)
        elif (length is not None) and (len(prefix) > length):
            return dict()

        child_dict = dict()

        for idx, character in enumerate(prefix):
            if idx >= self.max_depth:
                break

            node = node.get_child(character=character)

            if node is None:
                return dict()  # If we couldn't find the prefix, return an empty dictionary

        if (length is not None) and ((length >= self.max_depth) or (len(prefix) == length)):
            adjusted_length = None
        else:
            adjusted_length = max(length - len(prefix), 0) if length is not None else None

        return node.get_child_characters(length=adjusted_length)

    def get_max_log_prob(self, prefix: str, single_char_counts: Counter, length: int) -> float:
        if len(prefix) >= length:
            return 0.0

        num_remaining = len(prefix) - length
        node = self.get_node_for(prefix)

        if node is None:
            total_count = sum(single_char_counts.values())
            most_common_prob = single_char_counts.most_common(1)[0][1]
            return -1 * num_remaining * np.log(most_common_prob)
        else:
            remaining_prob = 1.0

            while (node is not None) and (num_remaining > 0):
                children = node.get_children()
                total_count = sum(children.values())
                probs = {key: (count / total_count) for key, count in children.items()}

                keys = [key for key in children.keys()]
                key_probs = [probs[key] for key in keys]
                max_idx = np.argmax(key_probs)

                remaining_prob *= key_probs[max_idx]
                node = node.get_child(character=keys[max_idx])
                num_remaining -= 1

            return -1 * np.log(remaining_prob)

    def get_num_nodes(self) -> int:

        def get_num_nodes_helper(root: TrieNode) -> int:
            if len(root.children) == 0:
                return 1

            return 1 + sum(get_num_nodes_helper(child) for child in root.children.values())

        return get_num_nodes_helper(root=self._root) - 1

    def get_node_for(self, prefix: str) -> TrieNode:
        node = self._root
        if prefix == '':
            return node

        idx = 0
        while (node is not None) and (idx < len(prefix)):
            node = node.get_child(character=prefix[idx])
            idx += 1

        return node

    def get_words_for(self, prefixes: Iterable[str], max_num_results: int, min_length: Optional[int], max_count_per_prefix: Optional[int]) -> Iterable[str]:
        # Get the root nodes for all prefixes and add to a priority queue for later searching
        node_queue = PriorityQueue()
        recommended_queues: Dict[str, PriorityQueue] = dict()

        for prefix in prefixes:
            node = self.get_node_for(prefix)
            recommended_queues[prefix] = PriorityQueue(maxsize=(max_count_per_prefix if max_count_per_prefix is not None else max_num_results))

            if node is not None:
                item = PrioritizedItem(priority=-1 * node.count, item=(node, prefix, prefix))
                node_queue.put(item)

        # Run Dijkstra's algorithm on the Trie to recover the most frequent words
        global_min_score = BIG_NUMBER
        num_results = 0

        while not node_queue.empty():
            item = node_queue.get()
            (node, string, prefix) = item.item

            prefix_queue = recommended_queues[prefix]
            prefix_min_score = get_min_priority(prefix_queue)

            if (node.is_end) and (min_length is None or (len(string) >= min_length)):
                score = node.count - sum(c for c in node.get_child_characters(length=None).values())

                if prefix_queue.full() and (score > prefix_min_score):
                    prefix_queue.get()
                    prefix_queue.put(PrioritizedItem(priority=score, item=string))
                    prefix_min_score = min(prefix_min_score, score) if (prefix_min_score > 0) else score
                elif (not prefix_queue.full()):
                    prefix_queue.put(PrioritizedItem(priority=score, item=string))
                    prefix_min_score = min(prefix_min_score, score) if (prefix_min_score > 0) else score

            global_min_score = min(global_min_score, prefix_min_score)
            if (node.count < global_min_score) and all(pq.full() for pq in recommended_queues.values()):
                continue

            for child in node.children.values():
                next_item = PrioritizedItem(priority=-1 * child.count, item=(child, string + child.character, prefix))
                node_queue.put(next_item)

        total_count = self._root.count
        words: List[Tuple[str, float]] = []

        for prefix_queue in recommended_queues.values():
            while not prefix_queue.empty():
                item = prefix_queue.get()
                count, word = item.priority, item.item

                if word.endswith(END_CHAR):
                    words.append((word.replace(END_CHAR, ''), count / total_count))

        return list(reversed(sorted(words, key=lambda t: t[1])))[0:max_num_results]


def get_count_for_depth(root: TrieNode, depth: int) -> int:
    if depth == 0 and root.is_end:
        return root.count
    elif depth <= 0:
        return 0
    else:
        return sum(get_count_for_depth(child, depth=(depth - 1)) for child in root.children.values())
