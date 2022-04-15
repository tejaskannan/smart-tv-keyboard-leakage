from typing import Optional, Dict, Any, List


class TrieNode:
    
    def __init__(self, character: str, count: int):
        self._character = character
        self._count = count
        self._children = []

    @property
    def count(self) -> int:
        return self._count

    @property
    def character(self) -> str:
        return self._character

    @property
    def children(self) -> List[Any]:
        return self._children

    def increment_count(self):
        self._count += 1

    def add_child(self, character: str):
        node = self.get_child(character)

        if node is None:
            node = TrieNode(character=character, count=1)
            self._children.append(node)
        else:
            node.increment_count()

        return node

    def get_child(self, character: str):
        for child in self._children:
            if child.character == character:
                return child

        return None

    def get_child_characters(self) -> Dict[str, int]:
        children: Dict[str, int] = dict()

        for child in self._children:
            children[child.character] = child.count

        return children


class Trie:

    def __init__(self):
        self._root = TrieNode(character='<ROOT>', count=0)

    def add_string(self, string: str):
        node = self._root

        for character in string:
            node = node.add_child(character)

    def get_next_characters(self, prefix: str) -> Dict[str, int]:
        node = self._root
        for character in prefix:
            node = node.get_child(character=character)

            if node is None:
                return dict()

        return node.get_child_characters()

    def get_num_nodes(self) -> int:

        def get_num_nodes_helper(root: TrieNode) -> int:
            if len(root.children) == 0:
                return 1

            return 1 + sum(get_num_nodes_helper(child) for child in root.children)

        return get_num_nodes_helper(root=self._root) - 1
