from __future__ import annotations
from typing import (
    DefaultDict,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Union
)

from collections import (
    defaultdict,
    deque
)

class TrieNode:
    """TrieNode class.

    Represents a node in a trie, stores relevant information to the node, as well as its parent node, depth level and frequency of the combination of tokens leading up to this node.

    Args:
        token (str): Value of the node.
        level (int): Depth level of the node in the trie.
        parent (TrieNode, optional): Parent node, None if it doesn't have parent node.

    Attributes:
        parent (Union[TrieNode, None]): Where parent node is stored.
        token (str): Where value of the node is stored.
        children (DefaultDict[str, TrieNode]): key-value mapping to store children of node in the trie, key being the value of the child node and value its TrieNode.
        level (int): Depth level of node in the trie.
        freq (int): Frequency of the combination of tokens leading up to this node in the trie.
    """
    parent : Union[TrieNode, None]
    token : str
    original_token : str
    children : DefaultDict[str, TrieNode]
    level : int
    freq : int
    def __init__(self, token : str, level : int, original_token : Optional[str] = None, parent : Optional[TrieNode] = None) -> None:
        self.parent = parent
        self.token = token
        self.original_token = original_token if original_token is not None else token
        self.children = defaultdict(lambda: TrieNode("", -1))
        self.level = level
        self.freq = 0

    def insert(self, token : str) -> TrieNode:
        """Inserts a new node onto the trie. Overrides node if it already exists.

        Args:
            token (str): Value of the token.

        Returns:
            TrieNode: Node inserted.
        """
        child = TrieNode(token, self.level + 1, parent=self)
        self.children[token] = child
        return child

    def get(self, token : str) -> Union[TrieNode, None]:
        """Access the TrieNode whose key has value equal to `token`, if it doesn't exist `get` returns `None`.

        Args:
            token (str): Value of the token.

        Returns:
            Union[TrieNode, None]: Returns the resulting node if it exists, otherwise returns `None`.
        """
        result = self.children[token]
        if result.level == -1:
            return None
        return result

    def update_freq(self) -> None:
        """Increases the frequency of the combination of nodes leading up to this one.
        """
        self.freq += 1

    def get_sequence(self) -> List[TrieNode]:
        """Get the sequence of tokens forming up to this node.

        Returns:
            List[TrieNode]: List containing the nodes in order that form the combination of tokens of present node.
        """
        result = []
        current = self
        while current.level > 0:
            result.insert(0, current)
            current = current.parent
        
        return result

    def get_sequence_str(self, sep : str = " ") -> str:
        """Get the value of the sequence of tokens forming up to this node, joining the tokens by the separator specified. 

        Args:
            sep (str, optional): Separator for tokens. Defaults to " ".

        Returns:
            str: Resulting string of the sequence of tokens.
        """
        return sep.join([node.token for node in self.get_sequence()])

    def __repr__(self) -> str:
        """Representation of TrieNode, including the tokens leading up until this one and the frequency of the combination.

        Returns:
            str: Resulting string of TrieNode representation.
        """
        if self.level == 0:
            return "RootTrieNode[]"

        return f"TrieNode['{self.get_sequence_str()}' : {self.freq}]"

class Trie:
    """Trie class.

    Represents the trie structure.

    Attributes:
        root (TrieNode): Root of the trie, automatically generated to serve as common node of all token occurences.
    """
    root : TrieNode
    def __init__(self) -> None:
        self.root = TrieNode("", 0)
    
    def insert_tokens(self, tokens : Union[List[str], Tuple[str]]) -> None:
        """Inserts the combination of tokens in order in the Trie.

        Trie will update frequency of this combination of nodes internally.

        Args:
            tokens (Union[List[str], Tuple[str]]): List of tokens to be inserted.
        """
        current = self.root
        for token in tokens:
            node = current.get(token)
            if node is None:
                node = current.insert(token)
            
            current = node

        current.update_freq()

    def search(self, tokens : Union[List[str], Tuple[str]]) -> List[TrieNode]:
        """Searchs the sequence of tokens in input and returns the list of tokens correspondent.

        Args:
            tokens (Union[List[str], Tuple[str]]): Combination of tokens to be searched, in order.

        Returns:
            List[TrieNode]: Resulting nodes of the search, returning an empty list if no result is found.
        """
        current = self.root
        result = []
        for token in tokens:
            node = current.get(token)
            if node is None:
                return []
            result.append(node)
            current = node
        return result
    
    def get_level(self, level : int) -> List[TrieNode]:
        """Get all nodes at depth level specified. 

        Args:
            level (int): Depth level.

        Returns:
            List[TrieNode]: List containing all nodes at the depth level specified.
        """
        result = []
        to_visit : Deque[TrieNode] = deque()
        to_visit.append(self.root)
        
        while to_visit:
            current = to_visit.popleft()

            if current.level >= level:
                continue

            if current.level == level - 1:
                result.extend(current.children.values())

            else:
                to_visit.extend(current.children.values())
        
        return result

    def get_level_freq_table(self, level : int) -> DefaultDict[str, int]:
        """Get the frequency table for the level specified of the Trie.

        Args:
            level (int): Depth level.

        Returns:
            Dict[str, int]: Dictionary containing the keys and its frequencies. Key is the token or the combination of tokens forming that node on the level. Non-existant keys default to 0.
        """
        result = defaultdict(lambda: 0)
        to_visit : Deque[TrieNode] = deque()
        to_visit.append(self.root)

        while to_visit:
            current = to_visit.popleft()

            if current.level > level:
                continue
            if current.level == level:
                token = current.get_sequence_str()
                result[token] = current.freq
            else:
                to_visit.extend(current.children.values())

        return result

    def get_most_freq(self, level : int) -> TrieNode:
        """Gets the combination of tokens that is the most frequent at level specified.

        Args:
            level (int): Depth level.

        Returns:
            TrieNode: Node at level specified with the maximum frequency.
        """
        max_node = TrieNode("", -1)
        max_freq = 0

        to_visit : Deque[TrieNode] = deque()
        to_visit.append(self.root)

        while to_visit:
            current = to_visit.popleft()

            if current.level >= level:
                continue

            if current.level == level - 1:
                max_child = max(current.children.values(), key=lambda x: x.freq, default=TrieNode("", -1))

                if max_child.freq > max_freq:
                    max_node = max_child
                    max_freq = max_child.freq
            else:
                to_visit.extend(current.children.values())

        return max_node
