import unittest

from src.data_struct.trie import (
    TrieNode,
    Trie
)

class TestTrie(unittest.TestCase):
    def testTrieNode(self) -> None:
        node = TrieNode("trieNode", 0, None)

        self.assertEqual("trieNode", node.token, "Value of token on the node should match.")
    
    def testTrieNodeInsert(self) -> None:
        node = TrieNode("root", 0)

        child = node.insert("child1")

        self.assertEqual("child1", child.token, "Value of token on the child node should match.")
        self.assertEqual(1, child.level, "Depth level of child node should match.")
        self.assertEqual(node.token, child.parent.token, "Value of token of parent node should match with the actual parent node.")

        self.assertIsNotNone(node.get("child1"))

        child_no_exist = node.get("child2")
        self.assertIsNone(child_no_exist, "If token doesn't exist in the children, it should return None.")

    def testTrieSequence(self) -> None:
        node = TrieNode("root", 0)

        child1 = node.insert("child1")
        child2 = child1.insert("child2")

        result1 = node.get_sequence()
        self.assertEqual(0, len(result1), "Sequence of root node should be empty.")

        result2 = child1.get_sequence()
        self.assertEqual(1, len(result2), "Sequence of node on first level must have lenght 1.")
        self.assertEqual("child1", " ".join([x.token for x in result2]), "Sequence should only contain token of the node on level 1.")

        result3 = child2.get_sequence()
        self.assertEqual(2, len(result3), "Sequence of node on second level must have lenght 2.")
        self.assertEqual("child1 child2", " ".join([x.token for x in result3]), "Sequence should contain the correct sequence of tokens.")

        self.assertEqual("child1 child2", result3[-1].get_sequence_str(), "Sequence should contain correct order of tokens, same sequence as test above.")

    def testTrie(self) -> None:
        trie = Trie()

        self.assertIsNotNone(trie.root, "Root node of trie shouldn't be None.")
        self.assertEqual(0, trie.root.level, "Root node of trie should have depth level 0.")

    def testTrieInsertTokensLevel1(self):
        trie = Trie()

        trie.insert_tokens(["Token1"])

        ins_node = trie.root.get("Token1")
        self.assertIsNotNone(ins_node, "Node with token inserted should exist.")
        self.assertEqual(1, ins_node.level, "Node of token inserted should have depth level 1.")
        self.assertEqual(1, ins_node.freq, "Node of token inserted should have frequency of 1.")

        trie.insert_tokens(["Token2"])

        ins_node2 = trie.root.get("Token2")
        self.assertIsNotNone(ins_node, "Node previously inserted should exist.")
        self.assertEqual(1, ins_node.freq, "Node previously inserted should have frequency of 1.")
        self.assertIsNotNone(ins_node2, "Node with second token inserted should exist.")
        self.assertEqual(1, ins_node2.level, "Node of second token inserted should have depth level 1.")
        self.assertEqual(1, ins_node2.freq, "Node of second token inserted should have frequency of 1.")

        trie.insert_tokens(["Token1"])
        ins_node = trie.root.get("Token1")
        self.assertEqual(2, ins_node.freq, "Node with token inserted two times should have frequency of 2.")
    
    def testTrieInsertTokensMultiLevel(self):
        trie = Trie()

        trie.insert_tokens(["Token1"])

        trie.insert_tokens(["Token1", "Token1_1"])

        node_token1 = trie.root.get("Token1")

        self.assertIsNotNone(node_token1, "Node of first layer should exist.")

        node_token1_1 = node_token1.get("Token1_1")

        self.assertIsNotNone(node_token1_1, "Node of second layer should exist.")
        self.assertEqual(node_token1.token, node_token1_1.parent.token, "Parent of node of second layer should match the node on the first layer.")
        self.assertEqual(1, node_token1.level, "Node of first layer should have depth level of 1.")
        self.assertEqual(2, node_token1_1.level, "Node of second layer should have depth level of 2.")
        self.assertEqual(1, node_token1.freq, "Node of first layer should have a frequency of 1.")
        self.assertEqual(1, node_token1_1.freq, "Node of second layer should have a frequency of 1.")

    def testTrieSearch(self):
        trie = Trie()

        trie.insert_tokens(["Token1", "Token1_1"])

        result = trie.search(["Token1", "Token1_2"])

        self.assertEqual(0, len(result), "Result of search of combination of tokens that wasn't inserted should return empty list.")

        result = trie.search(["Token1", "Token1_1"])

        self.assertEqual(2, len(result), "Result of search of combination of tokens should return 2 nodes.")

        self.assertEqual("Token1", result[0].token, "First node of the sequence should match the first token of the combination of tokens in the query.")
        self.assertEqual("Token1_1", result[1].token, "Second node of the sequence should match the second token of the combination of tokens in the query.")

    def testTrieMostFreq(self):
        trie = Trie()

        result0 = trie.get_most_freq(0)
        self.assertEqual("", result0.get_sequence_str(), "Sequence at root level should be empty.")
        self.assertEqual(-1, result0.level, "Level of resulting node should be -1 (invalid).")

        trie.insert_tokens(["Token1", "Token2"])
        trie.insert_tokens(["Token1", "Token3"])
        trie.insert_tokens(["Token1", "Token3"])
        trie.insert_tokens(["Token1", "Token2", "Token3"])
        trie.insert_tokens(["Token1"])
        trie.insert_tokens(["Token1"])
        trie.insert_tokens(["Token2"])

        result1 = trie.get_most_freq(0)
        self.assertEqual("", result1.get_sequence_str(), "Sequence at root level should be empty.")
        self.assertEqual(-1, result1.level, "Level of resulting node should be -1 (invalid).")

        result2 = trie.get_most_freq(1)
        self.assertEqual("Token1", result2.get_sequence_str(), "Sequence at root level should be empty.")
        
        result3 = trie.get_most_freq(2)
        self.assertEqual("Token1 Token3", result3.get_sequence_str(), "Sequence at root level should be empty.")
        
        result4 = trie.get_most_freq(3)
        self.assertEqual("Token1 Token2 Token3", result4.get_sequence_str(), "Sequence at root level should be empty.")

        result5 = trie.get_most_freq(4)
        self.assertEqual("", result5.get_sequence_str(), "Sequence at non existant level should be empty.")
        self.assertEqual(-1, result5.level, "Level of resulting node should be -1 (invalid).")

    def testTrieGetLevel(self):
        trie = Trie()

        result1 = trie.get_level(0)
        self.assertEqual(0, len(result1), "Root level should return an empty list.")

        result2 = trie.get_level(1)
        self.assertEqual(0, len(result2), "Empty or non-existant level should return an empty list.")

        trie.insert_tokens(["Token1", "Token2", "Token3"])
        trie.insert_tokens(["Token2", "Token2"])
        trie.insert_tokens(["Token3"])

        result3 = trie.get_level(1)
        self.assertEqual(3, len(result3), f"First level of trie should contain 3 nodes. Got {result3}.")
        expected = ["Token1", "Token2", "Token3"]
        for node in result3:
            if node.token not in expected:
                self.fail(f"Result missing expected token {node.token}")
            expected.remove(node.token)
        
        self.assertEqual(0, len(expected), "Result doesn't contain all nodes expected on the level 1.")

        result4 = trie.get_level(2)
        self.assertEqual(2, len(result4), "Second level of trie should contain 2 nodes.")
        expected = ["Token2", "Token2"]
        for node in result4:
            if node.token not in expected:
                self.fail(f"Result missing expected token {node.token}")
            expected.remove(node.token)
        
        self.assertEqual(0, len(expected), "Result doesn't contain all nodes expected on the level 2.")

        result5 = trie.get_level(3)
        self.assertEqual(1, len(result5), "Third level of trie should contain 1 node.")
        expected = ["Token3"]
        for node in result5:
            if node.token not in expected:
                self.fail(f"Result missing expected token {node.token}")
            expected.remove(node.token)
        
        self.assertEqual(0, len(expected), "Result doesn't contain all nodes expected on the level 3.")        

if __name__ == "__main__":
    unittest.main()