from collections import defaultdict
from collections import defaultdict
from typing import DefaultDict, Tuple

from data_struct.trie import Trie

from nltk.util import (
    everygrams
)

from nltk.tokenize import (
    word_tokenize,
    sent_tokenize
)

class TPU:
    ngrams: Trie
    pos_tags: DefaultDict[str, str]
    ngram_max: int
    def __init__(self, ngram_max : int = 4):
        self.ngrams = Trie()
        self.pos_tags = defaultdict(lambda: "UNKNOWN")
        self.ngram_max = ngram_max if ngram_max > 0 else -1

    def process(self, adu_tokens : str, language='english') -> None:
        sentences = sent_tokenize(adu_tokens, language=language)

        for sentence in sentences:
            tokens = word_tokenize(sentence, language=language)

            # Process n-grams
            ngrams = everygrams(tokens, min_len=1, max_len=self.ngram_max, pad_left=True, left_pad_symbol="<sent>", pad_right=True, right_pad_symbol="</sent>")
            
            for ngram in ngrams:
                self.ngrams.insert_tokens(ngram)
