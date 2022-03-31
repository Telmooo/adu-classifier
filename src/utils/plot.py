from wordcloud import WordCloud

from data_struct.trie import Trie

def get_wordcloud(trie : Trie, level : int, **kwargs) -> WordCloud:
    """Get the wordcloud for the level of the trie specified.

    Args:
        trie (Trie): Trie structure containing the nodes with tokens and its frequencies.
        level (int): Depth level.
        kwargs: Extra arguments to be passed to creation of WordCloud.

    Returns:
        WordCloud: Resulting wordcloud object
    """
    wc = WordCloud(**kwargs)

    freq_table = trie.get_level_freq_table(level)
    wc.generate_from_frequencies(freq_table)

    return wc