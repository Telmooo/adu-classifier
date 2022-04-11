from typing import DefaultDict, List, Literal, Set, Tuple, Union, Optional, Dict

from collections import defaultdict
from unicodedata import normalize
import csv

import pandas as pd
import numpy as np

from deep_translator import GoogleTranslator

from nltk.util import (
    everygrams
)
from nltk.tokenize import (
    word_tokenize,
    sent_tokenize
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)

import spacy

class TPU:
    vectorizer: Union[CountVectorizer, TfidfVectorizer]
    tfidf_matrix: np.ndarray
    efeature_matrix: np.ndarray
    ngram_max: int
    language: str
    allow_stopwords: bool
    stopwords: Set[str]
    exclude_stopwords: Set[str]
    spacyModel: spacy.language.Language
    translator: GoogleTranslator
    dictionary_path: str
    dictionary: Dict[str, str]
    senti_analyser: SentimentIntensityAnalyzer
    neg_adv: Set[str]
    cconj_adver: Set[str]

    def __init__(self, type : Literal['count', 'tfidf'], ngram_max : int = 4,
                allow_stopwords : bool = False, stopwords : Optional[Set[str]] = None, exclude_stopwords : Optional[Set[str]] = None,
                dictionary_path : Optional[str] = None, use_idf : bool = True):
        self.ngram_max = ngram_max if ngram_max > 0 else -1
        
        self.allow_stopwords = allow_stopwords
        self.stopwords = set(stopwords) if stopwords is not None else set()
        self.exclude_stopwords = set(exclude_stopwords) if exclude_stopwords is not None else set()

        self.tfidf_matrix = []
        self.efeatures = ["n_entities", "unique_entities", "org_entities", "loc_entities", "per_entities", "misc_entities",
                            "adu_polarity",
                            "adj_count", "adv_count", "cconj_count", "sconj_count", "noun_count", "det_count", "verb_count",
                            "intj_count", "part_count", "pron_count", "propn_count", "punct_count"
                        ]
        self.efeature_matrix = np.zeros(shape=(0, len(self.efeatures)))
        self.spacyModel = spacy.load("pt_core_news_md")

        self.translator = GoogleTranslator(source="pt", target="en")
        self.dictionary_path = dictionary_path if dictionary_path is not None else "tpu_dictionary.temp.csv"

        self.dictionary = {
            "sim": "yes",
            "não": "no",
            "nunca": "never",
            "nem": "nor",
            "jamais": "never",
        }

        if dictionary_path is not None:
            with open(dictionary_path, "r") as dictionary_csv:
                reader = csv.reader(dictionary_csv)
                next(reader)
                self.dictionary = dict(reader)

        self.senti_analyser = SentimentIntensityAnalyzer()
        self.neg_adv = set(["não", "nunca", "nada", "nem", "jamais"])
        self.cconj_adver = set(["contudo", "entretanto", "mas", "porém", "todavia", "apesar"])
        self.pos_analyze = set(["NOUN", "ADJ", "ADV", "VERB"])

        if type == 'count':
            self.vectorizer = CountVectorizer(
                input='content',
                encoding='utf-8',
                lowercase=True,
                ngram_range=(1, self.ngram_max),
                tokenizer=self.__tokenize,
                analyzer='word',
            )
        else:
            self.vectorizer = TfidfVectorizer(
                input='content',
                encoding='utf-8',
                lowercase=True,
                ngram_range=(1, self.ngram_max),
                norm='l2',
                use_idf=use_idf,
                smooth_idf=True,
                sublinear_tf=False,
                tokenizer=self.__tokenize,
                analyzer='word',
            )

    @staticmethod
    def __get_polarity(polarity_scores: dict):
        pos, neu, neg = polarity_scores["pos"], polarity_scores["neu"], polarity_scores["neg"]
        amortized_polarity = pos / (1 + neg + neu) - neg / (1 + pos + neu) + neu / (2 - pos) - neu / (2 - neg)
        return (amortized_polarity + polarity_scores["compound"]) / 2.0

    def save_dictionary(self, dictionary_path : Optional[str] = None):
        path = dictionary_path if dictionary_path is not None else self.dictionary_path

        with open(path, "w", newline="") as dictionary_csv:
            writer = csv.DictWriter(dictionary_csv, fieldnames=["pt", "en"])
            writer.writeheader()
            writer.writerows(self.dictionary)

    def __tokenize(self, adu_tokens : str):
        doc = self.spacyModel(adu_tokens)
        lemmas = []
        pos_tags = defaultdict(int)

        translated_tokens = self.translator.translate(adu_tokens)
        polarity = self.__get_polarity(self.senti_analyser.polarity_scores(translated_tokens))
        calculated_polarity = None
        neg_factor = 1 # 1 when neutral, -0.5 when in presence of a negation

        for token in doc:
            lemma = normalize("NFKD", token.lemma_).encode(encoding="ascii", errors="ignore").decode("utf-8")
            if not lemma:
                lemma = token.lemma_ if token.lemma_ else token.text

            if self.allow_stopwords or lemma in self.exclude_stopwords or (lemma not in self.stopwords and not token.is_stop and not token.is_punct):
                lemmas.append(lemma)
                pos_tags[token.pos_] += 1

            if not token.text.isdecimal() and (token.pos_ in self.pos_analyze or token.text in self.neg_adv):
                if token.text in self.dictionary:
                    translated_token = self.dictionary[token.text]
                else:
                    try:
                        translated_token = self.translator.translate(token.text)
                    except:
                        translated_token = token.text
                        print(token.text)
                    self.dictionary[token.text] = translated_token
                    self.f.writerow([token.text, translated_token])

        #         polarity_scores = self.senti_analyser.polarity_scores(translated_token)

        #         if calculated_polarity is None:
        #             calculated_polarity = neg_factor * self.__get_polarity(polarity_scores)
        #         else:
        #             calculated_polarity = np.mean([polarity, neg_factor * self.__get_polarity(polarity_scores)])

        #     morph_polarity = token.morph.get("Polarity")
        #     is_neg = bool(token.pos_ == "ADV" and morph_polarity and morph_polarity[0] == "Neg")
        #     if is_neg or token.text in self.neg_adv:
        #         neg_factor = -0.5
        #     elif token.text in self.cconj_adver:
        #         neg_factor = 1.0

        # if calculated_polarity is not None:
        #     polarity = np.mean([polarity, calculated_polarity])

        # n_entities = len(doc.ents)
        # unique_entities = len(set( [ ent.ent_id for ent in doc.ents ] ))
        # entity_types = defaultdict(int)

        # for ent in doc.ents:
        #     entity_types[ent[0].ent_type_] += 1
        
        # efeatures = np.array(
        #     [   n_entities, unique_entities, entity_types["ORG"], entity_types["LOC"], entity_types["PER"], entity_types["MISC"],
        #         polarity,
        #         pos_tags["ADJ"], pos_tags["ADV"], pos_tags["CCONJ"], pos_tags["SCONJ"], pos_tags["NOUN"], pos_tags["DET"], pos_tags["VERB"],
        #         pos_tags["INTJ"], pos_tags["PART"], pos_tags["PRON"], pos_tags["PROPN"], pos_tags["PUNCT"]
        #     ])
        
        # self.efeatures = np.vstack((self.efeatures, efeatures))
            
        return lemmas

    def process(self, df: pd.DataFrame, token_col : str):
        self.tfidf_matrix = self.vectorizer.fit_transform(
                df[token_col]
            )
