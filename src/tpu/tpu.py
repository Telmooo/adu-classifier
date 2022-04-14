from typing import DefaultDict, List, Literal, Set, Tuple, Union, Optional, Dict

from collections import defaultdict
import sys
import os
from unicodedata import normalize
import csv
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.sparse import (
    save_npz,
    spmatrix
)

from deep_translator import GoogleTranslator

from nltk.util import (
    everygrams
)
from nltk.tokenize import (
    word_tokenize,
    sent_tokenize
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
import joblib

import spacy

from utils.io import write_csv

class TPU:
    vectorizer: Union[CountVectorizer, TfidfVectorizer]
    tfidf_matrix: spmatrix
    efeature_matrix: np.ndarray
    ngram_range: Tuple[int, int]
    language: str
    allow_stopwords: bool
    stopwords: Set[str]
    exclude_stopwords: Set[str]
    spacyModel: spacy.language.Language
    translator: GoogleTranslator
    dictionary_path: str
    dictionary: Dict[str, str]
    adu_dictionary_path: str
    adu_dictionary: Dict[str, str]
    senti_analyser: SentimentIntensityAnalyzer
    neg_adv: Set[str]
    cconj_adver: Set[str]
    enable_extra_features: bool
    processing_row: int
    total_rows: int
    def __init__(self, type : Literal['count', 'tfidf'], ngram_range : Tuple[int, int] = (1, 4),
                allow_stopwords : bool = False, stopwords : Optional[Set[str]] = None, exclude_stopwords : Optional[Set[str]] = None,
                dictionary_path : Optional[str] = None, adu_dictionary_path : Optional[str] = None,
                use_idf : bool = True, enable_extra_features : Optional[bool] = False):
        self.ngram_range = ngram_range
        
        self.allow_stopwords = allow_stopwords
        self.stopwords = set(stopwords) if stopwords is not None else set()
        self.exclude_stopwords = set(exclude_stopwords) if exclude_stopwords is not None else set()

        self.tfidf_matrix = []
        self.efeatures = ["token_len", "n_entities", "unique_entities", "org_entities", "loc_entities", "per_entities", "misc_entities",
                            "adu_polarity", "token_polarity", "blob_polarity", "blob_subjectivity",
                            "adj_count", "adv_count", "cconj_count", "sconj_count", "noun_count", "det_count", "verb_count",
                            "intj_count", "part_count", "pron_count", "propn_count", "punct_count"
                        ]

        self.processing_row = 0
        self.total_rows = 0
        self.enable_extra_features = bool(enable_extra_features)
        self.efeature_matrix = np.zeros(shape=(self.total_rows, len(self.efeatures)))
        self.spacyModel = spacy.load("pt_core_news_md")

        self.translator = GoogleTranslator(source="pt", target="en")
        self.dictionary_path = dictionary_path if dictionary_path is not None else "tpu_dictionary.temp.csv"
        self.adu_dictionary_path = adu_dictionary_path if adu_dictionary_path is not None else "adu_dictionary.temp.csv"

        self.dictionary = {
            "sim": "yes",
            "não": "no",
            "nunca": "never",
            "nem": "nor",
            "jamais": "never",
        }

        self.adu_dictionary = {}

        if dictionary_path is not None:
            with open(dictionary_path, mode="r", encoding="utf-8") as dictionary_csv:
                reader = csv.reader(dictionary_csv)
                next(reader)
                self.dictionary.update(dict(reader))
                print("Loaded dictionary!")

        if adu_dictionary_path is not None:
            with open(adu_dictionary_path, mode="r", encoding="utf-8") as dictionary_csv:
                reader = csv.reader(dictionary_csv)
                next(reader)
                self.adu_dictionary.update(dict(reader))
                print("Loaded ADU dictionary!")

        self.senti_analyser = SentimentIntensityAnalyzer()
        self.neg_adv = set(["não", "nunca", "nada", "nem", "jamais"])
        self.cconj_adver = set(["contudo", "entretanto", "mas", "porém", "todavia", "apesar"])
        self.pos_analyze = set(["NOUN", "ADJ", "ADV", "VERB"])

        if type == 'count':
            self.vectorizer = CountVectorizer(
                input='content',
                encoding='utf-8',
                lowercase=True,
                ngram_range=self.ngram_range,
                tokenizer=self.__tokenize,
                analyzer='word',
            )
        else:
            self.vectorizer = TfidfVectorizer(
                input='content',
                encoding='utf-8',
                lowercase=True,
                ngram_range=self.ngram_range,
                norm='l2',
                use_idf=use_idf,
                smooth_idf=True,
                sublinear_tf=True,
                tokenizer=self.__tokenize,
                analyzer='word',
            )

    @staticmethod
    def __get_polarity(polarity_scores: dict) -> float:
        """Calculates the estimated polarity score given a structure with the probability of each polarity and the compound polarity score.

        Probabilities are such that: P(POS) + P(NEU) + P(NEG) = 1.

        Args:
            polarity_scores (dict): Polarity scores calculated via Vader sentiment and intensity analyser.

        Returns:
            float: The estimated polarity score.
        """
        pos, neu, neg = polarity_scores["pos"], polarity_scores["neu"], polarity_scores["neg"]
        amortized_polarity = pos / (1 + neg + neu) - neg / (1 + pos + neu) + neu / (2 - pos) - neu / (2 - neg)
        return (amortized_polarity + polarity_scores["compound"]) / 2.0

    def save_vectorizer(self, vectorizer_path : str, feature_matrix_path : str):
        joblib.dump(self.vectorizer, vectorizer_path)
        save_npz(feature_matrix_path, self.tfidf_matrix)

    def save_dictionary(self, dictionary_path : Optional[str] = None):
        path = dictionary_path if dictionary_path is not None else self.dictionary_path

        with open(path, "w", newline="", encoding="utf-8") as dictionary_csv:
            writer = csv.writer(dictionary_csv)
            writer.writerow(["pt", "en"])
            writer.writerows(self.dictionary.items())

    def save_adu_dictionary(self, adu_dictionary_path : Optional[str] = None):
        path = adu_dictionary_path if adu_dictionary_path is not None else self.adu_dictionary_path

        with open(path, "w", newline="", encoding="utf-8") as adu_dictionary_csv:
            writer = csv.writer(adu_dictionary_csv)
            writer.writerow(["pt", "en"])
            writer.writerows(self.adu_dictionary.items())

    def __generate_adu_dictionary(self, adu_tokens : str) -> str:
        self.__progress_bar()

        in_tokens = adu_tokens.lower()

        out_tokens = ""
        try:
            out_tokens = self.translator.translate(in_tokens)
        except:
            sys.stderr(f"Invalid ADU token found {in_tokens} at row {self.processing_row}.")
        
        self.adu_dictionary[in_tokens] = out_tokens

        self.processing_row += 1
        return out_tokens

    def generate_adu_dictionary(self, df: pd.DataFrame, token_col : str) -> None:
        start_time = datetime.now()
        print(f"Start of ADU dictionary generation at: {start_time.strftime('%H:%M:%S')}")

        self.processing_row = 0
        self.total_rows = df.shape[0]

        out = pd.DataFrame(columns=["pt"])
        out.loc[:, "pt"] = df[token_col]
        df[token_col].apply(self.__generate_adu_dictionary)

        self.save_adu_dictionary()

        self.__progress_bar(finished=True)

        end_time = datetime.now()
        seconds = int( (end_time - start_time).total_seconds() )
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        seconds, _ = divmod(seconds, 1)
        print(f"Ended ADU dictionary generation at {end_time.strftime('%H:%M:%S')}, taking {hours}h {minutes}min {seconds}s.")
        print(f"Saved generated dictionary onto \"{os.path.abspath(self.adu_dictionary_path)}\"")


    def __tokenize(self, adu_tokens : str):
        self.__progress_bar()

        doc = self.spacyModel(adu_tokens)
        lemmas = []
        
        if self.enable_extra_features:
            pos_tags = defaultdict(int)
            
            translated_tokens = self.adu_dictionary.get(adu_tokens, "")
            if not translated_tokens:
                try:
                    translated_tokens = self.translator.translate(adu_tokens)
                except:
                    translated_tokens = adu_tokens
                self.dictionary[adu_tokens] = translated_tokens

            if translated_tokens is None:
                translated_tokens = adu_tokens
            
            
            polarity = self.__get_polarity(self.senti_analyser.polarity_scores(translated_tokens))
            calculated_polarity = None
            neg_factor = 1 # 1 when neutral, -0.5 when in presence of a negation

            token_len = 0

            blob = TextBlob(translated_tokens)
            blob_polarity = blob.sentiment.polarity
            blob_subjectivity = blob.sentiment.subjectivity

        for token in doc:
            lemma = normalize("NFKD", token.lemma_).encode(encoding="ascii", errors="ignore").decode("utf-8")
            if not lemma:
                lemma = token.lemma_ if token.lemma_ else token.text

            if self.allow_stopwords or lemma in self.exclude_stopwords or (lemma not in self.stopwords and not token.is_stop and not token.is_punct):
                lemmas.append(lemma)

                if self.enable_extra_features:
                    pos_tags[token.pos_] += 1
                    token_len += 1

            if self.enable_extra_features:
                if not token.text.isdecimal() and (token.pos_ in self.pos_analyze or token.text in self.neg_adv):
                    translated_token = self.dictionary.get(token.text, "")
                    if not translated_token:
                        try:
                            translated_token = self.translator.translate(token.text)
                        except:
                            translated_token = token.text
                        self.dictionary[token.text] = translated_token

                    if translated_token is None:
                        translated_token = token.text

                    polarity_scores = self.senti_analyser.polarity_scores(translated_token)

                    if calculated_polarity is None:
                        calculated_polarity = neg_factor * self.__get_polarity(polarity_scores)
                    else:
                        calculated_polarity = np.mean([calculated_polarity, neg_factor * self.__get_polarity(polarity_scores)])

                morph_polarity = token.morph.get("Polarity")
                is_neg = bool(token.pos_ == "ADV" and morph_polarity and morph_polarity[0] == "Neg")
                if is_neg or token.text in self.neg_adv:
                    neg_factor = -0.5
                elif token.text in self.cconj_adver:
                    neg_factor = 1.0

        if self.enable_extra_features:
            if calculated_polarity is None:
                calculated_polarity = 0

            n_entities = len(doc.ents)
            unique_entities = len(set( [ ent.ent_id for ent in doc.ents ] ))
            entity_types = defaultdict(int)

            for ent in doc.ents:
                entity_types[ent[0].ent_type_] += 1
            
            efeatures = np.array(
                [   token_len, n_entities, unique_entities, entity_types["ORG"], entity_types["LOC"], entity_types["PER"], entity_types["MISC"],
                    polarity, calculated_polarity, blob_polarity, blob_subjectivity,
                    pos_tags["ADJ"], pos_tags["ADV"], pos_tags["CCONJ"], pos_tags["SCONJ"], pos_tags["NOUN"], pos_tags["DET"], pos_tags["VERB"],
                    pos_tags["INTJ"], pos_tags["PART"], pos_tags["PRON"], pos_tags["PROPN"], pos_tags["PUNCT"]
                ])
            
            self.efeature_matrix[self.processing_row] = efeatures
        
        self.processing_row += 1

        return lemmas

    def __progress_bar(self, finished : bool = False):
        bar_len = 60
        perc = self.processing_row / float(self.total_rows)
        filled_len = int(round(bar_len * perc))

        percents = round(100.0 * perc, 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        finisher = '\n' if finished else '\r'

        sys.stdout.write('[%s] %s%s ...%s/%s rows%s' % (bar, percents, '%', self.processing_row, self.total_rows, finisher))
        sys.stdout.flush()

    def process(self, df: pd.DataFrame, token_col : str):
        start_time = datetime.now()
        print(f"Start of processing at: {start_time.strftime('%H:%M:%S')}")

        self.processing_row = 0
        self.total_rows = df.shape[0]

        if self.enable_extra_features:
            self.efeature_matrix = np.zeros(shape=(self.total_rows, len(self.efeatures)))
            print(f"Allocating array of shape {self.efeature_matrix.shape}")

        self.tfidf_matrix = self.vectorizer.fit_transform(
                df[token_col]
            )

        self.__progress_bar(finished=True)

        end_time = datetime.now()
        seconds = int( (end_time - start_time).total_seconds() )
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        seconds, _ = divmod(seconds, 1)
        print(f"Ended processing at {end_time.strftime('%H:%M:%S')}, taking {hours}h {minutes}min {seconds}s.")
