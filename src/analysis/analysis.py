import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from yellowbrick.text import freqdist

from utils.io import (
    read_excel,
    write_csv,
    save_figure
)
from utils.plot import get_wordcloud
from tpu.tpu import TPU

# Set plot style
plt.style.use("seaborn")

# Set maximum columns to be displayed by Pandas
pd.set_option("display.max_columns", 500)

DATA_DIR = "../data"
OUT_DIR = "./out"
GENERATE_CHARTS = False

# Full articles
articles = read_excel("OpArticles.xlsx", directory=DATA_DIR)
# Article ADUs
articles_adu = read_excel("OpArticles_ADUs.xlsx", directory=DATA_DIR)

fig, ax = plt.subplots()
sns.countplot(data=articles_adu, x="label", ax=ax)
ax.set_title("Distribution of labels on ADUs")
ax.set_xlabel("Label")
ax.set_ylabel("Count")
save_figure(fig, "label_distribution.png", directory=OUT_DIR, format="png", dpi=150)
plt.clf()

fig, ax = plt.subplots()
sns.kdeplot(x=articles_adu.loc[:, "tokens"].apply(len), hue=articles_adu["label"], ax=ax)
ax.set_title("Distribution of token length per class of ADU")
ax.set_xlabel("ADU token length")
save_figure(fig, "adu_token_length_distribution.png", directory=OUT_DIR, format="png", dpi=150)
plt.clf()

"""Analysis of raw dataset
"""

# raw_tpu = TPU(
#     type="count",
#     ngram_max=1,
#     allow_stopwords=True,
#     stopwords=None,
#     exclude_stopwords=None,
#     use_idf=False
# )

# raw_tpu.process(articles_adu, "tokens")

# unigrams = raw_tpu.tfidf_matrix
# features = raw_tpu.vectorizer.get_feature_names_out()

# df = pd.DataFrame.sparse.from_spmatrix(data=unigrams, columns=features)

# fig, ax = plt.subplots(figsize=(10, 10))
# visualizer = freqdist(
#     features=features,
#     X=df.loc[:, df.columns != "label"],
#     ax=ax,
#     n=50,
#     orient="h",
#     show=False,
# )

# ax.set_title("Frequency Distribution of Top 50 tokens on ADUs")
# ax.set_xlabel("Token count")

# save_figure(fig, "unfiltered_token_distribution.png", directory=OUT_DIR, format="png", dpi=150)
# plt.clf()

# write_csv(
#     df.T.sum(axis=1).sort_values(ascending=False).reset_index(),
#     "unfiltered_most_common_tokens.csv",
#     OUT_DIR,
#     header=["Token", "Count"],
#     index=False
# )

# df["label"] = articles_adu["label"]
# fig, ((ax_fact, ax_policy, _ax), (ax_nvalue, ax_value, ax_pvalue)) = plt.subplots(ncols=3, nrows=2, figsize=(20, 20))
# fig.delaxes(_ax)

# for label, ax in zip(("Fact", "Policy", "Value(-)", "Value", "Value(+)"), (ax_fact, ax_policy, ax_nvalue, ax_value, ax_pvalue)):
#     X = df.loc[df["label"] == label, df.columns != "label"]
#     visualizer = freqdist(
#         features=features,
#         X=X,
#         ax=ax,
#         n=50,
#         orient="h",
#         show=False,
#     )
#     ax.set_title(f"{label}-ADUs")
#     freq_df = X.T.sum(axis=1).sort_values(ascending=False)
#     dividers = np.array([.25, .5, .75]) * np.max(freq_df)
#     ax.vlines(x=dividers, ymin=0, ymax=1, transform=ax.get_xaxis_transform(), colors="r")
#     write_csv(
#         freq_df.reset_index(),
#         f"{label}_unfiltered_most_common_tokens.csv",
#         OUT_DIR,
#         header=["Token", "Count"],
#         index=False
#     )

# fig.tight_layout()
# fig.suptitle("Frequency Distribution of Top 50 tokens")
# fig.subplots_adjust(top=0.95)
# save_figure(fig, "class_unfiltered_token_distribution.png", directory=OUT_DIR, format="png", dpi=150)
# plt.clf()

stopwords_set = [
    "da", "de", "do", "das", "dos",
    "e", "a", "o", "as", "os",
    "um", "uma", "uns", "umas",
    "em", "que", "para", "com", "se", "ao", "ou",
    "na", "no", "nas", "nos",
    ",", "(", ")", ".", "-",
]

"""Analysis after removal of relevant stopwords from the previous analysis 
"""

stopword_tpu = TPU(
    type="count",
    ngram_max=1,
    allow_stopwords=False,
    stopwords=stopwords_set,
    exclude_stopwords=["nao", "nunca", "nem", "jamais", "sim"],
    use_idf=False,
    dictionary_path="tpu_dictionary.csv",
    adu_dictionary_path="adu_dictionary.csv",
    enable_extra_features=True
)

#stopword_tpu.generate_adu_dictionary(articles_adu, "tokens")

stopword_tpu.process(articles_adu, "tokens")
stopword_tpu.save_dictionary()
stopword_tpu.save_adu_dictionary()

unigrams = stopword_tpu.tfidf_matrix
features = stopword_tpu.vectorizer.get_feature_names_out()

df = pd.DataFrame.sparse.from_spmatrix(data=unigrams, columns=features)

# fig, ax = plt.subplots(figsize=(10, 10))
# visualizer = freqdist(
#     features=features,
#     X=df.loc[:, df.columns != "label"],
#     ax=ax,
#     n=50,
#     orient="h",
#     show=False,
# )

# ax.set_title("Frequency Distribution of Top 50 tokens on ADUs")
# ax.set_xlabel("Token count")

# save_figure(fig, "token_distribution.png", directory=OUT_DIR, format="png", dpi=150)
# plt.clf()

# write_csv(
#     df.T.sum(axis=1).sort_values(ascending=False).reset_index(),
#     "most_common_tokens.csv",
#     OUT_DIR,
#     header=["Token", "Count"],
#     index=False
# )

df["label"] = articles_adu["label"]
# fig, ((ax_fact, ax_policy, _ax), (ax_nvalue, ax_value, ax_pvalue)) = plt.subplots(ncols=3, nrows=2, figsize=(20, 20))
# fig.delaxes(_ax)

# for label, ax in zip(("Fact", "Policy", "Value(-)", "Value", "Value(+)"), (ax_fact, ax_policy, ax_nvalue, ax_value, ax_pvalue)):
#     X = df.loc[df["label"] == label, df.columns != "label"]
#     visualizer = freqdist(
#         features=features,
#         X=X,
#         ax=ax,
#         n=50,
#         orient="h",
#         show=False,
#     )
#     ax.set_title(f"{label}-ADUs")
#     freq_df = X.T.sum(axis=1).sort_values(ascending=False)
#     dividers = np.array([.25, .5, .75]) * np.max(freq_df)
#     ax.vlines(x=dividers, ymin=0, ymax=1, transform=ax.get_xaxis_transform(), colors="r")
#     write_csv(
#         freq_df.reset_index(),
#         f"{label}_most_common_tokens.csv",
#         OUT_DIR,
#         header=["Token", "Count"],
#         index=False
#     )

# fig.tight_layout()
# fig.suptitle("Frequency Distribution of Top 50 tokens")
# fig.subplots_adjust(top=0.95)
# save_figure(fig, "class_token_distribution.png", directory=OUT_DIR, format="png", dpi=150)
# plt.clf()

"""Adjusting dataset
"""
def get_value_type(label):
    if label == "Value(-)":
        return 0
    if label == "Value(+)":
        return 2
    return 1

# Negative connotation has value 0, neutral connotation has value 1, positive connotation has value 2
df["value_type"] = df["label"].apply(get_value_type)
df.loc[df["label"].str.startswith("Value"), "label"] = "Value"
# write_csv(
#     df,
#     "features.csv",
#     OUT_DIR,
#     index=False
# )

edf = pd.DataFrame(stopword_tpu.efeature_matrix, columns=stopword_tpu.efeatures)
edf[["label", "value_type"]] = df[["label", "value_type"]]

write_csv(
    edf,
    "efeatures.csv",
    OUT_DIR,
    index=False
)