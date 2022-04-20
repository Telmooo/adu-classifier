# ADU Classifier

## Introduction

* The initial approach upon building the classifier was to modelling a data structure that not only would be able to store the token, but also create multiple instances of this token, for the same sentence, but in different n-grams.

* Thus, it was thought of using a trie (or prefix tree), that is a tree data structure used for locating specific keys from within a set. The actual token could be used as a key, instead of an individual character.

* In that way, it would be possible to analyze different n-gram scenarios for the same sentence, where the greater the depth level, the greater the n-value.

* However, this approach proved not to be very effective and it was discarded. 

* So the next step was to create a structure that could already do the pre-processing of the token. That structure is called Token Processing Unit (TPU) and it will be explained in the next sections.

## Preprocessing
* The first step was to load the article data.

* As was later noticed, the model was not doing well with
portuguese words. So it was decided to translate them
using google translate library.

* The process was very time consuming, so in order to improve the preprocessing time, it was created two dictionaries: the first one containing the adu original and translated sentences, and the second one containing original and translated words.

* In order to determine polarity, it was also considered:
    * sentiment analysis 
    * adverbs of negation
    * conjunctive adverbs of contrast

* The calculus made uses the sentiment analysis polarity function, that was also improved with an amortization technique.

* It was also taken in consideration a negation factor, where if the token has an adverb of negation or a conjunctive adverb of contrast, it would
reflect "partially" on the polarity of the word. 

* In other words, the negation factor was not reverting entirely the polarity in above cases case, because there is a sentimental difference in senteces that are affected by an adverb of negation or a conjunctive adverb of contrast. For instance:

    * I love this - A very positive sentence (+1.0)
    * I do like this - A positive sentence (+0.5)
    * I do _not_ like this - A positive sentence negated (+0.5 * -0.5 = -0.25)
    * I hate this - A very negative sentence (-1.0)

* This also proved to be a very interesting approach in terms of negating a negative sentence, as in the case:

    * I do _not_ hate this - A very negative sentence (-1), but negated (-0.5), turns out to be a possible positive sentence (-1.0 * -0.5 = +0.5).


* POS tagging was also used, but it was later verified that it had little influence on the model.

* It was also noticed that the use of tf-idf was an important weighting factor in preprocessing.

<!-- The parts of the dataset that you have managed to use, and any preprocessing that you needed to do before arriving at an actual dataset for the proposed classification task. [2 slides] -->

## Feature Engineering
The exact classification task(s) that you have addressed, and a brief exploratory data analysis for the classes considered. [2 slides]

## Modelling

The representation techniques you have followed and the selected machine learning algorithms and their parameterizations. [2 slides]

## Testing
The results obtained and a short error analysis (confusion matrices, sample inspection, ...). [2 slides]

## Result Analysis
The results obtained and a short error analysis (confusion matrices, sample inspection, ...). [2 slides]
## Conclusions 
[1 slide]

<!-- ## Objective


## Analysis -->
