"""
Implements a custom scikit-learn Transformer, which returns an embedding matrix for the input sentences.
For each sentence it computes a 300 dimensional embedding which is the result of the average of the word embeddings
in the sentence.
"""

from src import helpers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceEmbedding(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.emb_dim = 300
        self.spacy_nlp = helpers.get_nlp()

    def fit(self, X):
        return self

    def transform(self, X):
        emb_matrix = np.zeros((len(X), self.emb_dim))
        for i, x in enumerate(X):
            # average pooling of the word embeddings (built-in in spacy)
            emb = self.spacy_nlp(x).vector
            emb_matrix[i] = emb
        return emb_matrix
