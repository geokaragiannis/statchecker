import logging
from sklearn.feature_extraction.text import TfidfVectorizer

from src.featurizer.sentence_embedding import SentenceEmbedding
from src import helpers


class FeatureExtractor:
    def __init__(self, max_features=None, mode="word-embeddings"):
        self.nlp = helpers.get_nlp()
        if mode == "tfidf":
            self.featurizer = TfidfVectorizer(sublinear_tf=True, min_df=1, smooth_idf=True, norm="l2", encoding="utf-8",
                                              analyzer="word", ngram_range=(1,2))
        elif mode == "word-embeddings":
            self.featurizer = SentenceEmbedding()
        else:
            self.featurizer = None
        self.features = None
        self.logger = logging.getLogger(__name__)

    def featurize_claims(self, tokenized_claims):
        """
        Returns anf fits features of sentences
        :param tokenized_claims: List of strings of the tokenized claims
        :return: list of lists
        """
        self.features = self.featurizer.fit_transform(tokenized_claims)
        return self.features

    def featurize_test(self, tokenized_test_list):
        """
        Returns the features of a test set (does not fit the featurizer)
        """

        return self.featurizer.transform(tokenized_test_list)
