import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src import helpers


class FeatureExtractor:
    def __init__(self, max_features=None):
        self.nlp = helpers.get_nlp()
        self.featurizer = TfidfVectorizer(sublinear_tf=True, min_df=1, smooth_idf=True, norm="l2", encoding="utf-8",
                                          analyzer="word", ngram_range=(1,2))
        self.features = None
        self.logger = logging.getLogger(__name__)

    def featurize_claims(self, tokenized_claims):
        """
        Returns featurized claims
        :param tokenized_claims: List of strings of the tokenized claims
        :return: list of lists
        """
        self.features = self.featurizer.fit_transform(tokenized_claims)
        return self.features
