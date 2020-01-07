import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.featurizer.sentence_embedding import SentenceEmbedding
from src import helpers


class FeatureExtractor:
    def __init__(self, max_features=None, mode="word-embeddings"):
        self.nlp = helpers.get_nlp()
        if mode == "tfidf":
            self.tfidf_words = TfidfVectorizer(sublinear_tf=True, min_df=1, smooth_idf=True, norm="l2", encoding="utf-8", analyzer="word", ngram_range=(1, 2))
            self.tfidf_chars = TfidfVectorizer(sublinear_tf=True, min_df=1, smooth_idf=True, norm="l2", encoding="utf-8", analyzer="char", ngram_range=(3, 3))
            self.featurizer = FeatureUnion([("words", self.tfidf_words), ("chars", self.tfidf_chars)])
        elif mode == "word-embeddings":
            self.featurizer = SentenceEmbedding()
        else:
            self.featurizer = None
        self.features = None
        self.logger = logging.getLogger(__name__)

    def featurize_train(self, tokenized_utt):
        """
        Returns anf fits features of sentences
        :param tokenized_utt: List of strings of the tokenized utterance
        :return: list of lists
        """
        self.features = self.featurizer.fit_transform(tokenized_utt)
        return self.features

    def featurize_test(self, tokenized_test_list):
        """
        Returns the features of a test set (does not fit the featurizer)
        """

        return self.featurizer.transform(tokenized_test_list)
