import logging
from src import helpers


class TokenizerDriver:
    def __init__(self):
        self.tokenized_claims = []
        self.nlp = helpers.get_nlp()
        self.logger = logging.getLogger(__name__)

    def tokenize_claims(self, claim_spans):
        """
        Given a list of Spacy spans representing the sentences of claims, return a list of tokenized claims
        :param claim_spans: list of Spacy Spans
        :return: list of lists
        """
        for sent in claim_spans:
            self.tokenized_claims.append(" ".join([x.lower_ for x in sent if self._pass_filter(x)]))
        return self.tokenized_claims

    @staticmethod
    def _pass_filter(tok):
        if tok.is_punct or tok.is_stop or tok.text == "\n":
            return False
        return True
