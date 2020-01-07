import logging
from src import helpers


class TokenizerDriver:
    def __init__(self):
        self.tokenized_claims = []
        self.nlp = helpers.get_nlp()
        self.logger = logging.getLogger(__name__)

    def tokenize_data(self, claim_spans):
        """
        Given a list of sentences, return a list of tokenized sentences
        :param claim_spans: list of str sentences
        :return: list of lists
        """
        tokenized_claims = []
        for sent in claim_spans:
            tokenized_claims.append(" ".join([x.lower_ for x in self.nlp(str(sent)) if self._pass_filter(x)]))
        return tokenized_claims

    @staticmethod
    def _pass_filter(tok):
        if tok.is_punct or tok.is_stop or tok.text == "\n":
            return False
        return True
