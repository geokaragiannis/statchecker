"""
Class that represents a Claim
"""
from src.value import Value
from src import helpers


class Claim:

    def __init__(self, claim_value, claim_text=None):
        self.claim_value = Value(claim_value)
        self.claim_text = claim_text
        nlp = helpers.get_nlp()
        self.doc = nlp(claim_text) if claim_text else None
        # each element contains a (query, similarity_to_claim) tuple
        self.queries_list = []

    def add_query_to_list(self, query):
        """
        Adds a query to self.queries_list by first computing how many columns from the query's expression occur in
        the claim's text
        :param query: Query object
        :return: None
        """
        self.queries_list.append((query, self._claim_expression_similarity(query.expr)))

    def _claim_expression_similarity(self, expression):
        """
        Returns the number of times a numeric column from the query's expression occurs in the claim text
        :param expression: Expression object
        :return: int
        """
        sim = 0
        if self.doc is None:
            return sim
        for tok in self.doc:
            if tok.text in expression.cols_dict.keys():
                sim += 1
            elif tok.lemma_ in expression.cols_dict.keys():
                sim += 1
        return sim

    def __str__(self):
        return "claim text {} \n claim_value: {}".format(self.claim_text, self.claim_value)