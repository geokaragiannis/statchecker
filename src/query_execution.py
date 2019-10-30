"""
Executes the generated queries and checks if a set of queries matches the input claims
"""
import logging
from collections import OrderedDict


class QueryExecution:

    def __init__(self, query_generation_obj):
        """
        :param query_generation_obj (QueryGeneration object): the query generation object
        """
        self.logger = logging.getLogger(__name__)
        self.query_generation_obj = query_generation_obj

    def get_queries_from_claim(self, claim):
        """
        Executes the generated queries and checks if the claim_value is close to the result of the query
        :param claim (Claim object): the input claim
        :return: None
        """
        for i, query in enumerate(self.query_generation_obj.candidate_queries):
            if i % 10000 == 0:
                self.logger.info("Executing query number {} out of {}".
                                 format(i, len(self.query_generation_obj.candidate_queries)))
            query_result_df = query.execute()
            if query_result_df is not None:
                if self._compare_claim_with_query(claim, query):
                    claim.add_query_to_list(query)

        self.logger.info("Resulting {} matched queries for claim {}".format(len(claim.queries_list), claim))

    def get_queries_from_claims(self, claims):
        """
        Executes the generated queries and checks for each claim if the claim_value is close to the result of the query
        :param claims (list of Claims) : the input claims
        :return None
        """
        for claim in claims:
            self.get_queries_from_claim(claim)

    def _compare_claim_with_query(self, claim, query):
        """
        Compares if the claim and the query result are close in value.
        !!Note: We assume that the query_result DataFrame has only one column
        :param claim (Claim Object):
        :param query (Query Object):
        :return: True/False
        """
        # Get the Value object for the result of the query
        query_value_list = query.get_single_column_query_result()
        if query_value_list is None:
            return False

        # if at least one element of the query_value_list is approximately the same as the claim_value, return True
        for query_value in query_value_list:
            if claim.claim_value.new_value in query_value.round():
                return True

        return False



