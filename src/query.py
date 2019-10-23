"""
Represent a query
"""
from src.value import Value

import pandasql as ps
import logging


class Query:

    def __init__(self, query, type=None, table=None):
        """

        :param query (str): the query text
        :param type (str): the type of the query (e.g existential or percentage_increase)
        """
        self.logger = logging.getLogger(__name__)
        self.query = query
        self.type = type
        self.result_df = None
        self.table = table

    def execute(self):
        """
        executes the query and produce a dataframe of the result
        :return: DataFrame or None
        """
        if self.result_df:
            return self.result_df
        try:
            # create a variable with the same name as the table name, which points to the table dataframe
            # e.g create variable if self.table.name = 'table1', then we create a variable table1 = self.table.df
            # Necessary as sqldf looks for the variable name which holds in the dataframe in the namespace
            exec(self.table.name + "= self.table.df")
            self.result_df = ps.sqldf(self.query)
            return self.result_df
        except:
            print("could not execute")
            return

    def get_single_query_result(self):
        """
        if self.result has only one element, return this element
        :return: the single element in the self.result DataFrame
        """
        if self.result_df is None:
            return
        if self.result_df.shape != (1, 1):
            self.logger.warning("Getting single result from a query with more than one result. Query: {}".format(self.query))
            return

        return Value(self.result_df.values.tolist()[0][0])
