"""
Represent a query
"""
from src.value import Value

import pandasql as ps
import logging


class Query:

    def __init__(self, query, expr=None, type=None, table=None):
        """
        :param query (str): the query text
        :param type (str): the type of the query (e.g existential or percentage_increase)
        """
        self.logger = logging.getLogger(__name__)
        self.query = query
        self.type = type
        self.expr = expr
        self.result_df = None
        self.table = table
        self.not_executable = False

    def execute(self):
        """
        executes the query and produce a dataframe of the result
        :return: DataFrame or None
        """
        if self.result_df is not None:
            return self.result_df
        try:
            # create a variable with the same name as the table name, which points to the table dataframe
            # e.g create variable if self.table.name = 'table1', then we create a variable table1 = self.table.df
            # Necessary as sqldf looks for the variable name which holds in the dataframe in the namespace
            exec(self.table.name + "= self.table.df")
            self.result_df = ps.sqldf(self.query)
            return self.result_df
        except:
            self.not_executable = True
            return

    def get_single_column_query_result(self):
        """
        if self.result has only one column, return a list of values from this column
        :return (list of Value Obj.): the single column in the self.result DataFrame as a list of Value objects
        """
        if self.result_df is None:
            return
        if self.result_df.shape[1] != 1:
            self.logger.warning("Getting result from a query with more than one column. Query: {}".format(self.query))
            return

        # self.result_df.values.tolist() is a list of 1-element lists, which contain the column elements
        return [Value(v[0]) for v in self.result_df.values.tolist()]

    def __str__(self):
        return "Query: {} \n type: {}".format(self.query, self.type)
