"""
Generates a list of query objects based on the input tables
"""
from src.query import Query
import logging


class QueryGeneration:

    def __init__(self, tables):
        """

        :param tables (list of Table obj): tables for which to generate queries for
        """
        self.logger = logging.getLogger(__name__)
        self.tables = tables
        self.candidate_queries = []

    def generate_queries(self):
        """
        Generates the queries by calling functions that generate a specific query type
        :return: a list of Query objects
        """
        for table in self.tables:
            self._generate_existence_queries(table)
            self._generate_percentage_increase_queries(table)
            self._generate_percentage_decrease_queries(table)

    def _generate_existence_queries(self, table):
        """
        Generate existence queries for the input table. i.e for each table, generate queries that return each cell
        :param table (Table object): input table for which to generate the queries
        :return: None
        """
        existence_queries = []
        columns = list(table.df.columns)
        rows = list(table.df.row_name)
        for column in columns:
            for row in rows:
                query_text = "SELECT t.'{}' from {} as t WHERE t.'row_name' == '{}'".format(column, table.df.name, row)
                query = Query(query_text, type="existence", table=table)
                existence_queries.append(query)

        self.logger.info("Created {} existence queries for table {}".format(len(existence_queries), table.df.name))
        self.candidate_queries += existence_queries

    def _generate_percentage_increase_queries(self, table):
        """
        Generate percentage increase queries for each column combination. I.e for columns [col1, col2, col3] generate
        percentage of increase of [col1, col2], [col1, col3] and [col2, col3] for each of their rows.
        :param table (Table object): input table for which to generate the queries
        :return: None
        """
        percentage_increase_queries = []
        columns = list(table.df.columns)
        rows = list(table.df.row_name)
        column_tuples = self._get_column_tuples(columns)
        for column_tuple in column_tuples:
            for row in rows:
                query_text = "SELECT 100.0*(t.'{}' - t.'{}')/t.'{}' from {} as t WHERE t.'row_name' == '{}'".format(column_tuple[1], column_tuple[0], column_tuple[0], table.df.name, row)
                query = Query(query_text, type="percentage_increase", table=table)
                percentage_increase_queries.append(query)

        self.logger.info("Created {} percentage increase queries for table {}".format(len(percentage_increase_queries), table.df.name))
        self.candidate_queries += percentage_increase_queries

    def _generate_percentage_decrease_queries(self, table):
        """
        Generate percentage decrease queries for each column combination. I.e for columns [col1, col2, col3] generate
        percentage of decrease of [col1, col2], [col1, col3] and [col2, col3] for each of their rows.
        :param table (Table object): input table for which to generate the queries
        :return: None
        """
        percentage_decrease_queries = []
        columns = list(table.df.columns)
        rows = list(table.df.row_name)
        column_tuples = self._get_column_tuples(columns)
        for column_tuple in column_tuples:
            for row in rows:
                query_text = "SELECT 100.0*(t.'{}' - t.'{}')/t.'{}' from {} as t WHERE t.'row_name' == '{}'".format(column_tuple[0], column_tuple[1], column_tuple[1], table.df.name, row)
                query = Query(query_text, type="percentage_decrease", table=table)
                percentage_decrease_queries.append(query)

        self.logger.info("Created {} percentage decrease queries for table {}".format(len(percentage_decrease_queries), table.df.name))
        self.candidate_queries += percentage_decrease_queries

    @staticmethod
    def _get_column_tuples(columns):
        """
         Given a columns list ["a", "b", "c", "d"] return a list
         [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')]
        :param columns (list): the columns of the table
        :return: list of tuples
        """
        tuple_list = []
        for idx, col in enumerate(columns):
            t = tuple(columns[idx:])
            first_item = t[0]
            for a in t[1:]:
                tuple_list.append((first_item, a))
        return tuple_list


    def __str__(self):
        s = ""
        for query in self.candidate_queries:
            s += query.query + "\n"
        return s
