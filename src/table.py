"""
Represent the input tables as a list of table objects
"""

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


class Table:

    def __init__(self, path, name, delimiter=",", metadata=None):
        """

        :param path (str): the path to read the table from
        :param name (str): name of the table
        :param delimiter (str): delimiter to read pandas DataFrame
        :param metadata (dict): extra info about the table
        """
        self.path = path
        self.name = name
        self.metadata = metadata
        self.df = pd.read_csv(self.path, delimiter=delimiter)
        self.df.rename(columns={"Unnamed: 0": "row_name"}, inplace=True)
        self.df = self.df.astype(float, errors="ignore")
        self.df.name = self.name
        self.numeric_columns, self.string_columns = self.get_numeric_string_columns()

    def get_numeric_string_columns(self):
        numeric_columns = []
        string_columns = []
        for column in self.df.columns:
            if is_string_dtype(self.df[column]):
                string_columns.append(column)
            if is_numeric_dtype(self.df[column]):
                numeric_columns.append(column)

        return numeric_columns, string_columns

