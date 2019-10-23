"""
Represent the input tables as a list of table objects
"""

import pandas as pd


class Table:

    def __init__(self, path, name, metadata=None):
        """

        :param path (str): the path to read the table from
        :param name (str): name of the table
        :param metadata (dict): extra info about the table
        """
        self.path = path
        self.name = name
        self.metadata = metadata
        self.df = pd.read_csv(self.path)
        self.df.rename(columns={"Unnamed: 0": "row_name"}, inplace=True)
        self.df.name = self.name

