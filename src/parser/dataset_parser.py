"""
Given the path of the dataset, return
(1) dataframe for row_index prediction
(2) dataframe for formula prediction
"""

import pandas as pd
import re

from src.regex.regex import Regex
from src.templates.template_transformer import TemplateTransformer


class DatasetParser:
    def __init__(self, data_path):
        self.main_df = pd.read_csv(data_path)
        self.row_df = None
        self.formula_df = None
        self.regex_obj = Regex()

    def _cleanup_formula_df(self, row):
        if "LOOKUP" in str(row["claim"]) or "LOOKUP" in str(row["formula"]):
            return False
        if "Fig" in str(row["formula"]):
            return False
        elif len(re.findall(self.regex_obj.formula_regex, str(row["formula"]))) == 0:
            return False
        elif len(re.findall(self.regex_obj.formula_regex, str(row["claim"]))) > 0:
            return False
        elif len(re.findall(self.regex_obj.other_file_ref_regex, str(row["formula"]))) > 0:
            return False
        else:
            return True

    def get_formula_df(self, create_templates=True):
        """
        Returns a dataframe, which only consists of rows that have valid formulas.
        If create_templates is true, then the dataframe will also contain a column of the template of each
        formula
        """
        # remove unwanted rows
        self.main_df["keep"] = self.main_df.apply(self._cleanup_formula_df, axis=1)
        self.formula_df = self.main_df[self.main_df.keep == True]
        self.formula_df = self.formula_df.drop(columns="keep")
        if create_templates:
            template_transformer = TemplateTransformer(self.formula_df)
            self.formula_df = template_transformer.transform_formula_df()
        return self.formula_df

    def get_lookup_df(self):
        self.row_df = self.main_df[self.main_df["formula"].str.contains("LOOKUP")].reset_index(drop=True)
        return self.row_df
