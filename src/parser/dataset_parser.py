"""
Given the path of the dataset, return
(1) dataframe for row_index prediction
(2) dataframe for formula prediction
"""

import pandas as pd
import re
import numpy as np
import scipy
import json

from src.regex.regex import Regex
from src.templates.template_transformer import TemplateTransformer
from src import helpers


class DatasetParser:
    def __init__(self, data_path):
        self.main_df = self.create_main_df(data_path)
        self.row_df = None
        self.formula_df = None
        self.regex_obj = Regex()

    def create_main_df(self, data_path):
        csv_files = helpers.get_files_from_dir(data_path)
        df = pd.DataFrame()
        for file in csv_files:
            chapter_df = pd.read_csv(file)
            chapter_df["file"] = file
            df = pd.concat([df, chapter_df])
        # rename the dataframe
        df = df.rename(columns={"Text": "sent", "Claim": "claim", "Calculation Equation": "formula",
                                "LOOKUP and FORMULA Dictionaries": "dicts", "Annotation Tab": "tab", 
                                "Calculation Value": "calculation_value"})
        return df

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

    def expanded_reference_is_valid(self, ref):
        if "!" in ref:
            return False
        return True
    
    def extend_formula(self, row):
        """
        Go through the cell references in the formula and for each cell, check whether this cell can be replaced by its references
        Example: formula = G11 + G12 and G11 = G124+G332/2. Then extended_formula = G124+G332/2 + G12 
        """
        dicts = json.loads(row["dicts"])
        formula_dict = dicts["formula_dict"]
        formula = row["formula"]
        cell_references = re.findall(self.regex_obj.formula_regex, formula)
        for ref in cell_references:
            # the cells that the cells in the fomrula reference
            expanded_reference_formula_dict = formula_dict.get(ref, dict())
            expanded_reference = expanded_reference_formula_dict.get("formula", ref)
            if self.expanded_reference_is_valid(expanded_reference):
                formula = formula.replace(ref, expanded_reference)
        return formula

    def get_formula_df(self, create_templates=True):
        """
        Returns a dataframe, which only consists of rows that have valid formulas.
        If create_templates is true, then the dataframe will also contain a column of the template of each
        formula
        """
        # remove unwanted rows
        self.formula_df = self.main_df
        self.formula_df["keep"] = self.formula_df.apply(self._cleanup_formula_df, axis=1)
        self.formula_df = self.formula_df[self.formula_df.keep == True]
        self.formula_df = self.formula_df.drop(columns="keep")
        self.formula_df["extended_formula"] = self.formula_df.apply(self.extend_formula, axis=1)
        if create_templates:
            template_transformer = TemplateTransformer(self.formula_df)
            self.formula_df = template_transformer.transform_formula_df()
        return self.formula_df

    def decouple(self, row, item):
        """
        For the specified item (i.e row_index, year, tab, ...), return a set of all the mentioned items.
        Example return set("row_idx1", "row_idx2", "row_idx3") if the lookup_dict contains those 3 row_indexes
        """
        ret_set = set()
        row_dicts = json.loads(row["dicts"])
        lookup_dict = row_dicts.get("lookup_dict", dict())
        for _, cell_dict in lookup_dict.items():
            item_value = cell_dict.get(item)
            if isinstance(item_value, list):
                for w in item_value:
                    ret_set.add(w)
            elif item_value:
                ret_set.add(item_value)
        return ret_set if len(ret_set) > 0 else None

    def add_item_column_to_df(self, df, item, column):
        """
        df: dataframe which will be added a new column
        column: name of the new column to be added
        item: name of the item in the "dicts" column of the df
        """
        df[column] = df.apply(self.decouple, item=item, axis=1)
        df = df[df[column].notnull()]
        return df

    def get_lookup_df(self):
        self.row_df = self.main_df
        self.row_df = self.add_item_column_to_df(self.row_df, "row index", "row_index")
        return self.row_df
    
    def get_column_df(self):
        column_df = self.main_df
        column_df = self.add_item_column_to_df(column_df, "year", "column")
        return column_df
    
    def get_region_df(self):
        region_df = self.main_df
        region_df = self.add_item_column_to_df(region_df, "region", "region")
        return region_df

    def get_file_df(self):
        file_df = self.main_df
        file_df = self.add_item_column_to_df(file_df, "path", "file")
        return file_df

    def get_complete_df(self):
        # start with the formula_df and keep adding new items (row_idx, columns, ...)
        ret_df = self.get_formula_df()
        for item, col in [("row index", "row_index"), ("year", "column"), ("region", "region"), ("path", "file")]:
            ret_df = self.add_item_column_to_df(ret_df, item, col)
        return ret_df

    def create_cv_dataset(self, df, label_column, min_samples):
        """
        Only keep labels which appear more than cv times in the df
        """
        return df.groupby(label_column).filter(lambda x: len(x) > min_samples)

    def get_test_train_cv_split(self, df, label_column, min_samples=3, frac=0.8):
        """
        Retruns a train and test df, where the in the train_df all labels appear more than
        `min_samples` times and in the test_df, there doesn't exist a label not present in the test_df
        """
        train_df = df.sample(frac=frac)
        test_df = df.drop(train_df.index)
        train_df = self.create_cv_dataset(train_df, label_column, min_samples)
        train_labels = list(train_df[label_column])
        # only keep data points, which have a label seen in the training data
        test_df = test_df[test_df[label_column].isin(train_labels)]

        return train_df, test_df

    @staticmethod
    def get_features_union(features_s, features_c):
        if isinstance(features_c, scipy.sparse.csr.csr_matrix):
            features_c = features_c.toarray()
        if isinstance(features_s, scipy.sparse.csr.csr_matrix):
            features_s = features_s.toarray()
        return np.concatenate((features_s, features_c), axis=1)

    def get_feature_union_train(self, df, tokenizer, featurizer_emb, featurizer_tf):
        sents = list(df["sent"])
        claims = list(df["claim"])
        tokenized_sents = tokenizer.tokenize_data(sents)
        tokenized_claims = tokenizer.tokenize_data(claims)
        features_sents = featurizer_emb.featurize_train(tokenized_sents)
        features_claims = featurizer_tf.featurize_train(tokenized_claims)
        features_union = self.get_features_union(features_sents, features_claims)
        print("training features extracted")
        print("Sentence features shape: {}".format(features_sents.shape))
        print("Claims features shape: {}".format(features_claims.shape))
        print("Union features shape: {}".format(features_union.shape))
        return features_union

    def get_feature_union_test(self, df, tokenizer, featurizer_emb, featurizer_tf):
        sents = list(df["sent"])
        claims = list(df["claim"])
        tokenized_sents = tokenizer.tokenize_data(sents)
        tokenized_claims = tokenizer.tokenize_data(claims)
        features_sents = featurizer_emb.featurize_test(tokenized_sents)
        features_claims = featurizer_tf.featurize_test(tokenized_claims)
        features_union = self.get_features_union(features_sents, features_claims)
        print("test features extracted")
        print("Sentence features shape: {}".format(features_sents.shape))
        print("Claims features shape: {}".format(features_claims.shape))
        print("Union features shape: {}".format(features_union.shape))
        return features_union