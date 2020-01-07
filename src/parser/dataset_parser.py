"""
Given the path of the dataset, return
(1) dataframe for row_index prediction
(2) dataframe for formula prediction
"""

import pandas as pd
import re
import numpy as np
import scipy

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