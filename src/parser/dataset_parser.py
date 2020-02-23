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
from src.parser.classification_task import ClassificationTask


class DatasetParser:
    def __init__(self, data_path):
        self.main_df = self.create_main_df(data_path)
        self.row_df = None
        self.formula_df = None
        self.regex_obj = Regex()
        self.config = helpers.load_yaml("src/config.yml")
        self.classification_tasks_dict = self.get_classification_tasks()

    def get_classification_tasks(self):
        """
        Return a list of ClassificationTask objects, which are all the possible
        classifications that we can parse/do.
        """
        task_dict = dict()
        for task_name, task_value in self.config["classification_tasks"].items():
            task_obj = ClassificationTask(task_value["init_name"], task_value["name"], 
                                          task_value["hash_name"], task_value["label_task"],
                                          task_value["priority"], task_value["verification_cost"],
                                          task_value["derivation_cost"])
            task_dict[task_name] = task_obj
        return task_dict

    def create_main_df(self, data_path):
        csv_files = helpers.get_files_from_dir(data_path)
        df = pd.DataFrame()
        for file in csv_files:
            chapter_df = pd.read_csv(file)
            chapter_df["chapter"] = file.split("/")[-1].split(".")[0]
            df = pd.concat([df, chapter_df], sort=False)
        # rename the dataframe
        df = df.rename(columns={"Text": "sent", "Claim": "claim", "Calculation Equation": "formula",
                                "LOOKUP and FORMULA Dictionaries": "dicts", "Annotation Tab": "section", 
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
        try:
            dicts = json.loads(row["dicts"])
        except json.decoder.JSONDecodeError:
            return None
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
        self.formula_df = self.formula_df.drop_duplicates(subset=["sent", "claim"]).reset_index(drop=True)
        return self.formula_df

    def decouple(self, row, item):
        """
        For the specified item (i.e row_index, year, tab, ...), return a set of all the mentioned items.
        Example return set("row_idx1", "row_idx2", "row_idx3") if the lookup_dict contains those 3 row_indexes
        """
        ret_set = set()
        try:
            row_dicts = json.loads(row["dicts"])
            lookup_dict = row_dicts.get("lookup_dict", dict())
        except json.decoder.JSONDecodeError:
            return None
        for _, cell_dict in lookup_dict.items():
            item_value = cell_dict.get(item)
            if isinstance(item_value, list):
                for w in item_value:
                    ret_set.add(str(w))
            elif item_value:
                ret_set.add(str(item_value))
        return "-".join(sorted(ret_set)) if len(ret_set) > 0 else None
    
    def apply_hash(self, row, task):
        """
        Apply a hash function to the item (row_index, column, tab,...) of the row and store
        the mapping in a dict
        Arguments:
            row {Pandas series}
            task {ClassificationTask object} -- the task to apply a hash
        """
        hash_value = task.hash_(row[task.name])
        return hash_value

    def add_item_column_to_df(self, df, task):
        """
        df: dataframe which will be added a new column
        task: ClassificationTask obj 
        """
        df[task.name] = df.apply(self.decouple, item=task.init_name, axis=1)
        # add a new column which has the hash value of df[column]
        # if column is "row_index", then create "row_index_hash" as the new hash column
        df[task.hash_name] = df.apply(self.apply_hash, task=task, axis=1)
        df = df[df[task.name].notnull()]
        df = df.drop_duplicates(subset=["sent", "claim"]).reset_index(drop=True)
        return df

    def get_lookup_df(self):
        self.row_df = self.main_df
        task = self.classification_tasks_dict["row_index"]
        self.row_df = self.add_item_column_to_df(self.row_df, task)
        return self.row_df
    
    def get_column_df(self):
        column_df = self.main_df
        task = self.classification_tasks_dict["column"]
        column_df = self.add_item_column_to_df(column_df, task)
        return column_df
    
    def get_region_df(self):
        region_df = self.main_df
        task = self.classification_tasks_dict["region"]
        region_df = self.add_item_column_to_df(region_df, task)
        return region_df

    def get_file_df(self):
        file_df = self.main_df
        task = self.classification_tasks_dict["file"]
        file_df = self.add_item_column_to_df(file_df, task)
        return file_df

    def get_tab_df(self):
        tab_df = self.main_df
        task = self.classification_tasks_dict["tab"]
        tab_df = self.add_item_column_to_df(tab_df, task)
        return tab_df

    def apply_subsection(self, row):
        return "-".join([row["chapter"], str(row["section"])])

    def get_complete_df(self):
        # start with the formula_df and keep adding new items (row_idx, columns, ...)
        ret_df = self.get_formula_df()
        for task, task_obj in self.classification_tasks_dict.items():
            if task == "template_formula":
                continue
            ret_df = self.add_item_column_to_df(ret_df, task_obj)
        # add the subsection (i.e Ch2_intro)
        ret_df["subsection"] = ret_df.apply(self.apply_subsection, axis=1)
        return ret_df 

    def set_task_values(self, df):
        """
        Sets all the possible values a task can take
        Arguments:
            df {[type]} -- [description]
        """
        for task_name, task in self.classification_tasks_dict.items():
            task.populate_all_values(list(df[task_name])) 

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
