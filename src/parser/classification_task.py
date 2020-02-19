"""
This class holds information about different classification tasks like 
row_index, tab, file prediction.
"""
import json
import os
from src import helpers


class ClassificationTask:
    def __init__(self, init_name, name, hash_name, label_task, priority, ver_cost, der_cost, val_acc=0.0):
        """
        Arguments:
            init_name {str} -- [name that appears in the original csv files]
            name {str} -- [new name that is used after parsing the original csv files]
            hash_name {str} -- [name of the column that contains the hash value of the 
                               classification task]
            label_task {str} -- [Classification type. Either "single-label" or "multi-label"]
            priority {int} -- {a lower number is a more important task}
            ver_cost {int} -- {verification cost}
            der_cost {int} -- {derivation cost}
            val_acc {float} -- [accuracy in the validation set]
        """
        self.init_name = init_name
        self.name = name
        self.hash_name = hash_name
        self.label_task = label_task
        self.priority = priority
        self.ver_cost = ver_cost
        self.der_cost = der_cost
        self.topn = None
        self.val_acc = val_acc
        # True if we have hashed the values of the Classification Task
        self.has_hash = True
        if name == hash_name:
            self.has_hash = False
        
        self.config = helpers.load_yaml("src/config.yml")

        self.all_values = set()

        self.classifier_name = self.name + "_classifier"
        self.featurizer_tf_name = self.name + "_featurizer_tf"
        self.featurizer_emb_name = self.name + "_featurizer_emb" 
        # store components needed for classification
        self.is_trained = False
        self.featurizer_tf = None
        self.featurizer_emb = None
        self.classifier = None

        # dict to translate a label to a unique value
        self.label_to_hash_dict = dict()
        self.hash_to_label_dict = dict()
        self.hash_to_label_dict_name = self.name + "_hash_to_label_dict.json"
        self.label_to_hash_dict_name = self.name + "_label_to_hash_dict.json"
        # hash_counter will always be incremented when we add a new label to a task's hash_dict
        self.hash_counter = 0

    def populate_all_values(self, df_column_list):
        """
        Go through the values of the list of values and add them to the set of all possible
        values. Note, because some tasks have values joined by "-" ("2013-2017-2011")
        we split by "-" and add to the set
        Arguments:
            df_column_list {list} -- [list of strings]
        """
        for val in df_column_list:
            self.all_values.update(val.split("-"))

    
    def hash_(self, label):
        """
        For a given label for this task create 2 maps
        1. get a unique integer from a label
        2. get a label from a unique integer
        return the unique value for this label
        """
        if label in self.label_to_hash_dict:
            return self.label_to_hash_dict[label]
        else:
            self.hash_counter += 1
            self.label_to_hash_dict[label] = self.hash_counter
            self.hash_to_label_dict[str(self.hash_counter)] = label

        return self.label_to_hash_dict[label]

    def export_hash_dicts(self):
        fname1 = os.path.join(self.config["models_dir"], self.hash_to_label_dict_name)
        with open(fname1, "w") as fp:
            json.dump(self.hash_to_label_dict, fp)

        fname2 = os.path.join(self.config["models_dir"], self.label_to_hash_dict_name)
        with open(fname2, "w") as fp:
            json.dump(self.label_to_hash_dict, fp)

    def load_hash_dicts(self):
        fname1 = os.path.join(self.config["models_dir"], self.hash_to_label_dict_name)
        with open(fname1, "r") as fp:
            self.hash_to_label_dict = json.load(fp)

        fname2 = os.path.join(self.config["models_dir"], self.label_to_hash_dict_name)
        with open(fname2, "r") as fp:
            self.label_to_hash_dict = json.load(fp)
        # restore the counter after loading
        if self.hash_to_label_dict:
            self.hash_counter = int(max(list(self.hash_to_label_dict.keys())))