import logging
import logging.config
import os
import yaml
import spacy
import json
import pickle
import pathlib
import pandas as pd

from src.table import Table

logger = logging.getLogger(__name__)
NLP = None
CONTRAINT_FILE = "data/constraints_dicts"


def get_nlp():
    global NLP
    NLP = NLP if NLP else spacy.load("en_core_web_md")
    return NLP



def safe_mkdir(path):
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

def save_model_to_dir(dir, fname, obj):
    safe_mkdir(dir)
    fpath = os.path.join(dir, fname)
    f = open(fpath, "wb")
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model_from_dir(dir, fname):
    fpath = os.path.join(dir, fname)
    f = open(fpath, "rb")
    return pickle.load(f)

def save_df_to_dir(dir, fname, df):
    if not df:
        return
    safe_mkdir(dir)
    fpath = os.path.join(dir, fname)
    df.to_csv(fpath, index=False)

def load_df_from_dir(dir, fname):
    fpath = os.path.join(dir, fname)
    return pd.read_csv(fpath)

def set_up_logging():
    default_config = "logging.yml"
    default_level = logging.INFO

    if os.path.exists(default_config):
        with open(default_config, 'rt') as f:
            config = yaml.safe_load(f.read())

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def load_yaml(filename):
    with open(filename, "r") as ymlfile:
        config_file = yaml.safe_load(ymlfile)
    return config_file


def get_tables_list(table_dir_path):
    tables_list = []
    table_num = 0
    for root, d_names, file_names in os.walk(table_dir_path):
        for file in file_names:
            file_prefix = file.split(".")[0]
            file_suffix = file.split(".")[1]
            if file_suffix == "csv":
                tables_list.append(Table(os.path.join(table_dir_path, file), file_prefix))
    return tables_list



def get_files_from_dir(basepath):
    """
    Returns a list of files inside the directory of the specified path
    """
    files = []
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            files.append(os.path.join(basepath, entry))
    return files


def get_possible_values(source_value, target_value):
    """
    Returns a list of string values (not value objects), which
    are possible to be obtained from the source_value.
    For example if the source_value is of type "tab" with value "tab1" 
    and the  target_value is of type "row_index" with value "row3", 
    then we return all the values from the "tab2rowIndex.json" file for 
    key="tab1". 
    Arguments:
        source_value {Value obj} -- [Value used as a key]
        target_value {Value obj} -- [Value to be obtained]
    Returns: list of strings or the empty list
    """
    constraint_file = None
    source_property_name = source_value.property.property_name
    target_property_name = target_value.property.property_name
    # hacky way to get the filename
    file_name = "2".join([source_property_name, target_property_name]) + ".json"
    constraint_file = os.path.join(CONTRAINT_FILE, file_name)
    try:
        f = open(constraint_file)
    except FileNotFoundError:
        # print("constraint file {} not found".format(constraint_file))
        return []
    
    constraint_dict = json.load(f)
    # return the list of possible value strings or the empty list, if not found
    return constraint_dict.get(source_value.value, [])
