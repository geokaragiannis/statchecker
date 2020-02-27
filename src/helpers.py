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
# doct which contains the constraint dicts
constraint_dicts_dict = dict()

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
    if df is None:
        return
    safe_mkdir(dir)
    fpath = os.path.join(dir, fname)
    df.to_csv(fpath, index=False)

def load_df_from_dir(dir, fname):
    fpath = os.path.join(dir, fname)
    try:
        return pd.read_csv(fpath)
    except FileNotFoundError:
        return None

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

def load_contraint_file(fname):
    """
    returns a dict from the json of the contraint file
    
    Arguments:
        fname {str} -- [file name]
    """
    constraint_file = os.path.join(CONTRAINT_FILE, fname)
    try:
        f = open(constraint_file)
    except FileNotFoundError:
        print("constraint file {} not found".format(constraint_file))
        return dict()
    
    return json.load(f)
    

def get_constraint_dict(source_prop, target_prop):
    """
    Returns the constraint file for the given properties
    or None if not found
    """
    constraint_file = None
    source_property_name = source_prop.property_name
    target_property_name = target_prop.property_name
    # hacky way to get the filename
    file_name = "2".join([source_property_name, target_property_name]) + ".json"
    constraint_file = os.path.join(CONTRAINT_FILE, file_name)
    if file_name in constraint_dicts_dict:
        return constraint_dicts_dict[file_name]
    else:
        try:
            f = open(constraint_file)
        except FileNotFoundError:
            # print("constraint file {} not found".format(constraint_file))
            return None
    
    constraint_dict = json.load(f)
    constraint_dicts_dict[file_name] = constraint_dict
    # return the list of possible value strings or the empty list, if not found
    return constraint_dict

def value_meets_contraints(source_value, target_value):
    """
    Return true if the target value exists in the constraint files
    """
    
    
