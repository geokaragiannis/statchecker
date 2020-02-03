import logging
import logging.config
import os
import yaml
import spacy
import json

from src.table import Table

logger = logging.getLogger(__name__)
NLP = None
CONTRAINT_FILE = "data/constraints_dicts"


def get_nlp():
    global NLP
    NLP = NLP if NLP else spacy.load("en_core_web_md")
    return NLP


def set_up_logging():
    default_config = "logging.yml"
    default_level = logging.INFO

    if os.path.exists(default_config):
        with open(default_config, 'rt') as f:
            config = yaml.safe_load(f.read())

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


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
