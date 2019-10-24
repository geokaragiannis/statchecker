import logging
import logging.config
import os
import yaml

from src.table import Table


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
