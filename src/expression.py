"""
Defines an arithmetic expression which consists of columns and operators
"""
from collections import Counter


class Expression:
    def __init__(self):
        self.cols_dict = Counter()
        self.ops_dict = Counter()
        self.expr_text = None

    def update_cols(self, col):
        self.cols_dict.update([col])

    def update_ops(self, op):
        self.ops_dict.update([op])

    def __str__(self):
        return "cols_dict: {} \n ops_dict: {} \n expr_text = {}".format(self.cols_dict, self.ops_dict, self.expr_text)