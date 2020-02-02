"""
Receives a df containing formulas from the annotated claims, and transforms them to template formulas
by replacing cell references and more.
Note: the input df has to be cleaned up, such that it only contains rows, which contain a formula
"""
import re
from src.regex.regex import Regex


class TemplateTransformer:
    def __init__(self, df):
        self.df = df
        self.regex_obj = Regex()
        variables = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.variables_list = [v for v in variables]

    @staticmethod
    def get_variables_for_formula(cell_references, variables):
        """
        :param cell_references: list which contains the variables of the formulas
                                (example: ["G11", "G14", "G22", "G11"])
        :param variables: list containing all the letters
        returns a dict that maps the cell_references to the variables that should be replaced in the formula
        for the exmple above, we return {"G11":"a", "G14":"b", "G22": "c"]
        """
        ret_dict = dict()
        var_idx = 0
        for ref in cell_references:
            if ref not in ret_dict:
                ret_dict[ref] = variables[var_idx]
                var_idx += 1
                if var_idx > 50:
                    break
        return ret_dict

    @staticmethod
    def replace_variables_in_formula(formula, ref_var_dict):
        """
        claim: str of the formula (example: G11-G21/3)
        ref_var_dict: dict with keys the cell references (exist in claim) and values the variables that
                      should replace them
        """
        ret_formula = formula
        for ref, var in ref_var_dict.items():
            ret_formula = ret_formula.replace(ref, var)
        return ret_formula

    @staticmethod
    def replace_str_in_formula(formula, str_list):
        """
        replace all the constant strings in the formula with STR
        """
        const_str = "STR"
        ret_formula = formula
        for s in str_list:
            ret_formula = ret_formula.replace(s, const_str)
        return ret_formula

    @staticmethod
    def replace_if_formula(formula, if_reference):
        """
        replaces the if statements in the formula. E.g the formula "IF(a<b, "OK", "FALSE") will become a<b
        """
        # if we get one element in if_reference, this means that the extraction is correct. So we return the `body` of
        # the if statement. Otherwise, the extraction was not correct, and return the original formula
        if len(if_reference) == 1:
            return if_reference[0]
        else:
            return formula

    @staticmethod
    def remove_white_space(s):
        return s.replace(" ", "")

    def create_template_formulas(self, row):
        """
        applied to each row, returning the template for each formula, by substituting vars for cell references and
        strings with a string constant
        """
        formula = row.extended_formula
        # if there was a parsing error
        if not formula:
            return None
        # G12, G1, ... etc.
        cell_references = re.findall(self.regex_obj.formula_regex, formula)
        string_references = re.findall(self.regex_obj.str_const_regex, formula)
        if_references = re.findall(self.regex_obj.if_regex, formula)
        ref_vars_dict = self.get_variables_for_formula(cell_references, self.variables_list)
        template_formula = self.replace_if_formula(formula, if_references)
        template_formula = self.replace_variables_in_formula(template_formula, ref_vars_dict)
        template_formula = self.replace_str_in_formula(template_formula, string_references)
        template_formula = self.remove_white_space(template_formula)
        return template_formula

    def transform_formula_df(self):
        """
        Transforms self.df by creating templates for the formulas in an extra column called `template_formula`
        returns a Dataframe with this extra column
        """
        self.df["template_formula"] = self.df.apply(self.create_template_formulas, axis=1)
        return self.df

