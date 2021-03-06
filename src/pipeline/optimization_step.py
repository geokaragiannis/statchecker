"""
This class receives a claim object without predictions and a classification_step object
which contains the necessary featurizers and classifiers for prediction. Here we:
1. Compute the number of values to display per property
2. Compute the number of properties to ask about
3. Get predictions for the claim ???
4. Return the subset of properties we ask about along with their proposed values

!!!! Note that we expect that we have trained our model !!!!
"""
from src import helpers
from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value

import math
from collections import Counter
import numpy as np


class OptimizationStep:
    def __init__(self, classification_step):
        self.classification_step = classification_step
        self.max_num_prop = None
        self.init_number_formulas = self.get_init_number_formulas()

    def optimize_claim(self, claim, features):

        self.set_topn(claim)
        max_properties_questions = self.get_max_num_properties_to_ask(lamda=4)
        self.get_preds_from_claim(claim, features)
        # use the preds and re-calculate topn and clip (if necessary) the extra preds
        # self.set_topn_entropy_and_clip_preds(claim)
        if max_properties_questions < len(claim.available_properties):
            num_properties_questions = max_properties_questions 
        else:
            num_properties_questions = len(claim.available_properties)
        # self.get_subset_props_to_ask(claim, num_properties_questions)

    
    def optimize_claim_only_verification(self, claim):
        """
        This method will only be used in simulations. Here we set the top_n predictions 
        for each property to be all the labels we have in the training data. This way, 
        the outputs of the classifier, will be the maximum possible, which will result in 
        only having verifications and not a single derivation.
        
        Arguments:
            claim {Claim obj} -- [description]
        """

        for task in self.classification_step.classification_tasks_dict.values():
            task.topn = len(task.all_values)
        # ask about all claims
        num_properties_questions = len(claim.available_properties)
        self.get_preds_from_claim(claim)
        self.get_subset_props_to_ask(claim, num_properties_questions)
        

    def get_preds_from_claim(self, claim, features):

        for prop in claim.available_properties:
            classifier = prop.task.classifier
            pred_labels, pred_probs = classifier.predict_utt_top_n(features, n=prop.task.topn)
            if prop.property_name == "template_formula":
                values_list = [Value(label, prob, prop) for label, prob in zip(pred_labels, pred_probs)]
            else:
                hash_to_label_dict = prop.task.hash_to_label_dict
                # try:
                values_list = [Value(hash_to_label_dict[str(int(label))], prob, prop) for label, prob in zip(pred_labels, pred_probs)]
                # except: 
                #     values_list = [Value(label, prob, prop) for label, prob in zip(pred_labels, pred_probs)]
            prop.candidate_values = values_list
            prop.set_entropy()
        claim.set_uncertainty()
        claim.set_expected_cost()

    def shrink_candidate_values_with_constraints(self, claim, target_prop):
        """
        For the given target property and possible values, exclude those values that do not meet the constraints 
        from previous property preds. E.g If prop = "row_index" then each value should exist in the 
        constraint files for file2row_index and tab2row_index. Note that the previous predicted properties 
        should have a ground truth.
        Arguments:
            claim {Claim obj} -- [description]
            target_prop {Property obj} -- [the property we want to exclude values from]
            values)list {list of Values} -- [the predicted values for the target_property]
        """
        # probability mass excluded
        extra_prob_mass = 0.0
        for source_prop in claim.available_properties:
            if source_prop.property_name == target_prop.property_name:
                break
            constraint_dict = helpers.get_constraint_dict(source_prop, target_prop)
            if constraint_dict is None:
                continue
            # if target_prop.property_name == "row_index":
            #     print("source: ", source_prop.property_name)
            for value in target_prop.candidate_values:
                # we only need one value to be (or not) in the constraints in order to include it (or not)
                str_value = value.value.split("-")[0]
                # find all the possible values from the ground truth. If we gen an empty list, we do not exclude the value
                possible_values = self._get_possible_values(constraint_dict, source_prop.ground_truth)
                if len(possible_values) > 0 and str_value not in possible_values:
                    extra_prob_mass += value.prob
                    value.exclude = True

        # if target_prop.property_name == "row_index":
        #     for val in target_prop.candidate_values:
        #         if val.exclude:
        #             if val.value == target_prop.ground_truth:
        #                 print("\n\nooooooooooooopppppps\n\n")

        #TODO: transfer prob mass from the excluded values to the remaining ones
        target_prop.candidate_values = [value for value in target_prop.candidate_values if not value.exclude]

    def _get_possible_values(self, constraint_dict, ground_truth):
        """
        get the possible values that the ground truth gives from the constraint_dict
        """
        ret_list = []
        for t in ground_truth.split("-"):
            ret_list.extend(constraint_dict.get(ground_truth, []))
        return ret_list

    def set_topn(self, claim):
        """
        For each property, compute the number of options to display
        """
        for task in self.classification_step.classification_tasks_dict.values():
            task.topn = math.floor(task.der_cost/task.ver_cost)
    
    def set_topn_entropy_and_clip_preds(self, claim):
        """
        For each property modify the previosuly calculated topn for the task by incorporating the 
        entropy of the predictions. The higher the entropy the more uncertain we are.
        After computing the new topn using the entropy of preds, we potentially clip the number of predictions
        per property. Example, if the number of predictions was initially 5 and after the predictions topn=2, 
        we need to only keep the first two predictions
        """
        for prop in claim.available_properties:
            # prop.topn = math.ceil(prop.task.topn * (1 - prop.entropy))
            prop.topn = 10
            # only keep the new topn number of preds
            prop.candidate_values = prop.candidate_values[:prop.topn]
    
    def get_max_num_properties_to_ask(self, lamda=4):
        """
        Get the maximum number of properties we can ask about. I.e if the max
        number of properties is 2, we can ask about e.g file, row only.
        lamda is a constant. The higher it is, the more questions we can ask
        """
        # assuming derivation cost of all properties except template_formula is the same
        der_cost_prop = self.classification_step.classification_tasks_dict["row_index"].der_cost
        der_cost_formula = self.classification_step.classification_tasks_dict["template_formula"].der_cost
            
        return math.floor(der_cost_formula*(lamda-2)/(2*der_cost_prop))

    
    def get_subset_props_to_ask(self, claim, num_properties_questions):
        """
        Populate the "ask" field of the claim's properties an ordered list of properties to ask.
        Here we assume that we have the classifiers predictions on the claims. 
        Arguments:
            claim {Claim obj} -- [The claim for which we wish to know the prperties to ask]
        """
        number_remaining_formulas = self.init_number_formulas
        # print("init number of formulas: ", number_remaining_formulas)
        idx = 0
        # print("statring props to ask")
        # print("num properties questions: ", num_properties_questions)
        if num_properties_questions > len(claim.available_properties) - 1:
            # ask about all claims (minus the template formula)
            num_to_ask = len(claim.available_properties) - 1
        else:
            num_to_ask = num_properties_questions

        # print("num to ask: ", num_to_ask)
        for i in range(num_to_ask):
            max_factor_property = None
            max_prunning_factor = 0
            for prop in claim.available_properties:
                if prop.property_name == "template_formula":
                    prop.ask = True
                    continue
                idx += 1
                if not prop.ask:
                    prop.ask = True
                    prunning_factor = self.get_number_formulas_excluded_per_var(number_remaining_formulas, claim)
                    # print("property: {}, prunning factor: {}".format(prop.property_name, prunning_factor))
                    prop.ask = False
                    if prunning_factor > max_prunning_factor:
                        max_prunning_factor = prunning_factor
                        max_factor_property = prop

            max_factor_property.ask = True
            number_remaining_formulas = number_remaining_formulas - max_prunning_factor
            # print("step: {}, max_factor_prop: {}, remaining formulas: {}".format(i, max_factor_property.property_name, number_remaining_formulas))



    def get_number_formulas_excluded_per_var(self, number_remaining_formulas, claim):
        """
        Return the number of formulas a specific value for a property (i.e TPEDTOTAL)
        excludes per variable. I.e the number of formulas that does not have as row_index TPEDTOTAL
        Arguments:
            number_remaining_formulas {int} -- [contains the number of times each property value occurs]
            property_value {str} -- [TPEDTOTAL or file1 or ...]
            property_name {str} -- [row_index or file or column]
        """

        return number_remaining_formulas - self.get_remaining_number_formulas(claim)



    def get_remaining_number_formulas(self, claim):
        """
        Get the remaining number of total formulas per variable after picking
        some properties already
        
        Arguments:
            properties_picked {list of str} -- [description]
        """

        remaining_properties = []
        file_asked = False
        row_not_asked = False
        file_top_pred = None
        for prop in claim.available_properties:
            if prop.property_name == "template_formula":
                continue
            if prop.property_name == "file" and len(prop.candidate_values) > 0:
                file_top_pred = prop.candidate_values[0].value
                file_asked = True
            if not prop.ask:
                remaining_properties.append(prop)
                if prop.property_name == "row_index":
                    row_not_asked = True
                if prop.property_name == "file":
                    file_asked = False

        # if we have selected a file and not a row_idx use the contraints
        use_constraints = file_asked and row_not_asked

        # print("remaining properties: ", [prop.property_name for prop in remaining_properties])


        remaining_num = 1
        for rem_prop in remaining_properties:
            if use_constraints and rem_prop.property_name == "row_index":
                # get number of possible rows from constraint dict. If it fails consider all the rows
                try:
                    file_to_row_constraint_dict = helpers.load_contraint_file("file2row_index.json")
                    num_all_values = file_to_row_constraint_dict.get[file_top_pred]
                except:
                    num_all_values = len(self.classification_step.classification_tasks_dict[rem_prop.property_name].all_values)
            else:
                num_all_values = len(self.classification_step.classification_tasks_dict[rem_prop.property_name].all_values)
            remaining_num *= num_all_values 

        return remaining_num


    def get_init_number_formulas(self):
        """
        Get the inital number of all template formulas for one variable
        """
        # no contraints
        all_files = self.classification_step.classification_tasks_dict["file"].all_values
        all_row_indexes = self.classification_step.classification_tasks_dict["row_index"].all_values
        all_columns = self.classification_step.classification_tasks_dict["column"].all_values
        file_to_row_constraint_dict = helpers.load_contraint_file("file2row_index.json")
        # print("sizes of files: {}, row: {}, col: {} ".format(len(all_files), len(all_row_indexes), len(all_columns)) )
        # get all rows available from the constraint
        all_files_rows = 0
        for file in all_files:
            num_rows_from_file = len(file_to_row_constraint_dict.get(file, all_row_indexes))
            all_files_rows += num_rows_from_file

        return all_files_rows * len(all_columns)

