from src.pipeline.classification_step import ClassificationStep
from src.pipeline.optimization_step import OptimizationStep
from src.parser.classification_task import ClassificationTask
from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value
from colorama import Fore, Style
import math


DATA_PATH = "data/claims_01-23-2020/"
class Simulation:
    def __init__(self, class_pipeline, opt_pipeline):
        self.classification_pipeline = class_pipeline
        self.optimization_pipeline = opt_pipeline
        self.z = 0

    def create_claims_from_df(self, test_df, get_preds=False, get_ground_truth=False):
        """
        return a list of Claim objects by populating them with the sentence and claim strings
        from the test_df. 
        """
        test_claims = []
        for idx, test_row in test_df.iterrows():
            available_properties = [Property(t.name, task=t) for t in 
                                    list(self.classification_pipeline.classification_tasks_dict.values())]
            available_properties = sorted(available_properties, key=lambda x: x.task.priority)
            if get_ground_truth:
                for prop in available_properties: 
                    prop.ground_truth = test_row[prop.property_name]
            test_claim = Claim(test_row["sent"], test_row["claim"], test_row["subsection"], available_properties)
            if get_preds:
                self.get_preds_from_claim(test_claim)
            test_claims.append(test_claim)
        return test_claims

    def get_preds_from_claim(self, claim):
        """
        Get predictions for each property of the claim
        """
        not_found_hash = 0
        for prop in claim.available_properties:
            featurizer_tf = prop.task.featurizer_tf
            featurizer_emb = prop.task.featurizer_emb
            classifier = prop.task.classifier
            features = self.classification_pipeline.get_feature_union([claim.sent], [claim.claim], self.classification_pipeline.tok_driver, 
                                                           featurizer_emb, featurizer_tf, mode="test")
            pred_labels, pred_probs = classifier.predict_utt_top_n(features, n=5)
            if prop.property_name == "template_formula":
                values_list = [Value(label, prob, prop) for label, prob in zip(pred_labels, pred_probs)]
            else:
                hash_to_label_dict = prop.task.hash_to_label_dict
                # print(len(hash_to_label_dict))
                try:
                    values_list = [Value(hash_to_label_dict[str(label)], prob, prop) for label, prob in zip(pred_labels, pred_probs)]
                except: 
                    not_found_hash += 1
                    values_list = [Value(label, prob, prop) for label, prob in zip(pred_labels, pred_probs)]
            prop.candidate_values = values_list
        
        self.z += not_found_hash

    def ask_questions_about_claim(self, claim, test_df_row):
        print(Fore.RED + ">> Sentence: {}".format(claim.sent))
        print(Fore.GREEN + ">> Claim: {}".format(claim.claim))
        print(Style.RESET_ALL)
        for prop in claim.available_properties:
            print(">> Are any of the below correct for the property: {}".format(prop.property_name))
            for idx, value in enumerate(prop.candidate_values):
                print(">> {}.  {} with probability {}".format(idx, value.value, value.prob))
            choice_idx = int(input())
            ground_truth = test_df_row[prop.property_name]
            prop.ground_truth = ground_truth
            print("ground truth: ", ground_truth)
            if str(prop.candidate_values[choice_idx].value) == str(ground_truth):
                print(">> Great! You got it correctly!")
            else:
                print(">> You made a mistake")
        print("\n\n")

    def print_preds(self, claim):
        for prop in claim.available_properties:
            if prop.verified_index == -1:               
                print("------Derived-----")
            else:
                print("-----Verified: {} -------".format(prop.verified_index))
            print("Ground Truth: ", prop.ground_truth)
            print("Prop Name: {}".format(prop.property_name))
            print("preds: ")
            for val in prop.candidate_values:
                print("\t Value: {}, Prob: {}".format(val.value, val.prob))
            print("Entropy: ", prop.entropy)

    def get_cost_of_claim_from_preds(self, claim):
        cost = 0
        for prop in claim.available_properties:
            # find index of ground_truth in the preds. If not found, count it as a derivation cost
            cand_values_str = [v.value for v in prop.candidate_values]
            
            try:
                idx = cand_values_str.index(prop.ground_truth)
                cost += (idx+1)*prop.task.ver_cost
                prop.verified_index = idx
            except ValueError:
                # derivation
                # print("prop: {}, preds: {}".format(prop.property_name, cand_values_str))
                # print("ground_truth: ", prop.ground_truth)
                cost += len(cand_values_str)*prop.task.ver_cost + prop.task.der_cost
                prop.verified_index = -1
        # self.print_preds(claim)
        return cost

    def get_cost_random_order_opt(self, test_df):
        """
        Assuming that we have a trained model, go through the test claims, get the predictions
        and calculate the actual cost. We use the optimization to get the number of predictions
        and the subset of properties we ask about.
        Arguments:
            test_df {DataFrame} -- [claims to be verified]
        """
        # claims without preds with the groundtruth
        claims = self.create_claims_from_df(test_df, get_ground_truth=True)
        total_cost = 0
        for claim in claims:
            self.optimization_pipeline.optimize_claim(claim)
            total_cost += self.get_cost_of_claim_from_preds(claim)
        return total_cost, claims

    def get_cost_random_order_only_verification(self, test_df):
        """
        Assuming that we have a trained model, go through the test claims, get the predictions
        and calculate the actual cost. We only ask for a verification. never a derivation
        Arguments:
            test_df {DataFrame} -- [claims to be verified]
        """
        # claims without preds with the groundtruth
        claims = self.create_claims_from_df(test_df, get_ground_truth=True)
        total_cost = 0
        for claim in claims:
            self.optimization_pipeline.optimize_claim_only_verification(claim)
            total_cost += self.get_cost_of_claim_from_preds(claim)
        return total_cost, claims


    def run(self):

        print(">> Do you want to train?")
        train = bool(int(input()))

        if train:
            complete_df = self.classification_pipeline.parser.get_complete_df()
            self.classification_pipeline.train(complete_df)
            train_df = self.classification_pipeline.train_df
            val_df = self.classification_pipeline.val_df
            test_df = self.classification_pipeline.test_df
        else:
            self.classification_pipeline.load_models()
            train_df, val_df, test_df = self.classification_pipeline.load_dfs()
        
        print("size of training, val and test: {}, {}, {}".format(len(train_df), len(val_df), len(test_df)))
        test_claims = self.create_claims_from_df(test_df)

        claims_to_retrain = []
        for idx, test_claim in enumerate(test_claims):
            test_df_row = test_df.iloc[idx]
            self.ask_questions_about_claim(test_claim, test_df_row)
            claims_to_retrain.append(test_claim)
            if (idx + 1) % 5 == 0:
                self.classification_pipeline.retrain(claims_to_retrain)
                claims_to_retrain = []

if __name__ == "__main__":
    class_pipeline = ClassificationStep(DATA_PATH, simulation=True, min_samples=1)
    opt_pipeline = OptimizationStep(class_pipeline)
    simulation = Simulation(class_pipeline, opt_pipeline)
    simulation.run()

