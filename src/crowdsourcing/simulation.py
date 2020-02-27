from src.pipeline.classification_step import ClassificationStep
from src.pipeline.optimization_step import OptimizationStep
from src.parser.classification_task import ClassificationTask
from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value
from colorama import Fore, Style
import math
import pandas as pd


DATA_PATH = "data/claims_01-23-2020/"
class Simulation:
    def __init__(self, class_pipeline, opt_pipeline, active_pipeline):
        self.classification_pipeline = class_pipeline
        self.optimization_pipeline = opt_pipeline
        self.active_pipeline = active_pipeline
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
            features = self.classification_pipeline.get_feature_union([claim.sent], [claim.claim],
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

    def get_features_from_claim_list(self, claims):
        feat_tf = self.classification_pipeline.featurizer_tf
        feat_emb = self.classification_pipeline.featurizer_emb
        sents = [claim.sent for claim in claims]
        claims = [claim.claim + " " + claim.subsection for claim in claims]
        return self.classification_pipeline.get_feature_union(sents, claims, feat_emb, feat_tf, mode="test")

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

    def get_cost_of_claim_from_preds(self, claim, acc_dict):
        """
        For a given claim with preds, get the cost and accuracy across properties.
        NOTE: we shrink the preds of some properties depending on previous ground truth
        assignments. I.e remove impossible row_index predictions depending on file, tab and region
        preds. This exclusion of preds is done BEFORE we clip the predictions according to entropy 
        (or otherwise). So after that, we also run the last stage of the optimization pipeline, where
        we clip the number of preds we show to the fact checkers.
        """
        cost = 0
        for prop in claim.available_properties:
            # remove impossible preds and clip preds according to the optimization pipeline
            self.optimization_pipeline.shrink_candidate_values_with_constraints(claim, prop)
            self.optimization_pipeline.set_topn_entropy_and_clip_preds(claim)            
            # x2 = len(prop.candidate_values)
            # if prop.property_name == "row_index":
            #     print("num predicted valies values: ", x2)

            # find index of ground_truth in the preds. If not found, count it as a derivation cost
            cand_values_str = [v.value for v in prop.candidate_values]
            
            try:
                idx = cand_values_str.index(prop.ground_truth)
                cost += (idx+1)*prop.task.ver_cost
                prop.verified_index = idx
                acc_dict[prop.property_name][0] += 1
            except ValueError:
                # derivation
                # print("prop: {}, preds: {}".format(prop.property_name, cand_values_str))
                # print("ground_truth: ", prop.ground_truth)
                cost += len(cand_values_str)*prop.task.ver_cost + prop.task.der_cost
                prop.verified_index = -1
                acc_dict[prop.property_name][1] += 1
        # self.print_preds(claim)
        return cost
    
    def get_init_training_cost(self, train_df):
        """
        Here, we don't have trained models, so we derive everything.
        cost is num_claims * sum(der_cost of properties)
        """
        claims = self.create_claims_from_df(train_df)
        cost = 0
        if len(claims) == 0:
            return cost
        cost = sum([prop.task.der_cost for prop in claims[0].available_properties])
        return len(claims) * cost

    def get_cost_sequential_order_opt(self, test_df):
        """
        Assuming that we have a trained model, go through the test claims, get the predictions
        and calculate the actual cost. We use the optimization to get the number of predictions
        and the subset of properties we ask about.
        Arguments:
            test_df {DataFrame} -- [claims to be verified]
        """
        # claims without preds with the groundtruth
        claims = self.create_claims_from_df(test_df, get_ground_truth=True)
        print("optimizing over {} claims".format(len(claims)))
        features_list = self.get_features_from_claim_list(claims)
        print("extracted {} number of features".format(len(features_list)))
        total_cost = 0
        acc_dict = dict()
        for task_name, task in self.classification_pipeline.classification_tasks_dict.items():
            acc_dict[task_name] = [0,0]
        for features, claim in zip(features_list, claims):
            self.optimization_pipeline.optimize_claim(claim, features.reshape(1,-1))
            total_cost += self.get_cost_of_claim_from_preds(claim, acc_dict)
        return total_cost, claims, acc_dict

    def get_cost_milp_opt(self, test_claims):
        """
        Here we have preds for all the claims and we just get the cost
        Arguments:
            test_claims {list of Claims} -- [claims to be verified]
        """
        total_cost = 0
        acc_dict = dict()
        for task_name, task in self.classification_pipeline.classification_tasks_dict.items():
            acc_dict[task_name] = [0,0]
        for claim in test_claims:
            total_cost += self.get_cost_of_claim_from_preds(claim, acc_dict)
        return total_cost, acc_dict

    def get_cost_sequential_order_opt_retraining(self, train_df, test_df, batch_idx_list):
        """
        Assuming that we have a trained model, go through the test claims, get the predictions
        and calculate the actual cost and retrain. We use the optimization to get the number of predictions
        and the subset of properties we ask about.
        Arguments:
            test_df {DataFrame} -- [claims to be verified]
            batch_idx_list {list of ints} -- [i.e [20, 40, 60, 100]] are the sizes for each batch. 
                                         Each element is an index of the dataframe
        """
        total_cost = self.get_init_training_cost(train_df)
        print("init cost: ", total_cost)
        all_claims = []
        acc_list = []    
        i = 0
        for j in batch_idx_list:
            next_k_df = test_df.iloc[i:j]
            batch_cost, batch_claims, batch_acc_dict = self.get_cost_sequential_order_opt(next_k_df)
            print("batch cost is {} for {} claims".format(batch_cost, len(batch_claims)))
            for k, v in batch_acc_dict.items():
                batch_acc_dict[k] = float(v[0]/(v[0]+v[1]))
            batch_acc_dict["num_training_data"] = i+10
            print("acc_dict: ", batch_acc_dict)
            acc_list.append(batch_acc_dict)
            total_cost += batch_cost
            all_claims.extend(batch_claims)
            # retrain classifiers using the next_k_df
            new_train_df = pd.concat([self.classification_pipeline.train_df, next_k_df], 
                                      axis=0, ignore_index=True, sort=False)
            self.classification_pipeline.train_for_user_study(new_train_df)
            i=j

        print(acc_list)
        return total_cost, all_claims

    def _get_next_k_milp(self, test_df, k, skim_cost=5):
        # claims without preds
        claims = self.create_claims_from_df(test_df, get_ground_truth=True)
        print("optimizing over {} claims".format(len(claims)))
        features_list = self.get_features_from_claim_list(claims)
        print("extracted {} number of features".format(len(features_list)))
        for features, claim in zip(features_list, claims):
            # gets preds
            self.optimization_pipeline.optimize_claim(claim, features.reshape(1,-1))
        # now we can get the next_k from milp
        return self.active_pipeline.get_next_k_milp(claims, batch_size=k, skim_cost=skim_cost)

    def get_cost_active_learning_milp_opt_retraining(self, train_df, test_df, batch_idx_list, skim_cost=5):
        """
        Assuming that we have a trained model, go through the test claims by selecting the next k by milp, 
        get the predictions and calculate the actual cost and retrain. We use the optimization to get the number 
        of predictions and the subset of properties we ask about.
        Arguments:
            test_df {DataFrame} -- [claims to be verified]
            batch_idx_list {list of ints} -- [i.e [20, 40, 60, 100]] are the sizes for each batch. 
                                         Each element is an index of the dataframe
        """
        total_cost = self.get_init_training_cost(train_df)
        print("init cost: ", total_cost)
        all_claims = []
        i = 0
        acc_list = []
        for j in batch_idx_list:
            num_next_k = j-i
            next_k_claims = self._get_next_k_milp(test_df, num_next_k, skim_cost=skim_cost)
            batch_cost, batch_acc_dict = self.get_cost_milp_opt(next_k_claims)
            print("batch cost is {} for {} claims".format(batch_cost, len(next_k_claims)))
            if j == 190:
                for claim in next_k_claims[:1]:
                    self.print_preds(claim)
                    print("\n")
            for k, v in batch_acc_dict.items():
                batch_acc_dict[k] = float(v[0]/(v[0]+v[1]))
            batch_acc_dict["num_training_data"] = len(train_df)
            print("acc_dict: ", batch_acc_dict)
            acc_list.append(batch_acc_dict)
            total_cost += batch_cost
            all_claims.extend(next_k_claims)
            next_k_df = self.classification_pipeline.transform_claim_list_to_df(next_k_claims)
            # remove the selected next_k_df from test_df
            test_df = next_k_df.merge(test_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
            test_df.drop(columns=["_merge"], inplace=True)
            # retrain classifiers using the next_k_df
            train_df = pd.concat([train_df, next_k_df], 
                                      axis=0, ignore_index=True, sort=False)
            self.classification_pipeline.train_for_user_study(train_df)
            i=j

        print("acc_list: ", acc_list)
        return total_cost, all_claims

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

# if __name__ == "__main__":
#     class_pipeline = ClassificationStep(DATA_PATH, simulation=True, min_samples=1)
#     opt_pipeline = OptimizationStep(class_pipeline)
#     simulation = Simulation(class_pipeline, opt_pipeline)
#     simulation.run()

