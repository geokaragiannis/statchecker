from src.pipeline.classification_step import ClassificationStep
from src.parser.classification_task import ClassificationTask
from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value
from colorama import Fore, Style


DATA_PATH = "data/claims_01-23-2020/"
class Simulation:
    def __init__(self, data_path):
        self.classification_pipeline = ClassificationStep(data_path, simulation=True, min_samples=1)
        self.z = 0

    def create_claims_from_df(self, test_df):
        """
        return a list of Claim objects by populating them with the sentence and claim strings
        from the test_df. 
        """
        test_claims = []
        for idx, test_row in test_df.iterrows():
            available_properties = [Property(t.name, task=t) for t in 
                                    list(self.classification_pipeline.classification_tasks_dict.values())]
            available_properties = sorted(available_properties, key=lambda x: x.task.priority)
            test_claim = Claim(test_row["sent"], test_row["claim"], available_properties)
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
    simulation = Simulation(DATA_PATH)
    simulation.run()

