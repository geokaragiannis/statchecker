from src.pipeline.classification_step import ClassificationStep
from src.parser.classification_task import ClassificationTask
from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value
from random import sample
import pandas as pd
import os

DATA_PATH = "data/claims_01-23-2020/"


class ActiveLearningStep:
    def __init__(self, data_path):
        self.classification_pipeline = ClassificationStep(data_path, simulation=False, export=False)

    def select_next_k_random(self, df, k=100):
        """
        Select the next k claims randomly from a dataframe
        
        Keyword Arguments:
            df {DataFrame}
            k {int} -- [number of claims to select] (default: {100})
        """
        if len(df) <= k:
            return df
        return df.sample(n=k)

    def create_claims_from_df(self, df, predict=False):
        """
        return a list of Claim objects by populating them with the sentence and claim strings
        from the df. 
        """
        claims_list = []
        for idx, test_row in df.iterrows():
            available_properties = [Property(t.name, task=t) for t in 
                                    list(self.classification_pipeline.classification_tasks_dict.values())]
            available_properties = sorted(available_properties, key=lambda x: x.task.priority)
            test_claim = Claim(test_row["sent"], test_row["claim"], available_properties)
            # if predict:
            #     self.get_preds_from_claim(test_claim)
            claims_list.append(test_claim)
        return claims_list

    def export_val_acc_list_to_csv(self, val_acc_list, fpath):
        val_acc_df = pd.DataFrame(val_acc_list)
        val_acc_df.to_csv(fpath, index=False)

    
    def run_random_experiment(self, k=100):
        """
        Run an ective learning experiment where we choose the next k questions to ask
        randomly from the list of remaining claims
        """
        # list of dicts, where for each property we keep the val_accuracy
        val_accuracy_list = []
        remaining_df = self.classification_pipeline.parser.get_complete_df()
        # dataframe that will contain the next picked data and the previously picked ones
        train_df = self.select_next_k_random(remaining_df, k=k)
        while len(remaining_df) != 0:
            # train_df = pd.concat([train_df, self.select_next_k_random(remaining_df, k=k)], axis=0, ignore_index=True,
            #                      sort=False)
            # train the model using the remaining_df as validation set, where we get accuracy by using the 
            # trained model
            next_k_df = self.select_next_k_random(remaining_df, k=k)
            self.classification_pipeline.train_for_active(train_df, next_k_df)
            train_df = pd.concat([train_df, next_k_df], axis=0, ignore_index=True,
                                 sort=False)
            remaining_df = train_df.merge(remaining_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
            remaining_df.drop(columns=["_merge"], inplace=True)
            print("after removing. Len is: ", len(remaining_df))
            property_val_accuracy_dict = dict()
            for task_name, task in self.classification_pipeline.classification_tasks_dict.items():
                property_val_accuracy_dict[task_name] = task.val_acc
            
            val_accuracy_list.append(property_val_accuracy_dict)
        self.export_val_acc_list_to_csv(val_accuracy_list, os.path.join("data", "active_learning",
                                                                        "val_accuracy_list_random2.csv"))
        return val_accuracy_list


if __name__ == "__main__":
    active_learning = ActiveLearningStep(DATA_PATH)
    print(active_learning.run_random_experiment())