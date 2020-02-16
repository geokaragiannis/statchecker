from src.pipeline.classification_step import ClassificationStep
from src.pipeline.clustering_step import ClusteringStep
from src.parser.classification_task import ClassificationTask
from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value
from random import sample
import pandas as pd
import os
import numpy as np

DATA_PATH = "data/claims_01-23-2020/"


class ActiveLearningStep:
    def __init__(self, data_path, num_clusters=10):
        self.classification_pipeline = ClassificationStep(data_path, simulation=False, export=False)
        self.num_clusters = num_clusters
        self.clustering_pipeline = ClusteringStep(num_clusters=num_clusters)
        

    def cluster_claims(self, df, num_clusters=10):
        clusters_list = self.clustering_pipeline.cluster_claims(df, num_clusters=num_clusters)
        df["cluster_id"] = clusters_list

    def select_next_k_random(self, df, k=100):
        """
        Select the next k claims randomly from a dataframe
        
        Keyword Arguments:
            df {DataFrame}
            k {int} -- [number of claims to select] (default: {100})
        """
        if len(df) <= (1.5)*k:
            return df
        return df.sample(n=k)

    def get_preds_from_df(self, df):
        """
        Returns the predictions for a given dataframe for each classification task
        
        Arguments:
            df {DataFrame} -- Sents and claims to get preds from
        Returns:
            np array where each column [i, j] is the pred prob of claim i for task j
        """
        sents = df["sent"]
        claims = df["claim"]

        featurizer_tf = self.classification_pipeline.featurizer_tf
        featurizer_emb = self.classification_pipeline.featurizer_emb

        num_class_tasks = len(self.classification_pipeline.classification_tasks_dict.items())
        # each row will have the first pred prob for each task. Each column is the task
        first_pred_matrix = np.zeros((len(df), num_class_tasks))
        features = self.classification_pipeline.get_feature_union(sents, claims, self.classification_pipeline.tok_driver, 
                                                           featurizer_emb, featurizer_tf, mode="test")
        task_num = 0
        for task_name, task in self.classification_pipeline.classification_tasks_dict.items():
            classifier = task.classifier
            
            preds_list = classifier.predict_batch_top_n(features, topn=5)
            # get the first probability for each (sent, claim) pair
            pred_probs = [p[1][0] for p in preds_list]
            first_pred_matrix[:, task_num] = pred_probs
            task_num += 1
        
        return first_pred_matrix
        

    def select_next_k_most_unsure(self, df, k=100):
        print("\n getting preds of size: ", len(df))
        preds_matrix = self.get_preds_from_df(df)
        # sort by computing the average pred prob across tasks
        avg_across_tasks = np.average(preds_matrix, axis=1)
        df["unsure"] = avg_across_tasks
        sorted_df = df.sort_values(by=["unsure"], ascending=False)
        df.drop(columns=["unsure"], inplace=True)
        if len(sorted_df) > 1.5*k:
            return sorted_df.head(k)
        else:
            return sorted_df

    def select_next_k_most_sure(self, df, k=100):
        print("\n getting preds of size: ", len(df))
        preds_matrix = self.get_preds_from_df(df)
        # sort by computing the average pred prob across tasks
        avg_across_tasks = np.average(preds_matrix, axis=1)
        df["unsure"] = avg_across_tasks
        sorted_df = df.sort_values(by=["unsure"], ascending=True)
        df.drop(columns=["unsure"], inplace=True)
        if len(sorted_df) > 1.5*k:
            return sorted_df.head(k)
        else:
            return sorted_df

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

    def handle_cluster_one_model(self, train_df, cluster_df, cluster_id, k=100):
        """
        Here we have one set of property classifiers for ALL clusters
        For this cluster, receive a trained model already, and pick the next
        k claims for which we are most unsure about based on the previosuly 
        trained model
        
        Arguments:
            train_df {DataFrame} -- [sents and claims from previous trainings]
            cluster_df {Dataframe} -- [sents and claims for this current cluster]
            cluster_id {str} -- [the id of the cluster]
        
        Keyword Arguments:
            k {int} -- [number of claims to select next] (default: {100})
        Returns:
            the trained dataframe, which now includes the whole cluster 
            and a list of validation acciracies dictionaries for each property
        """
        cluster_val_accuracy_list = []
        while len(cluster_df) != 0:
            next_k_df = self.select_next_k_most_unsure(cluster_df, k=k)
            self.classification_pipeline.train_for_active(train_df, next_k_df)
            train_df = pd.concat([train_df, next_k_df], axis=0, ignore_index=True,
                                 sort=False)
            cluster_df = train_df.merge(cluster_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
            cluster_df.drop(columns=["_merge"], inplace=True)
            property_val_accuracy_dict = dict()
            for task_name, task in self.classification_pipeline.classification_tasks_dict.items():
                property_val_accuracy_dict[task_name] = task.val_acc
            cluster_val_accuracy_list.append(property_val_accuracy_dict)
        return train_df, cluster_val_accuracy_list
            

    def run_clustering_experiment(self, k=100):
        # list of dicts, where for each property we keep the val_accuracy
        val_accuracy_list = []
        remaining_df = self.classification_pipeline.parser.get_complete_df()
        train_df = self.select_next_k_random(remaining_df, k=k)
        # train initial model with 100 random claims
        self.classification_pipeline.train_for_active(train_df, None)
        remaining_df = train_df.merge(remaining_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
        remaining_df.drop(columns=["_merge"], inplace=True)
        self.cluster_claims(remaining_df)
        grouped_df = remaining_df.groupby("cluster_id")
        # go through each cluster (starting from the largest)
        for cluster_id, cluster_df in sorted(grouped_df, key=lambda x: len(x[1]), reverse=True):
            print("\n cluster id: {} \n".format(cluster_id))
            print("size of cluster: ", len(cluster_df))
            train_df, cluster_val_acc_list = self.handle_cluster_one_model(train_df, cluster_df, cluster_id)
            val_accuracy_list.extend(cluster_val_acc_list)

        fname = "val_accuracy_list_cluster_one_model_{}_clusters.csv".format(self.num_clusters)
        self.export_val_acc_list_to_csv(val_accuracy_list, os.path.join("data", "active_learning",
                                        fname))
    
    def run_sure_experiment(self, k=100):
        """
        Run an ective learning experiment where we choose the next k questions to ask
        by selecting the ones we are most unsure about
        """
        # list of dicts, where for each property we keep the val_accuracy
        val_accuracy_list = []
        remaining_df = self.classification_pipeline.parser.get_complete_df()
        # dataframe that will contain the next picked data and the previously picked ones
        train_df = self.select_next_k_random(remaining_df, k=k)
        init_iter = True
        print("initial length of df: ", len(remaining_df))
        while len(remaining_df) != 0:
            # train the model using the remaining_df as validation set, where we get accuracy by using the 
            # trained model
            if init_iter:
                print("random choice of next k")
                next_k_df = self.select_next_k_random(remaining_df, k=k)
            else:
                print("unsure choice of next k")
                next_k_df = self.select_next_k_most_sure(remaining_df, k=k)

            init_iter = False   
            self.classification_pipeline.train_for_active(train_df, next_k_df)
            train_df = pd.concat([train_df, next_k_df], axis=0, ignore_index=True,
                                 sort=False)
            remaining_df = train_df.merge(remaining_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
            remaining_df.drop(columns=["_merge"], inplace=True)
            print("after removing. Len is: ", len(remaining_df))
            inter_bool = len(pd.merge(train_df, remaining_df, how="inner", on=["sent", "claim"])) == 0
            print("no intersection: ", inter_bool)
            property_val_accuracy_dict = dict()
            for task_name, task in self.classification_pipeline.classification_tasks_dict.items():
                property_val_accuracy_dict[task_name] = task.val_acc
            
            val_accuracy_list.append(property_val_accuracy_dict)
        self.export_val_acc_list_to_csv(val_accuracy_list, os.path.join("data", "active_learning",
                                                                        "val_accuracy_list_sure.csv"))
        return val_accuracy_list

    
    def run_unsure_experiment(self, k=100):
        """
        Run an ective learning experiment where we choose the next k questions to ask
        by selecting the ones we are most unsure about
        """
        # list of dicts, where for each property we keep the val_accuracy
        val_accuracy_list = []
        remaining_df = self.classification_pipeline.parser.get_complete_df()
        # dataframe that will contain the next picked data and the previously picked ones
        train_df = self.select_next_k_random(remaining_df, k=k)
        init_iter = True
        print("initial length of df: ", len(remaining_df))
        while len(remaining_df) != 0:
            # train the model using the remaining_df as validation set, where we get accuracy by using the 
            # trained model
            if init_iter:
                print("random choice of next k")
                next_k_df = self.select_next_k_random(remaining_df, k=k)
            else:
                print("unsure choice of next k")
                next_k_df = self.select_next_k_most_unsure(remaining_df, k=k)

            init_iter = False   
            self.classification_pipeline.train_for_active(train_df, next_k_df)
            train_df = pd.concat([train_df, next_k_df], axis=0, ignore_index=True,
                                 sort=False)
            remaining_df = train_df.merge(remaining_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
            remaining_df.drop(columns=["_merge"], inplace=True)
            print("after removing. Len is: ", len(remaining_df))
            inter_bool = len(pd.merge(train_df, remaining_df, how="inner", on=["sent", "claim"])) == 0
            print("no intersection: ", inter_bool)
            property_val_accuracy_dict = dict()
            for task_name, task in self.classification_pipeline.classification_tasks_dict.items():
                property_val_accuracy_dict[task_name] = task.val_acc
            
            val_accuracy_list.append(property_val_accuracy_dict)
        self.export_val_acc_list_to_csv(val_accuracy_list, os.path.join("data", "active_learning",
                                                                        "val_accuracy_list_unsure.csv"))
        return val_accuracy_list

    
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
    active_learning = ActiveLearningStep(DATA_PATH, num_clusters=8)
    # print(active_learning.run_random_experiment())
    # print(active_learning.run_sure_experiment())
    active_learning.run_clustering_experiment()