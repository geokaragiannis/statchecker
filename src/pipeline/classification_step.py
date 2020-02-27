from src.parser.dataset_parser import DatasetParser
from src.tokenizer.tokenizer_driver import TokenizerDriver
from src.featurizer.feature_extractor import FeatureExtractor
from src.classifier.classifier_linear_svm import ClassifierLinearSVM
from src import helpers

import pandas as pd
import numpy as np
import scipy


class ClassificationStep:
    def __init__(self, data_path, min_samples=5, topn=5, simulation=True, export=True):
        self.min_samples = min_samples
        self.topn = topn
        self.data_path = data_path
        self.simulation = simulation
        self.export = export
        self.config = helpers.load_yaml("src/config.yml")

        self.parser = DatasetParser(self.data_path)
        # dict of classification_tasks to be included in the classification.
        self.classification_tasks_dict = self.parser.classification_tasks_dict
        # common featurizer objects for all tasks
        self.featurizer_tf = None
        self.featurizer_emb = None
        self.complete_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def create_min_samples_dataset(self, df, task):     
        return df.groupby(task.hash_name).filter(lambda x: len(x) > self.min_samples)

    def get_test_df_simulation(self, test_frac=0.05):
        """
        Returns a dataframe which will be used for simulation and will not be 
        part of training. Remove this test_df from the original complete dataset.
        Saves dataframe to directory specified by config.
        Keyword Arguments:
            test_frac {float} -- [Fraction of the whole data for test_df] (default: {0.05})
        """
        test_df = self.complete_df.sample(frac=test_frac)
        if self.export:
            helpers.save_df_to_dir(self.config["data_dir"], self.config["test_df_name"], test_df)
            print("saved test_df successfully")
        return test_df

    def get_train_val_test_splits(self, df, train_frac=0.9):
        train_df = df.sample(frac=train_frac)
        # get remaining rows in the dataframe not in train_df
        val_df = train_df.merge(df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
        return train_df, val_df

    def train_single_task(self, X_train, y_train, task):
        classifier = ClassifierLinearSVM(task)
        model = classifier.train(X_train, y_train)
        return classifier

    def get_featurizers(self):
        if self.featurizer_tf is None:
            self.featurizer_tf = FeatureExtractor(mode="tfidf")

        if self.featurizer_emb is None:
            self.featurizer_emb = FeatureExtractor(mode="word-embeddings")

        return self.featurizer_tf, self.featurizer_emb

    def set_featurizers(self, f_tf, f_emb):
        self.featurizer_tf = f_tf
        self.featurizer_emb = f_emb

    def train_for_active(self, train_df, val_df):

        self.train_df = train_df
        self.val_df = val_df
        self.parser.set_task_values(self.train_df)
        print("training on {} number of samples".format(len(self.train_df)))

        featurizer_tf, featurizer_emb =  self.get_featurizers()
        sents_train = list(self.train_df["sent"])
        claims_train = list(self.train_df["claim"])
        X_train = self.get_feature_union(sents_train, claims_train,
                                             featurizer_emb, featurizer_tf, mode="train")
        if val_df is not None:
            self.parser.set_task_values(self.val_df)
            print("validating on {} number of samples".format(len(self.val_df)))
            sents_val = list(self.val_df["sent"])
            claims_val = list(self.val_df["claim"])
            X_val = self.get_feature_union(sents_val, claims_val, featurizer_emb,
                                            featurizer_tf, mode="test")
        
        self.set_featurizers(featurizer_tf, featurizer_emb)

        for _, task in self.classification_tasks_dict.items():
            print("Training classifier for task: {}".format(task.name))
            
            y_train = list(self.train_df[task.hash_name])
            task_classifier = self.train_single_task(X_train, y_train, task)
            task.featurizer_tf = featurizer_tf
            task.featurizer_emb = featurizer_emb
            task.classifier = task_classifier
            if val_df is not None:
                y_val = list(self.val_df[task.hash_name])
                val_preds, val_acc = task_classifier.get_pred_and_accuracy(X_val, y_val, topn=self.topn)
                task.val_acc = val_acc
            task.is_trained = True
            if self.export:
                task_classifier.export()
                task.export_hash_dicts()
            print("val acc for task {} is {}".format(task.name, task.val_acc))

        if self.export:
            featurizer_tf.export()
            featurizer_emb.export()
            # export train_df and val_df to disk
            helpers.save_df_to_dir(self.config["data_dir"], self.config["train_df_name"], self.train_df)
            helpers.save_df_to_dir(self.config["data_dir"], self.config["val_df_name"], self.val_df)

    def train_for_user_study(self, df):
        """
        Only use a single training dataframe. No validation is done here.
        
        Arguments:
            df {[type]} -- [description]
        """
        self.train_df = df
        # each task has now all the possible values it can take
        self.parser.set_task_values(self.train_df)

        featurizer_tf = FeatureExtractor(mode="tfidf")
        featurizer_emb = FeatureExtractor(mode="word-embeddings")
        sents_train = list(self.train_df["sent"])
        claims_train = list(self.train_df["claim"] + " " + self.train_df["subsection"])
        X_train = self.get_feature_union(sents_train, claims_train, 
                                             featurizer_emb, featurizer_tf, mode="train")
        self.set_featurizers(featurizer_tf, featurizer_emb)
        print("training for user study. Shape of training data: ", X_train.shape)
        for _, task in self.classification_tasks_dict.items():
            print("Training classifier for task: {}".format(task.name))
            
            y_train = list(self.train_df[task.hash_name])
            task_classifier = self.train_single_task(X_train, y_train, task)
            task.featurizer_tf = featurizer_tf
            task.featurizer_emb = featurizer_emb
            task.classifier = task_classifier
            task.is_trained = True
            if self.export:
                task_classifier.export()
                task.export_hash_dicts()

        if self.export:
            featurizer_tf.export()
            featurizer_emb.export()
            # export train_df to disk
            helpers.save_df_to_dir(self.config["data_dir"], self.config["train_df_name"], self.train_df)

    def train(self, df, train_frac=0.9):
        self.complete_df = df
        self.parser.set_task_values(self.complete_df)
        
        if self.simulation:
            self.test_df = self.get_test_df_simulation(test_frac=0.05)
            # only keep data not picked for testing
            self.complete_df = self.test_df.merge(self.complete_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
            self.complete_df.drop(columns=["_merge"], inplace=True)

        featurizer_tf, featurizer_emb =  self.get_featurizers()
        self.train_df, self.val_df = self.get_train_val_test_splits(self.complete_df, train_frac=train_frac)
        sents_train = list(self.train_df["sent"])
        claims_train = list(self.train_df["claim"])
        X_train = self.get_feature_union(sents_train, claims_train, 
                                             featurizer_emb, featurizer_tf, mode="train")
        sents_val = list(self.val_df["sent"])
        claims_val = list(self.val_df["claim"])
        X_val = self.get_feature_union(sents_val, claims_val, featurizer_emb,
                                        featurizer_tf, mode="test")
        
        for _, task in self.classification_tasks_dict.items():
            print("Training classifier for task: {}".format(task.name))
            
            y_train = list(self.train_df[task.hash_name])
            y_val = list(self.val_df[task.hash_name])
            task_classifier = self.train_single_task(X_train, y_train, task)
            val_preds, val_acc = task_classifier.get_pred_and_accuracy(X_val, y_val, topn=self.topn)
            task.featurizer_tf = featurizer_tf
            task.featurizer_emb = featurizer_emb
            task.classifier = task_classifier
            task.val_acc = val_acc
            task.is_trained = True
            if self.export:
                task_classifier.export()
                task.export_hash_dicts()
            print("val acc for task {} is {}".format(task.name, val_acc))

        if self.export:
            featurizer_tf.export()
            featurizer_emb.export()
            # export train_df and val_df to disk
            helpers.save_df_to_dir(self.config["data_dir"], self.config["train_df_name"], self.train_df)
            helpers.save_df_to_dir(self.config["data_dir"], self.config["val_df_name"], self.val_df)

    def retrain(self, claims_list):
        """
        Given a list of new verified claims, we retrain our model
        by combining the old training data (if any) with the new data
        
        Arguments:
            claims_list {List of Claim objects} -- [New annotated data from crowdsourcing]
        """
        new_data_df = self.transform_claim_list_to_df(claims_list)
        if self.train_df is None:
            print("Retraining from 0 training data")
            self.train_df = new_data_df
        else:
            self.train_df = pd.concat([self.train_df, new_data_df], axis=0, ignore_index=True, sort=False)
        
        featurizer_tf, featurizer_emb =  self.get_featurizers()
        # self.train_df, self.val_df = self.get_train_val_test_splits(self.train_df)

        sents_train = list(self.train_df["sent"])
        claims_train = list(self.train_df["claim"])
        X_train = self.get_feature_union(sents_train, claims_train,
                                             featurizer_emb, featurizer_tf, mode="train")
        print("retraining with train size: ", len(self.train_df))
        sents_val = list(self.val_df["sent"])
        claims_val = list(self.val_df["claim"])
        X_val = self.get_feature_union(sents_val, claims_val, featurizer_emb,
                                        featurizer_tf, mode="test")
        
        self.set_featurizers(featurizer_tf, featurizer_emb)

        for _, task in self.classification_tasks_dict.items():
            print("Retraining classifier for task: {}".format(task.name))
            
            y_train = list(self.train_df[task.hash_name])
            y_val = list(self.val_df[task.hash_name])
            task_classifier = self.train_single_task(X_train, y_train, task)
            val_preds, val_acc = task_classifier.get_pred_and_accuracy(X_val, y_val, topn=self.topn)
            task.featurizer_tf = featurizer_tf
            task.featurizer_emb = featurizer_emb
            task.classifier = task_classifier
            task.val_acc = val_acc
            task.is_trained = True
            if self.export:
                task_classifier.export()
                task.export_hash_dicts()
            print("val acc for task {} is {}".format(task.name, val_acc))

        if self.export:
            featurizer_tf.export()
            featurizer_emb.export()
            # export new train_df and val_df to disk
            helpers.save_df_to_dir(self.config["data_dir"], self.config["train_df_name"], self.train_df)
            helpers.save_df_to_dir(self.config["data_dir"], self.config["val_df_name"], self.val_df)


    def transform_claim_list_to_df(self, claims_list):
        """
        Transforms a list of claims objects into a Dataframe that has the same format
        as the train_df
        
        Arguments:
            claims_list {list of Claim objects} -- [description]
        Returns:
            pandas Dataframe with the new training data
        """
        # get columns from the properties of a claim
        if len(claims_list) > 0:
            cols = list(claims_list[0].convert_to_pandas_row().keys())
        else:
            print("Empty list")
            return pd.DataFrame()
        ret_df = pd.DataFrame(columns=cols)
        for claim in claims_list:
            row_dict = claim.convert_to_pandas_row()
            ret_df = ret_df.append(row_dict,  ignore_index=True)
        print("successfully transformed {} new data points".format(len(ret_df)))

        return ret_df

    def load_models(self):
        common_featurizer_tf =  FeatureExtractor(mode="tfidf")
        common_featurizer_tf.load()
        common_featurizer_emb = FeatureExtractor(mode="word-embeddings")
        common_featurizer_emb.load()
        self.set_featurizers(common_featurizer_tf, common_featurizer_emb)
        for _, task in self.classification_tasks_dict.items():
            task_classifier = ClassifierLinearSVM(task)
            task_classifier.load()
            task.classifier = task_classifier
            task.featurizer_tf = common_featurizer_tf
            task.featurizer_emb = common_featurizer_emb
            task.is_trained = True
            task.load_hash_dicts()
            print("loaded models for {} task successfully".format(task.name))

    def load_dfs(self):
        self.test_df = helpers.load_df_from_dir(self.config["data_dir"], self.config["test_df_name"])
        self.train_df = helpers.load_df_from_dir(self.config["data_dir"], self.config["train_df_name"])
        self.val_df = helpers.load_df_from_dir(self.config["data_dir"], self.config["val_df_name"])
        self.parser.set_task_values(self.train_df)
        if self.test_df is not None:
            self.parser.set_task_values(self.test_df)
        if self.val_df is not None:
            self.parser.set_task_values(self.val_df)
        print("loaded train_df, val_df and test_df successfully")
        return self.train_df, self.val_df, self.test_df

    @staticmethod
    def concat_features(features_s, features_c):
        if isinstance(features_c, scipy.sparse.csr.csr_matrix):
            features_c = features_c.toarray()
        if isinstance(features_s, scipy.sparse.csr.csr_matrix):
            features_s = features_s.toarray()
        return np.concatenate((features_s, features_c), axis=1)

    def get_feature_union(self, sents, claims, featurizer_emb, featurizer_tf, mode="train"):
        if mode == "train":
            features_sents = featurizer_emb.featurize_train(sents)
            features_claims = featurizer_tf.featurize_train(claims)
        else:
            features_sents = featurizer_emb.featurize_test(sents)
            features_claims = featurizer_tf.featurize_test(claims)
        features_union = self.concat_features(features_sents, features_claims)
        # print("training features extracted")
        # print("Sentence features shape: {}".format(features_sents.shape))
        # print("Claims features shape: {}".format(features_claims.shape))
        # print("Union features shape: {}".format(features_union.shape))
        return features_union
