from src.parser.dataset_parser import DatasetParser
from src.tokenizer.tokenizer_driver import TokenizerDriver
from src.featurizer.feature_extractor import FeatureExtractor
from src.classifier.classifier_linear_svm import ClassifierLinearSVM
from src import helpers

import pandas as pd
import numpy as np
import scipy


class ClassificationStep:
    def __init__(self, data_path, cv=3, min_samples=5, topn=5, simulation=True):
        self.cv = cv
        self.min_samples = min_samples
        self.topn = topn
        self.data_path = data_path
        self.simulation = simulation
        self.config = helpers.load_yaml("src/config.yml")

        self.parser = DatasetParser(self.data_path)
        # dict of classification_tasks to be included in the classification.
        self.classification_tasks_dict = self.parser.classification_tasks_dict
        self.tok_driver = TokenizerDriver()

        self.complete_df = self.parser.get_complete_df()
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
        helpers.save_df_to_dir(self.config["data_dir"], self.config["test_df_name"], test_df)
        print("saved test_df successfully")
        return test_df

    def get_train_val_test_splits(self, df, task, train_frac=0.9, val_frac=0.5):
        train_df = df.sample(frac=train_frac)
        # get remaining rows in the dataframe not in train_df
        val_test_df = train_df.merge(df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
        train_labels = list(train_df[task.hash_name])
        # only keep data points, which have a label seen in the training data
        val_test_df = val_test_df[val_test_df[task.hash_name].isin(train_labels)]
        val_df = val_test_df.sample(frac=val_frac)
        test_df = val_test_df.drop(val_df.index)
        print("No intersection of train, test set: {}".format(len(pd.merge(train_df, val_df, how="inner", on=["sent", "claim"])) == 0))
        return train_df, val_df, test_df

    def train_single_task(self, X_train, y_train, task):
        classifier = ClassifierLinearSVM(task)
        model = classifier.train(X_train, y_train)
        return classifier
    
    def train(self, val_frac=1.0):
        
        if self.simulation:
            self.test_df = self.get_test_df_simulation(test_frac=0.05)
            # only keep data not picked for testing
            self.complete_df = self.test_df.merge(self.complete_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
            self.complete_df.drop(columns=["_merge"], inplace=True)

        for _, task in self.classification_tasks_dict.items():
            print("Training classifier for task: {}".format(task.name))
            featurizer_tf = FeatureExtractor(task, mode="tfidf")
            featurizer_emb = FeatureExtractor(task, mode="word-embeddings")
            min_samples_df = self.create_min_samples_dataset(self.complete_df, task)
            train_df, val_df, _ = self.get_train_val_test_splits(min_samples_df, task, 
                                                                           val_frac=val_frac)
            sents_train = list(train_df["sent"])
            claims_train = list(train_df["claim"])
            X_train = self.get_feature_union(sents_train, claims_train, self.tok_driver, 
                                             featurizer_emb, featurizer_tf, mode="train")
            sents_val = list(val_df["sent"])
            claims_val = list(val_df["claim"])
            X_val = self.get_feature_union(sents_val, claims_val, self.tok_driver, featurizer_emb,
                                           featurizer_tf, mode="test")
            y_train = list(train_df[task.hash_name])
            y_val = list(val_df[task.hash_name])
            task_classifier = self.train_single_task(X_train, y_train, task)
            val_preds, val_acc = task_classifier.get_pred_and_accuracy(X_val, y_val, topn=self.topn)
            task.featurizer_tf = featurizer_tf
            task.featurizer_emb = featurizer_emb
            task.classifier = task_classifier
            task.val_acc = val_acc
            task.is_trained = True
            task_classifier.export()
            featurizer_tf.export()
            featurizer_emb.export()
            print("val acc for task {} is {}".format(task.name, val_acc))

    def load_models(self):
        for _, task in self.classification_tasks_dict.items():
            task_classifier = ClassifierLinearSVM(task)
            task_classifier.load()
            task_featurizer_tf = FeatureExtractor(task, mode="tfidf")
            task_featurizer_tf.load()
            task_featurizer_emb = FeatureExtractor(task, mode="word-embeddings")
            task_featurizer_tf.load()
            task.classifier = task_classifier
            task.featurizer_tf = task_featurizer_tf
            task.featurizer_emb = task_featurizer_emb
            task.is_trained = True
            print("loaded models for {} task successfully".format(task.name))

    def load_test_df(self):
        self.test_df = helpers.load_df_from_dir(self.config["data_dir"], self.config["test_df_name"])
        print("loaded test_df successfully")
        return self.test_df

    @staticmethod
    def concat_features(features_s, features_c):
        if isinstance(features_c, scipy.sparse.csr.csr_matrix):
            features_c = features_c.toarray()
        if isinstance(features_s, scipy.sparse.csr.csr_matrix):
            features_s = features_s.toarray()
        return np.concatenate((features_s, features_c), axis=1)

    def get_feature_union(self, sents, claims, tokenizer, featurizer_emb, featurizer_tf, mode="train"):
        tokenized_sents = tokenizer.tokenize_data(sents)
        tokenized_claims = tokenizer.tokenize_data(claims)
        if mode == "train":
            features_sents = featurizer_emb.featurize_train(tokenized_sents)
            features_claims = featurizer_tf.featurize_train(tokenized_claims)
        else:
            features_sents = featurizer_emb.featurize_test(tokenized_sents)
            features_claims = featurizer_tf.featurize_test(tokenized_claims)
        features_union = self.concat_features(features_sents, features_claims)
        # print("training features extracted")
        # print("Sentence features shape: {}".format(features_sents.shape))
        # print("Claims features shape: {}".format(features_claims.shape))
        # print("Union features shape: {}".format(features_union.shape))
        return features_union
