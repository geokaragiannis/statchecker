from src.parser.dataset_parser import DatasetParser
from src.pipeline.classification_step import ClassificationStep
from src.tokenizer.tokenizer_driver import TokenizerDriver
from src.featurizer.feature_extractor import FeatureExtractor
from src.cluster.kmeans_cluster import KmeansCluster
from src import helpers

import pandas as pd
import numpy as np
import scipy

class ClusteringStep:
    def __init__(self, num_clusters=10):
        self.num_clusters = num_clusters
        self.config = helpers.load_yaml("src/config.yml")

        self.tok_driver = TokenizerDriver()
        # common featurizer objects for all tasks
        self.featurizer_tf = None
        self.featurizer_emb = None
        self.clusterer = KmeansCluster(num_clusters=num_clusters)

    def get_featurizers(self):
        if self.featurizer_tf is None:
            self.featurizer_tf = FeatureExtractor(mode="tfidf")

        if self.featurizer_emb is None:
            self.featurizer_emb = FeatureExtractor(mode="word-embeddings")

        return self.featurizer_tf, self.featurizer_emb

    def set_featurizers(self, f_tf, f_emb):
        self.featurizer_tf = f_tf
        self.featurizer_emb = f_emb
    
    def cluster_claims(self, df, num_clusters=10):
        print("clustering...")
        featurizer_tf, featurizer_emb =  self.get_featurizers()
        sents = list(df["sent"])
        claims = list(df["claim"])
        X = self.get_feature_union(sents, claims, self.tok_driver, 
                                   featurizer_emb, featurizer_tf, mode="train")
        return self.clusterer.get_clusters(X)

    def get_clusters_class_pipeline_obj(self, data_path):
        """
        For each cluster_id create a classificationStep object, which will 
        hold the logic for creating specific classifiers for the clusters
        
        Arguments:
            data_path {str} -- needed for creating a classification step obj.
        """
        ret_dict = dict()
        for label in self.clusterer.kmeans.labels_:
            ret_dict[label] = ClassificationStep(data_path, simulation=False, export=False)
        return ret_dict

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

        return features_union