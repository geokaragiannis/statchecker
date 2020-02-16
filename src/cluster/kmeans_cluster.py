"""
Clusters the given features using the KMeans algorithm
"""
import logging
import pandas as pd
from sklearn.cluster import KMeans


class KmeansCluster:
    def __init__(self, num_clusters=18):
        self.kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        self.claim_clusters = None
        self.logger = logging.getLogger(__name__)

    def get_clusters(self, features):
        # cluster index for each feature
        self.claim_clusters = self.kmeans.fit_predict(features)
        return self.claim_clusters