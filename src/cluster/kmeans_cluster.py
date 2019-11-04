"""
Clusters the given features using the KMeans algorithm
"""
import logging
import pandas as pd
from sklearn.cluster import KMeans


class KmeansCluter:
    def __init__(self, num_clusters=18):
        self.kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        self.claim_clusters = None
        self.df = None
        self.logger = logging.getLogger(__name__)

    def get_clusters(self, features):
        # cluster index for each feature
        self.claim_clusters = self.kmeans.fit_predict(features)
        return self.claim_clusters

    def get_claims_cluster_df(self, claims_list):
        """
        For each claim, get the corresponding cluster index (from self.claim_clusters). claims_list and
        self.claim_clusters have the same index. I.e the 3rd claim in claims_list will have as cluster
        self.claim_clusters[4]
        :param claims_list: List of claims
        :return: Dataframe with two columns: claim sentence and cluster index
        """
        if self.claim_clusters is None:
            self.logger.critical("Requesting Claims clusters Dataframe with None clusters")
            return
        print("Claims Num: {}, Clusters Num: {}".format(len(claims_list), len(self.claim_clusters)))
        self.df = pd.DataFrame(columns=["claim", "cluster"])
        for claim, cluster_idx in zip(claims_list, self.claim_clusters):
            self.df = self.df.append({"claim": claim, "cluster": cluster_idx}, ignore_index=True)
        return self.df