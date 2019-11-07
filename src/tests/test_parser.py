from src.parser.document_parser import DocumentParser
from src.tokenizer.tokenizer_driver import TokenizerDriver
from src.featurizer.feature_extractor import FeatureExtractor
from src.cluster.kmeans_cluster import KmeansCluster

from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt


REPORTS_PATH = "data/demo_reports/"

doc_parser = DocumentParser(REPORTS_PATH)
tok_driver = TokenizerDriver()
featurizer = FeatureExtractor(mode="word-embeddings")
claim_sents = doc_parser.parse_documents()
print("Number of claims: ", len(claim_sents))
print(claim_sents[:10])

tokenized_sents = tok_driver.tokenize_claims(claim_sents)

print("Tokenized Sentences: ")
print(tokenized_sents[:5])

features = featurizer.featurize_claims(tokenized_sents)

print("features 1 length: ", features[0].shape)
print("features 2 length: ", features[0].shape)
print("Features: ")
print(features[:5])

wcss = []
for i in range(1, 50):
    print("cluster num: ", i)
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 50), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans_cluster = KmeansCluster()
clusters = kmeans_cluster.get_clusters(features)
claims_cluster_df = kmeans_cluster.get_claims_cluster_df(tokenized_sents)

print(claims_cluster_df.head(20))
