"""
This experiment only uses a dataset for template predictions
"""
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import random
import argparse

from src.parser.dataset_parser import DatasetParser
from src.tokenizer.tokenizer_driver import TokenizerDriver
from src.featurizer.feature_extractor import FeatureExtractor
from src.classifier.classifier_linear_svm import ClassifierLinearSVM


DATA_PATH = "data/main_annotated_dataset_12-16-2019.csv"

arg_parser = argparse.ArgumentParser(description='Arguments for inference client')
arg_parser.add_argument('--num_runs', type=int, help='number of times to run the experiment', required=True)
arg_parser.add_argument('--cv', type=int, help="cross validation for fitting the sigmoid", required=True)
arg_parser.add_argument('--min_samples_per_label', type=int, help="Minumum number of samples each label should have", required=True)
arg_parser.add_argument('--topn', type=int, default=5, help="Top predictions to consider")
args = arg_parser.parse_args()

def join_formula_join_df(formula_df, lookup_df):
    joined_df = pd.merge(formula_df, lookup_df, on=["sent", "claim"], how="left")
    joined_df = joined_df[["sent", "claim", "formula_x", "template_formula", "row_index_y", "file_x"]]
    joined_df = joined_df[joined_df.row_index_y.notnull()]
    joined_df = joined_df.rename(columns={"formula_x": "formula", "row_index_y": "row_index", "file_x": "file"})
    unique_formula_df = joined_df.drop_duplicates(subset=["sent", "claim", "template_formula"])
    unique_formula_lookup_df = unique_formula_df.drop_duplicates(subset=["sent", "claim", "row_index"])
    return unique_formula_lookup_df

parser = DatasetParser(DATA_PATH)
template_df = parser.get_formula_df()
template_df = template_df.drop_duplicates(subset=["sent", "claim"]).reset_index(drop=True)
template_df = template_df[template_df.template_formula.notnull()]
template_df = template_df.drop_duplicates(subset=["sent", "claim", "template_formula"]).reset_index(drop=True)
cv = args.cv
k = args.num_runs
topn = args.topn
min_samples_per_label = args.min_samples_per_label
print("number of samples before prunning: {}".format(len(template_df)))
# main_df = create_cv_dataset(lookup_df, "row_index", cv=cv)
# print_stats(main_df)

print("topn = {}".format(topn))
print("min number of samples per label: {}".format(min_samples_per_label))
print("cv = {}".format(cv))
print("num_runs = {}".format(k))

tok_driver = TokenizerDriver()
print("Running classifier for template formulas")
acc = 0
for run in range(0, k):
    print("Run: {}".format(run+1))
    featurizer_tf = FeatureExtractor(mode="tfidf")
    featurizer_emb = FeatureExtractor(mode="word-embeddings")
    classifier = ClassifierLinearSVM(cv=cv)

    train_df, test_df = parser.get_test_train_cv_split(template_df, "template_formula", min_samples=min_samples_per_label)
    print("number of samples after prunning for run number {}: {}".format(run+1, len(train_df) + len(test_df)))

    X_train = parser.get_feature_union_train(train_df, tok_driver, featurizer_emb, featurizer_tf)
    X_test = parser.get_feature_union_test(test_df, tok_driver, featurizer_emb, featurizer_tf)
    y_train = list(train_df["template_formula"])
    y_test = list(test_df["template_formula"])
    model = classifier.train(X_train, y_train)
    _, step_acc = classifier.get_pred_and_accuracy(X_test, y_test, topn=topn)
    print("accuracy for run {} is: {}".format(run+1, step_acc))
    acc += step_acc
acc = float(acc/k)

print("accuracy for {} number of runs for template formulas {} for topn = {}".format(k, acc, topn))



