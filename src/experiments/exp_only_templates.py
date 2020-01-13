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
from src.simulate_crowdworkers.crowdworker import Crowdworker


DATA_PATH = "data/main_annotated_dataset_12-16-2019.csv"

arg_parser = argparse.ArgumentParser(description='Arguments for inference client')
arg_parser.add_argument('--num_runs', type=int, help='number of times to run the experiment', required=True)
arg_parser.add_argument('--cv', type=int, help="cross validation for fitting the sigmoid", required=True)
arg_parser.add_argument('--min_samples_per_label', type=int, help="Minumum number of samples each label should have", required=True)
arg_parser.add_argument('--topn', type=int, default=5, help="Top predictions to consider")
args = arg_parser.parse_args()

def print_predictions(test_df, predictions):
    pred_labels = [k[0] for k in predictions]
    pred_probs = [k[1] for k in predictions]
    test_df["pred_labels"] = pred_labels
    test_df["pred_probs"] = pred_probs
    test_df.to_csv("data/experiments/template_pred_01_10_2020.csv", index=False)

def get_stats(predictions, y_test, topn):
    """
    Returns statistics from the predictions and true values
    """
    topn_freq = [0]*(topn + 1)
    sum_p1_correct = 0.0
    num_p1_correct = 0.0
    sum_p1_incorrect = 0.0
    num_p1_incorrect = 0.0
    sum_p1_minus_p_correct = 0.0
    num_p1_minus_p_correct = 0.0

    num_p1_more_0_5_correct = 0.0
    num_p1_more_0_5 = 0.0
    for pred, label in zip(predictions, y_test):
        pred_labels = pred[0]
        pred_probs = pred[1]
        correct_pred_idx = 0
        if pred_probs[0] >= 0.5:
            num_p1_more_0_5 += 1
        # get sum of probabilities and number of times, the first prediction is correct
        if pred_labels[0] == label:
            sum_p1_correct += pred_probs[0]
            num_p1_correct += 1
            if pred_probs[0] >= 0.5:
                num_p1_more_0_5_correct += 1
        else:
            sum_p1_incorrect += pred_probs[0]
            num_p1_incorrect += 1
            try:
                correct_pred_idx = list(pred_labels).index(label)
                p_correct = pred_probs[correct_pred_idx]
                sum_p1_minus_p_correct += (pred_probs[0] - p_correct)
                num_p1_minus_p_correct += 1
            except ValueError:
                correct_pred_idx = -1

        topn_freq[correct_pred_idx] += 1

    topn_freq = list(map(lambda x: x/len(y_test), topn_freq))
    print("\n ---------- stats ----------\n")
    print("topn_freq: {}".format(topn_freq))
    print("average prediction probability that the first prediction is correct: {}".format(sum_p1_correct/num_p1_correct))
    print("average prediction probability that the first prediction is incorrect: {}".format(sum_p1_incorrect/num_p1_incorrect))
    print("average difference of the first prediction probability minus the correct probability {}".format(sum_p1_minus_p_correct/num_p1_minus_p_correct))
    print("percentage of times p1>=0.5 and first prediction is correct: {}".format(num_p1_more_0_5_correct/num_p1_more_0_5))
    print("\n")

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
acc3 = 0
acc5 = 0
for run in range(0, k):
    print("Run: {}".format(run+1))
    featurizer_tf = FeatureExtractor(mode="tfidf")
    featurizer_emb = FeatureExtractor(mode="word-embeddings")
    classifier = ClassifierLinearSVM(cv=cv)

    train_df, test_df = parser.get_test_train_cv_split(template_df, "template_formula", min_samples=min_samples_per_label)
    print("number of samples after prunning for run number {}: {}".format(run+1, len(train_df) + len(test_df)))
    print("number of unique labels for run number {}: {}".format(run+1, len(train_df.drop_duplicates(subset="template_formula"))))
    print("No intersection of train, test set: {}".format(len(pd.merge(train_df, test_df, how="inner", on=["sent", "claim"])) == 0))

    X_train = parser.get_feature_union_train(train_df, tok_driver, featurizer_emb, featurizer_tf)
    X_test = parser.get_feature_union_test(test_df, tok_driver, featurizer_emb, featurizer_tf)
    y_train = list(train_df["template_formula"])
    y_test = list(test_df["template_formula"])
    model = classifier.train(X_train, y_train)
    
    # top3
    print("-----top3-----\n")
    preds3, step_acc3 = classifier.get_pred_and_accuracy(X_test, y_test, topn=3)
    get_stats(preds3, y_test, 3)
    print_predictions(test_df, preds3)
    print("accuracy for run {} is: {}".format(run+1, step_acc3))
    acc3 += step_acc3
    # top5
    print("-----top5-----\n")
    preds5, step_acc5 = classifier.get_pred_and_accuracy(X_test, y_test, topn=5)
    get_stats(preds5, y_test, 5)
    print_predictions(test_df, preds5)
    print("accuracy for run {} is: {}".format(run+1, step_acc5))
    acc5 += step_acc5
    # crowdworker = Crowdworker(preds3, test_df, "template_formula", 1, 2)
    # crowdworker.get_ask_cost_for_all_templates()
    # crowdworker.get_overall_cost_exp1(prob_thres=0.5)
    # crowdworker.get_overall_cost_exp1(prob_thres=0.4)
    # crowdworker.get_overall_cost_exp2()

acc3 = float(acc3/k)
acc5 = float(acc5/k)

print("accuracy for {} number of runs for template formulas {} for topn = {}".format(k, acc3, 3))
print("accuracy for {} number of runs for template formulas {} for topn = {}".format(k, acc5, 5))



