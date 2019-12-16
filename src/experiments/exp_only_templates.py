"""
This experiment only uses a dataset for template predictions
"""

"""
Create a dataset, which contains sentence, claim pairs, which have a valid formula and a valid row_index
"""
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import random

from src.parser.dataset_parser import DatasetParser
from src.tokenizer.tokenizer_driver import TokenizerDriver
from src.featurizer.feature_extractor import FeatureExtractor


DATA_PATH = "data/main_annotated_dataset_12-16-2019.csv"


def join_formula_join_df(formula_df, lookup_df):
    joined_df = pd.merge(formula_df, lookup_df, on=["sent", "claim"], how="left")
    joined_df = joined_df[["sent", "claim", "formula_x", "template_formula", "row_index_y", "file_x"]]
    joined_df = joined_df[joined_df.row_index_y.notnull()]
    joined_df = joined_df.rename(columns={"formula_x": "formula", "row_index_y": "row_index", "file_x": "file"})
    unique_formula_df = joined_df.drop_duplicates(subset=["sent", "claim", "template_formula"])
    unique_formula_lookup_df = unique_formula_df.drop_duplicates(subset=["sent", "claim", "row_index"])
    return unique_formula_lookup_df


def create_cv_dataset(df, label_column, cv=3):
    """
    Only keep labels which appear more than cv times in the df
    """
    return df.groupby(label_column).filter(lambda x: len(x) > cv)


def print_stats(df):
    print("Stats of main dataset, to be used for training")
    print("number of samples: {}".format(len(df)))
    print("number of formula labels: {}".format(len(df.drop_duplicates(subset="template_formula"))))


def tokenize(tokenizer, data_list):
    return tokenizer.tokenize_claims(data_list)


def featurize(featurizer, tokenized_data_list):
    return featurizer.featurize_claims(tokenized_data_list)


def get_features_from_list(tokenizer, featurizer, data_list):
    tokenized_list = tokenize(tokenizer, data_list)
    features = featurize(featurizer, tokenized_list)
    return features


def get_features_union(features_s, features_c):
    if isinstance(features_c, scipy.sparse.csr.csr_matrix):
        features_c = features_c.toarray()
    if isinstance(features_s, scipy.sparse.csr.csr_matrix):
        features_s = features_s.toarray()
    return np.concatenate(features_s, features_c)


def _linear_scale_confidence(confidences):
    """
    return the ratio of prob according to the sum of top n probabilities for the predicted intents.
    if probs = [p1, p2, p3] then the return probabilities will be scaled as
    [p1/sum(p1,p2,p3), p2/sum(p1,p2,p3), p3/sum(p1,p2,p3)]
    Args:
        confidences: probabilities of intents
    Returns:
        numpy array: the scaled confidences
    """
    s = np.sum(confidences)
    return confidences/s


def predict_utt_top_n(mod, featurized_utt, n=3):
    """
    predict the top3 intents along with the confidence probability for each one.
    Note that model.classes_ contains the trained labels in alphabetical order. Here, we sort the
    confidences together with the labels, and return the top3 from this sorted order
    Args:
        featurized_utt (str): featurized and tokenized single utterance
    Returns:
        One list of strings and one list of floats
    """
    raw_confidences = mod.predict_proba(featurized_utt)[0]
    # indices of sorted confidences from high to low confidence
    sorted_conf_idx = np.argsort(raw_confidences)[::-1][:n]
    labels = np.take(mod.classes_, sorted_conf_idx)
    confidences = np.take(raw_confidences, sorted_conf_idx)
    scaled_confidences = _linear_scale_confidence(confidences)

    return labels, scaled_confidences


def get_test_train_split(features, labels, test_size=0.20, random_state=2):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def fit_classifier(X_train, y_train, cv):
    model = LinearSVC(dual=False, max_iter=3000)
    model.fit(X_train, y_train)
    final_model = CalibratedClassifierCV(base_estimator=model, cv="prefit")
    final_model.fit(X_train, y_train)
    return final_model


def get_pred_and_accuracy(model, X_test, y_test, topn=5):
    predictions = [predict_utt_top_n(model, test.reshape(1, -1), n=topn) for test in X_test]
    num_correct = 0
    for test, pred in zip(y_test, predictions):
        topn_list = pred[0]
        if test in topn_list:
            num_correct += 1
    return predictions, num_correct/len(y_test)


def get_accuracy_from_k_runs(features, labels, cv, topn=5, k=3):
    acc = 0
    for run in range(0, k):
        print("Run number: {}".format(run))
        random_state = random.randint(0, 1000)
        X_train, X_test, y_train, y_test = get_test_train_split(features, labels, random_state=random_state)
        model = fit_classifier(X_train, y_train, cv)
        _, step_acc = get_pred_and_accuracy(model, X_test, y_test, topn=topn)
        acc += step_acc
    return float(acc/k)


parser = DatasetParser(DATA_PATH)
formula_df = parser.get_formula_df()
formula_df = formula_df.drop_duplicates(subset=["sent", "claim"]).reset_index(drop=True)
main_df = formula_df.drop_duplicates(subset=["sent", "claim"])
cv = 5
main_df = create_cv_dataset(main_df, "template_formula", cv=cv)
print_stats(main_df)

tok_driver = TokenizerDriver()
featurizer_tf = FeatureExtractor(mode="tfidf")
featurizer_emb = FeatureExtractor(mode="word-embeddings")

sents_list = list(main_df["sent"])
claims_list = list(main_df["claim"])

features_sents = get_features_from_list(tok_driver, featurizer_emb, sents_list)
features_claims = get_features_from_list(tok_driver, featurizer_tf, claims_list)

features_union = np.concatenate((features_sents, features_claims.toarray()), axis=1)

print("features extracted")
print("Sentence features shape: {}".format(features_sents.shape))
print("Claims features shape: {}".format(features_claims.shape))
print("Union features shape: {}".format(features_union.shape))

k = 3
# for formula predictions
print("Running classifier for template formulas")
print("cross validation for sigmoid training is {}".format(cv))
f_labels = list(main_df["template_formula"])
f_topn = 5
f_acc_k_runs = get_accuracy_from_k_runs(features_union, f_labels, cv, topn=f_topn)
print("accuracy for {} number of runs for template formulas {} for topn = {}".format(k, f_acc_k_runs, f_topn))
# fx_train, fx_test, fy_train, fy_test = get_test_train_split(features_union, f_labels)
#
# f_model = fit_classifier(fx_train, fy_train)
# f_preds, f_acc = get_pred_and_accuracy(f_model, fx_test, fy_test)
# print("accuracy for template formulas {}".format(f_acc))


