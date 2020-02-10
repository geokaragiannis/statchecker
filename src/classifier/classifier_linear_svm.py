import numpy as np

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle

from src import helpers

class ClassifierLinearSVM:
    def __init__(self, task, cv=3):
        self.cv = cv
        self.model = None
        self.calibrated_model = None
        # name of the property
        self.task = task
        self.config = helpers.load_yaml("src/config.yml")

    def train(self, X_train, y_train):
        if self.task.label_task == "single-label":
            self.model = LinearSVC(dual=True, max_iter=3000)
        elif self.task.label_task == "multi-label":
            self.model = OneVsRestClassifier(LinearSVC(dual=True, max_iter=3000))

        self.model.fit(X_train, y_train)
        self.calibrated_model = CalibratedClassifierCV(base_estimator=self.model, cv="prefit")
        self.calibrated_model.fit(X_train, y_train)
        return self.calibrated_model
    
    @staticmethod
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

    def predict_utt_top_n(self, featurized_utt, n=3):
        """
        predict the topn predictions along with the confidence probability for each one.
        Note that model.classes_ contains the trained labels in alphabetical order. Here, we sort the
        confidences together with the labels, and return the top3 from this sorted order
        Args:
            featurized_utt (str): featurized and tokenized single utterance
        Returns:
            One list of strings and one list of floats
        """
        raw_confidences = self.calibrated_model.predict_proba(featurized_utt)[0]
        # indices of sorted confidences from high to low confidence
        sorted_conf_idx = np.argsort(raw_confidences)[::-1][:n]
        labels = np.take(self.calibrated_model.classes_, sorted_conf_idx)
        confidences = np.take(raw_confidences, sorted_conf_idx)
        scaled_confidences = self._linear_scale_confidence(confidences)

        return labels, scaled_confidences

    def predict_batch_top_n(self, X_test, topn=5):
        """
        predict the topn predictions for the whole batch.
        Returns a list of tuples, where each tuple is a list
        """
        return [self.predict_utt_top_n(test.reshape(1, -1), n=topn) for test in X_test]

    def get_pred_and_accuracy(self, X_test, y_test, topn=5):
        """
        Returns predictions and accuracy for the test set
        """
        predictions = [self.predict_utt_top_n(test.reshape(1, -1), n=topn) for test in X_test]
        num_correct = 0
        for test, pred in zip(y_test, predictions):
            topn_list = pred[0]
            if test in topn_list:
                num_correct += 1
        return predictions, num_correct/len(y_test)

    def load(self):
        self.calibrated_model = helpers.load_model_from_dir(self.config["models_dir"], self.task.classifier_name)

    def export(self):
        helpers.save_model_to_dir(self.config["models_dir"], self.task.classifier_name, self.calibrated_model)