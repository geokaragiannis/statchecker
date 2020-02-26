import math
from scipy.stats import entropy
class Property:
    def __init__(self, property_name, confidence=1.0, candidate_values=[], claim=None, task=None):
        # file, tab, row_index, ...
        self.property_name = property_name
        self.confidence = confidence
        # the claim this property refers to
        self.claim = claim
        # the values that this property can take (output of the classifiers)
        self.candidate_values = candidate_values
        # entropy of the topn predictions
        self.entropy = None
        self.topn = None
        # ClassificationTask object
        self.task = task
        # a single string that constitutes the correct value
        self.ground_truth = None
        # -1 if the property was derived. Otherwise keep track of the verification position
        self.verified_index = None
        # True if the property will be included in the crowdsourcing questions
        self.ask = False

    def set_entropy(self):
        probs = [v.prob for v in self.candidate_values]
        ent = entropy(probs, base=10)
        # making sure maximum entropy we get is 1 (which is the case unless we have arithmetic miscalc.)
        self.entropy = min(1, ent)
