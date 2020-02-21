class Property:
    def __init__(self, property_name, confidence=1.0, candidate_values=[], claim=None, task=None):
        # file, tab, row_index, ...
        self.property_name = property_name
        self.confidence = confidence
        # the claim this property refers to
        self.claim = claim
        # the values that this property can take (output of the classifiers)
        self.candidate_values = candidate_values
        # ClassificationTask object
        self.task = task
        # a single string that constitutes the correct value
        self.ground_truth = None
        # -1 if the property was derived. Otherwise keep track of the verification position
        self.verified_index = None
        # True if the property will be included in the crowdsourcing questions
        self.ask = False
