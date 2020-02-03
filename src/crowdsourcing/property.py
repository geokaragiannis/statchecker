class Property:
    def __init__(self, property_name, confidence=1.0, candidate_values=[], claim=None):
        # file, tab, row_index, ...
        self.property_name = property_name
        self.confidence = confidence
        # the claim this property refers to
        self.claim = claim
        # the values that this property can take (output of the classifiers)
        self.candidate_values = candidate_values