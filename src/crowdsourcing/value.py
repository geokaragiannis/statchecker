class Value:
    def __init__(self, value, prob, property_obj):
        self.value = value
        self.prob = prob
        self.property = property_obj
        self.exclude = False