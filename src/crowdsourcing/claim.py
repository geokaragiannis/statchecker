import src.helpers as helpers
import numpy as np


class Claim:
    def __init__(self, sent, claim, available_properties, ver_cost=1, der_cost=5):
        self.sent = sent
        self.claim = claim
        self.expected_cost = 0.0
        self.real_cost = 0.0
        # list of Property objects, which are relevant for this claim
        self.available_properties = available_properties
        self.verification_cost = ver_cost
        self.derivation_cost = der_cost
    

    def get_optimal_property_order(self):
        """
        Go through each singleton property, and compute the optimal
        property order for this singleton as a starting property.
        Return the property order with the minumum cost
        
        Returns:
            [type] -- [description]
        """
        best_property_order = None
        min_expected_verification_cost = np.inf
        for candidate_property in self.available_properties:
            candidate_order, candidate_cost = self.get_property_order(candidate_property)
            if candidate_cost < min_expected_verification_cost:
                best_property_order = candidate_order
                min_expected_verification_cost = candidate_cost
        return (best_property_order, min_expected_verification_cost)


    def get_property_order(self, candidate_property):
        """
        Starting from the candidate property (singleton property) we derive the best property order.
        We stop when we have visited all properties.
        Arguments:
            candidate_property {Property obj.} -- [the singleton property from which we start from]
        Returns:
            [tuple of list and float] -- [return the ordered list of properties and the total
                                          expected verification cost for that order]
        """
        existing_ordered_properties = [candidate_property]
        remaining_candidates = list(set(self.available_properties) - set(existing_ordered_properties))
        property_order_cost = 0.0
        while len(remaining_candidates) > 0:
            next_best_property, next_best_cost = self.find_next_best_property(existing_ordered_properties, remaining_candidates)
            existing_ordered_properties.append(next_best_property)
            property_order_cost += next_best_cost
            remaining_candidates =  list(set(self.available_properties) - set(existing_ordered_properties))

        return (existing_ordered_properties, property_order_cost)

    def find_next_best_property(self, existing_ordered_properties, remaining_property_candidates):
        """
        Find the next property that can be added to the existing optimal order.
        We choose a property from the remaining candidates, which will minimize
        the expected cost, when added.
        Arguments:
            existing_ordered_properties {list of Property obj.} -- [properties that have already been picked]
            remaining_property_candidates {list of Property obj.} -- [properties that remain to be picked]
        Returns:
            [tuple] -- [best property and minumim expected cost]
        """
        best_property = None
        min_expected_verification_cost = np.inf
        # list of properties that we can choose from
        # remaining_candidates = list(set(self.available_properties) - set(existing_ordered_properties))
        for candidate_property in remaining_property_candidates:
            for candidate_value in candidate_property.candidate_values:
                expected_verification_cost = self.get_expected_verification_cost(existing_ordered_properties, candidate_value)
                if expected_verification_cost < min_expected_verification_cost:
                    min_expected_verification_cost = expected_verification_cost
                    best_property = candidate_property
        return (best_property, min_expected_verification_cost)
                    

    def get_expected_verification_cost(self, existing_ordered_properties, candidate_value):
        """
        Given a list of properties chosen already and a new value we want to estimate
        the cost for we do the following:
            1. For all values in the existing properties
            2. Get values that make the candidate_value impossible
            3. Find the probability that the canidate_value can be picked
            4. Calculate expected cost
        Arguments:
            existing_ordered_properties {list of property obj.} -- [properties that have already been picked]
            candidate_value {Value obj.} -- [Value we want to add]
        
        Returns:
            [float] -- [Expected verification cost]
        """
        excluding_values = self.get_excluding_values(existing_ordered_properties, candidate_value)
        prob_not_excluded = 1.0
        # assuming independence
        for excluding_value in excluding_values:
            prob_not_excluded *= (1-excluding_value.prob)

        expected_verification_cost = prob_not_excluded * self.verification_cost
        return expected_verification_cost


    def get_excluding_values(self, existing_ordered_properties, candidate_value):
        """
        Return a list of Values which exclude the candidate value. 
        I.e find all the values from the already picked properties, which make
        the candidate_value impossible
        
        Arguments:
            existing_ordered_properties {list of Property obj.} -- [properties that have already been picked]
            candidate_value {Value obj} -- [the value whose property we want to add]
        Returns:
            list of Value objects, which exclude the candidate value
        """
        # list of Value objects
        excluding_values = []
        for prop in existing_ordered_properties:
            for val in prop.candidate_values:
                possible_values = helpers.get_possible_values(val, candidate_value)
                # if the candidate value is not part of the possible values that we can get
                # then the current value excludes it.
                if not (val.value in possible_values):
                    excluding_values.append(val)
        print(len(excluding_values))
        return excluding_values

    