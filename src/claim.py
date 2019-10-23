"""
Class that represents a Claim
"""
from src.value import Value


class Claim:

    def __init__(self, claim_value, claim_text=None):
        self.claim_value = Value(claim_value)
        self.claim_text = claim_text