"""
Class which holds information about a claim value or a result of a query
"""

import math
import numpy as np


class Value:

    def __init__(self, raw_value):
        """

        :param raw_value: The initial value passed. E.g: it could be 11.0
        """
        self.raw_value = raw_value
        # float, string, int, etc.
        self.instance = None
        self.num_digits = None
        self.num_decimal_digits = None
        self.new_value = self.raw_value
        self._populate_value()

    def _populate_value(self):
        """
        Populates the instance attributes
        :return: None
        """

        if isinstance(self.new_value, str):
            try:
                self.new_value = float(self.new_value)
                self.instance = float
            except ValueError:
                self.instance = str

        if isinstance(self.new_value, int):
            self.instance = int

        if isinstance(self.new_value, float):
            if self.new_value.is_integer():
                self.instance = int
                self.new_value = int(self.new_value)
            else:
                self.instance = float

        self._get_num_digits()

    def _get_num_digits(self):
        if self.instance == int:
            self.num_digits = len(str(self.new_value))

        if self.instance == float:
            s = str(self.new_value).split(".")
            self.num_digits = len(s[0])
            self.num_decimal_digits = len(s[1])

    def round(self):
        if self.instance == int:
            return self._round_int(self.new_value, self.num_digits)
        if self.instance == float:
            # create an int such that it can be rounded
            int_value = self.new_value * (10**self.num_decimal_digits)
            rounded_list = self._round_int(int_value, self.num_digits+self.num_decimal_digits)
            return list(map(lambda x: x/(10**self.num_decimal_digits), rounded_list))
        else:
            return []

    def _round_int(self, value, num_digits):
        """
        rounds self.new_value according to its number of digits
        :return (list): rounded numbers for different rounded numbers
        """
        if num_digits is None:
            return []
        if num_digits == 1:
            return [value]
        if num_digits == 2:
            return [value, self._round_int_helper(value, n=10)]
        if num_digits == 3:
            return [value, self._round_int_helper(value, n=10), self._round_int_helper(value, n=100)]
        else:
            tens_list = [10*(10**i) for i in np.arange(num_digits-2)]
            round_list = [self._round_int_helper(value, n=n) for n in tens_list]
            return [value] + round_list

    @staticmethod
    def _round_int_helper(value, n=10):
        """
        rounds x to the nearest n
        :param value (int): value to be rounded
        :param n (int): the nearest power of 10
        :return (int): the rounded number
        """
        res = math.ceil(value / n) * n
        if (value % n < n / 2) and (value % n > 0):
            res -= n
        return res

    def __str__(self):
        return "raw value: {}, new value: {}, instance: {}, num_digits: {}, num_decimals: {}".\
            format(self.raw_value,
                   self.new_value,
                   self.instance,
                   self.num_digits,
                   self.num_decimal_digits)
