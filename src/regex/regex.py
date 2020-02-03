class Regex:
    def __init__(self):
        self.formula_regex =  "[A-Z]+[0-9]+"
        self.other_file_ref_regex = "\'.*\'"
        self.str_const_regex = "(?<![0-9])[a-z]+(?![0-9])"
        self.if_regex = "(?<=IF\()[^\"]*(?=,)"
	self.year_regex="^[12][1209]\d{2}$"
