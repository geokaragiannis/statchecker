class Regex:
    def __init__(self):
        self.formula_regex =  "[A-Z|$G$$A$]+[0-9]+"
        self.other_file_ref_regex = "\'[a-zA-z0-9.]*\'!"
        self.str_const_regex = "(?<![0-9])[a-z]+(?![0-9])"
        self.if_regex = "(?<=IF\()[^\"]*(?=,)"
