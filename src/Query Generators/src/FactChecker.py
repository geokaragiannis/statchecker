from src.Book_FactChecker import Book_FactChecker
from src.QueryGenerator import QueryGenerator
from src.TableBuilder import TableBuilder

class FactChecker:

    """
    Args
    ====
    -classifier_output: lists of files,row indices, years, and operations
                     Example:classifier_output={'files': ['/home/barbacou/Desktop/Fact Checking/Supervised Learning/World_Output_2018_NPS.xlsx'],
	                                       'row_indices': ['TPEDrenew', 'PGElecDemand', 'Prod_Steel'],
                                               'years': [2040, 2016, 2017],
                                               'formulas': ['a', 'a+b', 'POWER(a/b,1/(y2-y1))-1']}


    Instance Variables
    ==================

    -files: list of files
    -row_indices: list of row indices
    -years: list of years
    -domains: dictionary of row indices and years
    -formulas: list of formulas


    Methods
    =======

    -FactCheck:

     Args:
    -true_value: the true value to check

     Returns:  list of possible solutions if any

    -generate_queries:

     Args:
    -solution: list of all possible assignments of generic variables
    -formula2query: dictionary to translate from formula to query

     Returns: list of possible queries

    """


    def __init__(self,classifier_output):
        self.files=classifier_output['files']

        self.row_indices=classifier_output['row_indices']
        self.years=classifier_output['years']
        self.domains={'row_indices':self.row_indices,'years':self.years}
        self.formulas=classifier_output['formulas']
        self.bfcs={k:None for k in self.files}
        self.solutions=None
    def FactCheck(self,true_value):
        solutions=[]
        for file in self.files:
            bfc=Book_FactChecker(file,self.domains)
            for formula in self.formulas:
                sols=bfc.book_FactCheck(formula,true_value)
                solutions.append(sols)
            self.bfcs[file]=bfc
        solutions_flatenned=[xxx  for x in solutions for xx in x for xxx in xx ]

        return solutions_flatenned
    def generate_queries(self,solutions,formula2query):
        self.solutions=solutions
        QG=QueryGenerator(self.solutions)
        return QG.generate_queries(formula2query)
    
    def execute_query(self,query):
        file=self.solutions[0]['generic_variables']['a']['file']
        tab=self.solutions[0]['generic_variables']['a']['tab']
        bfc=self.bfcs[file]#.sfcs[tab]
        tb=TableBuilder(bfc)
        tb.build_table()
        return tb.query_table(query)

    def release_mem(self):
        for x in self.bfcs:
            self.bfcs[x].book.book.close()
            del self.bfcs[x].book.book





