from src.Book import Book
from src.Formula_Calculator import Formula_Calculator
from src.Sheet_FactChecker import Sheet_FactChecker


class Book_FactChecker:

    """

    Args
    ====

    -file_path: the file path

    -domains: dict  containing list of possible row indices and years
              Example: {'row_indices': ['TPEDrenew', 'PGElecDemand', 'Prod_Steel'],
                        'years': [2040, 2016, 2017]}



    Instance Variables
    ==================

    -book: Book object
    -file_path: the file path
    -row_indices: list of row indices
    -years: lost of years


    Methods
    =======

    -book_FactCheck: 

     Args: 


    -formula: the formula string used
              Example: "a+b"


    -true_value: The actual value to verify



     Returns:  list of possible solutions if any

    """


    def __init__(self,file_path,domains):
        
        self.book=Book(file_path)
        self.file_path=file_path
        
        self.row_indices=domains['row_indices']
        self.years=domains['years']
        self.sfcs={k:None for k in self.book.sheetnames}
        self.solutions=[]
       
    def book_FactCheck(self,formula,true_value):
        
        self.formula=Formula_Calculator(formula)
        
        file_solutions=[]
        self.domain_cells=[]
        for sheetname in self.book.sheetnames:
         #   print(sheetname)
            sheet=self.book.get_sheet(sheetname)
            sfc=Sheet_FactChecker(sheet,self.row_indices,self.years,self.formula)
            self.sfcs[sheetname]=sfc
            self.domain_cells.extend(sfc.domain_cells)

        sheet_solution=self.formula.verify_formula(self.domain_cells,true_value)
        file_solutions.append(sheet_solution)

        self.solutions=self.formula.solutions
        return file_solutions
