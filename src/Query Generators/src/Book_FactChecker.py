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



        
    def book_FactCheck(self,formula,true_value):


	#change string formula to Formula_Calculator object        
        self.formula=Formula_Calculator(formula)
        
        file_solutions=[]
        
        for sheetname in self.book.sheetnames:
         #   print(sheetname)

            sheet=self.book.get_sheet(sheetname)


            # Fact Check object for each Sheet
            sfc=Sheet_FactChecker(sheet,self.row_indices,self.years,self.formula)

            # brute force domains to find value equal to true_value
            sheet_solution=self.formula.verify_formula(sfc.domain_cells,true_value)


            file_solutions.append(sheet_solution)

        return file_solutions
