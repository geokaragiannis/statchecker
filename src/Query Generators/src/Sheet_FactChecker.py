from src.Domain import Domain
from itertools import product


"""

Args
====

-sheet: Sheet object
-row_indices: list of row indices
-years: list of years:
-operation: formula 





Instance Variables
==================

-sheet: Sheet object
-file: file name
-row_indices: list of row indices
-years: list of years:
-domains: dict of row indices and years
-domain_tuples: tuples of row indices and years
-domains: Domain objects of the domain tuples 
-domain_cells: Domain cell objects 
-operation: formula

"""







class Sheet_FactChecker:
    
    def __init__(self,sheet,row_indices,years,operation):
        """years should be integers"""
        
        
        
        self.sheet=sheet
        self.file=self.sheet.file
        
        self.row_indices=row_indices
        self.years=years

        self.domains={'row_indices':self.row_indices,'years':self.years}
        self.domain_tuples=list(product(*(self.domains[key] for key in self.domains)))
        self.domains=[Domain(self.sheet,x) for x in self.domain_tuples]
        
        self.domain_cells=[xx for x in self.domains for xx in x.domain_cells]
        
        self.operation=operation
        
        


        
