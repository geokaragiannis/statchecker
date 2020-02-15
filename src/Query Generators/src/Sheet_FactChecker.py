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

        df_tuples=list(set([(x.row_index,x.year,x.domain_value) for x in self.domain_cells if x.domain_value and isinstance(x.domain_value,float)]))
      #  print(df_tuples)
        df_tuples=sorted(df_tuples)


 
        #remove increase/growth tuples
        tuples_values=[x[2] for x in df_tuples]
        tuples2remove=list(set([x for x in tuples_values if tuples_values.count(x)>1]))
        #print("tUPLES@remove",tuples2remove)
        self.domain_cells=[x for x in self.domain_cells if x.domain_value not in tuples2remove]





        
        self.operation=operation
        
        


        
