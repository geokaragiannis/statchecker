import re
from itertools import permutations

class Formula_Calculator:
    
    """

    Args
    ====

    -formula: string formula 


    Instance Variables
    ==================

    -generic_variables: generic variables of the formula
    -num_vars= number of generic variables
    -year_variables= year generic variables
    -num_year_variables: number of year generic variables
    -formula: string formula adapted for python
    -excel_formula: string excel formula




    Methods
    =======

    -verify_formula:

     Args:
    -domains: Domain cell objects
    -true_value: the actual true value

    ################################################
    # TODO: Implement verification for near values.#
    ################################################
     Returns: set of variable assignments if any

    """

    @staticmethod
    def fix_formula(formula):
        if(formula=='(a/b)^(1/(y1-y2))-1'):
            formula='POWER(a/b,1/(y1-y2))-1'
        return formula.replace('POWER','pow')
    
    
    def __init__(self,formula):
        self.generic_variables=re.findall("([a-xz])",formula) #assuming y will neber be a generic variable
        self.num_vars=len(self.generic_variables)
        self.year_variables=re.findall("y\d+",formula)
        self.num_year_variables=len(self.year_variables)
        self.formula=self.fix_formula(formula)
        self.excel_formula=formula
        self.solutions=[]
            
    def verify_formula(self,domains,true_value):
        possible_solutions=[]
        domains=[x for x in domains if x.domain_value]
        prms=list(permutations(domains,self.num_vars))
        years=list(set([x.year for x in domains if x.year]))
        year_prms=list(permutations(years,self.num_year_variables))
        for prm in prms:
            
            f=self.formula
            
            for i in range(self.num_vars):      
                f=f.replace(self.generic_variables[i],str(prm[i].domain_value))
            #print(f)
            intermed_f=f
            for y_prm in year_prms:
                f=intermed_f
                for j in range(self.num_year_variables):
                    f=f.replace(self.year_variables[j],str(y_prm[j]))
                #print(f)


                try:
      
                    if(abs((eval(f)-true_value)/true_value)<0.01):
                            self.solutions.append(prm)
                            possible_solutions.append({**{'generic_variables':dict(zip(self.generic_variables,    \

                            [{'file':x.file,'tab':x.sheetname, \
                              'row_index':x.row_index,\
                              'year':x.year,'value':x.domain_value} 
                              for x in prm]

                                      ))},**{'formula':self.excel_formula}, \
                                          **{'year_variables':dict(zip(self.year_variables,list(y_prm)))}})

                except:
                    pass
        return possible_solutions
