class QueryGenerator:


    """

    Args
    ====

    -solutions: set of possible solutions


    Instance Variables
    ==================

    -solutions: set of possible solutions


    Methods
    =======

    -generate_queries: 

     Returns:  list of query tuples, first query is the fully nested query, second query is the query with replaced values

    """
      

    def __init__(self,solutions):
        self.solutions=solutions
        
    def generate_queries(self,formula2query_dict):
        
        queries=[]
        
        for sol in self.solutions:
            lu_query_dict={}
            value_dict={}
            formula=sol['formula']
            gvs=sol['generic_variables']
            yvs=sol['year_variables']
            for key in gvs :
                
                gv_dict=gvs[key]
                lu_query_dict[key]='SELECT '+'Y'+str(gv_dict['year'])+" FROM df"+' WHERE RowIndex='+ \
                     "'"+gv_dict['row_index']+'\''
                value_dict[key]=str(gv_dict['value'])
                
            
            query_formula=formula2query_dict[formula]
            query_value=query_formula
            
            for key in lu_query_dict:
                query_formula=query_formula.replace('('+key+')','('+lu_query_dict[key]+')')
            for key in yvs:
                query_formula=query_formula.replace(key,str(yvs[key]))
                
            for key in value_dict:
                query_value=query_value.replace('('+key+')','('+value_dict[key]+')')
            
            for key in yvs:
                query_value=query_value.replace(key,str(yvs[key]))
            if(formula=='a'):
                query_formula=query_formula[1:-1]
                query_value=query_value[1:-1]
                
            queries.append([query_formula,query_value])
        return queries
    
	    

            
