import json
from src.FactChecker import FactChecker

formula2query=json.load(open('src/formula2query.json'))



classifier_output={'files': ['files/World_Output_2018_NPS.xlsx'],
 'row_indices': ['TPEDrenew', 'PGElecDemand', 'Prod_Steel'],
 'years': [2040, 2016, 2017],
 'formulas': ['a', 'a+b', 'POWER(a/b,1/(y1-y2))-1']}


true_value=0.030407493463541435


fc=FactChecker(classifier_output)
sols=fc.FactCheck(true_value)



print("Query: "+ fc.generate_queries(sols,formula2query)[0][0])

print("Query with values: "+ fc.generate_queries(sols,formula2query)[0][1])


