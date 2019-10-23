import os
from src.table import Table
from src.query_generation import QueryGeneration
from src.query_execution import QueryExecution
from src.claim import Claim

import pandasql as ps

TABLES = "data/demo_tables/"

table1_path = os.path.join(TABLES, "demo_table1.csv")
table2_path = os.path.join(TABLES, "demo_table2.csv")
table3_path = os.path.join(TABLES, "demo_table3.csv")
paths = [table1_path, table2_path, table3_path]

tables = [Table(table1_path, "table1"), Table(table2_path, "table2"), Table(table3_path, "table3")]

query_generation_obj = QueryGeneration(tables)

query_generation_obj.generate_queries()

print("num queries: ", len(query_generation_obj.candidate_queries))

query_execution_obj = QueryExecution(query_generation_obj)

claim = Claim(45)

queries1 = query_execution_obj.get_queries_from_claim(claim)

print(queries1)
if len(queries1) > 0:
    print(queries1[0].query)
# print(globals())