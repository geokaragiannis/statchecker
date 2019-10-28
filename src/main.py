import logging

from src.query_generation import QueryGeneration
from src.query_execution import QueryExecution
from src.claim import Claim

from src import helpers

helpers.set_up_logging()
logger = logging.getLogger(__name__)

TABLES_PATH = "data/demo_tables/"

tables = helpers.get_tables_list(TABLES_PATH)

query_generation_obj = QueryGeneration([tables[2]])

query_generation_obj.generate_queries()

logger.info("num queries: {}".format(len(query_generation_obj.candidate_queries)))
print("qqq: ", query_generation_obj.candidate_queries[0])

not_exec_queries = 0
for query in query_generation_obj.candidate_queries:
    if query.not_executable:
        print(query)
        not_exec_queries += 1

logger.info("Not executable queries: {}".format(not_exec_queries))

query_execution_obj = QueryExecution(query_generation_obj)

claim = Claim(26000)

queries1 = query_execution_obj.get_queries_from_claim(claim)

logger.info("Generated {} matched queries".format(len(queries1)))

for i, q in enumerate(queries1):
    logger.info("Matched queries: {}".format(q))