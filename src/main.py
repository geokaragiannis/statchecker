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

claim1 = Claim(26000)
claim2 = Claim(0.45)
claims = [claim1, claim2]

queries_dict = query_execution_obj.get_queries_from_claims(claims)


num_generated_queries = 0
for key, value in queries_dict.items():
    num_generated_queries += len(value)
    logger.info("Generated {} queries for claim {}".format(len(value), key.claim_value))
    logger.info("Matched queries for claim {} :".format(key.claim_value))
    for query in value:
        logger.info("\t {}".format(query))
    print("\n\n")

logger.info("Generated {} matched queries".format(num_generated_queries))
