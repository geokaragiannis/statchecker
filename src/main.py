import logging

from src.query_generation import QueryGeneration
from src.query_execution import QueryExecution
from src.claim import Claim

from src import helpers

helpers.set_up_logging()
logger = logging.getLogger(__name__)

TABLES_PATH = "data/demo_tables/"

tables = helpers.get_tables_list(TABLES_PATH)

query_generation_obj = QueryGeneration(tables)

query_generation_obj.generate_queries()

logger.info("num queries: {}".format(len(query_generation_obj.candidate_queries)))

query_execution_obj = QueryExecution(query_generation_obj)

claim = Claim(45)

queries1 = query_execution_obj.get_queries_from_claim(claim)

logger.info("Generated {} canidate queries".format(len(queries1)))
for i, q in enumerate(queries1):
    logger.info(q)