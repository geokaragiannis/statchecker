import logging

from src.query_generation import QueryGeneration
from src.query_execution import QueryExecution
from src.claim import Claim

from src import helpers

helpers.set_up_logging()
helpers.get_nlp()
logger = logging.getLogger(__name__)

TABLES_PATH = "data/demo_tables/"

tables = helpers.get_tables_list(TABLES_PATH)

query_generation_obj = QueryGeneration(tables)

query_generation_obj.generate_queries()

logger.info("num queries: {}".format(len(query_generation_obj.candidate_queries)))
print("qqq: ", query_generation_obj.candidate_queries[0])

query_execution_obj = QueryExecution(query_generation_obj)

claim1 = Claim(37000, claim_text="While  total electricity generated increases in the Sustainable Development Scenario by nearly 45% to reach 37,000 Twh by 2040 the share of renewables in generation grows more than two and a half times to 66%.")
claim2 = Claim(0.45, claim_text="While  total electricity generated increases in the Sustainable Development Scenario by nearly 45% to reach 37,000 Twh by 2040 the share of renewables in generation grows more than two and a half times to 66%.")
claim3 = Claim(500, claim_text="The average carbon intensity of electricity generated continues its decline from around 500 g CO2/kWh today to around 70 g CO2/kWh in 2040 ")
claim4 = Claim(70, claim_text="The average carbon intensity of electricity generated continues its decline from around 500 g CO2/kWh today to around 70 g CO2/kWh in 2040 ")
claim5 = Claim(35.8, claim_text="CO2 emissions are set to rise gradually to 35.8 gigatonnes (Gt) in 2040.")
claims = [claim1, claim2, claim3, claim4, claim5]

query_execution_obj.get_queries_from_claims(claims)

not_exec_queries = 0
for query in query_generation_obj.candidate_queries:
    if query.not_executable:
        print("Not exec: ", query)
        not_exec_queries += 1

logger.info("Not executable queries: {}".format(not_exec_queries))


num_generated_queries = 0
for claim in claims:
    num_generated_queries += len(claim.queries_list)
    sorted_list_queries = sorted(claim.queries_list, key=lambda x: (len(x[0].expr.cols_dict.keys()), -x[1]))
    logger.info("For Claim: {} \n\n\n\n".format(claim))
    for query, sim in sorted_list_queries:
        logger.info("\t {} --sim = {}".format(query, sim))
    print("------------\n---\n--------------\n\n\n\n\n")

# for key, value in queries_dict.items():
#     num_generated_queries += len(value)
#     logger.info("Generated {} queries for claim {}".format(len(value), key))
#     logger.info("Matched queries for claim {} :".format(key.claim_value))
#     for query in sorted(value, key=lambda x: len(x.query)):
#         logger.info("\t {}".format(query))
#     print("\n\n")

logger.info("Generated {} matched queries total".format(num_generated_queries))
