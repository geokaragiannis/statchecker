from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value

p1 = Property("file")
p2 = Property("tab")
p3 = Property("region")
p4 = Property("row_index")

v_file1 = Value("f1", 0.65, p1)
v_file2 = Value("f2", 0.25, p1)
v_file3 = Value("f3", 0.1, p1)
v_tab1 = Value("t1", 0.80, p2)
v_tab2 = Value("t2", 0.19, p2)
v_tab3 = Value("t3", 0.01, p2)
v_region1 = Value("r1", 0.55, p3)
v_region2 = Value("r2", 0.44, p3)
v_region3 = Value("r3", 0.01, p3)
v_row_idx1 = Value("i1", 0.95, p4)
v_row_idx2 = Value("i2", 0.06, p4)

p1.candidate_values = [v_file1, v_file2, v_file3]
p2.candidate_values = [v_region1, v_region2, v_region3]
p3.candidate_values = [v_tab1, v_tab2, v_tab3]
p4.candidate_values = [v_row_idx1, v_row_idx2]

claim = Claim("sentence1", "claim2", [p1,p2,p3,p4])

best_order, best_cost = claim.get_optimal_property_order()

print("ok")
print("best cost: ", best_cost)
for i, p in enumerate(best_order):
    print("index: {}, property: {}".format(i, p.property_name))