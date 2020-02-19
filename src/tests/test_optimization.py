from src.pipeline.classification_step import ClassificationStep
from src.pipeline.optimization_step import OptimizationStep

DATA_PATH = "data/claims_01-23-2020/"
class_pipeline = ClassificationStep(DATA_PATH, simulation=False, min_samples=1) 

complete_df = class_pipeline.parser.get_complete_df()
class_pipeline.parser.set_task_values(complete_df)

optim_step = OptimizationStep(class_pipeline)

all_instances_dict = optim_step.get_all_instances_of_var()

print(len(all_instances_dict))

print("len file: ", len(all_instances_dict["file"]))
print("len row: ", len(all_instances_dict["row_index"]))
print("\n")
print("len col: ", len(all_instances_dict["column"]))