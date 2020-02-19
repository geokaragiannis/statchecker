from src.pipeline.classification_step import ClassificationStep
from src.pipeline.optimization_step import OptimizationStep

DATA_PATH = "data/claims_01-23-2020/"
class_pipeline = ClassificationStep(DATA_PATH, simulation=False, min_samples=1) 

complete_df = class_pipeline.parser.get_complete_df()
class_pipeline.parser.set_task_values(complete_df)

optim_step = OptimizationStep(class_pipeline)

print("init number of formulas: ", optim_step.init_number_formulas)