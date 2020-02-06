from src.pipeline.classification_step import ClassificationStep

DATA_PATH = "data/claims_01-23-2020/"

classification_pipeline = ClassificationStep(DATA_PATH)

# classification_pipeline.complete_df.to_csv("data/complete.csv", index=False)
# df_train.to_csv("data/train_demo.csv", index=False)
# df_val.to_csv("data/val_demo.csv", index=False)
# df_test.to_csv("data/test_demo.csv", index=False)
print("len complete df: ", len(classification_pipeline.complete_df))

# for task_name, task in classification_pipeline.classification_tasks_dict.items():
#     print("task name: ", task_name)
#     df_train, df_val, df_test = classification_pipeline.get_train_val_test_splits(
#         classification_pipeline.complete_df, task, val_frac=1.0)
#     print("len train: ", len(df_train))
#     print("len val: ", len(df_val))
#     print("len test: ", len(df_test))
#     min_df = classification_pipeline.create_min_samples_dataset(task)
#     print("len min_df: ", len(min_df))
    
# min_df.to_csv("data/min_df.csv", index=False)

# classification_pipeline.train()

classification_pipeline.load_models()
test_df = classification_pipeline.load_test_df()
print(len(test_df))

for task_name, task in classification_pipeline.classification_tasks_dict.items():
    print("task name: ", task_name)
    print("feat_tf trained: {}, feat_emb trained: {}, classifier trained: {}".format(
        task.featurizer_tf is not None, task.featurizer_emb is not None, 
        task.classifier is not None
    ))
