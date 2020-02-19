"""
This file reads the 40 claims intended for the user study, removes them from the training data and
trains a model on these remaining data. We output the predictions for the 40 claims in a csv
"""
import pandas as pd
from src.pipeline.classification_step import ClassificationStep
from src.pipeline.optimization_step import OptimizationStep
from src.parser.classification_task import ClassificationTask
from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value
import json

DATA_PATH = "data/claims_01-23-2020/"

def create_claims_from_df(test_df, class_pipeline):
    """
    return a list of Claim objects by populating them with the sentence and claim strings
    from the test_df. 
    """
    test_claims = []
    for idx, test_row in test_df.iterrows():
        available_properties = [Property(t.name, task=t) for t in 
                                list(class_pipeline.classification_tasks_dict.values())]
        available_properties = sorted(available_properties, key=lambda x: x.task.priority)
        test_claim = Claim(test_row["sent"], test_row["claim"], available_properties)
        # get_preds_from_claim(test_claim, class_pipeline)
        test_claims.append(test_claim)
    return test_claims

def get_preds_from_claim(claim, class_pipeline):
    """
    Get predictions for each property of the claim
    """
    not_found_hash = 0
    for prop in claim.available_properties:
        featurizer_tf = prop.task.featurizer_tf
        featurizer_emb = prop.task.featurizer_emb
        classifier = prop.task.classifier
        features = class_pipeline.get_feature_union([claim.sent], [claim.claim], class_pipeline.tok_driver, 
                                                        featurizer_emb, featurizer_tf, mode="test")
        pred_labels, pred_probs = classifier.predict_utt_top_n(features, n=5)
        if prop.property_name == "template_formula":
            values_list = [Value(label, prob, prop) for label, prob in zip(pred_labels, pred_probs)]
        else:
            hash_to_label_dict = prop.task.hash_to_label_dict
            # print(len(hash_to_label_dict))
            # print(hash_to_label_dict)
            
            values_list = [Value(hash_to_label_dict[str(int(label))], prob, prop) for label, prob in zip(pred_labels, pred_probs)]
            
        prop.candidate_values = values_list

def get_preds_from_df(df, class_pipeline):
    """
    Returns the predictions for a given dataframe for each classification task
    
    Arguments:
        df {DataFrame} -- Sents and claims to get preds from
    Returns:
        np array where each column [i, j] is the pred prob of claim i for task j
    """
    sents = df["sent"]
    claims = df["claim"]

    featurizer_tf = class_pipeline.featurizer_tf
    featurizer_emb = class_pipeline.featurizer_emb
    task_num = 0
    for task_name, task in class_pipeline.classification_tasks_dict.items():
        classifier = task.classifier
        task_pred_lists = []
        for sent, claim in zip(sents, claims):
            features = class_pipeline.get_feature_union([sent], [claim], class_pipeline.tok_driver, 
                                                        featurizer_emb, featurizer_tf, mode="test")
            
            pred_labels, pred_probs = classifier.predict_utt_top_n(features, n=5)
            
            if task_name == "template_formula":
                pred_labels_unhashed = pred_labels
            else:
                try:
                    pred_labels_unhashed = [task.hash_to_label_dict[str(int(p))] for p in pred_labels]
                except:
                    print([str(int(p)) for p in pred_labels])
                    print(task.hash_to_label_dict)

            pred_labels_and_probs = [{"label": z[0], "prob": z[1]} for z in zip(pred_labels_unhashed, pred_probs)]
            task_pred_lists.append(json.dumps(pred_labels_and_probs))
        df[task.name] = task_pred_lists
    return df

class_step = ClassificationStep(DATA_PATH, simulation=False, export=False)
opt_step = OptimizationStep(class_step)
user_study_file = "data/user_study_40_claims.csv"
user_study_df = pd.read_csv(user_study_file)

complete_df = class_step.parser.get_complete_df()[:200]

# remove the claims selected for the user study
complete_df = user_study_df.merge(complete_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
complete_df.drop(columns=["_merge"], inplace=True)
class_step.train(complete_df)
claims = create_claims_from_df(user_study_df, class_step)
for claim in claims:
    opt_step.optimize_claim(claim)
    print(claim.sent)
    for prop in claim.available_properties:
        print("prop name: ", prop.property_name)
        print("property ask: ", prop.ask)
        for val in prop.candidate_values:
            print("val: {} prob: {}".format(val.value, val.prob))
    break

# preds_df = get_preds_from_df(user_study_df, class_step)
# print(preds_df.head())
# print(len(preds_df))

# preds_df.to_csv("data/user_study_predictions.csv", index=False)
