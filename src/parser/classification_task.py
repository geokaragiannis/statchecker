"""
This class holds information about different classification tasks like 
row_index, tab, file prediction.
"""


class ClassificationTask:
    def __init__(self, init_name, name, hash_name, label_task, val_acc=0.0):
        """
        Arguments:
            init_name {str} -- [name that appears in the original csv files]
            name {str} -- [new name that is used after parsing the original csv files]
            hash_name {str} -- [name of the column that contains the hash value of the 
                               classification task]
            label_task {str} -- [Classification type. Either "single-label" or "multi-label"]
            val_acc {float} -- [accuracy in the validation set]
        """
        self.init_name = init_name
        self.name = name
        self.hash_name = hash_name
        self.label_task = label_task
        self.val_acc = val_acc
        # True if we have hashed the values of the Classification Task
        self.has_hash = True
        if name == hash_name:
            self.has_hash = False
        
        # store components needed for classification
        self.is_trained = False
        self.featurizer_tf = None
        self.featurizer_emb = None
        self.classifier = None
