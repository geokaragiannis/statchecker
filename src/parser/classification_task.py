"""
This class holds information about different classification tasks like 
row_index, tab, file prediction.
"""


class ClassificationTask:
    def __init__(self, init_name, name, hash_name, label_task, priority, val_acc=0.0):
        """
        Arguments:
            init_name {str} -- [name that appears in the original csv files]
            name {str} -- [new name that is used after parsing the original csv files]
            hash_name {str} -- [name of the column that contains the hash value of the 
                               classification task]
            label_task {str} -- [Classification type. Either "single-label" or "multi-label"]
            priority {int} -- {a lower number is a more important task}
            val_acc {float} -- [accuracy in the validation set]
        """
        self.init_name = init_name
        self.name = name
        self.hash_name = hash_name
        self.label_task = label_task
        self.priority = priority
        self.val_acc = val_acc
        # True if we have hashed the values of the Classification Task
        self.has_hash = True
        if name == hash_name:
            self.has_hash = False
        
        self.classifier_name = self.name + "_classifier"
        self.featurizer_tf_name = self.name + "_featurizer_tf"
        self.featurizer_emb_name = self.name + "_featurizer_emb" 
        # store components needed for classification
        self.is_trained = False
        self.featurizer_tf = None
        self.featurizer_emb = None
        self.classifier = None

        # dict to translate a label to a unique value
        self.label_to_hash_dict = dict()
        self.hash_to_label_dict = dict()
        # hash_counter will always be incremented when we add a new label to a task's hash_dict
        self.hash_counter = 0

    
    def hash_(self, label):
        """
        For a given label for this task create 2 maps
        1. get a unique integer from a label
        2. get a label from a unique integer
        return the unique value for this label
        """
        if label in self.label_to_hash_dict:
            return self.label_to_hash_dict[label]
        else:
            self.hash_counter += 1
            self.label_to_hash_dict[label] = self.hash_counter
            self.hash_to_label_dict[self.hash_counter] = label

        return self.label_to_hash_dict[label]