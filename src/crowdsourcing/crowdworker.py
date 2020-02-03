"""
Simulates a crowd worker, who verifies whether a prediction is correct or not
"""

class Crowdworker:
    def __init__(self, predictions, test_df, label_col, verification_cost=1, ask_cost=2):
        self.predictions = predictions
        self.test_df = test_df
        self.label_col = label_col
        self.verification_cost = verification_cost
        self.ask_cost = ask_cost

    def get_ask_cost_for_all_templates(self):
        cost = len(self.test_df) * self.ask_cost
        print("------- get_ask_cost_for_all_templates -------- \n")
        print("overall cost: {}".format(cost))

    def first_prediction_correct(self, prediction, label):
        """
        Check if we get a correct prediction (first predicted label is correct)
        """
        pred_labels = prediction[0]
        pred_probs = prediction[1]     
        return pred_labels[0] == label
        
    def get_overall_cost_exp1(self, prob_thres=0.5):
        """
        if pred of first template >= prob_thres verify the template
        else ask
        """
        cost = 0
        num_verification = 0
        num_ask = 0
        num_correct = 0.0
        labels = self.test_df[self.label_col]
        for pred, label in zip(self.predictions, labels):
            pred_labels = pred[0]
            pred_probs = pred[1] 
            if pred_probs[0] >= prob_thres:
                cost += self.verification_cost
                num_verification += 1
                if self.first_prediction_correct(pred, label):
                    num_correct += 1
                else:
                    cost += self.ask_cost
                    num_ask += 1
            else:
                cost += self.ask_cost
                num_ask += 1
        
        print("-------- get_overall_cost_exp1 ------ \n")
        print("prob threshold: {}".format(prob_thres))
        print("Total cost: {}".format(cost))
        print("accuracy: {}".format(num_correct/len(self.test_df)))
        print("Number of verifications: {}".format(num_verification))
        print("Number of asks: {}".format(num_ask))


    def get_prediction_verification_cost(self, prediction, label):
        """
        Get the verification cost for each prediction.
        Every time we verify, we add the  verification cost, until we get it correct
        """
        pred_ver_cost = 0.0
        pred_labels = prediction[0]
        pred_probs = prediction[1]
        correct_pred_topn = False
        num_verifications = 0
        for pred_label, pred_prob in zip(pred_labels, pred_probs):
            if pred_label == label:
                pred_ver_cost += self.verification_cost
                correct_pred_topn = True
                num_verifications += 1
                break
            else:
                pred_ver_cost += self.verification_cost
                num_verifications += 1

        return pred_ver_cost, correct_pred_topn, num_verifications

    def get_overall_cost_exp2(self):
        verification_cost = 0
        num_verification = 0
        num_ask = 0
        ask_cost = 0
        labels = self.test_df[self.label_col]
        for pred, label in zip(self.predictions, labels):
            pred_ver_cost, correct_topn, pred_num_ver = self.get_prediction_verification_cost(pred, label)
            verification_cost += pred_ver_cost
            num_verification += pred_num_ver
            if not correct_topn:
                num_ask += 1
                ask_cost += self.ask_cost

        print("-------- get_overall_cost_exp2 ------ \n")
        print("Total cost: {}".format(verification_cost + ask_cost))
        print("Number of verifications: {}".format(num_verification))
        print("Number of asks: {}".format(num_ask))