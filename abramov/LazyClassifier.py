from __future__ import print_function
import pandas as pd
import numpy as np
import copy

C = 0.55
C2 = 0.55
L = 0.05

attrib_names = [str(i) for i in range(1, 28)]
attrib_names.append('class')

def accuracy(res):
    return float(res["positive_positive"] + res["negative_negative"]) / max(1, res["positive_positive"] + res["negative_negative"] + res["positive_negative"] + res["negative_negative"] + res["contradictory"])

def precision(res):
    return float(res["positive_positive"]) / max(1, res["positive_positive"] + res["negative_negative"])

def recall(res):
    return float(res["positive_positive"]) / max(1, res["positive_positive"] + res["positive_negative"])
    
def true_neg_pred_val(res):
    return float(res["negative_negative"]) / max(1, res["negative_negative"] + res["positive_negative"])
    
def false_pos_rate(res):
    return float(res["negative_positive"]) / max(1, res["negative_positive"] + res["negative_negative"])    

def false_neg_rate(res):
    return float(res["positive_negative"]) / max(1, res["positive_positive"] + res["positive_negative"])
    
def false_disc_rate(res):
    return float(res["negative_positive"]) / max(1, res["positive_positive"] + res["negative_positive"])

def F1_score(res):
    prec = precision(res)
    rec = recall(res)
    return 2 * prec * rec / max(1, prec + rec)

def summary(res):
    stats = {}
    stats["accuracy"] = accuracy(res)
    stats["precision"] = precision(res)
    stats["recall"] = recall(res)
    stats["f1"] = F1_score(res)
    stats["TN_Pred_Rate"] = true_neg_pred_val(res)
    stats["FP_Rate"] = false_pos_rate(res)
    stats["FN_Rate"] = false_neg_rate(res)
    stats["FDISC_Rate"] = false_disc_rate(res)
    return stats
    

def k_fold_cross_validation(X, K, algorithm = 4, randomise = True):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    res = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "TN_Pred_Rate": 0,
    'FP_Rate' : 0, 'FN_Rate': 0, "FDISC_Rate": 0}
    if randomise: from random import shuffle; X=list(X); shuffle(X)
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        #print(len(training))

        validation = [x for i, x in enumerate(X) if i % K == k]
        #print(len(validation))
        classifier = LazyClassifier(training, validation)
        cl = classifier.classify(algorithm = algorithm)
        res["accuracy"] += accuracy(cl)
        res["precision"] += precision(cl)
        res["recall"] += recall(cl)
        res["f1"] += F1_score(cl)
        res["TN_Pred_Rate"] += true_neg_pred_val(cl)
        res["FP_Rate"] += false_pos_rate(cl)
        res["FN_Rate"] += false_neg_rate(cl)
        res["FDISC_Rate"] += false_disc_rate(cl)
        print('Cross valid: k = ', k, '; res: ', res)
    
    for i in res:
        res[i] /= K
    
    #print(res)
    return res

class LazyClassifier:
    
    def __init__(self, train = [], test = []):
        self.cv_res = {
         "positive_positive": 0,
         "positive_negative": 0,
         "negative_positive": 0,
         "negative_negative": 0,
         "contradictory": 0,
        }

        self.context_plus = [a for a in train if a[-1] == 'positive']
        self.context_minus = [a for a in train if a[-1] == 'negative']
        self.test = test

    def classify(self, train = [], test = [], algorithm = 4):
        #print(len(self.context_plus))
        #print(len(self.context_minus))
        if len(train) > 0:
            self.context_plus = [a for a in train if a[-1] == 'positive']
            self.context_minus = [a for a in train if a[-1] == 'negative']
        if len(test) > 0:
            self.test = test
        
        if len(self.context_plus) == 0 or len(self.context_minus) == 0:
            raise AttributeError('Sorry, train set must not be empty')
            
        if len(self.test) == 0:
            raise AttributeError('Sorry, test set must not be empty')
        
        i = 0
        #print(self.test)
        for elem in self.test:
            i += 1
            print(i, '/', len(self.test))
            #print(elem)
            if algorithm == 1:
                self.check_hypothesis_algorithm_1(elem)
            elif algorithm == 2:
                self.check_hypothesis_algorithm_2(elem)
            elif algorithm == 3:
                self.check_hypothesis_algorithm_3(elem)
            elif algorithm == 4:
                self.check_hypothesis_algorithm_4(elem)
            else:
                raise AttributeError('Sorry, but algorithm must be an integer between 1 and 4')
            #if i == 15: break
        
        return self.cv_res
        
    
    def make_intent(self, example):
        global attrib_names
        #print(example)
        return set([i+':'+str(k) for i,k in zip(attrib_names, example)])
       
    def check_hypothesis_algorithm_1(self, example):
        example_intent = self.make_intent(example)
        #print(example[-1])
        #print(example_intent)
        example_intent.discard('class:positive')
        example_intent.discard('class:negative')

        labels = {"negative": 0, "positive": 0}
    
        # For plus context
        #print(self.context_plus)
        for p_example in self.context_plus:
            #print(p_example)
            p_example_intent = self.make_intent(p_example)
            #print(p_example)
            candidate_intent = p_example_intent & example_intent
            #print(p_example)
            #print(candidate_intent)
            falsification = [self.make_intent(i) for i in self.context_minus if self.make_intent(i).issuperset(candidate_intent)]
            if not falsification and candidate_intent:
                labels["positive"] += 1

        # For minus context

        for n_example in self.context_minus:
            n_example_intent = self.make_intent(n_example)
            candidate_intent = n_example_intent & example_intent
            falsification = [self.make_intent(i) for i in self.context_plus if self.make_intent(i).issuperset(candidate_intent)]
            if not falsification and candidate_intent:
                labels["negative"] += 1
        
        #print(labels['positive'])
        #print(labels['negative'])
        #print('-------')

        if labels["positive"] > labels["negative"]:
            if example[-1] == "positive":
                self.cv_res["positive_positive"] += 1
            else:
                self.cv_res["negative_positive"] += 1
        elif labels["positive"] < labels["negative"]:
            if example[-1] == 'negative':
                self.cv_res["negative_negative"] += 1
            else:
                self.cv_res["positive_negative"] += 1
        else:
            self.cv_res["contradictory"] += 1
            
        return self.cv_res

    def check_hypothesis_algorithm_4(self, example):
        example_intent = self.make_intent(example)
        #print(example_intent)
        example_intent.discard('class:positive')
        example_intent.discard('class:negative')
    
        labels = {"negative": 0, "positive": 0}
    
        # For plus context
        #print(C)
        for p_example in self.context_plus:
            p_example_intent = self.make_intent(p_example)
            candidate_intent = p_example_intent & example_intent
        
            intent_power = len(candidate_intent) * 1.0 / len(example_intent)
        
            # support 
            support = [self.make_intent(i) for i in self.context_plus if self.make_intent(i).issuperset(candidate_intent)]
            support_idx = float(len(support)) / len(self.context_plus)
        
            # falsification
            falsification = [self.make_intent(i) for i in self.context_minus if self.make_intent(i).issuperset(candidate_intent)]
            falsification_idx = float(len(falsification)) / len(self.context_minus)

            if intent_power > C2 and support_idx > falsification_idx:
                labels["positive"] += (support_idx - falsification_idx) * 1.0 #/ (len(self.context_plus))

        # For minus context

        for n_example in self.context_minus:
            n_example_intent = self.make_intent(n_example)
            candidate_intent = n_example_intent & example_intent
        
            intent_power = len(candidate_intent) * 1.0 / len(example_intent)
        
            # support 
            support = [self.make_intent(i) for i in self.context_minus if self.make_intent(i).issuperset(candidate_intent)]
            support_idx = float(len(support)) / len(self.context_minus)
        
            # falsification
            falsification = [self.make_intent(i) for i in self.context_plus if self.make_intent(i).issuperset(candidate_intent)]
            falsification_idx = float(len(falsification)) / len(self.context_plus)
        
            if intent_power > C2 and support_idx > falsification_idx:
                labels["negative"] += (support_idx - falsification_idx) * 1.0 #/ (len(self.context_minus))
     
     
        p = (labels['positive'] - labels['negative'])/(labels['positive'] + labels['negative']) 
        #print(labels['positive'])   
        #print(labels['negative'])
        #print(p)
        #print('-------')
    
        if abs(p) > L:
            if p > 0:
                if example[-1] == "positive":
                    self.cv_res["positive_positive"] += 1
                else:
                    self.cv_res["negative_positive"] += 1
            else:
                if example[-1] == "negative":
                    self.cv_res["negative_negative"] += 1
                else:
                    self.cv_res["positive_negative"] += 1
        else:
            self.cv_res["contradictory"] += 1
            
        return self.cv_res

    def check_hypothesis_algorithm_3(self, example):
        example_intent = self.make_intent(example)
        #print(example_intent)
        example_intent.discard('class:positive')
        example_intent.discard('class:negative')
        labels = {"negative": 0, "positive": 0}
    
        # For plus context
        #print(C)
        for p_example in self.context_plus:
            p_example_intent = self.make_intent(p_example)
            candidate_intent = p_example_intent & example_intent
        
            intent_power = float(len(candidate_intent)) / len(example_intent)
            #print(intent_power)
            # support 
            support = [self.make_intent(i) for i in self.context_plus if self.make_intent(i).issuperset(candidate_intent)]
            support_idx = float(len(support)) / len(self.context_plus)
            #print(support_idx)
        
            # falsification
            falsification = [self.make_intent(i) for i in self.context_minus if self.make_intent(i).issuperset(candidate_intent)]
            falsification_idx = float(len(falsification)) / len(self.context_minus)
            #print(falsification_idx)
            #print('-----')

            if intent_power > C and support_idx > falsification_idx:
                labels["positive"] += 1.0

        # For minus context

        for n_example in self.context_minus:
            n_example_intent = self.make_intent(n_example)
            candidate_intent = n_example_intent & example_intent
        
            intent_power = float(len(candidate_intent)) / len(example_intent)
        
            # support 
            support = [self.make_intent(i) for i in self.context_minus if self.make_intent(i).issuperset(candidate_intent)]
            support_idx = float(len(support)) / len(self.context_minus)
        
            # falsification
            falsification = [self.make_intent(i) for i in self.context_plus if self.make_intent(i).issuperset(candidate_intent)]
            falsification_idx = float(len(falsification)) / len(self.context_plus)
        
            if intent_power > C and support_idx > falsification_idx:
                labels["negative"] +=  1.0
     
        p = (labels['positive'] - labels['negative'])/(labels['positive'] + labels['negative']) 
        #print(labels['positive'] / len(self.context_plus))   
        #print(labels['negative'] / len(self.context_minus))
        #print(p)
        #print('-------')
    
        if abs(p) > L:
            if p > 0:
                if example[-1] == "positive":
                    self.cv_res["positive_positive"] += 1
                else:
                    self.cv_res["negative_positive"] += 1
            else:
                if example[-1] == "negative":
                    self.cv_res["negative_negative"] += 1
                else:
                    self.cv_res["positive_negative"] += 1
        else:
            self.cv_res["contradictory"] += 1
            
        return self.cv_res

    def check_hypothesis_algorithm_2(self, example):
        example_intent = self.make_intent(example)
        example_intent.discard('class:positive')
        example_intent.discard('class:negative')
        labels = {"negative": 0, "positive": 0}
    
        # For plus context
    
        for p_example in self.context_plus:
            p_example_intent = self.make_intent(p_example)
            candidate_intent = p_example_intent & example_intent
            support = [self.make_intent(i) for i in self.context_plus if self.make_intent(i).issuperset(candidate_intent)]
            labels["positive"] += float(len(support)) / len(self.context_plus)

        # For minus context

        for n_example in self.context_minus:
            n_example_intent = self.make_intent(n_example)
            candidate_intent = n_example_intent & example_intent
            support = [self.make_intent(i) for i in self.context_minus if self.make_intent(i).issuperset(candidate_intent)]
            labels["negative"] += float(len(support)) / len(self.context_minus)
     
        #print(labels['positive'])   
        #labels["positive"] = float(labels["positive"]) / len(self.context_plus)
        #print(labels['positive'])
        #print(labels['negative'])  
        #labels["negative"] = float(labels["negative"]) / len(self.context_minus)
        #print(labels['negative']) 
    
        #print('-------')

        if labels["positive"] > labels["negative"]:
            if example[-1] == "positive":
                self.cv_res["positive_positive"] += 1
            else:
                self.cv_res["negative_positive"] += 1
        elif labels["positive"] < labels["negative"]:
            if example[-1] == "negative":
                self.cv_res["negative_negative"] += 1
            else:
                self.cv_res["positive_negative"] += 1
        else:
            self.cv_res["contradictory"] += 1
            
        return self.cv_res

def dummy_encode_categorical_columns(data):
    result_data = copy.deepcopy(data)
    for column in data.columns.values:
        result_data = pd.concat([result_data, pd.get_dummies(result_data[column], prefix = column, prefix_sep = ': ')], axis = 1)
        del result_data[column]
    return result_data.astype(int)

def parse_file(name):
    df = pd.read_csv(name, sep=',')
    df = df.replace(to_replace='positive', value = 'positive')
    df = df.replace(to_replace='negative', value = 'negative')
    y = np.array(df['V10'])
    del(df['V10'])
    bin_df = dummy_encode_categorical_columns(df)
    bin_df['res'] = y
    return np.array(bin_df).tolist()

def find_constants(algorithm = 4):
    arrC = [0.5, 0.55, 0.6, 0.65, 0.7]
    m = 11
    global C
    global C2
    for c in arrC:
        C = c
        C2 = c
        res = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "TN_Pred_Rate": 0,
        'FP_Rate' : 0, 'FN_Rate': 0, "FDISC_Rate": 0}
        for i in range(1, m):
            train = parse_file('train' + str(i) + '.csv')
            unknown = parse_file('test' + str(i) + '.csv')

            print(i)

            res_temp = k_fold_cross_validation(unknown, K = 7, algorithm = algorithm)
            res['accuracy'] += res_temp['accuracy']
            res['precision'] += res_temp['precision']
            res['recall'] += res_temp['recall']
            res['f1'] += res_temp['f1']
            res["TN_Pred_Rate"] += res_temp["TN_Pred_Rate"]
            res["FP_Rate"] += res_temp["FP_Rate"]
            res["FN_Rate"] += res_temp["FN_Rate"]
            res["FDISC_Rate"] += res_temp["FDISC_Rate"]
            #print('Set #', i, ': ', res)
            #print('Set #', i, ': ', res2)
        for i in res:
            res[i] /= (m - 1)
        print('For C = ', C, '; Res = ', res)

if __name__ == "__main__":

    #find_constants(4)
    p = 1
    train = parse_file('train' + str(p) + '.csv')
    unknown = parse_file('test' + str(p) + '.csv')

    res = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "TN_Pred_Rate": 0,
    'FP_Rate' : 0, 'FN_Rate': 0, "FDISC_Rate": 0}

    algorithm = 4
    classifier = LazyClassifier(train, unknown)
    res_temp = classifier.classify(algorithm = algorithm)
    res = summary(res_temp)
    print(res)