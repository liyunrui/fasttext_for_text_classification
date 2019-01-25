"""
Provide Customized interface for calculating evaluation metric such as precision, recall, f1-score, top_3_acc, etc.

@author: Ray

"""
import numpy as np
import pandas as pd
# logging
from utils import init_logging
import logging
# evaluation metric
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def top_3_acc(y_pred_top3, y_true, debug = False):
    """
    Return top 3 accuracy
    
    y_pred: list of list. For example, [[c1,c2,c3],[c3,c4,c5], ...]
    y_true: array-like
    """
    assert len(y_pred_top3) == len(y_true), "y_true and y_pred is mismatched"

    # top3 accuracy
    top_3_acc = 0
    ix = 0
    for pred, true in zip(y_pred_top3, y_true):
        ix += 1
        if true in set(pred):
            top_3_acc+=1
        if debug == True:
            if ix <= 10:
                print ('y_pred',pred)
                print ('y_true',true)
    top_3_acc = 1.0 * top_3_acc/len(y_true)
    return top_3_acc

# setting
init_logging('../logs')
#----------------------
# loading data
#----------------------
logging.info("load cleaned dataset result")
# preds
f_preds = open('../prediction/cleaned_test_top3.txt','r')
preds = []
predes_top3 = []
ix = 0
for pred in f_preds.readlines():
    ix += 1
    pred = pred.strip().split(' ')
    pred = [i[9:] for i in pred]
    preds.append(pred[0])
    predes_top3.append(pred)
    # if ix <= 10:
    #     print ('pred', pred)
# labels
f_labels = open('../cleaned_test.txt','r')
test_labels = [i.strip().split(' ')[0][9:] for i in f_labels.readlines()]
assert len(preds) == len(test_labels), 'the number of prediction and ground truch is mismatched'
#----------------------
# confusion matrix
#----------------------

#----------------------
# evaluation metric(whole data sete)
#----------------------
# accuracy
logging.info('accuracy : {}'.format(accuracy_score(np.array(preds), test_labels)))
# f1-score
logging.info('weighted f1-score : {}'.format(f1_score(np.array(preds), test_labels, average='weighted')))
logging.info('micro f1-score : {}'.format(f1_score(np.array(preds), test_labels, average='micro')))
logging.info('macro f1-score : {}'.format(f1_score(np.array(preds), test_labels, average='macro')))
# recall
logging.info('weighted recall : {}'.format(recall_score(np.array(preds), test_labels, average='weighted')))
logging.info('micro recall : {}'.format(recall_score(np.array(preds), test_labels, average='micro')))
logging.info('macro recall : {}'.format(recall_score(np.array(preds), test_labels, average='macro')))
# precision
logging.info('weighted precision : {}'.format(precision_score(np.array(preds), test_labels, average='weighted')))
logging.info('micro precision : {}'.format(precision_score(np.array(preds), test_labels, average='micro')))
logging.info('macro precision : {}'.format(precision_score(np.array(preds), test_labels, average='macro')))
# top3 accuracy
logging.info('top_3_acc : {}'.format(top_3_acc(predes_top3,test_labels)))
#----------------------
# evaluation metric(for imblamced problem)
#----------------------
df = pd.DataFrame({
    'pred':preds,
    'true':test_labels,
    'y_pred_top3':predes_top3,
})
df.to_csv('../prediction/cleaned_prediction.csv', index = False)
for flag in ['branded','unbranded']:
    logging.info('===================flag=================== \n {}'.format(flag))
    if flag == 'unbranded':
        test = df[df.true == 'no-brand']
        # number of sku
        logging.info('Number of SKU in testing dataset : {}'.format(len(test)))
        # f1-score
        logging.info('weighted f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='macro')))
        # recall
        logging.info('weighted recall : {}'.format(recall_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro recall : {}'.format(recall_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro recall : {}'.format(recall_score(test.pred.values, test.true.values, average='macro')))
        # precision
        logging.info('weighted precision : {}'.format(precision_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro precision : {}'.format(precision_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro precision : {}'.format(precision_score(test.pred.values, test.true.values, average='macro')))
        # accuracy
        logging.info('accuracy : {}'.format(accuracy_score(test.pred.values, test.true.values)))
        # top3 acc
        logging.info('top 3 accuracy : {}'.format(top_3_acc(test.y_pred_top3.values,test.true.values)))
    elif flag == 'branded':
        test = df[df.true != 'no-brand']
        # number of sku
        logging.info('Number of SKU in testing dataset : {}'.format(len(test)))
        # f1-score
        logging.info('weighted f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='macro')))
        # recall
        logging.info('weighted recall : {}'.format(recall_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro recall : {}'.format(recall_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro recall : {}'.format(recall_score(test.pred.values, test.true.values, average='macro')))
        # precision
        logging.info('weighted precision : {}'.format(precision_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro precision : {}'.format(precision_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro precision : {}'.format(precision_score(test.pred.values, test.true.values, average='macro')))
        # accuracy
        logging.info('accuracy : {}'.format(accuracy_score(test.pred.values, test.true.values)))
        # top3 acc
        logging.info('top 3 accuracy : {}'.format(top_3_acc(test.y_pred_top3.values,test.true.values)))
    else:
        print ('something wrong in true columns')



#----------------------
# loading data
#----------------------
logging.info("load raw dataset result")
# preds
f_preds = open('../prediction/raw_test_top3.txt','r')
preds = []
predes_top3 = []
ix = 0
for pred in f_preds.readlines():
    ix += 1
    pred = pred.strip().split(' ')
    pred = [i[9:] for i in pred]
    preds.append(pred[0])
    predes_top3.append(pred)
    # if ix <= 10:
    #     print ('pred', pred)

#----------------------
# evaluation metric(whole data sete)
#----------------------

# labels
f_labels = open('../raw_test.txt','r')
test_labels = [i.strip().split(' ')[0][9:] for i in f_labels.readlines()]
assert len(preds) == len(test_labels), 'the number of prediction and ground truch is mismatched'
# accuracy
logging.info('accuracy : {}'.format(accuracy_score(np.array(preds), test_labels)))
# f1-score
logging.info('weighted f1-score : {}'.format(f1_score(np.array(preds), test_labels, average='weighted')))
logging.info('micro f1-score : {}'.format(f1_score(np.array(preds), test_labels, average='micro')))
logging.info('macro f1-score : {}'.format(f1_score(np.array(preds), test_labels, average='macro')))
# recall
logging.info('weighted recall : {}'.format(recall_score(np.array(preds), test_labels, average='weighted')))
logging.info('micro recall : {}'.format(recall_score(np.array(preds), test_labels, average='micro')))
logging.info('macro recall : {}'.format(recall_score(np.array(preds), test_labels, average='macro')))
# precision
logging.info('weighted precision : {}'.format(precision_score(np.array(preds), test_labels, average='weighted')))
logging.info('micro precision : {}'.format(precision_score(np.array(preds), test_labels, average='micro')))
logging.info('macro precision : {}'.format(precision_score(np.array(preds), test_labels, average='macro')))
# top3 accuracy
logging.info('top_3_acc : {}'.format(top_3_acc(predes_top3,test_labels)))


#----------------------
# evaluation metric(for imblamced problem)
#----------------------
df = pd.DataFrame({
    'pred':preds,
    'true':test_labels,
    'y_pred_top3':predes_top3,
})
df.to_csv('../prediction/raw_prediction.csv', index = False)
for flag in ['branded','unbranded']:
    logging.info('flag : {}'.format(flag))
    if flag == 'unbranded':
        test = df[df.true == 'no-brand']
        # number of sku
        logging.info('Number of SKU in testing dataset : {}'.format(len(test)))
        # f1-score
        logging.info('weighted f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='macro')))
        # recall
        logging.info('weighted recall : {}'.format(recall_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro recall : {}'.format(recall_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro recall : {}'.format(recall_score(test.pred.values, test.true.values, average='macro')))
        # precision
        logging.info('weighted precision : {}'.format(precision_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro precision : {}'.format(precision_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro precision : {}'.format(precision_score(test.pred.values, test.true.values, average='macro')))
        # accuracy
        logging.info('accuracy : {}'.format(accuracy_score(test.pred.values, test.true.values)))
        # top3 acc
        logging.info('top 3 accuracy : {}'.format(top_3_acc(test.y_pred_top3.values,test.true.values)))
    elif flag == 'branded':
        test = df[df.true != 'no-brand']
        # number of sku
        logging.info('Number of SKU in testing dataset : {}'.format(len(test)))
        # f1-score
        logging.info('weighted f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro f1-score : {}'.format(f1_score(test.pred.values, test.true.values, average='macro')))
        # recall
        logging.info('weighted recall : {}'.format(recall_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro recall : {}'.format(recall_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro recall : {}'.format(recall_score(test.pred.values, test.true.values, average='macro')))
        # precision
        logging.info('weighted precision : {}'.format(precision_score(test.pred.values, test.true.values, average='weighted')))
        logging.info('micro precision : {}'.format(precision_score(test.pred.values, test.true.values, average='micro')))
        logging.info('macro precision : {}'.format(precision_score(test.pred.values, test.true.values, average='macro')))
        # accuracy
        logging.info('accuracy : {}'.format(accuracy_score(test.pred.values, test.true.values))) 
        # top3 acc
        logging.info('top 3 accuracy : {}'.format(top_3_acc(test.y_pred_top3.values,test.true.values)))
    else:
        print ('something wrong in true columns')


        
