"""

@author: Ray

Do hyper-parameter tuning for fastText using random search

"""
import os
import numpy as np
import random
import time
from utils import init_logging
import logging
#------
# tunning
#------
init_logging('../logs')
print ("ramdom search")


wordNgrams = [1,2,3,4,5]
window_size = [5,10,25] # window size
learning_rate = np.linspace(0.1,1.0, num = 5)
dims = [100, 300]
epoch = np.linspace(5, 50, num = 5)
losses = ['ns','hs','softmax'] # ns: negative sampling

tc = len(wordNgrams) * len(window_size) * len(learning_rate) * len(dims) * len(epoch) * len(losses)
print ('total combinations : {}'.format(tc))

already_used_combination = []

while len(already_used_combination) <= 100:
    n_gram =  random.choice(wordNgrams)
    ws =  random.choice(window_size)
    lr = random.choice(learning_rate)
    dim = random.choice(dims)
    ep = random.choice(epoch)
    ls = random.choice(losses)
    logging.info('learning_rate : {}'.format(lr))
    logging.info('wordNgrams : {}'.format(n_gram))
    logging.info('window_size : {}'.format(ws))
    logging.info('dims : {}'.format(dim))
    logging.info('epoch : {}'.format(ep))
    logging.info('loss : {}'.format(ls))
    s = time.time()
    if (n_gram,ws,lr,dim,ep,ls) not in already_used_combination:
        cmd = '../../../fastText/fasttext supervised -input ../data/train_for_tuning.txt -output ../model/model_tune -lr {} -epoch {} -wordNgrams {} -dim {} -loss {} -ws {} '.format(lr,ep,n_gram,dim,ls,ws)            
        os.system(cmd)
        # evaluate on vlsalidation set
        cmd = '../../../fastText/fasttext test ../model/model_tune.bin ../data/val.txt 1 > ../logs/lr_{}_ep_{}_ngram_{}_dim_{}_ls_{}_ws_{}.txt'.format(lr,ep,n_gram,dim,ls,ws) 
        # note: Here, precision@1 == accuracy
        os.system(cmd)
        f = open('../logs/lr_{}_ep_{}_ngram_{}_dim_{}_ls_{}_ws_{}.txt'.format(lr,ep,n_gram,dim,ls,ws), 'r')
        accuracy = [i.strip().replace('\t',' ') for i in f.readlines()][-1]
        logging.info('accuracy : {}'.format(accuracy))
        f.close()
        # terminating condition
        already_used_combination.append((n_gram,ws,lr,dim,ep,ls))
    e = time.time()
    training_time = round((e-s)/60.0, 2)
    logging.info('training time : {}'.format(training_time))

#------
# tuning result save
#------
data = []
for f_name in os.listdir('../logs/'):
    if f_name.startswith('lr'):
        try:
            paras_value = [e for i,e in enumerate(f_name.replace('.txt','').split('_')) if i%2 ==1] 
            #print (paras_value)
            f = open(os.path.join('../logs',f_name), 'r')
            accuracy = [i.strip().replace('\t',' ') for i in f.readlines()][-1]
            #print (accuracy)
            paras_value.append(accuracy)
            data.append(paras_value)
        except Exception as error:
            print (error)
df = pd.DataFrame(data, columns = ['learning_rate','epoch','ngram','dim','loss','window_size','accuracy'])
df.to_csv('../logs/tuning.csv', index = False)





