"""
Getting and preparing the data for FastText classification

@author : Ray

python3 create_training_data.py -tr_path /data/yunrui_li/seen_brand/brand_classification_train.csv -te_path /data/yunrui_li/seen_brand/brand_classification_test.csv -o_dir /data/yunrui_li/fastText_example/fast_text_brand_classification/data

"""
import os
import pandas as pd
import time
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

#----------------------
# standard argument config
#----------------------
parser = argparse.ArgumentParser()
parser.add_argument('-tr_path', '--train_path', type=str, required=True)
parser.add_argument('-te_path', '--test_path', type=str, required=True)
parser.add_argument('-o_dir', '--output_dir', type=str, required=True)

args = parser.parse_args()
train_path = args.train_path
test_path = args.test_path
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#--------------------
# loading data
#--------------------
#train_path = '/data/brand_classification/cleaned_train.csv'
#test_path = '/data/brand_classification/cleaned_test.csv'
train = pd.read_csv(train_path) # 5678025 rows
test = pd.read_csv(test_path) # 1419508 rows
train.dropna(subset = ['label','item_name'], inplace = True)
test.dropna(subset = ['label','item_name'], inplace = True)
train.drop_duplicates(subset = ['item_name'], inplace = True)
test.drop_duplicates(subset = ['item_name'], inplace = True)
# cannot have blank within the whole brand words, it causes some problems when running fasttext
train['brand_for_fasttext'] = train.label.apply(lambda x: x.replace(' ','_'))
test['brand_for_fasttext'] = test.label.apply(lambda x: x.replace(' ','_'))
# need a dict for convert the predicted brand into original version of the brand

#--------------------
# core
#--------------------
s = time.time()

required_col = ['brand_for_fasttext','item_name']
for ix, df in enumerate([train, test]):
    if ix == 0:
        f = open(os.path.join(output_dir, 'train.txt'),'w')
    elif ix == 1:
        f = open(os.path.join(output_dir, 'test.txt'),'w')
    else:
        assert False,"we only have two ah"
    for _, row in tqdm(df[required_col].iterrows()):
        prefix_FastText = '__label__'
        label = prefix_FastText+str(row.brand_for_fasttext)
        sentence = str(row.item_name)
        output_form = label + ' '+ sentence
        f.write(output_form)
        f.write('\n')
    f.close()
e = time.time()
print ('preparing take {} mins'.format((e-s)/60.0))

#--------------------
# hold-out split for hyper-parameter tuning
#--------------------
train, val = train_test_split(train, train_size= 0.8)

s = time.time()
for ix, df in enumerate([train, val]):
    if ix == 0:
        f = open(os.path.join(output_dir, 'train_for_tuning.txt'),'w')
    elif ix == 1:
        f = open(os.path.join(output_dir, 'val.txt'),'w')
    else:
        assert False,"we only have two ah"
    for _, row in tqdm(df[required_col].iterrows()):
        prefix_FastText = '__label__'
        label = prefix_FastText+str(row.brand_for_fasttext)
        sentence = str(row.item_name)
        output_form = label + ' '+ sentence
        f.write(output_form)
        f.write('\n')
    f.close()

e = time.time()
print ('preparing take {} mins'.format((e-s)/60.0))



# #--------------------
# # loading data
# #--------------------
# train_path = '/data/brand_classification/raw_train.csv'
# test_path = '/data/brand_classification/raw_test.csv'
# train = pd.read_csv(train_path) # 5678025 rows
# test = pd.read_csv(test_path) # 1419508 rows
# train.dropna(subset = ['brand','item_name'], inplace = True)
# test.dropna(subset = ['brand','item_name'], inplace = True)
# train.drop_duplicates(subset = ['item_name'], inplace = True)
# test.drop_duplicates(subset = ['item_name'], inplace = True)
# # cannot have blank within the whole brand words, it causes some problems when running fasttext
# train['brand_for_fasttext'] = train.brand.apply(lambda x: x.replace(' ','_'))
# test['brand_for_fasttext'] = test.brand.apply(lambda x: x.replace(' ','_'))

# #--------------------
# # core
# #--------------------
# required_col = ['brand_for_fasttext','item_name']
# ix = 0
# s = time.time()
# for ix,df in enumerate([train, test]):
#     if ix == 0:
#         f = open('../raw_train.txt','w')
#     elif ix == 1:
#         f = open('../raw_test.txt','w')
#     else:
#         assert False,"we only have two ah"
#     for _, row in tqdm(df[required_col].iterrows()):
#         prefix_FastText = '__label__'
#         label = prefix_FastText+str(row.brand_for_fasttext)
#         sentence = str(row.item_name)
#         output_form = label + ' '+ sentence
#         f.write(output_form)
#         f.write('\n')
#     f.close()
# e = time.time()
# print ('preparing take {} mins'.format((e-s)/60.0))





