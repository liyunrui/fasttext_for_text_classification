"""
Getting and preparing the data for FastText classification

@author : Ray

"""
import pandas as pd
import time
from tqdm import tqdm

#--------------------
# loading data
#--------------------
train_path = '/data/brand_classification/cleaned_train.csv'
test_path = '/data/brand_classification/cleaned_test.csv'
train = pd.read_csv(train_path) # 5678025 rows
test = pd.read_csv(test_path) # 1419508 rows
train.dropna(subset = ['brand','item_name'], inplace = True)
test.dropna(subset = ['brand','item_name'], inplace = True)
train.drop_duplicates(subset = ['item_name'], inplace = True)
test.drop_duplicates(subset = ['item_name'], inplace = True)
# cannot have blank within the whole brand words, it causes some problems when running fasttext
train['brand_for_fasttext'] = train.brand.apply(lambda x: x.replace(' ','_'))
test['brand_for_fasttext'] = test.brand.apply(lambda x: x.replace(' ','_'))
# need a dict for convert the predicted brand into original version of the brand

#--------------------
# core
#--------------------
s = time.time()

required_col = ['brand_for_fasttext','item_name']
for ix, df in enumerate([train, test]):
    if ix == 0:
        f = open('../cleaned_train.txt','w')
    elif ix == 1:
        f = open('../cleaned_test.txt','w')
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
# loading data
#--------------------
train_path = '/data/brand_classification/raw_train.csv'
test_path = '/data/brand_classification/raw_test.csv'
train = pd.read_csv(train_path) # 5678025 rows
test = pd.read_csv(test_path) # 1419508 rows
train.dropna(subset = ['brand','item_name'], inplace = True)
test.dropna(subset = ['brand','item_name'], inplace = True)
train.drop_duplicates(subset = ['item_name'], inplace = True)
test.drop_duplicates(subset = ['item_name'], inplace = True)
# cannot have blank within the whole brand words, it causes some problems when running fasttext
train['brand_for_fasttext'] = train.brand.apply(lambda x: x.replace(' ','_'))
test['brand_for_fasttext'] = test.brand.apply(lambda x: x.replace(' ','_'))

#--------------------
# core
#--------------------
required_col = ['brand_for_fasttext','item_name']
ix = 0
s = time.time()
for ix,df in enumerate([train, test]):
    if ix == 0:
        f = open('../raw_train.txt','w')
    elif ix == 1:
        f = open('../raw_test.txt','w')
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





