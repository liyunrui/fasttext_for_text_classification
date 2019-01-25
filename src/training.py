import os

#------
# clearned
#------
print ("training")
cmd = '../../../fastText/fasttext supervised -input ../data/train.txt -output ../model/model'
os.system(cmd)

# #------
# # raw
# #------
# print ("raw data")
# cmd = '../../../fastText/fasttext supervised -input ../raw_train.txt -output ../model/model_raw -loss hs'
# os.system(cmd)