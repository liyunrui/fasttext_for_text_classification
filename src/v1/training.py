import os

#------
# clearned
#------
print ("clearned data")
cmd = '../../../fastText/fasttext supervised -input ../cleaned_train.txt -output ../model/model_cleaned'
os.system(cmd)

#------
# raw
#------
print ("raw data")
cmd = '../../../fastText/fasttext supervised -input ../raw_train.txt -output ../model/model_raw -loss hs'
os.system(cmd)