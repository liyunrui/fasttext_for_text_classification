"""

@author: Ray

What's difference between hierachical softmax from the high-level explanation?
https://www.quora.com/What-is-hierarchical-softmax

"""
import os
#------
# clearned
#------
print ("clearned data")
# evaluate P@k and R@k
cmd = '../../../fastText/fasttext test ../model/model_cleaned.bin ../cleaned_test.txt'
os.system(cmd)
# obtain the k most likely labels for a piece of text
if not os.path.exists('../prediction/'):
    os.makedirs('../prediction/')
topK = 3
cmd = '../../../fastText/fasttext predict ../model/model_cleaned.bin ../cleaned_test.txt {} > ../prediction/cleaned_test_top3.txt'.format(topK)
os.system(cmd)
#------
# raw
#------
print ("raw data")
# evaluate P@k and R@k
cmd = '../../../fastText/fasttext test ../model/model_raw.bin ../raw_test.txt'
os.system(cmd)
# obtain the k most likely labels for a piece of text
topK = 3
cmd = '../../../fastText/fasttext predict ../model/model_raw.bin ../raw_test.txt {} > ../prediction/raw_test_top3.txt'.format(topK)
os.system(cmd)
