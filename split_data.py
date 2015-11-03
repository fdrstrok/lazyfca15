#-*- coding: utf8 -*-

import sys

def k_fold_cross_validation(X, K, randomise = False):
	"""
	Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
	if randomise: from random import shuffle; X=list(X); shuffle(X)
	for k in xrange(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation, k

if __name__ == '__main__':
    source = sys.argv[1]
    with open(source, 'r') as data:
            for training, validation, k in k_fold_cross_validation(data.readlines(), K = 7):  
                tf = open("%s_train_%d.txt" % (source, k), 'w')
                tf.writelines(training)
                tv = open("%s_validation_%d.txt" % (source, k), 'w')
                tv.writelines(validation)
                tf.close()
                tv.close()
