# Test of Implication Test by Threshold

import sys

#max_index = sys.argv[1]

max_files = 11
ithr_min  = 5           # min Threshold * 10
ithr_max  = 16          # max Threshold * 10

max_index = max_files
accuracy  = {}          # Accuracy for different Threshold
accuracyf = {}          # Accuracy for different number of files
acc_max   = 0           # best Accuracy for different Threshold
acc_maxf  = 0           # best Accuracy for different number of files
ithr_best = ithr_min    # index of best Accuracy for different Threshold
imax_best = 1           # index of best Accuracy for different number of files
totalf    = {}           # total number of examples through all files

for ithreshold in xrange(ithr_min, ithr_max, 1):
    threshold = 0.1*ithreshold
    
    execfile("test_impl.py")
    
    accuracy[ithreshold] = cv_res_p["positive_positive"] + cv_res_p["negative_negative"]    # Accuracy
    if accuracy[ithreshold] > acc_max :       
        acc_max   = accuracy[ithreshold]
        ithr_best = ithreshold
        
    print 'Threshold =', threshold, '\tAccuracy =', accuracy[ithreshold]
    print 'True  Positive :\t', cv_res["positive_positive"], '\t', cv_res_p["positive_positive"]
    print 'True  Negative :\t', cv_res["negative_negative"], '\t', cv_res_p["negative_negative"]
    print 'False Positive :\t', cv_res["negative_positive"], '\t', cv_res_p["negative_positive"]
    print 'False Negative :\t', cv_res["positive_negative"], '\t', cv_res_p["positive_negative"]
    print 'Contradictory  :\t', cv_res["contradictory"], '\t', cv_res_p["contradictory"]
    print 'Total          :\t', cv_res["total"], '\t', cv_res_p["total"]
    print '========================================================'
print '========================================================'
   
for max_index in xrange(2, int(max_files)+1):
    threshold = 0.1*ithr_best   # Best Threshold will be used
    
    execfile("test_impl.py")
    
    accuracyf[max_index] = cv_res_p["positive_positive"] + cv_res_p["negative_negative"]    # Accuracy
    if accuracyf[max_index] > acc_maxf :       
        acc_maxf  = accuracyf[max_index]
        imax_best = max_index
    totalf[max_index] = cv_res["total"]
    
    print 'Number of Files =', max_index-1, '\tAccuracy =', accuracyf[max_index]
    print 'True  Positive :\t', cv_res["positive_positive"], '\t', cv_res_p["positive_positive"]
    print 'True  Negative :\t', cv_res["negative_negative"], '\t', cv_res_p["negative_negative"]
    print 'False Positive :\t', cv_res["negative_positive"], '\t', cv_res_p["negative_positive"]
    print 'False Negative :\t', cv_res["positive_negative"], '\t', cv_res_p["positive_negative"]
    print 'Contradictory  :\t', cv_res["contradictory"], '\t', cv_res_p["contradictory"]
    print 'Total          :\t', cv_res["total"], '\t', cv_res_p["total"]
    print '========================================================'  
print "Threshold \tAccuracy"
print "--------- \t--------------"

for ithreshold in xrange(ithr_min, ithr_max, 1):
    if ithreshold == ithr_best:
        bestsign = "!!! Best !!!"
    else:
        bestsign = " "
    print 0.1*ithreshold, "\t\t", accuracy[ithreshold], "\t", bestsign

print '========================================================'

print "Files  \t\tExamples \tAccuracy"
print "-------\t\t-------- \t--------------"

for max_index in xrange(2, int(max_files)+1):
    if max_index == imax_best:
        bestsign = "!!! Best !!!"
    else:
        bestsign = " "
    print max_index-1, "\t\t", totalf[max_index], "\t\t", accuracyf[max_index], "\t", bestsign

print '========================================================'
