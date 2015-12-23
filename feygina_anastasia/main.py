from svm import *
from lazy_fca import *
from fca_votes import *
from supp_fca import *
from antisupp_fca import *
from supp_falsifiab import *
from smooth_supp_fca import *

df = read_csv('2008_100_300.csv')
df = replace_nan_with_mean(df)
df = category_to_binary(df)
test_df, train_df = divide_test_train(df)

df_for_test = train_df.ix[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                              24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, -9, -8,
                              -7, -6, -5, -4, -3, -2, -1]]
df_for_test = df_for_test.tail(140)


#try_svm_method(train_df, test_df, 2)
#lazy_fca(df_for_test, 2)
#votes_fca(df_for_test, 2)
#supp_fca(df_for_test, 2)

#anti_supp_fca(df_for_test, 2)
#supp_falsifiability(df_for_test, 2)
smooth_supp_fca(df_for_test, 2)
