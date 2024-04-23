# utility functions for use with all variations of GENN (standard, leap and mcge)
import sys
from sklearn.model_selection import KFold
import numpy as np
import re

# split using kfolds approach
# params
#   df input set as dataframe
#   number of folds
#   random seed (default=100) 
# return splits  
def split_kfolds(df, nfolds, seed=100):
#     kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    
    train_splits=[]
    test_splits=[]
    
    for i, (train_index, test_index) in enumerate(kf.split(df.index.values)):
        train_splits.append(train_index)
        test_splits.append(test_index)
    
    return train_splits,test_splits

# set up data for running GENN 
def prepare_split_data(df, train_indexes, test_indexes):
    traindf = df.iloc[train_indexes]
    testdf = df.iloc[test_indexes]
    
    train_rows = traindf.shape[0]
    train_cols = traindf.shape[1]-1
    
    X_train = np.zeros([train_rows,train_cols], dtype=float)
    Y_train = np.zeros([train_rows,], dtype=float)
    for i in range(train_rows):
        for j in range(train_cols):
            X_train[i,j] = traindf['x'+str(j)].iloc[i]
    for i in range(train_rows):
        Y_train[i] = traindf['y'].iloc[i]
    
    test_rows=testdf.shape[0]
    test_cols=testdf.shape[1]-1

    X_test = np.zeros([test_rows,test_cols], dtype=float)
    Y_test = np.zeros([test_rows,], dtype=float)
    for i in range(test_rows):
        for j in range(test_cols):
            X_test[i,j] = testdf['x'+str(j)].iloc[i]
    for i in range(test_rows):
        Y_test[i] = testdf['y'].iloc[i]
    
    X_train = np.transpose(X_train)
    X_test = np.transpose(X_test)
    
    return X_train,Y_train,X_test,Y_test
    

# update grammar file to include correct number of input variables
# params
#   grammar: string 
#   nvars: number of variables in data
# return updated grammar  
def update_grammar_file(grammar, nvars):
    print(grammar)
    updated_grammar=""
    for i,line in enumerate(grammar.splitlines()):
        if re.search("^\s*<v>",line):
            print(line)
            exit()
            line = "<v> ::= " + ' | '.join([f"x[{i}]" for i in range(nvars)])
        updated_grammar += line + "\n"
    print(updated_grammar)
    return updated_grammar
    