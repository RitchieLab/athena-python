import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import re
import numpy as np

def read_input_files(outcomefn, genofn, continfn, out_scale=False,
    contin_scale=False, geno_encode=None):
    """
    Read in data and construct pandas dataframe

        Parameters:
            outcomefn: Phenotypes (outcomes)
            genofn: SNP values
            continfn: any continuous data
            out_norm: scale outcome values from 0 to 1.0
            contin_norm: scale each continuous variable from 0 to 1.0
            geno_encode: encode genotype data. options are 'add_quad' and 'additive'

        Returns: 
            pandas dataframe
    """
    
    y_df = process_continfile(outcomefn, out_scale)
    y_df.columns = ['y']
    
    contin_df = None
    if continfn:
        contin_df = process_continfile(continfn, contin_scale)
    
    if genofn:
        geno_df = process_genofile(genofn, geno_encode)
    
    dataset_df = y_df
    if genofn:
        dataset_df = pd.concat([dataset_df, geno_df], axis=1)
    if continfn:
        dataset_df = pd.concat([dataset_df, contin_df], axis=1)
    
    return dataset_df
    

def normalize(val):
    minval = min(val)
    diff = max(val) - minval
    newval = (val - minval) / diff 
    return newval


def process_continfile(fn, scale):
    """
    Read in continuous data and construct dataframe from values

        Parameters:
            fn: Phenotypes (outcomes) file
            normalize: boolean for controlling normalization

        Returns: 
            pandas dataframe
    """
    data = pd.read_table(fn, delim_whitespace=True, header=0)
    
    if scale:
        data = data.apply(normalize, axis=0)

    return data
    
    
def additive_encoding(genos):
    return genos.map({'0':-1,'1':0, '2':1})

def add_quad_encoding_second(genos):
    return genos.map({'0': -1, '1':2, '2':-1})
    
def add_quad_encoding(df):
    df.iloc[:, ::2] = df.iloc[:, ::2].astype(str).apply(additive_encoding)
    df.iloc[:, 1::2] = df.iloc[:, 1::2].astype(str).apply(add_quad_encoding_second)
    return df

def process_genofile(fn, encoding):
    """
    Read in genotype data and construct dataframe from values

        Parameters:
            fn: Phenotypes (outcomes) file
            encoding: string for controlling encoding 

        Returns: 
            pandas dataframe
    """
    data = pd.read_table(fn, delim_whitespace=True, header=0)
    
    if encoding == 'additive':
        data = data.astype(str).apply(additive_encoding)
        
    if encoding == 'add_quad':
        new_df = data[data.columns.repeat(2)]
        columns = list(new_df.columns)
        columns[::2]= [ x + "-a" for x in new_df.columns[::2]]
        columns[1::2]= [ x + "-b" for x in new_df.columns[1::2]]
        new_df.columns = columns
        add_quad_encoding(new_df)
        data = new_df
            

    return data


def split_kfolds(df, nfolds, seed=100):
#     kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    
    train_splits=[]
    test_splits=[]
    
    for i, (train_index, test_index) in enumerate(kf.split(df.index.values)):
        train_splits.append(train_index)
        test_splits.append(test_index)
    
    return train_splits,test_splits
    
def split_statkfolds(df, nfolds, seed=100):
#     kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    
    train_splits=[]
    test_splits=[]
    
    for i, (train_index, test_index) in enumerate(kf.split(df.index.values,df['y'])):
        train_splits.append(train_index)
        test_splits.append(test_index)
    
    return train_splits,test_splits
    
    
def rename_variables(df):
    newcols = {}
    vmap = {}
    oldcols = list(df.drop('y', axis=1).columns)
    for i in range(len(oldcols)):
        newvar = 'x' + str(i)# + ']'
        newcols[oldcols[i]]=newvar
        vmap[newvar]=oldcols[i]

    df.rename(newcols, inplace=True, axis=1)

    return vmap

def process_grammar_file(grammarfn, data):
#     print(nvars)
    with open(grammarfn, "r") as text_file:
        grammarstr = text_file.read()

    nvars = len([xcol for xcol in data.columns if 'x' in xcol])
    updated_grammar=""
    for i,line in enumerate(grammarstr.splitlines()):
        if re.search("^\s*<v>",line):
#              print("MATCH")
             line = "<v> ::= " + ' | '.join([f"x[{i}]" for i in range(nvars)])
        updated_grammar += line + "\n"
#     print(updated_grammar)
    return updated_grammar

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
    # print(X_train)
    
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