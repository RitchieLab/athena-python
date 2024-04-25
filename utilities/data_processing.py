import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import re
import numpy as np
from genn.functions import pdiv

def read_input_files(outcomefn, genofn, continfn, out_scale=False,
    contin_scale=False, geno_encode=None, missing=None):
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
        contin_df = process_continfile(continfn, contin_scale, missing)
    
    if genofn:
        geno_df = process_genofile(genofn, geno_encode, missing)
    
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


def process_continfile(fn, scale, missing=None):
    """
    Read in continuous data and construct dataframe from values

        Parameters:
            fn: Phenotypes (outcomes) file
            normalize: boolean for controlling normalization
            missing: string identifying any missing data

        Returns: 
            pandas dataframe
    """
    data = pd.read_table(fn, delim_whitespace=True, header=0, keep_default_na=False)
    
    if missing:
        data.replace([missing], np.nan, inplace=True)
    
    data = data.astype(float)
    
    if scale:
        data = data.apply(normalize, axis=0)

    return data
    
    
def additive_encoding(genos):
    return genos.map({'0':-1,'1':0, '2':1, np.nan:np.nan})

def add_quad_encoding_second(genos):
    return genos.map({'0': -1, '1':2, '2':-1, np.nan:np.nan})
    
def add_quad_encoding(df):
    df.iloc[:, ::2] = df.iloc[:, ::2].astype(str).apply(additive_encoding)
    df.iloc[:, 1::2] = df.iloc[:, 1::2].astype(str).apply(add_quad_encoding_second)
    return df

def process_genofile(fn, encoding, missing=None):
    """
    Read in genotype data and construct dataframe from values

        Parameters:
            fn: Phenotypes (outcomes) file
            encoding: string for controlling encoding 
            missing: string identifying any missing data

        Returns: 
            pandas dataframe
    """
    data = pd.read_table(fn, delim_whitespace=True, header=0, keep_default_na=False)

    if missing:
        data.replace([missing], np.nan, inplace=True)
        
    
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
        vmap['x['+str(i) + ']']=oldcols[i]

    df.rename(newcols, inplace=True, axis=1)

    return vmap

def reset_variable_names(model, vmap):
    """
    Replace x variables with names in variable map

        Parameters:
            model: string of model
            vmap: dict with key as x variable and value as new name

        Returns: 
            string
    """
    return re.sub(r"((x\[\d+\]))", lambda g: vmap[g.group(1)], model)

def process_grammar_file(grammarfn, data):
    with open(grammarfn, "r") as text_file:
        grammarstr = text_file.read()

    nvars = len([xcol for xcol in data.columns if 'x' in xcol])
    updated_grammar=""
    for i,line in enumerate(grammarstr.splitlines()):
        if re.search(r"^\s*<v>",line):
             line = "<v> ::= " + ' | '.join([f"x[{i}]" for i in range(nvars)])
        updated_grammar += line + "\n"
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
    

def generate_splits(ncvs, fitness_type, df, have_test_file=False, test_df=None, rand_seed=1234):
    if ncvs > 1:
        if fitness_type== 'r-squared':
            (train_splits, test_splits) = split_kfolds(df, ncvs, 
                rand_seed)
        else:
            (train_splits, test_splits) = split_statkfolds(df, ncvs, 
                rand_seed)
    else:
        train_splits = np.zeros((1,df.shape[0]))
        train_splits[0] = np.array([i for i in range(df.shape[0])])
        if not have_test_file:
            test_splits = np.zeros((1,0))
        else:
            test_splits = np.zeros((1, test_df.shape[0]))
            test_splits[0] = np.array([i for i in range(df.shape[0], test_df.shape[0] + df.shape[0])])
            df = pd.concat([df, test_df], axis=0)

    return train_splits, test_splits, df

def evaluate_constant(match):
    new_constant = eval(match.group(1))
    first=""
#     if match.group()[0] == '('
#         first = '('
    return '(' + str(new_constant) + " * " + match.group()[-1]


def compress_expression(funct_name, model_str):
    pattern = re.compile(funct_name)
    included = set()
    new_model_str = model_str
    for match in pattern.finditer(model_str):
#         print(match.start())
#         print(match.end())
        if match.start() in included:
#             print("found already")
            continue
        nopen=0
        nclose=0
        start = match.start()
        for i in range(match.start()-2, 0, -1):
            if model_str[i] == '(':
                start = i+1
            else:
                break
#         print(f"start={start} match.start={match.start()} match.end={match.end()}")       
        
#         for i in range(match.end(), len(model_str)):
        for i in range(start, len(model_str)):
#             print(f"{i} --> {model_str[i]}")
            if model_str[i] == '(':
                nopen+=1
            elif model_str[i] == ')':
                nclose+=1
                if nclose == nopen:
#                     print(f"EVALUATE: {model_str[match.start():i+1]}")
#                     print(f"EVALUATE: {model_str[start:i+1]}")
#                     value = eval(model_str[match.start():i+1])
                    value = eval(model_str[start:i+1])
#                     print(model_str[match.start():i])
#                     new_model_str = new_model_str.replace(model_str[match.start():i], str(value))
                    new_model_str = new_model_str.replace(model_str[start-1:i+1], f" ({value} ")
#                     print(new_model_str)
#                     print(value)
                    break
            elif model_str[i] == 'p' or model_str[i] == 'f':
#                 print(f"added into included {i}")
                included.add(i)
#         exit()
    return new_model_str

def compress_weights(model_str):
    """
    Compresses the weights evolved in the models to simplify output. 
    Constants are defined by a parenthetical expression
    (float(2.18) + float(7.95))
    pdiv(float(8.08),float(0.65))

    :param model_str: string containing model
    :return: string with constants compressed
    """
#     print(model_str)
    new_str = compress_expression('pdiv', model_str)
#     print(new_str)
    
    new_str = compress_expression('float', new_str)
#     print(f"OLD STRING={model_str}")
#     print(f"NEW STRING={new_str}")
    return new_str
#     exit()
    
    
    # match each and then 
#     print(model_str)
# #     div_pattern = re.compile(r"(pdiv\(.+)\))")
# #     div_pattern = re.compile(r"pdiv\(.+?\))\s\*\s[x|P]")
#     div_pattern = re.compile(r"\((\(*pdiv\(.+?\))\s\*\s[x|P]")
# 
# #     new_constants = []
# #     for match in div_pattern.finditer(model_str):
# #         print(f"MATCH: {match.group(1)}")
# #         new_constants.append(eval(match.group(1)))
# #         print(new_constants[-1])
#         
#     new_model_str = div_pattern.sub(evaluate_constant, model_str)
#     
#     print("")
#     print(new_model_str)
#     print("")
#     
# #     float_pattern = re.compile(r"(float\(.+?\))\s\*\s[x|P]")
#     float_pattern = re.compile(r"\((\(*float\(.+?\))\s\*\s[x|P]")
# 
# #     for match in float_pattern.finditer(new_model_str):
# #         print(f"MATCH: {match.group(1)} <--> {match.group()}")
# #     final_model_str = float_pattern.sub(evaluate_constant, new_model_str)
# #     print(final_model_str)
#     
#     # first remove all pdivs
#     div_pattern = re.compile(r"pdiv")
#     pdiv_included = set()
#     new_model_str = model_str
#     for match in div_pattern.finditer(model_str):
#         print(match.start())
#         print(match.end())
#         if match.start() in pdiv_included:
#             print("found already")
#             continue
#         nopen=0
#         nclose=0
#         for i in range(match.end(), len(model_str)):
# #             print(f"{i} --> {model_str[i]}")
#             if model_str[i] == '(':
#                 nopen+=1
#             elif model_str[i] == ')':
#                 nclose+=1
#                 if nclose == nopen:
# #                     print(f"EVALUATE: {model_str[match.start():i]}")
#                     value = eval(model_str[match.start():i+1])
#                     print(model_str[match.start():i])
#                     new_model_str = new_model_str.replace(model_str[match.start():i], str(value))
#                     print(new_model_str)
#                     print(value)
#             elif model_str[i] == 'p':
# #                 print(f"added into pdiv_included {i}")
#                 pdiv_included.add(i)
# #         exit()
#     exit()
    
    
    
#     return final_model_str


def write_summary(filename, best_models, score_type, var_map, fitness_test,nmissing):
    """
    Writes summary file displaying best models and scores across all cross-validations.

    :param filename: Output filename
    :param best_models: List of individual objects from population
    :param score_type: String with fitness type used
    :param var_map: dict with x index as keys
    :return: Nothing
    """
    header = f"CV\tVariables\t{score_type} Training\tTesting\tTraining-missing\tTesting-missing\n"
    
    fh = open(filename, "w")
    fh.write(header)
    
    pattern = re.compile(r"(x\[\d+\])")
    

    for i,model in enumerate(best_models):
#         print(model.phenotype)
        fh.write(f"{i+1}\t")
        # extract variables from model
        for match in pattern.finditer(model.phenotype):
#             print(match.group(1))
            fh.write(f"{var_map[match.group(1)]} ")
        
        fh.write(f"\t{model.fitness.values[0]}")
        fh.write(f"\t{fitness_test[i]}")
        fh.write(f"\t{nmissing[i][0] * 100:.2f}%")
        fh.write(f"\t{nmissing[i][1] * 100:.2f}%")
        fh.write("\n")


    fh.write("\nCV\tModel\n")
    for i,model in enumerate(best_models):
#         compress_weights(model.phenotype)
        compressed = compress_weights(model.phenotype)
#         compressed = re.sub(r"((x\[\d+\]))", lambda g: var_map[g.group(1)], compressed)
        compressed = reset_variable_names(compressed, var_map)
#         fh.write(f"\t{i+1}\t{compress_weights(model.phenotype)}\n")
        fh.write(f"\t{i+1}\t{compressed}\n")
        
    fh.write("\n***** Original Networks *****")
    fh.write("\nCV\tModel\n")
    for i,model in enumerate(best_models):
#         compress_weights(model.phenotype)
        fh.write(f"\t{i+1}\t{model.phenotype}\n")
#         fh.write(f"\t{i+1}\t{compress_weights(model.phenotype)}\n")
    
    fh.close()

