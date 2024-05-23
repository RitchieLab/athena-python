import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import re
import numpy as np
from genn.functions import pdiv

from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph
import textwrap


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

def compress_weights(model_str):
    pattern = re.compile(r"([f|p].*?)\s\*\s[x|P]")
    new_model_str = model_str
    m = pattern.search(new_model_str)
    while m:
        match_start = m.start(1)
        i = m.start() - 2
        while new_model_str[i] == '(':
            match_start -= 1
            i -= 1
        value = eval(new_model_str[match_start:m.end(1)])
        value = "{weight:.2f}".format(weight=float(value))
        new_model_str = new_model_str[:match_start] + str(value) + new_model_str[m.end(1):]
        m = pattern.search(new_model_str)
    return new_model_str


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
        fh.write(f"{i+1}\t")
        # extract variables from model
        for match in pattern.finditer(model.phenotype):
            fh.write(f"{var_map[match.group(1)]} ")
        
        fh.write(f"\t{model.fitness.values[0]}")
        fh.write(f"\t{fitness_test[i]}")
        fh.write(f"\t{nmissing[i][0] * 100:.2f}%")
        fh.write(f"\t{nmissing[i][1] * 100:.2f}%")
        fh.write("\n")


    fh.write("\nCV\tModel\n")
    for i,model in enumerate(best_models):
        compressed = compress_weights(model.phenotype)
        compressed = reset_variable_names(compressed, var_map)
        fh.write(f"\t{i+1}\t{compressed}\n")
        
    fh.write("\n***** Original Networks *****")
    fh.write("\nCV\tModel\n")
    for i,model in enumerate(best_models):
        fh.write(f"\t{i+1}\t{model.phenotype}\n")
    
    fh.close()



class Node:
    def __init__(self, label=None, weight=None, to=None, num=None):
        self.label = label
        self.weight = weight
        self.to = to
        self.num = num

def construct_nodes(modelstr):
    """
    Returns node objects representing the network

    :param modelstr: String containing GE network
    :return: nodes constructed from the model
    """    
    model = modelstr.replace('([', ' [').replace('])', '] ').replace('(', ' ( ').replace(')', ' ) ')
    ignore = {','}
    elements = model.split()
    
    stack = deque()

    stack.append(elements[0])
    i = 1
    nodes = []
    
    # use stack to construct the nodes/edges
    while stack:
        if elements[i] in ignore:
            i+=1
        elif elements[i] == ')':
            enditem = elements[i]
            item = stack.pop()
            popitems = list()
            # pop and keep all the items that are not the matching enditem
            while item != '(':
                popitems.append(item)
                item = stack.pop()
            
            # this will now be 3 elements with the first being a node, second * and third the weight            
            if isinstance(popitems[0], Node):
                popitems[0].weight = popitems[2]
                node = popitems[0]
            else:
                node = Node(weight = popitems[2], num=len(nodes), label=popitems[0])
                nodes.append(node)
            
            # push the node back on to the stack
            stack.append(node)
            i += 1
        elif elements[i] == ']':
            # should only be nodes on stack until '['
            item=stack.pop()
            function_nodes = list()
            while item != '[':
                function_nodes.append(item.num)
                item=stack.pop()
            
            # element after should be a node
            item = stack.pop()
            if not isinstance(item, Node):
                node = Node(num=len(nodes), label=item)
                nodes.append(node)
            else:
                node = item
                
            for n in function_nodes:
                nodes[n].to = node.num
            # when empty all nodes have been processed
            if not stack:
                break
            else:
                stack.append(node)
                i += 1
        else:
            stack.append(elements[i])
            i+=1
    
    return nodes


def write_plots(basefn, best_models, var_map):
    """
    Writes jpg file displaying best models with one per cross-validation.

    :param filename: Output filename base
    :param best_models: List of individual objects from population
    :param var_map: dict with x index as key and original name as vallue
    :return: Nothing
    """

    for cv,model in enumerate(best_models,1):
        compressed = compress_weights(model.phenotype)
        modelstr = reset_variable_names(compressed, var_map)
        nodes = construct_nodes(modelstr)
        finalindex = len(nodes)-1
        node_labels={}
        edge_labels={}
        for node in nodes:
            node.num = abs(node.num - finalindex)
            node_labels[node.num] = node.label
            if node.to is not None:
                node.to = abs(node.to - finalindex)
                edge_labels[(node.num,node.to)]="{weight:.2f}".format(weight=float(node.weight))

        edges = []
        for node in nodes:
            if node.to is not None:
                edges.append((node.num,node.to))
        plt.clf()
        Graph(edges, node_layout='dot', arrows=True, node_labels = node_labels, 
            edge_labels=edge_labels, node_size=5)
        outputfn = basefn + ".cv" + str(cv) + ".png"
        plt.title("\n".join(textwrap.wrap(modelstr, 60)))
        plt.savefig(outputfn)

