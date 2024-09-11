"""Reads data files and transforms data


"""

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
import csv


def read_input_files(outcomefn: str, genofn: str, continfn: str, out_scale: bool=False,
    contin_scale: bool=False, geno_encode: str=None, missing: str=None, outcome: str=None,
    included_vars: list[str]=None) -> tuple[pd.DataFrame, dict, list]:
    """Read in data and construct pandas dataframe

    Args:
        outcomefn: Phenotypes (outcomes) filename
        genofn: SNP values filename
        continfn: any continuous data filename
        out_norm: scale outcome values from 0 to 1.0
        contin_norm: scale each continuous variable from 0 to 1.0
        geno_encode: encode genotype data. options are 'add_quad' and 'additive'
        outcome: column header in continfn to use for 'y'
        included_vars: list of variable names to include in analysis; all others excluded

    Returns:
        dataset_df: pandas dataframe
        inputs_map: dictionary with new label as key, original label as value
        unmatched: list of IDs that are not in all input files
    """
    
    y_df = process_continfile(outcomefn, out_scale)
   
    if outcome is None:
        dataset_df = y_df[['ID',y_df.columns[1]]]
    else:
        dataset_df = y_df[['ID',outcome]]

    dataset_df.columns = ['ID', 'y']

    if included_vars:
        included_vars.insert(0, 'ID')

    contin_df = None
    inputs_map = {}
    if continfn:
        contin_df = process_continfile(continfn, contin_scale, missing, included_vars)
        inputs_map={contin_df.columns[i]:contin_df.columns[i] for i in range(0,len(contin_df.columns))}

    if genofn:
        geno_df, geno_map = process_genofile(genofn, geno_encode, missing, included_vars)
        inputs_map.update(geno_map)
    
    geno_df = geno_df.sort_values('ID', ascending=False)
    unmatched = []

    if genofn:
        unmatched.extend(dataset_df[~dataset_df['ID'].isin(geno_df['ID'])]['ID'].tolist())
        unmatched.extend(geno_df[~geno_df['ID'].isin(dataset_df['ID'])]['ID'].tolist())
        dataset_df = pd.merge(dataset_df,geno_df,on="ID", validate='1:1')

    if continfn:
        unmatched.extend(dataset_df[~dataset_df['ID'].isin(contin_df['ID'])]['ID'].tolist())
        unmatched.extend(contin_df[~contin_df['ID'].isin(dataset_df['ID'])]['ID'].tolist())
        dataset_df = pd.merge(dataset_df, contin_df, on="ID", validate='1:1')

    dataset_df.drop(columns=['ID'], inplace=True)

    return dataset_df, inputs_map, unmatched
    

def normalize(val):
    minval = np.nanmin(val)
    diff = np.nanmax(val) - minval
    newval = (val - minval) / diff 
    return newval


def process_continfile(fn: str, scale: bool, missing: str=None, included_vars: list[str]=None) -> pd.DataFrame:
    """Read in continuous data and construct dataframe from values

    Args:
        fn: Phenotypes (outcomes) filename
        scale: normalize values if true
        missing: identifies any missing data in file
        included_vars: restrict set to only variables (column names) in list
            
    Returns: 
        pandas dataframe 
    """
    data = pd.read_table(fn, delim_whitespace=True, header=0, keep_default_na=False)

    if included_vars:
        data=data.loc[:, data.columns.isin(included_vars)]

    if missing:
        data.loc[:,data.columns!='ID'] = data.loc[:,data.columns!='ID'].replace(missing, np.nan)
        
    data.loc[:,data.columns!='ID'] = data.loc[:,data.columns!='ID'].astype(float)

    if scale:
        data.loc[:,data.columns!='ID'] = data.loc[:,data.columns!='ID'].apply(normalize, axis=0)

    return data
    
    
def additive_encoding(genos):
    return genos.map({'0':-1,'1':0, '2':1, np.nan:np.nan})

def add_quad_encoding_second(genos):
    return genos.map({'0': -1, '1':2, '2':-1, np.nan:np.nan})
    
def add_quad_encoding(df):
    df.iloc[:, ::2] = df.iloc[:, ::2].astype(str).apply(additive_encoding)
    df.iloc[:, 1::2] = df.iloc[:, 1::2].astype(str).apply(add_quad_encoding_second)
    return df

def process_genofile(fn: str, encoding: str, missing: str=None, included_vars: list[str]=None) ->  tuple[pd.DataFrame, dict]:
    """Read in genotype data and construct dataframe from values

    Args:
        fn: Phenotypes (outcomes) filename
        encoding: Genotype encoding type
        missing: identifies missing data in file
        included_vars: restrict set to only variables in list

    Returns: 
        data: pandas dataframe
        geno_map: dictionary with new label as key, original label as value
    """
    data = pd.read_table(fn, delim_whitespace=True, header=0, keep_default_na=False)

    if included_vars:
        data=data.loc[:, data.columns.isin(included_vars)]

    if missing:
        # data.replace([missing], np.nan, inplace=True)
        data.loc[:,data.columns!='ID'] = data.loc[:,data.columns!='ID'].replace(missing, np.nan)
    
#     oldcols = list(data.drop('y', axis=1).columns)
    labels = list(data.columns)
    geno_map={}
    
    if encoding == 'additive':
        data = data.astype(str).apply(additive_encoding)
        geno_map = {labels[i]:labels[i] for i in range(0,len(labels))}
        
    if encoding == 'add_quad':
        new_df = data[data.loc[:,data.columns!='ID'].columns.repeat(2)]

        columns = list(new_df.columns)
        columns[::2]= [ x + "-a" for x in new_df.columns[::2]]
        columns[1::2]= [ x + "-b" for x in new_df.columns[1::2]]
        
        geno_map = {columns[i]:data.columns[i//2] for i in range(0,len(columns))}
        
        new_df.columns = columns
        add_quad_encoding(new_df)
        # add back ID column
        new_df.insert(0,"ID",data['ID'])
        data = new_df

    return data, geno_map
    

class NodeColor:
    def __init__(self,name,color,category):
        self.name=name
        self.color=color
        self.category=category

class Category:
    def __init__(self, name,color):
        self.name=name
        self.color=color

class ColorMapping:
    def __init__(self, default_color='white', operator_color='lightgray'):
        self.categories = {}
        self.inputs = {}
        self.default_color = default_color
        self.operator_color = operator_color
    
    def add_category(self, name, color):
        self.categories[name]=Category(name,color)
    
    def get_categories(self):
        return list(self.categories.values())
    
    def add_input(self, label, category, color):
        self.inputs[label]=NodeColor(label,color,category)
    
    def get_category_color(self, name):
        return self.categories[name].color
    
    def get_input_category(self, label):
        return self.inputs[label].category
    
    def add_nodes(self, mapping, category):
        for name in mapping:
            self.inputs[name] = NodeColor(name,mapping[name],category)
    
    def get_input_color(self,name):
        if name in self.inputs:
            return self.inputs[name].color
        else:
            return self.default_color
    
    

def process_var_colormap(colorfn: str=None, node_color: str='lightgray', var_default: str='white',
    geno_encode: str='additive') -> dict:
    """Create color map for graphical output of networks. 

    Args:
        colorfn: name of file to process, when no fn provided only the network nodes
            (PA,PD,PM,PS) are included
        node_default: Colors for nodes not specified in the color map file
        var_default: Default colors for unspecified variables

    Returns: 
        color_map: node name as key and color as value
    """
    color_map = ColorMapping(default_color=var_default, operator_color=node_color)
    
    color_map.add_category('netnodes', color_map.operator_color)
    color_map.add_nodes({'PA':color_map.operator_color,'PD':color_map.operator_color,
    'PM':color_map.operator_color,'PS':color_map.operator_color},'netnodes')
    
    # header for file is category,color,inputs
    if colorfn:
        with open(colorfn) as csv_file:
            #skip header
            heading = next(csv_file)
            reader = csv.reader(csv_file)
            
            for row in reader:
                if not row:
                    continue
                # set category color
                color_map.add_category(row[0],row[1])
                for in_var in row[2:]:
                    color_map.add_input(in_var,row[0],row[1])
    return color_map
    

def split_kfolds(df, nfolds, seed=100):
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    
    train_splits=[]
    test_splits=[]
    
    for i, (train_index, test_index) in enumerate(kf.split(df.index.values)):
        train_splits.append(train_index)
        test_splits.append(test_index)
    
    return train_splits,test_splits
    
def split_statkfolds(df, nfolds, seed=100):
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    
    train_splits=[]
    test_splits=[]
    
    for i, (train_index, test_index) in enumerate(kf.split(df.index.values,df['y'])):
        train_splits.append(train_index)
        test_splits.append(test_index)
    
    return train_splits,test_splits
    
    
def rename_variables(df: pd.DataFrame) -> dict:
    """ Rename variables in dataframe to be indexed version of x

    Args:
        df: dataframe to alter
    
    Returns:
        vmap: new names are keys and original names are values

    """
    newcols = {}
    vmap = {}
    oldcols = list(df.drop('y', axis=1).columns)
    for i in range(len(oldcols)):
        newvar = 'x' + str(i)# + ']'
        newcols[oldcols[i]]=newvar
        vmap['x['+str(i) + ']']=oldcols[i]

    df.rename(newcols, inplace=True, axis=1)

    return vmap

def reset_variable_names(model: str, vmap: dict) -> str:
    """Replace x variables with names in variable map

    Args:
        model: evolved model containing variables with indexed x values ('x[0],x[1],...)
        vmap: dict with key as x variable and value as name to replace with

    Returns: 
        string: model string with variable names updated
    """
    return re.sub(r"((x\[\d+\]))", lambda g: vmap[g.group(1)], model)

def process_grammar_file(grammarfn: str, data: pd.DataFrame) -> str:
    """Reads grammar file into string and adds all x variables present in dataframe

    Args:
        grammarfn: grammar filename to read and modify
        data: dataset to be used with the grammar

    Returns: 
        updated_grammar: grammar text modified for number of variables in data
    """
    with open(grammarfn, "r") as text_file:
        grammarstr = text_file.read()

    nvars = len([xcol for xcol in data.columns if 'x' in xcol])
    updated_grammar=""
    for i,line in enumerate(grammarstr.splitlines()):
        if re.search(r"^\s*<v>",line):
             line = "<v> ::= " + ' | '.join([f"x[{i}]" for i in range(nvars)])
        updated_grammar += line + "\n"
    return updated_grammar

def prepare_split_data(df: pd.DataFrame, train_indexes: np.ndarray, 
                       test_indexes: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Create and return data arrays for training and testing using indexes passed.

    Args:
        df: data set to split
        train_indexes: rows in dataset to make training set
        test_indexes: rows in dataset to make test set

    Returns: 
        X_train: x values in training
        Y_train: y values in training
        X_test: x values for testing
        Y_test: y values for testing
    """
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
    

def generate_splits(ncvs: int, fitness_type: str, df: pd.DataFrame, have_test_file: bool=False, test_df: pd.DataFrame=None, 
                    rand_seed: int=1234) -> tuple[np.ndarray,np.ndarray,pd.DataFrame]:
    """Generate splits for training and testing based on number of cross-validation intervals
        requested.

    Args:
        ncvs: number of splits (cross-validations)
        fitness_type: for 'r-squared' split into specified number of folds, otherwise split balancing classes in data
        df: dataset to use for splitting
        have_test_file: when true use the test_df as the tesing set
        test_df: when using a test_file contains the testing dataset
        rand_seed: controls split


    Returns: 
        train_splits: 2-D array of indexes to use in traininig
        test_splits: 2-D array of indexes to use in testing
        df: dataset to use with these indexes, concatenated for training and testing when test dataset provided
    """
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


def write_summary(filename: str, best_models: list['deap.creator.Individual'], score_type: str, var_map: dict, 
                  fitness_test: list[float],nmissing: list[int]) -> None:
    """Produce summary file reporting results

    Args:
        filename: name of file to write
        best_models: deap Individual objects from run
        score_type: test used for scoring individuals
        var_map: key is value (x[0],x[1],etc) and value is original column name in dataset
        fitness_test: contains testing fitness scores for each individual
        nmissing: number of missing rows for individual


    Returns: 
        None
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


def write_plots(basefn: str, best_models: list['deap.Creator.Individual'], var_map: dict, 
                inputs_map: dict, color_map: ColorMapping) -> None:
    """Produces png file displaying best models with one per cross-validation.

    Parameters:
        basefn: name of file to write
        best_models: deap Individual objects from run
        var_map: key is value (x[0],x[1],etc) and value is name from dataset adjusted for multiple occurences (Ott encoding)
        inputs_map: key is name (adjusted for Ott encoding), value is original column name in input dataset
        color_map: contains colors to use in plot


    Returns: 
        None
    """

    inputs_map.update({'PA':'PA', 'PM':'PM', 'PS':'PS','PD':'PD'})
    for cv,model in enumerate(best_models,1):
        compressed = compress_weights(model.phenotype)
        modelstr = reset_variable_names(compressed, var_map)
        nodes = construct_nodes(modelstr)
        finalindex = len(nodes)-1
        node_labels={}
        edge_labels={}
        node_colors={}
        categories = set()
        node_size=8
        
        for node in nodes:
            node.num = abs(node.num - finalindex)
            node_labels[node.num] = node.label
            # possible colors: https://matplotlib.org/stable/users/explain/colors/colors.html
            node_colors[node.num] = color_map.get_input_color(inputs_map[node.label])
            
            if node_colors[node.num] != color_map.default_color and \
                node_colors[node.num]  != color_map.operator_color:
                categories.add(color_map.inputs[inputs_map[node.label]].category)
            
            if node.to is not None:
                node.to = abs(node.to - finalindex)
                edge_labels[(node.num,node.to)]="{weight:.2f}".format(weight=float(node.weight))

        edges = []
        for node in nodes:
            if node.to is not None:
                edges.append((node.num,node.to))
        plt.clf()
        fig, ax = plt.subplots()
        Graph(edges, node_layout='dot', arrows=True, node_labels = node_labels, 
            edge_labels=edge_labels, node_color=node_colors, node_size=node_size, ax=ax)
            
            
        if len(categories) > 0:
            # add legend
            node_proxy_artists = []
            for cat in categories:
                proxy =  plt.Line2D(
                    [], [],
                    linestyle='None',
                color=color_map.get_category_color(cat),
                marker='s',
                markersize=node_size,#//1.25,
                label=cat
                )
                node_proxy_artists.append(proxy)
            
            node_legend = ax.legend(handles=node_proxy_artists, loc='lower left')#, title='Categories')
            ax.add_artist(node_legend)
        
        outputfn = basefn + ".cv" + str(cv) + ".png"
        plt.title("\n".join(textwrap.wrap(modelstr, 60)))
        plt.savefig(outputfn, dpi=300)


