#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: scott dudek
"""

import grape.grape as grape
import grape.algorithms as algorithms
#from functions import add, sub, mul, pdiv, plog, exp, psqrt
from genn.genn_functions import activate_sigmoid, PA, PM, PS, PD, pdiv
from genn import genn_setup

import random
import sys

from sklearn.model_selection import train_test_split
import csv
import os
from os import path
import pandas as pd
import numpy as np
# from deap import creator, base, tools
from deap import tools
import genn.genn_utils
from utilities import *

import warnings
warnings.filterwarnings("ignore")

GENOME_REPRESENTATION = 'list'
MAX_TREE_DEPTH = 50
MAX_GENOME_LENGTH = None


if len(sys.argv) == 1:
    sys.argv.append("-h")
params = parameters.set_params(sys.argv[1:])

# print(params)
if not parameters.valid_parameters(params):
    sys.exit()
    
random.seed(params['RANDOM_SEED'])
np.random.seed(params['RANDOM_SEED'])
# print(params)

test_data = None
# process the input files to create the appropriate X and Y sets for testing training
data = data_processing.read_input_files(outcomefn=params['OUTCOME_FILE'], genofn=params['GENO_FILE'],
    continfn=params['CONTIN_FILE'], geno_encode=params['GENO_ENCODE'], 
    out_scale=params['SCALE_OUTCOME'], contin_scale=params['SCALE_CONTIN'],
    missing=params['MISSING'])

if params['TEST_OUTCOME_FILE']:
    test_data = data_processing.read_input_files(outcomefn=params['TEST_OUTCOME_FILE'], genofn=params['TEST_GENO_FILE'],
    continfn=params['TEST_CONTIN_FILE'], geno_encode=params['GENO_ENCODE'], 
    out_scale=params['SCALE_OUTCOME'], contin_scale=params['SCALE_CONTIN'],
    missing=params['MISSING'])

(train_splits, test_splits, data) = data_processing.generate_splits(fitness_type=params['FITNESS'],
    ncvs=params['CV'], df=data, have_test_file=params['TEST_OUTCOME_FILE'], 
    test_df=test_data, rand_seed=params['RANDOM_SEED'])

var_map = data_processing.rename_variables(data)

grammarstr = data_processing.process_grammar_file(params['GRAMMAR_FILE'], data)
# print(grammarstr)
BNF_GRAMMAR = grape.Grammar(grammarstr, params['CODON_CONSUMPTION'])

toolbox=genn_setup.configure_toolbox(params['GENOME_TYPE'], params['FITNESS'], params['SELECTION'])

REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max',
                'fitness_test', 
          'best_ind_length', 'avg_length', 
          'best_ind_nodes', 'avg_nodes', 
          'best_ind_depth', 'avg_depth', 
          'avg_used_codons', 'best_ind_used_codons', 
        #  'behavioural_diversity',
          'structural_diversity', #'fitness_diversity',
          'selection_time', 'generation_time', 'best_phenotype']

for cv in range(params['CV']):
    print("\nCV: ", cv+1, "\n")
    
    (X_train,Y_train,X_test,Y_test) = data_processing.prepare_split_data(data, 
        train_splits[cv], test_splits[cv])
    
    if params['INIT'] == 'random':
        population = toolbox.populationCreator(pop_size=params['POPULATION_SIZE'],
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_genome_length=params['MIN_INIT_GENOME_LENGTH'],
                                           max_init_genome_length=['MAX_INIT_GENOME_LENGTH'],
                                           max_init_depth=MAX_TREE_DEPTH,
                                           codon_size=params['CODON_SIZE'],
                                           codon_consumption=params['CODON_CONSUMPTION'],
                                           genome_representation=GENOME_REPRESENTATION
                                           )
    else:
        population = toolbox.populationCreator(pop_size=params['POPULATION_SIZE'],
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_depth=params['MIN_INIT_TREE_DEPTH'],
                                           max_init_depth=params['MAX_INIT_TREE_DEPTH'],
                                           codon_size=params['CODON_SIZE'],
                                           codon_consumption=params['CODON_CONSUMPTION'],
                                           genome_representation=GENOME_REPRESENTATION
                                            )
                                            
    # define the hall-of-fame object:
    hof = tools.HallOfFame(params['HALLOFFAME_SIZE'])
    
    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    
    
    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, 
                                              cxpb=params['P_CROSSOVER'], 
                                              mutpb=params['P_MUTATION'],
                                              ngen=params['GENERATIONS'], 
                                              elite_size=params['ELITE_SIZE'],
                                              bnf_grammar=BNF_GRAMMAR,
                                              codon_size=params['CODON_SIZE'],
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              max_genome_length=MAX_GENOME_LENGTH,
                                              points_train=[X_train, Y_train],
                                              points_test=[X_test, Y_test],
                                              codon_consumption=params['CODON_CONSUMPTION'],
                                              report_items=REPORT_ITEMS,
                                              genome_representation=GENOME_REPRESENTATION,
                                              stats=stats, halloffame=hof, verbose=False)

    import textwrap

    
    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")

    selection_time = logbook.select("selection_time")
    generation_time = logbook.select("generation_time")
    gen, invalid = logbook.select("gen", "invalid")
    avg_used_codons = logbook.select("avg_used_codons")
    best_ind_used_codons = logbook.select("best_ind_used_codons")
    
    fitness_test = logbook.select("fitness_test")
    
    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    structural_diversity = logbook.select("structural_diversity") 


    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
#     print("Test Fitness: ", genn_setup.fitness_eval(hof.items[0], [X_test,Y_test])[0])
    print("Test Fitness: ", fitness_test[-1])
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')

    import csv    
    header = REPORT_ITEMS
    
    with open(params['OUT'] +'.'+ str(cv+1) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value], 
                             fitness_test[value],
                             best_ind_length[value], 
                             avg_length[value], 
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value], 
                           #  behavioural_diversity[value],
                             structural_diversity[value],
                          #   fitness_diversity[value],
                             selection_time[value], 
                             generation_time[value]])
    