#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: scott dudek

Pyhon implementation of ATHENA software

"""

import grape.grape as grape
import grape.algorithms as algorithms
from genn.functions import activate_sigmoid, PA, PM, PS, PD, pdiv
from genn import alg_setup
from genn import parallel 

import random
import sys

from sklearn.model_selection import train_test_split
import csv
import os
from os import path
import pandas as pd
import numpy as np
from deap import tools
from utilities import *

import warnings
warnings.filterwarnings("ignore")

GENOME_REPRESENTATION = 'list'
MAX_TREE_DEPTH = 50
MAX_GENOME_LENGTH = None

proc_rank = 0
nprocs = 1
comm = None

if parallel.has_mpi:
    nprocs = parallel.get_nprocs()
    proc_rank = parallel.get_rank()

proceed = True

if len(sys.argv) == 1:
    sys.argv.append("-h")
    proceed = False
    
if proc_rank == 0:
    params = parameters.set_params(sys.argv[1:], has_mpi=parallel.has_mpi)
    if not parameters.valid_parameters(params):
        proceed = False
else:
    params = None
    if not proceed:
        sys.exit()

if nprocs > 1:
    parallel.continue_run(proc_rank, proceed)
    params = parallel.distribute_params(params,proc_rank)
elif not proceed:    
    sys.exit()
    
params['RANDOM_SEED'] += proc_rank*10

random.seed(params['RANDOM_SEED'])
np.random.seed(params['RANDOM_SEED'])
best_models = []
best_fitness_test = []
nmissing = []

data, train_splits, test_splits, var_map,BNF_GRAMMAR = None,None,None,None,None
if proc_rank == 0:
    # process the input files to create the appropriate X and Y sets for testing training
    data, inputs_map, unmatched = data_processing.read_input_files(outcomefn=params['OUTCOME_FILE'], genofn=params['GENO_FILE'],
        continfn=params['CONTIN_FILE'], geno_encode=params['GENO_ENCODE'], 
        out_scale=params['SCALE_OUTCOME'], contin_scale=params['SCALE_CONTIN'],
        missing=params['MISSING'], outcome=params['OUTCOME'], included_vars=params['INCLUDEDVARS'])
    if(len(unmatched)>0):
        print("\nWARNING: The following IDs are not found in all data input files and will be ignored:")
        print(''.join(unmatched))

    test_data = None
    if params['TEST_OUTCOME_FILE']:
        test_data = data_processing.read_input_files(outcomefn=params['TEST_OUTCOME_FILE'], genofn=params['TEST_GENO_FILE'],
        continfn=params['TEST_CONTIN_FILE'], geno_encode=params['GENO_ENCODE'], 
        out_scale=params['SCALE_OUTCOME'], contin_scale=params['SCALE_CONTIN'],
        missing=params['MISSING'])

    (train_splits, test_splits, data) = data_processing.generate_splits(fitness_type=params['FITNESS'],
        ncvs=params['CV'], df=data, have_test_file=params['TEST_OUTCOME_FILE'], 
        test_df=test_data, rand_seed=params['RANDOM_SEED'])
    
    var_map = data_processing.rename_variables(data)
    color_map = data_processing.process_var_colormap(params['COLOR_MAP_FILE'])

    grammarstr = data_processing.process_grammar_file(params['GRAMMAR_FILE'], data)
    BNF_GRAMMAR = grape.Grammar(grammarstr, params['CODON_CONSUMPTION'])
 
 # share data to subordinate processes when using paralllelization
if nprocs > 1:
    data,train_splits, test_splits, var_map, BNF_GRAMMAR = parallel.distribute_data(rank=proc_rank,
        data=data,train_splits=train_splits, test_splits=test_splits, vmap=var_map,
        grammar=BNF_GRAMMAR)

# set up deap toolbox for evolutionary algorithm 
toolbox=alg_setup.configure_toolbox(params['GENOME_TYPE'], params['FITNESS'], params['SELECTION'], params['INIT'])

# configure report items 
REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max',
                'fitness_test', 
          'best_ind_length', 'avg_length', 
          'best_ind_nodes', 'avg_nodes', 
          'best_ind_depth', 'avg_depth', 
          'avg_used_codons', 'best_ind_used_codons', 
        #  'behavioural_diversity',
          'structural_diversity', #'fitness_diversity',
          'selection_time', 'generation_time', 'best_phenotype']

# conduct evolution for specified number of cross-validations
for cv in range(params['CV']):
    if proc_rank == 0:
        print("\nCV: ", cv+1, "\n")
    
    (X_train,Y_train,X_test,Y_test) = data_processing.prepare_split_data(data, 
        train_splits[cv], test_splits[cv])
    
    if params['INIT'] == 'random':
        population = toolbox.populationCreator(pop_size=params['POPULATION_SIZE'],
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_genome_length=params['MIN_INIT_GENOME_LENGTH'],
                                           max_init_genome_length=params['MAX_INIT_GENOME_LENGTH'],
                                           max_init_depth=params['MAX_DEPTH'],
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
                                              max_tree_depth=params['MAX_DEPTH'],
                                              max_genome_length=MAX_GENOME_LENGTH,
                                              points_train=[X_train, Y_train],
                                              points_test=[X_test, Y_test],
                                              codon_consumption=params['CODON_CONSUMPTION'],
                                              report_items=REPORT_ITEMS,
                                              genome_representation=GENOME_REPRESENTATION,
                                              stats=stats, halloffame=hof, verbose=False,
                                              rank=proc_rank, 
                                              migrate_interval=params['GENS_MIGRATE'])

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
    best_fitness_test.append(fitness_test[-1])
    
    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    structural_diversity = logbook.select("structural_diversity") 
    best_phenotypes = logbook.select("best_phenotype")

    best = hof.items[0].phenotype
    best_models.append(hof.items[0])
    nmissing_test = logbook.select("test_missing")

    nmissing.append([hof.items[0].nmissing/len(Y_train),nmissing_test[-1]/len(Y_test)])
    
    # output report files
    if proc_rank == 0:
        print("Best individual:")#,"\n".join(textwrap.wrap(best,80)))
        print("\n".join(textwrap.wrap(data_processing.reset_variable_names(best, var_map),80)))
        print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
        print("Test Fitness: ", fitness_test[-1])
        print("Depth: ", hof.items[0].depth)
        print("Length of the genome: ", len(hof.items[0].genome))
        print(f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')
    
        import csv    
        header = REPORT_ITEMS
        
        with open(params['OUT'] +'.cv'+ str(cv+1) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
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
                                 generation_time[value],
                                 best_phenotypes[value]])

# create and write summary files and plots for run
if proc_rank == 0:
    data_processing.write_summary(params['OUT'] + '_summary.txt',
        best_models,params['FITNESS'], var_map, best_fitness_test, nmissing)
    data_processing.write_plots(params['OUT'], best_models, var_map, inputs_map, color_map)
    

