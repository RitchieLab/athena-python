""" Modified DEAP to work with GRAPE library. 
    docstrings modified to conform to google style 

    Original code at https://github.com/bdsul/grape
"""

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import math
import numpy as np
import time
import warnings

from genn import parallel 
from deap import tools

def varAnd(population: list, toolbox: 'deap.base.Toolbox', cxpb: float, 
           mutpb: float, bnf_grammar: "grape.grape.grammar", codon_size: int, 
           max_tree_depth: int, codon_consumption: str,
           genome_representation: str, max_genome_length: int) -> list:
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    Args:
        population: A list of individuals to vary.
        toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
        cxpb: The probability of mating two individuals.
        mutpb: The probability of mutating an individual.
        bnf_grammar: BNF grammar for mapping
        codon_size: maximum value in a codon of the genome
        max_tree_depth: maximum allowed tree depth while mapping genome
        codon_consumption: type of consumption ('eager' or 'lazy')
        genome_representation: 'list' or 'numpy'
        max_genome_length: maximum allowed number of codons in genome

    Returns: A list of varied individuals that are independent of their
              parents.

    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i],
                                                          bnf_grammar, 
                                                          max_tree_depth, 
                                                          codon_consumption,
                                                          genome_representation,
                                                          max_genome_length)

    for i in range(len(offspring)):
        offspring[i], = toolbox.mutate(offspring[i], mutpb,
                                       codon_size, bnf_grammar, 
                                       max_tree_depth, codon_consumption,
                                       max_genome_length)

    return offspring

class hofWarning(UserWarning):
    pass
    
def ge_eaSimpleWithElitism(population:list, toolbox:'deap.base.Toolbox', cxpb: float,
                mutpb:float, ngen:int, elite_size:int, 
                bnf_grammar:'grape.grape.grammar', codon_size:int, max_tree_depth:int, 
                max_genome_length:int=None,
                points_train:list=None, points_test:list=None, codon_consumption:str='eager', 
                report_items:list=None,
                genome_representation:str='list',
                stats:'deap.tools.Statistics'=None, halloffame:'eap.tools.HallOfFame'=None, 
                rank:int=0, migrate_interval:int=None,
                verbose:bool=__debug__) -> tuple [list, 'deap.tools.logbook']:
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_, with some adaptations to run GE
    on GRAPE.

    Args:
        population: A list of individuals.
        toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
        cxpb: The probability of mating two individuals.
        mutpb: The probability of mutating an individual.
        ngen: The number of generation.
        elite_size: The number of best individuals to be copied to the 
                    next generation.
        bnf_grammar: BNF grammar for mapping
        codon_size: Maximum value in codon of genome
        max_tree_depth: Maximum depth for a tree when mapping genome
        max_genome_length: Maximum number of codons in genome
        points_train: Points to use in training
        points_test: Points to use in testing
        codon_consumption: type of consumption, 'lazy' or 'eager'
        report_items: list of report items to include 
        genome_representation: 'list' or 'numpy'
        stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
        halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
        rank: processs rank in MPI run
        migrate_interval: number of generations between individual transfer
            for island model (used with parallelized version)
        verbose: Whether or not to log the statistics.

    Returns: 
        population: The final population
        logbook: A class`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    
    logbook = tools.Logbook()
    
    if halloffame is None:
        if elite_size != 0:
            raise ValueError("You should add a hof object to use elitism.") 
        else:
            warnings.warn('You will not register results of the best individual while not using a hof object.', hofWarning)
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if elite_size > halloffame.maxsize:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to ELITE_SIZE")         
        if points_test:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time', 'best_phenotype']
        else:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time', 'best_phenotype']

    start_gen = time.time()        
    # Evaluate the individuals with an invalid fitness
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
        
    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
    if len(valid0) != len(valid):
        warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid them.")
    invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals    
    
    list_structures = []
    if 'fitness_diversity' in report_items:
        list_fitnesses = []
    if 'behavioural_diversity' in report_items:
        behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
    
    #for ind in offspring:
    for idx, ind in enumerate(valid):
        list_structures.append(str(ind.structure))
        if 'fitness_diversity' in report_items:
            list_fitnesses.append(str(ind.fitness.values[0]))
        if 'behavioural_diversity' in report_items:
            behaviours[idx, :] = ind.fitness_each_sample
            
    unique_structures = np.unique(list_structures, return_counts=False)  
    if 'fitness_diversity' in report_items:
        unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
    if 'behavioural_diversity' in report_items:
        unique_behaviours = np.unique(behaviours, axis=0)
        
    n_inds = len(population)
    n_unique_structs = len(unique_structures)
    
    fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
    behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0

    length = [len(ind.genome) for ind in valid]
    nodes = [ind.nodes for ind in valid]
    depth = [ind.depth for ind in valid]
    used_codons = [ind.used_codons for ind in valid]

    sum_length = sum(length)
    n_length = len(length)
    sum_nodes = sum(nodes)
    n_nodes = len(nodes)
    sum_used_codons = sum(used_codons)
    n_used_codons = len(used_codons)
    sum_depth = sum(depth)
    n_depth = len(depth)

    if parallel.has_mpi and parallel.nprocs > 1:
        recv = parallel.send_log_info(length, nodes, depth, used_codons, invalid, n_inds, n_unique_structs)
        sum_length = recv['sum_length']
        n_length = recv['n_length']
        sum_nodes = recv['sum_nodes']
        n_nodes = recv['n_nodes']
        sum_used_codons = recv['sum_used_codons']
        n_used_codons = recv['n_used_codons']
        sum_depth = recv['sum_depth']
        n_depth = recv['n_depth']
        invalid = recv['invalid']
        n_inds = recv['n_inds']
        n_unique_structs = recv['n_unique_structs']

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(valid)
        best_ind_length = len(halloffame.items[0].genome) 
        best_ind_nodes = halloffame.items[0].nodes
        best_ind_depth = halloffame.items[0].depth
        best_ind_used_codons = halloffame.items[0].used_codons
        best_phenotype = halloffame.items[0].phenotype
        if not verbose and rank==0:
            print("gen =", 0, ", Best fitness =", halloffame.items[0].fitness.values)
    
    avg_length = sum_length / n_length
    avg_nodes = sum_nodes / n_nodes
    avg_used_codons = sum_used_codons / n_used_codons
    avg_depth = sum_depth / n_depth
    structural_diversity = n_unique_structs/n_inds
    
    end_gen = time.time()
    generation_time = end_gen-start_gen
        
    selection_time = 0
    
    if points_test:
        fitness_test = np.NaN
    
    record = stats.compile(valid0) if stats else {}
    if parallel.has_mpi and parallel.nprocs > 1:
        record = parallel.get_stats(stats,valid0)
    
    if points_test: 
        logbook.record(gen=0, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time, best_phenotype=best_phenotype)
    else:
        logbook.record(gen=0, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time, best_phenotype=best_phenotype)
    if verbose and rank==0:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(logbook.select("gen")[-1]+1, ngen + 1):
        start_gen = time.time()    
    
        # Select the next generation individuals
        start = time.time()    
        offspring = toolbox.select(valid, len(population)-elite_size)
        end = time.time()
        selection_time = end-start
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, genome_representation,
                           max_genome_length)

        # Evaluate the individuals with an invalid fitness
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
                
        #Update population for next generation
        population[:] = offspring
        #Include in the population the elitist individuals
        for i in range(elite_size):
            population.append(halloffame.items[i])
        
        #include best from other populations when using parellelization
        if migrate_interval and gen % migrate_interval == 0:
            halloffame.update(valid)
            best_indiv = halloffame.items[0]
            best_indivs = parallel.exchange_best(best_indiv)
            present = set()
            for ind in population:
                present.add(ind.phenotype)
            def sortByFitness(item):
                return item.fitness
            population.sort(key=sortByFitness, reverse=True)
            replace_index=-1
            for i in range(len(best_indivs)):
                if best_indivs[i].phenotype not in present:
                    population[replace_index]=best_indivs[i]
                    replace_index -=1
        
        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        if len(valid0) != len(valid):
            warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid in the statistics.")
        invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals
        
        list_structures = []
        if 'fitness_diversity' in report_items:
            list_fitnesses = []
        if 'behavioural_diversity' in report_items:
            behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
        
        for idx, ind in enumerate(valid):
            list_structures.append(str(ind.structure))
            if 'fitness_diversity' in report_items:
                list_fitnesses.append(str(ind.fitness.values[0]))
            if 'behavioural_diversity' in report_items:
                behaviours[idx, :] = ind.fitness_each_sample
                
        unique_structures = np.unique(list_structures, return_counts=False)
        if 'fitness_diversity' in report_items:
            unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
        if 'behavioural_diversity' in report_items:
            unique_behaviours = np.unique(behaviours, axis=0)
        
        
        n_inds = len(population)
        n_unique_structs = len(unique_structures)
        
        fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
        behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
                  
          
        length = [len(ind.genome) for ind in valid]
        nodes = [ind.nodes for ind in valid]
        depth = [ind.depth for ind in valid]
        used_codons = [ind.used_codons for ind in valid]

        sum_length = sum(length)
        n_length = len(length)
        sum_nodes = sum(nodes)
        n_nodes = len(nodes)
        sum_used_codons = sum(used_codons)
        n_used_codons = len(used_codons)
        sum_depth = sum(depth)
        n_depth = len(depth)

        if parallel.has_mpi and parallel.nprocs > 1:
            recv = parallel.send_log_info(length, nodes, depth, used_codons, invalid, n_inds, n_unique_structs)
            sum_length = recv['sum_length']
            n_length = recv['n_length']
            sum_nodes = recv['sum_nodes']
            n_nodes = recv['n_nodes']
            sum_used_codons = recv['sum_used_codons']
            n_used_codons = recv['n_used_codons']
            sum_depth = recv['sum_depth']
            n_depth = recv['n_depth']
            invalid = recv['invalid']
            n_inds = recv['n_inds']
            n_unique_structs = recv['n_unique_structs']

        nmissing_test = 0
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            best_phenotype = halloffame.items[0].phenotype
            if not verbose and rank==0:
                print("gen =", gen, ", Best fitness =", halloffame.items[0].fitness.values, ", Number of invalids =", invalid)
            if points_test:
                if gen < ngen:
                    fitness_test = np.NaN
                else:
                    nmissing = halloffame.items[0].nmissing
                    fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]
                    nmissing_test = halloffame.items[0].nmissing
                    halloffame.items[0].nmissing = nmissing
        
        avg_length = sum_length / n_length
        avg_nodes = sum_nodes / n_nodes
        avg_used_codons = sum_used_codons / n_used_codons
        avg_depth = sum_depth / n_depth
        structural_diversity = n_unique_structs/n_inds
        
        end_gen = time.time()
        generation_time = end_gen-start_gen        
        
        # Append the current generation statistics to the logbook
        record = stats.compile(valid0) if stats else {}
        if parallel.has_mpi and parallel.nprocs > 1:
            record = parallel.get_stats(stats,valid0)
        
        if points_test: 
            logbook.record(gen=gen, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time,
                       best_phenotype=best_phenotype,
                       test_missing=nmissing_test)
        else:
            logbook.record(gen=gen, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time,
                       test_missing=nmissing_test,
                       best_phenotype=best_phenotype)
                
        if verbose and rank==0:
            print(logbook.stream)

    return population, logbook
