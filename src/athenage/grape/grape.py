# -*- coding: utf-8 -*-
"""

Modified code from GRAPE: Grammatical Algorithms in Python for Evolution for use with ATHENA and includes multi-chromosome and LEAP genome implementations

"""

"""

Original license:

BSD 3-Clause License

Copyright (c) 2022-2023, BDS Research Group at University of Limerick

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""



import re
import math
from operator import attrgetter
import numpy as np
import random
import copy
from .genomes import Genome,LeapGenome,MCGEGenome


from math import modf


class Individual:
    """
    A GE individual.
    """

    def __init__(self, genome, grammar, max_depth, codon_consumption):
        """
        """
        
        self.genome = genome
        if codon_consumption == 'lazy':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_lazy(genome, grammar, max_depth)
        elif codon_consumption == 'eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_eager(genome, grammar, max_depth)
#             self.structure = mapper_eager_opt(genome, grammar, max_depth)
            
            
        else:
            raise ValueError("Unknown mapper")
            

def create_genome(genome_type:str, n_rules:int, codon_size:int) -> "grape.grape.Genome":
    """Create and return appropriate Genome 

    Args: 
        genome_type: Type to create (leap,mcge,standard)
        n_rules: number of rules in the grammar used to map this genome
        codon_size: maximum value for a codon 
    
    Returns:
        new Genome
    """
    if genome_type == 'leap':
        return LeapGenome(n_rules, codon_size)
    elif genome_type == 'mcge':
        return MCGEGenome(n_rules, codon_size)
    else:  
        return Genome(n_rules, codon_size)



def sensible_initialization(ind_class: 'deap.creator.MetaCreator', pop_size:int, 
                          bnf_grammar:'grape.grape.Grammar', 
                          min_init_depth:int, 
                          max_init_depth:int, codon_size:int, codon_consumption:str,
                          genome_representation:str, genome_type:str) -> list:
    """Sensible initialization ensures valid individuals. Half the generated individuals
    are up to the depth of the max_init_depth and half are limited to that depth but may be
    smaller

    Args:
        ind_class: Deap creator
        pop_size: number of individuals to create
        bnf_grammar: BNF grammar 
        min_init_depth: minimum depth for tree in mapping genome
        max_init_depth: maximum depth for tree in mapping genome
        codon_size: maximum to be stored in codon
        codon_consumption: type of consumption when mapping (ie. lazy or eager)
        genome_representation: either list or numpy
        genome_type: standard/mcge/leap
        

    Returns:
        population: list of individuals
    """ 
    #Calculate the number of individuals to be generated with each method
    is_odd = pop_size % 2
    n_grow = int(pop_size/2)
    
    n_sets_grow = max_init_depth - min_init_depth + 1
    set_size = int(n_grow/n_sets_grow)
    remaining = n_grow % n_sets_grow
    
    n_full = n_grow + is_odd + remaining #if pop_size is odd, generate an extra ind with "full"
    
    #TODO check if it is possible to generate inds with max_init_depth        
    
    population = []
    #Generate inds using "Grow"
    for i in range(n_sets_grow):
        max_init_depth_ = min_init_depth + i
        for j in range(set_size):
            remainders = [] #it will register the choices
            possible_choices = [] #it will register the respective possible choices
            rules_used = [] #registers which rule used for setting the frame

            phenotype = bnf_grammar.start_rule
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
            depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
            idx_branch = 0 #index of the current branch being grown
            while len(remaining_NTs) != 0:
                idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth_]
                Ch = random.choice(actual_options)
                phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                depths[idx_branch] += 1
                if codon_consumption == 'eager':
                    remainders.append(Ch[3])
                    possible_choices.append(len(total_options))
                    rules_used.append(idx_NT)
                elif codon_consumption == 'lazy':
                    if len(total_options) > 1:
                        remainders.append(Ch[3])
                        possible_choices.append(len(total_options))
                        rules_used.append(idx_NT)
                
                if Ch[2] > 1:
                    if idx_branch == 0:
                        depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    else:
                        depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                if Ch[1] == 'terminal':
                    idx_branch += 1
                
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
            
            #Generate the genome
            #remainders contain the choice for the rule
            #possible_choices simply tells you how many choices for that codon -- needed to generate actual codon value
            genome = create_genome(genome_type, bnf_grammar.nt_rule_size, codon_size)
            # genome = []
            if codon_consumption == 'eager' or codon_consumption == 'lazy':
                for k in range(len(remainders)):
                    codon = (random.randint(0,int(1e10)) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                    genome.add_codon(codon, rules_used[k])
                genome.finalize()
            else:
                raise ValueError("Unknown mapper")
            
            #Include a tail with 50% of the genome's size
            size_tail = int(0.5*genome.size())
            genome.add_tail(size_tail)

            #Initialize the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption)
            
            #Check if the individual was mapped correctly
            if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                raise Exception('error in the mapping')
            population.append(ind)

    for i in range(n_full):
        remainders = [] #it will register the choices
        possible_choices = [] #it will register the respective possible choices
        rules_used = [] #registers which rule used for setting the frame

        phenotype = bnf_grammar.start_rule
        remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
        depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
        idx_branch = 0 #index of the current branch being grown

        while len(remaining_NTs) != 0:
            idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
            total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
            actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth]
            recursive_options = [PR for PR in actual_options if PR[4]]
            if len(recursive_options) > 0:
                Ch = random.choice(recursive_options)
            else:
                Ch = random.choice(actual_options)
            phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
            depths[idx_branch] += 1
            if codon_consumption == 'eager':
                remainders.append(Ch[3])
                possible_choices.append(len(total_options))
                rules_used.append(idx_NT)
            elif codon_consumption == 'lazy':
                if len(total_options) > 1:
                    remainders.append(Ch[3])
                    possible_choices.append(len(total_options))
                    rules_used.append(idx_NT)

            if Ch[2] > 1:
                if idx_branch == 0:
                    depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                else:
                    depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
            if Ch[1] == 'terminal':
                idx_branch += 1
            
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]


        genome = create_genome(genome_type, bnf_grammar.nt_rule_size, codon_size)
        # genome = []
        if codon_consumption == 'eager' or codon_consumption == 'lazy':
            for k in range(len(remainders)):
                codon = (random.randint(0,int(1e10)) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                genome.add_codon(codon, rules_used[k])
            genome.finalize()
        else:
            raise ValueError("Unknown mapper")

        #Include a tail with 50% of the genome's size
        size_tail = int(0.5*genome.size())
        genome.add_tail(size_tail)
        
        #Initialise the individual and include in the population
        ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
                    
        #Check if the individual was mapped correctly
        if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
            raise Exception('error in the mapping')

        population.append(ind)    
    
    if genome_representation == 'list':
        return population
    elif genome_representation == 'numpy':
        for ind in population:
            ind.genome = np.array(ind.genome)
        return population
    else:
        raise ValueError("Unkonwn genome representation")

def random_initialization(ind_class: 'deap.creator.MetaCreator', pop_size:int, 
                          bnf_grammar:'grape.grape.Grammar', 
                          min_init_genome_length:int, max_init_genome_length:int,
                          max_init_depth:int, codon_size:int, codon_consumption:str,
                          genome_representation:str, genome_type:str) -> list:
    """Randomly generated genomes that may be invalid when mapped

    Args:
        ind_class: Deap creator
        pop_size: number of individuals to create
        bnf_grammar: BNF grammar 
        min_init_genome_length: minimum number of codons in genome
        max_init_genome_length: maximum number of codons in genome
        max_init_depth: maximum depth for tree in mapping genome
        codon_size: maximum to be stored in codon
        codon_consumption: type of consumption when mapping (ie. lazy or eager)
        genome_representation: either list or numpy
        genome_type: standard/mcge/leap
        

    Returns:
        population: list of individuals
    """ 

    population = []
    
    for i in range(pop_size):
        genome = create_genome(genome_type,bnf_grammar.nt_rule_size, codon_size)
        init_genome_length = random.randint(min_init_genome_length, max_init_genome_length)
        genome.random_fill(init_genome_length)
        genome.finalize()
        ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
        population.append(ind)
        
    if genome_representation == 'list':
        return population
    else:
        raise ValueError("Unkonwn genome representation")

def mapper_eager(genome: "grape.grape.Genome", grammar: "grape.grape.grammar",
                  max_depth: int) -> tuple[str, int, int, int, bool, list]:
    """Maps GE  genome. Uses eager mapping so that 
    every nonterminal in the grammar consumes a codon in the genome

    Args:
        genome: list of codons in genome
        grammar: BNF grammar used to create individual phenotype
        max_depth: maximum depth allowed for a tree during mapping
        

    Returns:
        phenotype: mapped phenotype 
        nodes: number of nodes in tree
        depth: max depth of tree used in mapping
        used_codons: number of codons consumed in mapping
        invalid: True if valid phenotype not constructed
        int for number of wraps used (0 for this implementation as wrapping not used)
        structure: list of production rules chosen in mapping, 
            used to monitor population diversity
    """

    # idx_genome = 0
    genome.reset_map_index()
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    # while next_NT and idx_genome < len(genome):
    while next_NT:
        NT_index = grammar.non_terminals.index(next_NT)
        if genome.consumed(NT_index):
            break
        # index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        index_production_chosen = genome.get_next_codon(NT_index) % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        # idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = genome.used_codons()
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure


def mapper_lazy(genome: list, grammar: "grape.grape.grammar",
                  max_depth: int) -> tuple[str, int, int, int, bool, list]:
    """ Uses lazy mapping so that 
    only nonterminals with more than one option in the grammar 
    consume a codon in the genome

    Args:
        genome: list of codons in genome
        grammar: BNF grammar used to create individual phenotype
        max_depth: maximum depth allowed for a tree during mapping
        

    Returns:
        phenotype: 
        nodes: number of nodes in tree
        depth: max depth of tree used in mapping
        used_codons: number of codons consumed in mapping
        invalid: True if valid phenotype not constructed
        int for number of wraps used (0 for this implementation as wrapping not used)
        structure: list of production rules chosen in mapping, 
            used to monitor population diversity
    """  
    
    genome.reset_map_index()
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT:
        NT_index = grammar.non_terminals.index(next_NT)
        if genome.consumed(NT_index):
            break        
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            index_production_chosen = genome.get_next_codon(NT_index) % grammar.n_rules[NT_index]
            structure.append(index_production_chosen)
            # idx_genome += 1
        
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
            
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = genome.used_codons()
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure


def reMap(ind: 'deap.creator.Individual', genome: list, 
          bnf_grammar:'grape.grape.Grammar', max_tree_depth:int, 
          codon_consumption:str) -> 'deap.creator.Individual':
    """Maps GE genome in individual to produce new phenotype

    Args:
        ind: individual to mutate
        bnf_grammar: BNF grammar 
        max_tree_depth: maximum depth allowed for a tree during mapping
        codon_consumption: type of consumption when mapping (ie. lazy or eager)
        
    Returns:
        new_ind: new individual with updating phenotype and other mapping information
    """ 

    ind.genome = genome
    if codon_consumption == 'lazy':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_lazy(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_eager(genome, bnf_grammar, max_tree_depth)
    else:
        raise ValueError("Unknown mapper")
        
    return ind


def submap_eager(genome: "grape.grape.Genome", grammar: "grape.grape.grammar",
                  start_codon: int, start_rule: int) -> tuple[str, int, int, int, bool, list]:
    """Maps GE  genome. Uses eager mapping so that 
    every nonterminal in the grammar consumes a codon in the genome

    Args:
        genome: list of codons in genome
        grammar: BNF grammar used to create individual phenotype
        start_codon: Index in genome of starting codon
        start_rule: Index of nonterminal in grammar
        

    Returns:
        codon_end: ending codon for subtree of the expression tree
    """

    genome.reset_map_index()
    start_codon = start_codon[0]
    n_codons_used=0
    genome.next_read = start_codon
    next_NT = grammar.non_terminals[start_rule]
    phenotype = next_NT
    
    # while next_NT and idx_genome < len(genome):
    while next_NT:
        NT_index = grammar.non_terminals.index(next_NT)
        if genome.consumed(NT_index):
            break
        index_production_chosen = genome.get_next_codon(NT_index) % grammar.n_rules[NT_index]
        n_codons_used += 1
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        
    if next_NT:
        invalid = True
    else:
        invalid = False
    
    return start_codon + n_codons_used -1 


def submap_lazy(genome: "grape.grape.Genome", grammar: "grape.grape.grammar",
                  start_codon: int, start_rule: int) -> tuple[str, int, int, int, bool, list]:
    """ Uses lazy mapping so that 
    only nonterminals with more than one option in the grammar 
    consume a codon in the genome

    Args:
        genome: list of codons in genome
        grammar: BNF grammar used to create individual phenotype
        start_codon: Index in genome of starting codon
        start_rule: Index of nonterminal in grammar
        

    Returns:
        codon_end: ending codon for subtree of the expression tree
    """  
    
    genome.reset_map_index()
    start_codon = start_codon[0]
    n_codons_used=0
    genome.next_read = start_codon
    next_NT = grammar.non_terminals[start_rule]
    phenotype = next_NT

    while next_NT:
        NT_index = grammar.non_terminals.index(next_NT)
        if genome.consumed(NT_index):
            break        
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            index_production_chosen = genome.get_next_codon(NT_index) % grammar.n_rules[NT_index]
            n_codons_used += 1
        
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
    
    if next_NT:
        invalid = True
    else:
        invalid = False
   
    return start_codon + n_codons_used -1 


def submap(ind: 'deap.creator.Individual', genome: list, 
          bnf_grammar:'grape.grape.Grammar', start_codon: int,
          start_rule:int, codon_consumption:str) -> int:
    """Maps GE genome in individual to produce new phenotype

    Args:
        ind: individual to mutate
        bnf_grammar: BNF grammar 
        start_codon: Index of starting codon for the subtree
        start_rule: Index of rule for starting codon
        codon_consumption: type of consumption when mapping (ie. lazy or eager)
        
    Returns:
        pos_end: index of ending position in genome for the subtree
    """ 

    # set initial ending to be rest of genome
    pos_end = ind.genome.total_codons()-1
    ind.genome = genome
    if codon_consumption == 'lazy':
        pos_end= submap_lazy(genome, bnf_grammar, start_codon, start_rule)
    elif codon_consumption == 'eager':
        pos_end = submap_eager(genome, bnf_grammar, start_codon, start_rule)
    else:
        raise ValueError("Unknown mapper")    
        
    return pos_end


def mutation_int_flip_per_codon(ind:'deap.creator.Individual', mut_probability:float, 
                                codon_size:int , bnf_grammar:'grape.grape.Grammar', max_depth:int, 
                                codon_consumption:str, 
                                max_genome_length:int=None) -> 'deap.creator.Individual':
    """Mutation operator for GE genome. Each codon within the effective used codons
    of the genome is checked

    Args:
        ind: individual to mutate
        mut_probability: chance for each codon to mutate
        codon_size: maximum value for a codon
        bnf_grammar: BNF grammar 
        max_depth: maximum depth allowed for a tree during mapping
        codon_consumption: type of consumption when mapping (ie. lazy or eager)
        max_genome_length: maximum allowed length of genome when not None
        
    Returns:
        new_ind: new individual incorporating any mutations
    """ 

    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = ind.genome.total_codons()
    else:
        possible_mutation_codons = min(ind.genome.used_codons(), ind.genome.total_codons())

    mutated_ = False

    for i in range(possible_mutation_codons):
        if random.random() < mut_probability:
            ind.genome.set_codon(i, random.randint(0,codon_size), ind.invalid)
            mutated_ = True

    if mutated_:
        new_ind = reMap(ind, ind.genome, bnf_grammar, max_depth, codon_consumption)
    else:
        new_ind = ind
    
    if new_ind.depth > max_depth:
        new_ind.invalid = True
        
    if max_genome_length:
        if len(new_ind.genome) > max_genome_length:
            new_ind.invalid = True

    if mutated_:
        del new_ind.fitness.values
    return new_ind,
    

def crossover_onepoint(parent1: 'deap.creator.Individual', parent2: 'deap.creator.Individual', 
                       bnf_grammar: 'grape.grape.Grammar', max_depth: int, codon_consumption:str, 
                    genome_representation:str='list', 
                    max_genome_length:int=None) -> tuple['deap.creator.Individual','deap.creator.Individual']:
    """One point crossover for GE individuals

    Args:
        parent1: individual to cross
        parent2: second individual to cross
        bnf_grammar: BNF grammar 
        max_depth: maximum depth allowed for a tree during mapping
        codon_consumption: type of consumption when mapping (ie. lazy or eager)
        genome_representation: either list or numpy
        max_genome_length: maximum allowed length of genome when not None
        
    Returns:
        new_ind0: new individual resulting from crossover
        new_ind1: second new individual resulting from crossover
    """ 
    # restrict crossover to effective genome when individual is valid
    if parent1.invalid: #used_codons = 0
        # possible_crossover_codons0 = len(parent0.genome)
        pos1 = parent1.genome.all_cross_loc()
    else:
        pos1 = parent1.genome.effective_cross_loc()
    if parent2.invalid:
        pos2 = parent2.genome.all_cross_loc()
    else:
        pos2 = parent2.genome.effective_cross_loc()


    new_genome1, new_genome2 = parent1.genome.crossover_onepoint(parent2.genome, pos1, pos2)

    new_ind0 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
    new_ind1 = reMap(parent2, new_genome2, bnf_grammar, max_depth, codon_consumption)

    if new_ind0.depth > max_depth:
        new_ind0.invalid = True
    if new_ind1.depth > max_depth:
        new_ind1.invalid = True

    if max_genome_length:
        if len(new_ind0.genome) > max_genome_length:
            new_ind0.invalid = True
        if len(new_ind1.genome) > max_genome_length:
            new_ind1.invalid = True

    del new_ind0.fitness.values, new_ind1.fitness.values
    return new_ind0, new_ind1   


def crossover_match(parent1: 'deap.creator.Individual', parent2: 'deap.creator.Individual', 
                       bnf_grammar: 'grape.grape.Grammar', max_depth: int, codon_consumption:str, 
                    genome_representation:str='list', 
                    max_genome_length:int=None) -> tuple['deap.creator.Individual','deap.creator.Individual']:
    """One point crossover for GE individuals restricted to locations where identical rule selections are made

    Args:
        parent1: individual to cross
        parent2: second individual to cross
        bnf_grammar: BNF grammar 
        max_depth: maximum depth allowed for a tree during mapping
        codon_consumption: type of consumption when mapping (ie. lazy or eager)
        genome_representation: either list or numpy
        max_genome_length: maximum allowed length of genome when not None
        
    Returns:
        new_ind0: new individual resulting from crossover
        new_ind1: second new individual resulting from crossover
    """ 

    # restrict crossover to effective genome when individual is valid
    if parent1.invalid: #used_codons = 0
        pos1 = parent1.genome.all_cross_loc()
    else:
        pos1 = parent1.genome.effective_cross_loc()
    rule1_idx = parent1.genome.rule_used(pos1[0])

    if parent2.invalid:
        pos2 = parent2.genome.all_cross_loc()
        check_len = parent2.genome.used_codons()
    else:
        pos2 = parent2.genome.effective_cross_loc()
        check_len = parent2.genome.total_codons()

    # start with selected position and then move up or down list to find match
    if random.random() < 0.5:
        range1 = range(pos2[0],check_len, 1)
        range2 = range(pos2[0]-1, -1, -1)
    else:
        range1 = range(pos2[0], -1, -1)
        range2 = range(pos2[0]+1,check_len,1)

    found = False
    for i in range1:
        if parent2.genome.rule_used(i) == rule1_idx:
            found = True
            pos2 = [i]
            break
    if not found:
        for i in range2:
            if parent2.genome.rule_used(i) == rule1_idx:
                found = True
                pos2 = [i]
                break

    new_genome1, new_genome2 = parent1.genome.crossover_onepoint(parent2.genome, pos1, pos2)

    new_ind0 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
    new_ind1 = reMap(parent2, new_genome2, bnf_grammar, max_depth, codon_consumption)

    if new_ind0.depth > max_depth:
        new_ind0.invalid = True
    if new_ind1.depth > max_depth:
        new_ind1.invalid = True

    if max_genome_length:
        if len(new_ind0.genome) > max_genome_length:
            new_ind0.invalid = True
        if len(new_ind1.genome) > max_genome_length:
            new_ind1.invalid = True

    del new_ind0.fitness.values, new_ind1.fitness.values
    return new_ind0, new_ind1   



def crossover_block(parent1: 'deap.creator.Individual', parent2: 'deap.creator.Individual', 
                       bnf_grammar: 'grape.grape.Grammar', max_depth: int, codon_consumption:str, 
                    genome_representation:str='list', 
                    max_genome_length:int=None) -> tuple['deap.creator.Individual','deap.creator.Individual']:
    """Two point crossover for GE individuals restricted to locations where identical rule selections are made
    Starts at matching point and ends with the expression tree portion terminated from that point.

    Args:
        parent1: individual to cross
        parent2: second individual to cross
        bnf_grammar: BNF grammar 
        max_depth: maximum depth allowed for a tree during mapping
        codon_consumption: type of consumption when mapping (ie. lazy or eager)
        genome_representation: either list or numpy
        max_genome_length: maximum allowed length of genome when not None
        
    Returns:
        new_ind0: new individual resulting from crossover
        new_ind1: second new individual resulting from crossover
    """ 

    # restrict crossover to effective genome when individual is valid
    if parent1.invalid: #used_codons = 0
        pos1 = parent1.genome.all_cross_loc()
    else:
        pos1 = parent1.genome.effective_cross_loc()
    rule1_idx = parent1.genome.rule_used(pos1[0])

    pos1_end = submap(parent1, parent1.genome, bnf_grammar, pos1, rule1_idx, codon_consumption)

    if parent2.invalid:
        pos2 = parent2.genome.all_cross_loc()
        check_len = parent2.genome.total_codons()
    else:
        pos2 = parent2.genome.effective_cross_loc()
        check_len = parent2.genome.used_codons()

    # start with selected position and then move up or down list to find match
    if random.random() < 0.5:
        range1 = range(pos2[0],check_len, 1)
        range2 = range(pos2[0]-1, -1, -1)
    else:
        range1 = range(pos2[0], -1, -1)
        range2 = range(pos2[0]+1,check_len,1)

    found = False
    for i in range1:
        if parent2.genome.rule_used(i) == rule1_idx:
            found = True
            pos2 = [i]
            break
    if not found:
        for i in range2:
            if parent2.genome.rule_used(i) == rule1_idx:
                found = True
                pos2 = [i]
                break
    
    rule2_idx = parent2.genome.rule_used(pos2[0])
    
    pos2_end = submap(parent2, parent2.genome, bnf_grammar, pos2, rule2_idx, codon_consumption)

    new_genome1, new_genome2 = parent1.genome.crossover_twopoint(parent2.genome, 
                                                                 [pos1[0],pos1_end], [pos2[0],pos2_end])

    new_ind0 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
    new_ind1 = reMap(parent2, new_genome2, bnf_grammar, max_depth, codon_consumption)

    if new_ind0.depth > max_depth:
        new_ind0.invalid = True
    if new_ind1.depth > max_depth:
        new_ind1.invalid = True

    if max_genome_length:
        if len(new_ind0.genome) > max_genome_length:
            new_ind0.invalid = True
        if len(new_ind1.genome) > max_genome_length:
            new_ind1.invalid = True

    del new_ind0.fitness.values, new_ind1.fitness.values
    return new_ind0, new_ind1   



class Grammar:
    """
    BNF Grammar 
    Attributes:
        non_terminals: list with each non-terminal (NT);
        start_rule: first non-terminal;
        nt_rule_size: number of nt rules used for frame size for set of codons in leap mapping 
            and for the number of chromosomes in MCGE
        production_rules: list with each production rule (PR), which contains in each position:
            - the PR itself as a string
            - 'non-terminal' or 'terminal'
            - the arity (number of NTs in the PR)
            - production choice label
            - True, if it is recursive, and False, otherwise
            - the minimum depth to terminate the mapping of all NTs of this PR
        n_rules: df
    
    """
    def __init__(self, bnf_grammar: str,codon_consumption:str ='eager'):
        """ Initializes Grammar instance by reading BNF grammar string 
        Args:
            bnf_grammar: text describing BNF grammar
            codon_consumption: 'eager' or 'lazy' controlling consumption during mapping

        """
        bnf_grammar = re.sub(r"\s+", " ", bnf_grammar)

        #self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>\s*::=",bnf_grammar)]
        self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>\s*::=",bnf_grammar)]
        self.start_rule = self.non_terminals[0]
        for i in range(len(self.non_terminals)):
            bnf_grammar = bnf_grammar.replace(self.non_terminals[i] + " ::=", "  ::=")
        rules = bnf_grammar.split("::=")
        del rules[0]
        rules = [item.replace('\n',"") for item in rules]
        rules = [item.replace('\t',"") for item in rules]
        
        #list of lists (set of production rules for each non-terminal)
        self.production_rules = [i.split('|') for i in rules]
        for i in range(len(self.production_rules)):
            #Getting rid of all leading and trailing whitespaces
            self.production_rules[i] = [item.strip() for item in self.production_rules[i]]
            for j in range(len(self.production_rules[i])):
                #Include in the list the PR itself, NT or T, arity and the production choice label
                #if re.findall(r"\<(\w+)\>",self.production_rules[i][j]):
                if re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]):                    
                    #arity = len(re.findall(r"\<(\w+)\>",self.production_rules[i][j]))
                    arity = len(re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]))
                    self.production_rules[i][j] = [self.production_rules[i][j] , "non-terminal", arity, j]
                else:
                    self.production_rules[i][j] = [self.production_rules[i][j] , "terminal", 0, j] #arity 0
        #number of production rules for each non-terminal
        self.n_rules = [len(list_) for list_ in self.production_rules]
  
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[i][j][0])
                NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
                unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
                recursive = False
                for NT_to_check in unique_NTs:
                    stack = [self.non_terminals[i]]  
                    if NT_to_check in stack:
                        recursive = True
                        break
                    else:
                        stack.append(NT_to_check)
                        recursive = self.check_recursiveness(NT_to_check, stack)
                        if recursive:
                            break
                        stack.pop()
                self.production_rules[i][j].append(recursive)
      
        #minimum depth from each non-terminal to terminate the mapping of all symbols
        NT_depth_to_terminate = [None]*len(self.non_terminals)
        #minimum depth from each production rule to terminate the mapping of all symbols
        part_PR_depth_to_terminate = list() #min depth for each non-terminal or terminal to terminate
        isolated_non_terminal = list() #None, if the respective position has a terminal
        #Separating the non-terminals within the same production rule
        for i in range(len(self.production_rules)):
            part_PR_depth_to_terminate.append( list() )
            isolated_non_terminal.append( list() )
            for j in range(len(self.production_rules[i])):
                part_PR_depth_to_terminate[i].append( list() )
                isolated_non_terminal[i].append( list() )
                if self.production_rules[i][j][1] == 'terminal':
                    isolated_non_terminal[i][j].append(None)
                    part_PR_depth_to_terminate[i][j] = 1
                    if not NT_depth_to_terminate[i]:
                        NT_depth_to_terminate[i] = 1
                else:
                    for k in range(self.production_rules[i][j][2]): #arity
                        part_PR_depth_to_terminate[i][j].append( list() )
                        #term = re.findall(r"\<(\w+)\>",self.production_rules[i][j][0])[k]
                        term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
                        isolated_non_terminal[i][j].append('<' + term + '>')
        continue_ = True
        while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
            if None not in NT_depth_to_terminate:
                continue_ = False 
            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(isolated_non_terminal[j][k])):
                            if self.non_terminals[i] == isolated_non_terminal[j][k][l]:
                                if NT_depth_to_terminate[i]:
                                    if not part_PR_depth_to_terminate[j][k][l]:
                                        part_PR_depth_to_terminate[j][k][l] = NT_depth_to_terminate[i] + 1
                                        if [] not in part_PR_depth_to_terminate[j][k]:
                                            if not NT_depth_to_terminate[j]:
                                                NT_depth_to_terminate[j] = part_PR_depth_to_terminate[j][k][l]
        PR_depth_to_terminate = []
        for i in range(len(part_PR_depth_to_terminate)):
            for j in range(len(part_PR_depth_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
                if type(part_PR_depth_to_terminate[i][j]) == int:
                    depth_ = part_PR_depth_to_terminate[i][j]
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                else:
                    depth_ = max(part_PR_depth_to_terminate[i][j])
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)

        self.rule_conversion = []
        idx = 0
        self.nt_rule_size = 0
        if codon_consumption == 'lazy':
            for i in range(len(self.non_terminals)):
                if len(self.production_rules[i]) > 1:
                    self.rule_conversion.append(idx)
                    self.nt_rule_size += 1
                    idx += 1
                else:
                    self.rule_conversion.append(-1)
        else: #codon consumption is lazy
            self.rule_conversion=[i for i in range(len(self.non_terminals))]
            self.nt_rule_size = len(self.rule_conversion)

        
    def check_recursiveness(self, NT: np.str_, stack: list) -> bool:
        """ Determines whether a non-terminal in the grammar is recursive 
        Args:
            NT: non-terminal checked
            stack: tracks non-terminals traversed
        
        Returns:
            True is recursive, False if not
        """
        idx_NT = self.non_terminals.index(NT)
        for j in range(len(self.production_rules[idx_NT])):
            NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[idx_NT][j][0])
            NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
            unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
            recursive = False
    #      while unique_NTs.size and not recursive:
            for NT_to_check in unique_NTs:
                if NT_to_check in stack:
                    recursive = True
                    return recursive
                else:
                    stack.append(NT_to_check) #Include the current NT to check it recursively
                    recursive = self.check_recursiveness(NT_to_check, stack)
                    if recursive:
                        return recursive
                    stack.pop() #If the inclusion didn't show recursiveness, remove it before continuing
        return recursive
  

def replace_nth(string:str, substring:str, new_substring:str, nth:int) -> str:
    """ utility function to replace nth instance of a substring
    
    Args:
        string: original str
        substring: substring to replace
        new_substring: new substring to use
        nth: instance of occurrence of substring to replace

    Returns;
        updated string
    
    """
    find = string.find(substring)
    i = find != -1
    while find != -1 and i != nth:
        find = string.find(substring, find + 1)
        i += 1
    if i == nth:
        return string[:find] + new_substring + string[find+len(substring):]
    return string


def selTournamentWithoutInvalids(individuals: list, k: int, tournsize: int, 
                                 fit_attr: str="fitness") -> list:
    """
    A simple tournament selection which avoids invalid individuals.

    Args:
        individuals: population to select from
        k: number to select
        tournsize: tournament size
        fit_attr: attribute to use in selecting individuals
    
    Returns:
        list of selected individuals
    """

    chosen = []
    valid_individuals = [i for i in individuals if not i.invalid]
    while len(chosen) < k:
        aspirants = random.sample(valid_individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


def selAthenaRoulette(individuals: list, k:int, fit_attr:str="fitness") -> list:
    """Matches the original C++ ATHENA roulette selection method

    Args:
        individuals: A list of individuals to select from.
        k: The number of individuals to select.
        fit_attr: The attribute of individuals to use as selection criterion

    Returns: 
        A list containing the k selected individuals
    """
    # sort the population in ascending order
    individuals = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
        
    psum=[]
    n = len(individuals)
    
    # set the probabilities for each i ndividual to be selected
    # if all identical then make all probabilities equal
    if individuals[0].fitness.values[0] == individuals[-1].fitness.values[0]:
        psum = [i.fitness/float(n) for i in individuals]
    else:
        maxfit = individuals[-1].fitness.values[0]
        minfit = individuals[0].fitness.values[0]
        # set first probability
        psum.append(-individuals[0].fitness.values[0] + maxfit + minfit)
        
        for i,ind in enumerate(individuals[1:], start=1):
            psum.append(-ind.fitness.values[0] + minfit + maxfit + psum[i-1])
        
        for i in range(n):
            psum[i] /= psum[-1]
        
    # select individuals
    chosen=[]
    for i in range(k):
        #use a binary search 
        cutoff = random.random()
        lowerbound = 0
        upperbound = n-1
        
        while(upperbound >= lowerbound):
            indindex=lowerbound + (upperbound-lowerbound)//2
            if psum[indindex] > cutoff:
                upperbound = indindex-1
            else:
                lowerbound = indindex+1
        
        lowerbound = min(n-1, lowerbound) 
        lowerbound = max(0, lowerbound)
        
        chosen.append(individuals[lowerbound])    
    
    return chosen    


def selTournRoulette(individuals: list, k:int, fit_attr:str="fitness") -> list:
    """Pick two individuals from the population using the RouletteWheel selection
    method.  Then return the better of the two individuals. Repeat to get k individuals
    to return. Matches the ATHENA tournament selection method

    Args:
        individuals: A list of individuals to select from.
        k: The number of individuals to select.
        fit_attr: The attribute of individuals to use as selection criterion
    
    Returns: 
        A list containing the k selected individuals

    Return:

    """
    # sort the population in ascending order
    individuals = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    
    psum=[]
    n = len(individuals)
    
    # set the probabilities for each individual to be selected
    # if all identical then make all probabilities equal
    if individuals[0].fitness.values[0] == individuals[-1].fitness.values[0]:
        psum = [i.fitness/float(n) for i in individuals]
    else:
        maxfit = individuals[-1].fitness.values[0]
        minfit = individuals[0].fitness.values[0]
        # set first probability
        psum.append(-individuals[0].fitness.values[0] + maxfit + minfit)
        
        for i,ind in enumerate(individuals[1:], start=1):
            psum.append(-ind.fitness.values[0] + minfit + maxfit + psum[i-1])
        
        for i in range(n):
            psum[i] /= psum[-1]
        
    # select individuals
    chosen=[]
    for i in range(k):
        #use a binary search 
        aspirants=[]
        for j in range(2):
            cutoff = random.random()
            lowerbound = 0
            upperbound = n-1
        
            while(upperbound >= lowerbound):
                indindex=lowerbound + (upperbound-lowerbound)//2
                if psum[indindex] > cutoff:
                    upperbound = indindex-1
                else:
                    lowerbound = indindex+1
        
            lowerbound = min(n-1, lowerbound) 
            lowerbound = max(0, lowerbound)
            
            aspirants.append(individuals[lowerbound])
        
        selected = (min(aspirants, key=attrgetter(fit_attr)))
        
        chosen.append(selected)    
        
    return chosen


def selBalAccLexicase(individuals: list, k: int) -> list:
    """Returns an individual that does the best on the fitness cases when
    considered one at a time in random order.
    http://faculty.hampshire.edu/lspector/pubs/lexicase-IEEE-TEC.pdf
    Matches will have 0 for a result while mismatches are 1
    nan also causes a candidate to drop out

    Args:
        individuals: A list of individuals to select from.
        k: The number of individuals to select.
    
    Returns: 
        A list of selected individuals.

    """
    
    selected_individuals = []

    nscores =  len(individuals[0].ptscores)

    for i in range(k):
        candidates = individuals
        cases = list(range(nscores))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            newcandidates = [x for x in candidates if x.ptscores[cases[0]] == 0]
            if not newcandidates:
                break
            candidates = newcandidates
            cases.pop(0)
            
        selected_individuals.append(random.choice(candidates))

    return selected_individuals

def selAutoEpsilonLexicase(individuals: list, k: int) -> list:
    """
    Adapted from DEAP selAutomaticEpsilonLexicase
    Returns an individual that does the best on the fitness cases when considered one at a
    time in random order.
    https://push-language.hampshire.edu/uploads/default/original/1X/35c30e47ef6323a0a949402914453f277fb1b5b0.pdf
    Implemented lambda_epsilon_y implementation.

    Args;
        individuals: A list of individuals to select from.
        k: The number of individuals to select.

    Returns: 
        A list of selected individuals.
    """

    selected_individuals = []
    nscores =  len(individuals[0].ptscores)

    for i in range(k):
        candidates = individuals
        cases = list(range(nscores))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = np.array([x.ptscores[cases[0]] for x in candidates])

            # check if all candidates are nan for this case
            if np.count_nonzero(np.isnan(errors_for_this_case)) == len(candidates):
                break
            nan_mask = np.isnan(errors_for_this_case)
            
            median_val = np.median(errors_for_this_case[~nan_mask])
            median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case[~nan_mask]])
            
            best_val_for_case = min(errors_for_this_case[~nan_mask])
            max_val_to_survive = best_val_for_case + median_absolute_deviation
            new_candidates = []
            for x in candidates:
                if not np.isnan(x.ptscores[cases[0]]) and x.ptscores[cases[0]] <= max_val_to_survive:
                    new_candidates.append(x)
            candidates = new_candidates
            cases.pop(0)

        selected_individuals.append(random.choice(candidates))

    return selected_individuals
