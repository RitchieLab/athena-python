# -*- coding: utf-8 -*-
"""
Created on Tue May 10 06:53:28 2022

@author: allan
"""

import re
import math
from operator import attrgetter
import numpy as np
import random
import copy

from math import modf


class Individual(object):
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
            

class LeapIndividual(object):
    """
    A GE LEAP mapping individual.
    genome is divided into frames - one codon per non-terminal in a frame
    stay in frame until need another NT that has already been consumed
    """

    def __init__(self, genome, grammar, max_depth, codon_consumption):
        """
        """
        
        self.genome = genome
        

        if codon_consumption == 'lazy':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_lazy_leap(genome, grammar, max_depth)
        elif codon_consumption == 'eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_eager_leap(genome, grammar, max_depth)
        else:
            raise ValueError("Unknown mapper")

class MCGEIndividual(object):
    """
    A  multichromosome GE mapping individual.
    genome is divided into chromosomes
    each chromosome contains codons for a non-terminal in the grammar
    """

    def __init__(self, genome, grammar, max_depth, codon_consumption):
        """
        """
        
        self.genome = genome
        if codon_consumption == 'lazy':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.consumed_codons = mapper_lazy_mcge(genome, grammar, max_depth)
        elif codon_consumption == 'eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.consumed_codons = mapper_eager_mcge(genome, grammar, max_depth)
        else:
            raise ValueError("Unknown mapper")



class Grammar(object):
    """
    Attributes:
    - non_terminals: list with each non-terminal (NT);
    - start_rule: first non-terminal;
    - nt_rule_size: number of nt rules used for frame size for set of codons in leap mapping 
        and for the number of chromosomes in MCGE
    - production_rules: list with each production rule (PR), which contains in each position:
        - the PR itself as a string
        - 'non-terminal' or 'terminal'
        - the arity (number of NTs in the PR)
        - production choice label
        - True, if it is recursive, and False, otherwise
        - the minimum depth to terminate the mapping of all NTs of this PR
    - n_rules: df
    
    """
    def __init__(self, bnf_grammar,codon_consumption='eager'):
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
                        recursive = check_recursiveness(self, NT_to_check, stack)
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

        
def check_recursiveness(self, NT, stack):
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
                recursive = check_recursiveness(self, NT_to_check, stack)
                if recursive:
                    return recursive
                stack.pop() #If the inclusion didn't show recursiveness, remove it before continuing
    return recursive

def selLexicaseFilterCount(individuals, k):
    """
   

    """
    selected_individuals = []
    #valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals
        
def mapper(genome, grammar, max_depth):
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
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
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_eager(genome, grammar, max_depth):
    """
    Identical to the previous one.
    Solve the names later.
    """    

    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
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
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure
    

def mapper_eager_leap(genome, grammar, max_depth):
    """
    'Eager' codon consumption mapping for LEAP genome
    """    

    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []

    # move along the genome in frames
    # consume codons in a frame until you find a repeat
    # shift to new frame 
    idx_frame = 0
    consumed_codons = [False for i in range(grammar.nt_rule_size)]
    n_frames = len(genome) / grammar.nt_rule_size
    
    while next_NT and idx_frame < n_frames:
        NT_index = grammar.non_terminals.index(next_NT)
        codon_frame_idx = grammar.rule_conversion[NT_index]
        if(consumed_codons[codon_frame_idx]):
            idx_frame += 1
            if idx_frame == n_frames:
                break
            consumed_codons = [False for i in range(grammar.nt_rule_size)]
        
        index_production_chosen = genome[idx_frame*grammar.nt_rule_size+codon_frame_idx] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        consumed_codons[codon_frame_idx] = True
        
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
        depth = max(list_depth)
        used_codons = idx_frame * grammar.nt_rule_size
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure


def mapper_eager_mcge(genome, grammar, max_depth):
    """
    Consumes condons even for non-terminals with only one production rule
    """    

    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    # track position in each chromosome
    consumed_codons = [ 0 for i in range(grammar.nt_rule_size)]
    
    while next_NT:
        NT_index = grammar.non_terminals.index(next_NT)
        chr_idx = grammar.rule_conversion[NT_index]
        if(consumed_codons[chr_idx] == len(genome[chr_idx])):
            break
            
        index_production_chosen = genome[chr_idx][consumed_codons[chr_idx]] % grammar.n_rules[NT_index]
        consumed_codons[chr_idx] += 1
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
        
    if next_NT:
        invalid = True
        used_codons = 0
        consumed_codons=[]
    else:
        invalid = False
        used_codons = sum(consumed_codons)
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure, consumed_codons



def mapper_lazy(genome, grammar, max_depth):
    """
    This mapper is similar to the previous one, but it does not consume codons
    when mapping a production rule with a single option."""
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            structure.append(index_production_chosen)
            idx_genome += 1
        
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
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure


def mapper_lazy_leap(genome, grammar, max_depth):
    """
    This mapper is similar to the standard mapper_lazy one, but modified to work with
    the leap mapping genome."""
    
#     idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
        
    # move along the genome in frames
    # consume codons in a frame until you find a repeat
    # shift to new frame 
    idx_frame = 0
    consumed_codons = [False for i in range(grammar.nt_rule_size)]
    n_frames = len(genome) / grammar.nt_rule_size
    
    while next_NT and idx_frame < n_frames:
        NT_index = grammar.non_terminals.index(next_NT)
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            # check if already used codon in this frame
            codon_frame_idx = grammar.rule_conversion[NT_index]
            if(consumed_codons[codon_frame_idx]):
                idx_frame += 1
                if idx_frame == n_frames:
                    break
                consumed_codons = [False for i in range(grammar.nt_rule_size)]
            index_production_chosen = genome[idx_frame*grammar.nt_rule_size+codon_frame_idx] % grammar.n_rules[NT_index]
            structure.append(index_production_chosen)
            consumed_codons[codon_frame_idx] = True
        
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
        used_codons = idx_frame * grammar.nt_rule_size
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure


def mapper_lazy_mcge(genome, grammar, max_depth):
    """
    This mapper is similar to the standard mapper_lazy one, but modified to work with
    the multiple chromosomes of MCGE."""
    
#     idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    # track position in each chromosome
    consumed_codons = [ 0 for i in range(grammar.nt_rule_size)]
        
    while next_NT:
        NT_index = grammar.non_terminals.index(next_NT)
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            chr_idx = grammar.rule_conversion[NT_index]
            # if the end of a chromosome has been reached, stop mapping
            if(consumed_codons[chr_idx] == len(genome[chr_idx])):
                break
            
            index_production_chosen = genome[chr_idx][consumed_codons[chr_idx]] % grammar.n_rules[NT_index]
            consumed_codons[chr_idx] += 1
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
            
        
    if next_NT:
        invalid = True
        used_codons = 0
        consumed_codons=[]
    else:
        invalid = False
        used_codons = sum(consumed_codons)
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure, consumed_codons
    

            
def random_initialisation(ind_class, pop_size, bnf_grammar, 
                          min_init_genome_length, max_init_genome_length,
                          max_init_depth, codon_size, codon_consumption,
                          genome_representation):
        """
        Randomly generated linear genome compatible with both Leap and standard mapping
        """
        population = []
        
        for i in range(pop_size):
            genome = []
            init_genome_length = random.randint(min_init_genome_length, max_init_genome_length)
            for j in range(init_genome_length):
                genome.append(random.randint(0, codon_size))
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            population.append(ind)
            
        if genome_representation == 'list':
            return population
        elif genome_representation == 'numpy':
            for ind in population:
                ind.genome = np.array(ind.genome)
            return population
        else:
            raise ValueError("Unkonwn genome representation")
        
            
def leap_random_initialisation(ind_class, pop_size, bnf_grammar, 
                          min_init_genome_length, max_init_genome_length,
                          max_init_depth, codon_size, codon_consumption,
                          genome_representation):
        """
        Randomly generated linear genome compatible with Leap
        """
        population = []
        # generate a complete set of frames for Leap 
        codon_size = bnf_grammar.nt_rule_size
        minsize = min_init_genome_length // codon_size + 1
        maxsize = max_init_genome_length // codon_size + 1

        for i in range(pop_size):
            genome = []
            init_genome_length = random.randint(minsize, maxsize) * codon_size
            for j in range(init_genome_length):
                genome.append(random.randint(0, codon_size))
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            population.append(ind)
            
        if genome_representation == 'list':
            return population
        elif genome_representation == 'numpy':
            for ind in population:
                ind.genome = np.array(ind.genome)
            return population
        else:
            raise ValueError("Unkonwn genome representation")
        
def mcge_random_initializaion(ind_class, pop_size, bnf_grammar, 
                          min_init_genome_length, max_init_genome_length,
                          max_init_depth, codon_size, codon_consumption,
                          genome_representation):
        """
        Randomly generated multiplie chromosome genome compatible MCGE
        """
        population = []

        for i in range(pop_size):
            genome = []
            for i in range(bnf_grammar.nt_rule_size):
                chrom = []
                init_genome_length = random.randint(min_init_genome_length, max_init_genome_length)
                for j in range(init_genome_length):
                    chrom.append(random.randint(0, codon_size))
                genome.append(chrom)

            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            population.append(ind)
            
        if genome_representation == 'list':
            return population
        elif genome_representation == 'numpy':
            for ind in population:
                ind.genome = np.array(ind.genome)
            return population
        else:
            raise ValueError("Unkonwn genome representation")


    
def sensible_initialisation(ind_class, pop_size, bnf_grammar, min_init_depth, 
                            max_init_depth, codon_size, codon_consumption,
                            genome_representation):
        """
        
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
                    elif codon_consumption == 'lazy':
                        if len(total_options) > 1:
                            remainders.append(Ch[3])
                            possible_choices.append(len(total_options))
                    
                    if Ch[2] > 1:
                        if idx_branch == 0:
                            depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        else:
                            depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    if Ch[1] == 'terminal':
                        idx_branch += 1
                    
                    remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
                
                #Generate the genome
                genome = []
                if codon_consumption == 'eager' or codon_consumption == 'lazy':
                    for k in range(len(remainders)):
                        codon = (random.randint(0,int(1e10)) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                        genome.append(codon)
                else:
                    raise ValueError("Unknown mapper")
                    
                #Include a tail with 50% of the genome's size
                size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                for j in range(size_tail):
                    genome.append(random.randint(0,codon_size))
                    
                #Initialise the individual and include in the population
                ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption)
                
                #Check if the individual was mapped correctly
#                 if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
#                     print(f"ind.structure={ind.structure} remainders={remainders}")
#                     raise Exception('error in the mapping')
                    
                population.append(ind)    
            
        for i in range(n_full):
            remainders = [] #it will register the choices
            possible_choices = [] #it will register the respective possible choices

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
                elif codon_consumption == 'lazy':
                    if len(total_options) > 1:
                        remainders.append(Ch[3])
                        possible_choices.append(len(total_options))

                if Ch[2] > 1:
                    if idx_branch == 0:
                        depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    else:
                        depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                if Ch[1] == 'terminal':
                    idx_branch += 1
                
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
            
            #Generate the genome
            genome = []
            if codon_consumption == 'eager' or codon_consumption == 'lazy':
            	for j in range(len(remainders)):
            		codon = (random.randint(0,int(1e10)) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
            		genome.append(codon)
            else:
            	raise ValueError("Unknown mapper")

            #Include a tail with 50% of the genome's size
            if codon_consumption == 'eager' or codon_consumption == 'lazy':
                size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
            
            for j in range(size_tail):
                genome.append(random.randint(0,codon_size))
                
            #Initialise the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)

            #Check if the individual was mapped correctly
#             if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
#                 raise Exception('error in the mapping')
                
            population.append(ind)    
    
        if genome_representation == 'list':
            return population
        elif genome_representation == 'numpy':
            for ind in population:
                ind.genome = np.array(ind.genome)
            return population
        else:
            raise ValueError("Unkonwn genome representation")


def leap_sensible_initialisation(ind_class, pop_size, bnf_grammar, min_init_depth, 
                            max_init_depth, codon_size, codon_consumption,
                            genome_representation):
        """
        
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
                # self.rule_conversion.append(idx)
                # self.nt_rule_size += 1
                genome = []
                if codon_consumption == 'eager' or codon_consumption == 'lazy':
                    frame = [-1 for i in range(bnf_grammar.nt_rule_size)]
                    for k in range(len(remainders)):
                        codon = (random.randint(0,int(1e10)) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                        # check if frame has been used
                        if frame[bnf_grammar.rule_conversion[rules_used[k]]] != -1:
                            # fill unused codons in frame randomly
                            frame = [random.randint(0,codon_size) if x == -1 else x for x in frame]
                            genome.extend(frame)
                            frame = [-1 for i in range(bnf_grammar.nt_rule_size)]
                        # TODO
                        frame[bnf_grammar.rule_conversion[rules_used[k]]]=codon
                    if frame.count(frame[0]) != len(frame) or frame[0] != -1:
                        frame = [random.randint(0,codon_size) if x == -1 else x for x in frame]
                        genome.extend(frame)
                            
                else:
                    raise ValueError("Unknown mapper")
                    
                #Include a tail with 50% of the genome's size
                # calculate based on number of frames and extend by complete frames
                n_frames = len(genome) / bnf_grammar.nt_rule_size
                size_tail = max(int(0.5*n_frames)*bnf_grammar.nt_rule_size, max(int(bnf_grammar.nt_rule_size),bnf_grammar.nt_rule_size)) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                
                for j in range(size_tail):
                    genome.append(random.randint(0,codon_size))
                    
                #Initialise the individual and include in the population
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
            
            #Generate the genome
            genome = []
            if codon_consumption == 'eager' or codon_consumption == 'lazy':
                frame = [-1 for i in range(bnf_grammar.nt_rule_size)]
                for j in range(len(remainders)):
                    codon = (random.randint(0,int(1e10)) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
                    if frame[bnf_grammar.rule_conversion[rules_used[j]]] != -1:
                        # fill unused codons in frame randomly
                        frame = [random.randint(0,codon_size) if x == -1 else x for x in frame]
                        genome.extend(frame)
                        frame = [-1 for i in range(bnf_grammar.nt_rule_size)]
                    frame[bnf_grammar.rule_conversion[rules_used[j]]]=codon
                if frame.count(frame[0]) != len(frame) or frame[0] != -1:
                    frame = [random.randint(0,codon_size) if x == -1 else x for x in frame]
                    genome.extend(frame)
            else:
                raise ValueError("Unknown mapper")

            #Include a tail with 50% of the genome's size
            # calculate based on number of frames and extend by complete frames
            n_frames = len(genome) / bnf_grammar.nt_rule_size
            size_tail = max(int(0.5*n_frames)*bnf_grammar.nt_rule_size, max(int(bnf_grammar.nt_rule_size),bnf_grammar.nt_rule_size)) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
            
            for j in range(size_tail):
                genome.append(random.randint(0,codon_size))
            
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


def mcge_sensible_initialisation(ind_class, pop_size, bnf_grammar, min_init_depth, 
                            max_init_depth, codon_size, codon_consumption,
                            genome_representation):
        """
        
        """
        #Calculate the number of individuals to be generated with each method
        is_odd = pop_size % 2
        n_grow = int(pop_size/2)
        
        n_sets_grow = max_init_depth - min_init_depth + 1
        set_size = int(n_grow/n_sets_grow)
        remaining = n_grow % n_sets_grow
        
        n_full = n_grow + is_odd + remaining #if pop_size is odd, generate an extra ind with "full"
        
        #TODO check if it is possible to generate inds with max_init_depth
        # this has not been done as I discovered before debugging my problem with GENN grammar
        
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
                # self.rule_conversion.append(idx)
                # self.nt_rule_size += 1
                
                # set up 2-D list 
                genome = [ [] for i in range(bnf_grammar.nt_rule_size)]
                if codon_consumption == 'eager' or codon_consumption == 'lazy':
                    for k in range(len(remainders)):
                        codon = (random.randint(0,int(1e10)) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                        genome[bnf_grammar.rule_conversion[rules_used[k]]].append(codon)
                            
                else:
                    raise ValueError("Unknown mapper")
                    
                #Include a tail with 50% of the genome's size
                # calculate based on number of frames and extend by complete frames
                # add tail to each chromosome
                for chrom in genome:
                    size_tail = max(int(0.5*len(chrom)), 10) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                    for j in range(size_tail):
                        chrom.append(random.randint(0,codon_size))
                
                #Initialise the individual and include in the population
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
            
            #Generate the genome
            genome = [ [] for i in range(bnf_grammar.nt_rule_size)]
            if codon_consumption == 'eager' or codon_consumption == 'lazy':
                for j in range(len(remainders)):
                    codon = (random.randint(0,int(1e10)) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
                    genome[bnf_grammar.rule_conversion[rules_used[j]]].append(codon)
            else:
                raise ValueError("Unknown mapper")

            #Include a tail with 50% of the genome's size
            for chrom in genome:
                size_tail = max(int(0.5*len(chrom)), 10) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                for j in range(size_tail):
                    chrom.append(random.randint(0,codon_size))
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


            
def crossover_onepoint(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       genome_representation='list', max_genome_length=None):
    """
    
    """
    # restrict crossover to effective genome when individual is valid
    if parent0.invalid: #used_codons = 0
        possible_crossover_codons0 = len(parent0.genome)
    else:
        possible_crossover_codons0 = min(len(parent0.genome), parent0.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_codons1 = len(parent1.genome)
    else:
        possible_crossover_codons1 = min(len(parent1.genome), parent1.used_codons)
    
    point0 = random.randint(1, possible_crossover_codons0)
    point1 = random.randint(1, possible_crossover_codons1)
  
    if genome_representation == 'list':
        #Operate crossover
        new_genome0 = parent0.genome[0:point0] + parent1.genome[point1:]
        new_genome1 = parent1.genome[0:point1] + parent0.genome[point0:]
    else:
        raise ValueError("Only 'list' representation is implemented")
    
    new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
    new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
    
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

    

def leap_crossover_onepoint(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       genome_representation, max_genome_length):
    """
    Onepoint crossover conducted in a leap genome 
    Select frames in each parent and then cross at the same codon position in the frame
    """
    if parent0.invalid: #used_codons = 0
        possible_crossover_codons0 = len(parent0.genome)
    else:
        possible_crossover_codons0 = min(len(parent0.genome), parent0.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_codons1 = len(parent1.genome)
    else:
        possible_crossover_codons1 = min(len(parent1.genome), parent1.used_codons)

    frame0 = random.randint(0,possible_crossover_codons0 // bnf_grammar.nt_rule_size -1 )
    frame1 = random.randint(0,possible_crossover_codons1 // bnf_grammar.nt_rule_size -1 )
        
    codon_cross = random.randint(0,bnf_grammar.nt_rule_size - 1)
    
    point0 = frame0*bnf_grammar.nt_rule_size+codon_cross
    point1 = frame1*bnf_grammar.nt_rule_size+codon_cross
      
    if genome_representation == 'numpy':
        #TODO This operations is not working in case of wrapping
        len0 = point0 + (len(parent1.genome) - point1)
        len1 = point1 + (len(parent0.genome) - point0)
        new_genome0 = np.zeros([len0], dtype=int)
        new_genome1 = np.zeros([len1], dtype=int)

        #Operate crossover
        new_genome0[0:point0] = parent0.genome[0:point0]
        new_genome0[point0:] = parent1.genome[point1:]
        new_genome1[0:point1] = parent1.genome[0:point1]
        new_genome1[point1:] = parent0.genome[point0:]
        
    elif genome_representation == 'list':
        #Operate crossover
        new_genome0 = parent0.genome[0:point0] + parent1.genome[point1:]
        new_genome1 = parent1.genome[0:point1] + parent0.genome[point0:]
    else:
        raise ValueError("Unknown genome representation")
      
    new_ind0 = reMap_leap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
    new_ind1 = reMap_leap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
    
    if new_ind0.depth > max_depth:
        new_ind0.invalid = True
    if new_ind1.depth > max_depth:
        new_ind1.invalid = True
    
    if max_genome_length:
        if len(new_ind0.genome) > max_genome_length:
            new_ind0.invalid = True
        if len(new_ind1.genome) > max_genome_length:
            new_ind1.invalid = True
        
    return new_ind0, new_ind1   


def mcge_crossover_onepoint(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       genome_representation, max_genome_length):
    """
    One point crossover conducted in a multi chromosome GE genome
    Select one chromosome type (matched NT rules) and cross at random points
    along that chromosome
    Restrict by the used portion of the chromosome
    """
    
    idx_chr = random.randint(0,bnf_grammar.nt_rule_size-1)
    
    if parent0.invalid or parent0.consumed_codons[idx_chr] == 0: 
        possible_crossover_codons0 = len(parent0.genome[idx_chr])
    else:
        possible_crossover_codons0 = min(len(parent0.genome[idx_chr]), parent0.consumed_codons[idx_chr]) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid or parent1.consumed_codons[idx_chr] == 0:
        possible_crossover_codons1 = len(parent1.genome[idx_chr])
    else:
        possible_crossover_codons1 = min(len(parent1.genome[idx_chr]), parent1.consumed_codons[idx_chr])

    #Set points for crossover within the effective part of the genomes
    point0 = random.randint(1, possible_crossover_codons0)
    point1 = random.randint(1, possible_crossover_codons1)
    
      
    if genome_representation == 'numpy':
        #TODO Update this code to work with MCGE
        #TODO This operations is not working in case of wrapping
        len0 = point0 + (len(parent1.genome) - point1)
        len1 = point1 + (len(parent0.genome) - point0)
        new_genome0 = np.zeros([len0], dtype=int)
        new_genome1 = np.zeros([len1], dtype=int)

        #Operate crossover
        new_genome0[0:point0] = parent0.genome[0:point0]
        new_genome0[point0:] = parent1.genome[point1:]
        new_genome1[0:point1] = parent1.genome[0:point1]
        new_genome1[point1:] = parent0.genome[point0:]
        
    elif genome_representation == 'list':
        #Operate crossover - copy parents and then cross over designated chromosome
        new_genome0 = copy.deepcopy(parent0.genome)
        new_genome1 = copy.deepcopy(parent1.genome)
        new_genome0[idx_chr] = parent0.genome[idx_chr][0:point0] + parent1.genome[idx_chr][point1:]
        new_genome1[idx_chr] = parent1.genome[idx_chr][0:point1] + parent0.genome[idx_chr][point0:]
        
    else:
        raise ValueError("Unknown genome representation")
        
        
    new_ind0 = reMap_mcge(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
    new_ind1 = reMap_mcge(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
       
    if new_ind0.depth > max_depth:
        new_ind0.invalid = True
    if new_ind1.depth > max_depth:
        new_ind1.invalid = True
 
    if max_genome_length:
        if len(new_ind0.genome) > max_genome_length:
            new_ind0.invalid = True
        if len(new_ind1.genome) > max_genome_length:
            new_ind1.invalid = True
        
    return new_ind0, new_ind1 



def mutation_int_flip_per_codon(ind, mut_probability, codon_size, bnf_grammar, max_depth, 
                                codon_consumption, max_genome_length=None):
    """

    """
    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length

    mutated_ = False
    
    for i in range(possible_mutation_codons):
        if random.random() < mut_probability:
            ind.genome[i] = random.randint(0, codon_size)
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
    


def leap_mutation_int_flip_per_codon(ind, mut_probability, codon_size, bnf_grammar, max_depth, 
                                codon_consumption, max_genome_length):
    """

    """
    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length

    mutated_ = False
    
    for i in range(possible_mutation_codons):
        if random.random() < mut_probability:
            ind.genome[i] = random.randint(0, codon_size)
            mutated_ = True

    if mutated_:
        new_ind = reMap_leap(ind, ind.genome, bnf_grammar, max_depth, codon_consumption)
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


# import copy
def mcge_mutation_int_flip_per_codon(ind, mut_probability, codon_size, bnf_grammar, max_depth, 
                                codon_consumption, max_genome_length):
    """
    Mutation operator for multi-chromosome GE genome
    Mutation done per chromosome
    """    
    # Operation mutation within the effective part of the genome
    mutated_ = False
    for idx_chr in range(len(ind.genome)):
        if ind.invalid: #used_codons = 0
            possible_mutation_codons = len(ind.genome[idx_chr])
        else:
            possible_mutation_codons = min(len(ind.genome[idx_chr]), ind.consumed_codons[idx_chr]) 

        for i in range(possible_mutation_codons):
            if random.random() < mut_probability:
                ind.genome[idx_chr][i] = random.randint(0, codon_size)
                mutated_ = True
    
    if mutated_:
        new_ind = reMap_mcge(ind, ind.genome, bnf_grammar, max_depth, codon_consumption)
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


def reMap(ind, genome, bnf_grammar, max_tree_depth, codon_consumption):
    ind.genome = genome
    if codon_consumption == 'lazy':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_lazy(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_eager(genome, bnf_grammar, max_tree_depth)
#         ind.structure = mapper_eager_opt(genome, bnf_grammar, max_tree_depth)
    else:
        raise ValueError("Unknown mapper")    
        
        
    return ind


def reMap_leap(ind, genome, bnf_grammar, max_tree_depth, codon_consumption):
    ind.genome = genome
    if codon_consumption == 'lazy':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_lazy_leap(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_eager_leap(genome, bnf_grammar, max_tree_depth)
    else:
        raise ValueError("Unknown mapper")
        
    return ind


def reMap_mcge(ind, genome, bnf_grammar, max_tree_depth, codon_consumption):
    ind.genome = genome
    if codon_consumption == 'lazy':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.consumed_codons = mapper_lazy_mcge(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.consumed_codons = mapper_eager_mcge(genome, bnf_grammar, max_tree_depth)
    else:
        raise ValueError("Unknown mapper")
        
    return ind

def replace_nth(string, substring, new_substring, nth):
    find = string.find(substring)
    i = find != -1
    while find != -1 and i != nth:
        find = string.find(substring, find + 1)
        i += 1
    if i == nth:
        return string[:find] + new_substring + string[find+len(substring):]
    return string

def selTournamentWithoutInvalids(individuals, k, tournsize, fit_attr="fitness"):
    """
    A simple tournament selection, which avoid invalid individuals.
    """
    chosen = []
    valid_individuals = [i for i in individuals if not i.invalid]
    while len(chosen) < k:
        aspirants = random.sample(valid_individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen

def selAthenaRoulette(individuals, k, fit_attr="fitness"):
    """Matches the ATHENA roulette selection method

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list containing the k selected individuals
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


def selTournRoulette(individuals, k, fit_attr="fitness"):
    """Pick two individuals from the population using the RouletteWheel selection
    method.  Then return the better of the two individuals. Repeat to get k individuals
    to return. Matches the ATHENA tournament selection method

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list containing the k selected individuals
    """
    # sort the population in ascending order
    individuals = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    
    psum=[]
    n = len(individuals)
    
    # set the probabilities for each individual to be selected
    # if all identical then make all probabilities equal
    if individuals[0].fitness.values[0] == individuals[-1].fitness.values[0]:
        print(" ")
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


def selBalAccLexicase(individuals, k):
    """Returns an individual that does the best on the fitness cases when
    considered one at a time in random order.
    http://faculty.hampshire.edu/lspector/pubs/lexicase-IEEE-TEC.pdf
    Matches will have 0 for a result while mismatches are 1
    nan also causes a candidate to drop out

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
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

def selAutoEpsilonLexicase(individuals, k):
    """
    Adapted from DEAP selAutomaticEpsilonLexicase
    Returns an individual that does the best on the fitness cases when considered one at a
    time in random order.
    https://push-language.hampshire.edu/uploads/default/original/1X/35c30e47ef6323a0a949402914453f277fb1b5b0.pdf
    Implemented lambda_epsilon_y implementation.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
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