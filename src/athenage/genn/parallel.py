"""Handles parrallel operations using MPI when available"""
try:
    from mpi4py import MPI
    has_mpi = True
    comm = MPI.COMM_WORLD
    proc_rank = comm.Get_rank()
    nprocs = comm.Get_size()
except ImportError:
    has_mpi = False
    proc_rank = 0
    nprocs = 1
    comm = None

import numpy as np
import pandas as pd
import itertools
    
def get_rank() -> int:
    """Return rank within MPI run for this process"""
    return comm.Get_rank()

def get_nprocs() -> int:
    """ Return number of processes in the run"""
    return comm.Get_size()
    
    
def distribute_params(params: dict, rank: int) -> dict:
    """Broadcast parameters to all processes in run
    
    Args:
        params: key is parameter and option is value
    
    Returns:
        params: dict containing parameters
    """

    params = comm.bcast(params, root=0)
    return params
    
def distribute_data(rank: int,data: pd.DataFrame,train_splits: np.ndarray, 
                    test_splits: np.ndarray, vmap: dict, 
                    grammar: str) -> tuple[pd.DataFrame,np.ndarray,np.ndarray,dict,str]:
    """Broadcast data and related values to all processes from root so that
        all processes are using the same data and splits.
    
    Args:
        rank: process number in MPI 
        data: dataset to analyze
        train_splits: contains indexes for managing training splits
        test_splits: contains indexes for managing testing splits
        vmap: dict mapping new variable name to old one
        grammar: contains grammar to use in GE 
    
    Returns:
        data: dataset to analyze
        train_splits: contains indexes for managing training splits
        test_splits: contains indexes for managing testing splits
        vmap: dict mapping new variable name to old one
        grammar: contains grammar to use in GE 
    """

    if rank != 0:
        data = None
        train_splits = None
        test_splits = None
        vmap = None
        
    data = comm.bcast(data, root=0)
    train_splits = comm.bcast(train_splits, root=0)
    test_splits = comm.bcast(test_splits, root=0)
    vmap = comm.bcast(vmap, root=0)
    grammar = comm.bcast(grammar, root=0)

    return data,train_splits, test_splits, vmap, grammar
    

def continue_run(rank: int, contin: bool) -> None:
    """ Stops run and ends MPI
    
    Args:
        rank: process rank in MPI
        contin: when False stop run
    
    Returns:
        None
    """

    if rank != 0:
        contin = None
    contin = comm.bcast(contin, root=0)
    if contin == False:
        exit()

def exchange_best(ind: "deap.creator.Individual") -> "deap.creator.Individual":
    """ Share best individual with all other processes"""
    return comm.allgather(ind)
    
def send_log_info(length: list, nodes: list, depth: list, used_codons: list, invalid: int, 
                  n_inds: int, n_unique_structs: int) -> dict:  
    """ Gather all logging information at root process
    
    Args:
        length: length of each individual 
        nodes: number of nodes for each individual
        depth: depth of each individual in population
        used_codons: number of used codons for each individual
        invalid: number of invalid individuals in population
        n_inds: number of individuals in population
        n_unique_structs: number of unique structures created by population
        
    Returns:
        log_data: dict containing compiled logging stats for all processes
    """      

    log_data = {'sum_length':sum(length), 'n_length':len(length), 'sum_nodes':sum(nodes),
        'n_nodes':len(nodes), 'sum_used_codons':sum(used_codons), 'n_used_codons':len(used_codons),
        'sum_depth':sum(depth), 'n_depth':len(depth), 'invalid':invalid, 'n_inds':n_inds,
        'n_unique_structs':n_unique_structs}
    recv = None
    recv = comm.gather(log_data, root=0)
    if proc_rank == 0:
        totals = recv[0]
        for i in range(1, len(recv)):
            for key,value in recv[i].items():
                totals[key] += recv[i][key]        
        return totals
    else:
        return log_data
            
def get_stats(stats: "deap.tools.Statistics", population: list) -> dict:
    """ Generate fitness lists for this process and send to the root
    
    Args:
        stats: deap statistics object
        population: individuals in population
    
    Returns:
        dict: contains statistics generated from fitness scores of the populations
    """
    # generate fitness lists and send to head proc
    scores = [ind.fitness.values[0] for ind in population if not ind.invalid]
    recv = None
    recv = comm.gather(scores, root=0)
    if proc_rank == 0:
        scores = [val for xs in recv for val in xs]
        
    return {'avg':np.nanmean(scores), 
        'std':np.nanstd(scores),
        'min':np.nanmin(scores),
        'max':np.nanmax(scores)}

