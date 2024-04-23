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
    
def get_rank():
    return comm.Get_rank()

def get_nprocs():
    return comm.Get_size()
    
    
def distribute_params(params, rank):
    params = comm.bcast(params, root=0)
    return params
    
def distribute_data(rank,data,train_splits, test_splits, vmap, grammar):
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
    

def continue_run(rank, contin):
    if rank != 0:
        contin = None
    contin = comm.bcast(contin, root=0)
    if contin == False:
        exit()

def exchange_best(ind):
    return comm.allgather(ind)
    
def send_log_info(length, nodes, depth, used_codons, invalid, n_inds, n_unique_structs):        
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
            
def get_stats(stats, population):
    # generate fitness lists and send to head proc
#     print(type(population[0]))
#     print(population[0].fitness.values)
#     vals = population[0].fitness.values
#     print(vals)
#     exit()
    scores = [ind.fitness.values[0] for ind in population if not ind.invalid]
#     print(scores)
    # pass all the scores to the root
    # run the functions set in stats on them
    # return the dictionary with the values for root
#     stats.register("avg", np.nanmean)
#     stats.register("std", np.nanstd)
#     stats.register("min", np.nanmin)
#     stats.register("max", np.nanmax)

    recv = None
    recv = comm.gather(scores, root=0)
    if proc_rank == 0:
#         print(f"proc_rank={proc_rank} recv={recv}")
#         print(recv[0][0].values[0])
#         print(recv)
        scores = [val for xs in recv for val in xs]
#         print(scores)
        
    return {'avg':np.nanmean(scores), 
        'std':np.nanstd(scores),
        'min':np.nanmin(scores),
        'max':np.nanmax(scores)}
    
#     if proc_rank==0:
#         print(compiled)
# #         scores = [x.values[0] 
# #             for xs in scores
# #             for x in xs]
# #         print(f"scores={scores}")
#     
#     exit()

