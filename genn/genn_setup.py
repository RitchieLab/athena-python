# import grape
from deap import creator, base, tools
import grape.grape as grape
from genn.genn_functions import activate_sigmoid, PA, PM, PS, PD, pdiv
import numpy as np
from sklearn.metrics import balanced_accuracy_score

INVALID_FITNESS = -1000

def configure_toolbox(genome_type, fitness):
    toolbox = base.Toolbox()
    if genome_type == 'standard':
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create('Individual', grape.Individual, fitness=creator.FitnessMax)
        toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual) 
    #     toolbox.register("mate", grape.crossover_onepoint)
        toolbox.register("mate", grape.crossover_onepoint_opt)
    #     toolbox.register("mutate", grape.mutation_int_flip_per_codon)#_leap)
        toolbox.register("mutate", grape.mutation_int_flip_per_codon_opt)#_leap) #faster
    #     toolbox.register("mutate", grape.mutation_per_ind)
    elif genome_type == 'leap':
        creator.create('Individual', grape.LeapIndividual, fitness=creator.FitnessMax)
        toolbox.register("populationCreator", grape.leap_sensible_initialisation, creator.Individual) 
        toolbox.register("mate", grape.leap_crossover_onepoint)#_leap2)
        toolbox.register("mutate", grape.leap_mutation_int_flip_per_codon)#_leap)
    elif genome_type == 'mcge':
        creator.create('Individual', grape.MCGEIndividual, fitness=creator.FitnessMax)
        toolbox.register("populationCreator", grape.mcge_sensible_initialisation, creator.Individual) 
        toolbox.register("mate", grape.mcge_crossover_onepoint)#_leap2)
        toolbox.register("mutate", grape.mcge_mutation_int_flip_per_codon)#_leap)
    else:
        raise ValueError("genome_type must be standard, leap or mcge")
    
    if fitness=='r-squared':
        toolbox.register("evaluate", fitness_rsquared)
    elif fitness == 'balanced_acc':
        toolbox.register("evaluate", fitness_balacc)
    else:
        raise ValueError("fitness must be fitness_rsquared or fitness_balacc")
    
    toolbox.register("select", tools.selTournament, tournsize=7) 

    return toolbox


# y actual values
# y_hat predicted values
def r_squared(y, y_hat):
    nan_mask = np.isnan(y_hat)
    y_bar = y[~nan_mask].mean()
    ss_tot = ((y[~nan_mask]-y_bar)**2).sum()
    ss_res = ((y[~nan_mask]-y_hat[~nan_mask])**2).sum()
    return 1 - (ss_res/ss_tot)
    

def fitness_rsquared(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]
    
    if individual.invalid == True:
        return INVALID_FITNESS,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return INVALID_FITNESS,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)
    
    try:
        fitness = r_squared(y,pred)
        individual.nmissing = np.count_nonzero(np.isnan(pred))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = INVALID_FITNESS
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise
        
    if fitness == float("inf"):
        return INVALID_FITNESS,
    
    return fitness,
    
def fitness_balacc(individual, points):
    x = points[0]
    y = points[1]
    
    if individual.invalid == True:
        return INVALID_FITNESS,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return INVALID_FITNESS,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)
    
    try:
        nan_mask = np.isnan(pred)
        # assign case/control status
        pred_nonan = np.where(pred[~nan_mask] < 0.5, 0, 1)
        fitness = balanced_accuracy_score(y[~nan_mask],pred_nonan)
        individual.nmissing = np.count_nonzero(np.isnan(pred))
        
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = INVALID_FITNESS
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise
        
    if fitness == float("inf"):
        return INVALID_FITNESS,
    
    return fitness,
    
