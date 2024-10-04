""" Setup of DEAP structures and additional functions used in GENN algorithm """
# import grape
from deap import creator, base, tools
import grape.grape as grape
from genn.functions import activate_sigmoid, PA, PM, PS, PD, pdiv
import numpy as np
from sklearn.metrics import balanced_accuracy_score

INVALID_FITNESS = -1000

def configure_toolbox(genome_type: str, fitness: str, selection: str, 
                      init:str ='sensible') -> base.Toolbox:
    """Configure the DEAP toolbox for controlling GE algorithm

    Args:
        genome_type: Phenotypes (outcomes) filename
        fitness: SNP values filename
        selection: any continuous data filename
        init: scale outcome values from 0 to 1.0

    Returns:
        DEAP base.Toolbox configured for a GE run
    """

    toolbox = base.Toolbox()
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if genome_type == 'standard':
        creator.create('Individual', grape.Individual, fitness=creator.FitnessMax)
        if init == 'sensible':
            toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual) 
        else:
            toolbox.register("populationCreator", grape.random_initialisation, creator.Individual) 
        toolbox.register("mate", grape.crossover_onepoint)
        toolbox.register("mutate", grape.mutation_int_flip_per_codon)
    elif genome_type == 'leap':
        creator.create('Individual', grape.LeapIndividual, fitness=creator.FitnessMax)
        if init=='sensible':
            toolbox.register("populationCreator", grape.leap_sensible_initialisation, creator.Individual) 
        else:
            toolbox.register("populationCreator", grape.leap_random_initialisation, creator.Individual) 
        toolbox.register("mate", grape.leap_crossover_onepoint)#_leap2)
        toolbox.register("mutate", grape.leap_mutation_int_flip_per_codon)#_leap)
    elif genome_type == 'mcge':
        creator.create('Individual', grape.MCGEIndividual, fitness=creator.FitnessMax)
        if init=='sensible':
            toolbox.register("populationCreator", grape.mcge_sensible_initialisation, creator.Individual) 
        else:
            toolbox.register("populationCreator", grape.mcge_random_initializaion, creator.Individual) 
        toolbox.register("mate", grape.mcge_crossover_onepoint)#_leap2)
        toolbox.register("mutate", grape.mcge_mutation_int_flip_per_codon)#_leap)
    else:
        raise ValueError("genome_type must be standard, leap or mcge")
    
    if fitness=='r-squared':
        if selection == 'epsilon_lexicase':
            toolbox.register("evaluate", fitness_rsquared_lexicase)
            toolbox.register("select", grape.selAutoEpsilonLexicase)#, tournsize=7)
        else:
            toolbox.register("evaluate", fitness_rsquared)
            toolbox.register("select", tools.selTournament, tournsize=2)
    elif fitness == 'balanced_acc':
        if selection=='lexicase':
            toolbox.register("evaluate", fitness_balacc_lexicase)
            toolbox.register("select", grape.selBalAccLexicase)
        else:
            toolbox.register("evaluate", fitness_balacc)
            toolbox.register("select", tools.selTournament, tournsize=2)
    else:
        raise ValueError("fitness must be fitness_rsquared or fitness_balacc")
    
    return toolbox


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate r-squared values

    Args:
        y: Observed values
        y_hat: Predicted values

    Returns:
        r-squared value
    """

    nan_mask = np.isnan(y_hat)
    y_bar = y[~nan_mask].mean()
    ss_tot = ((y[~nan_mask]-y_bar)**2).sum()
    ss_res = ((y[~nan_mask]-y_hat[~nan_mask])**2).sum()
    return 1 - (ss_res/ss_tot)
    

def fitness_rsquared(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate r-squared fitness for this individual using points passed

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calcualting fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        r-squared fitness
    """

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
    
    
def fitness_rsquared_lexicase(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate r-squared fitness for this individual and store differences in
        predicted vs observed outcomes for use in lexicase selection

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calcualting fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        r-squared fitness
    """

    #points = [X, Y]
    x = points[0]
    y = points[1]
    
    if individual.invalid == True:
        individual.ptscores = np.full(len(y), np.nan)
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
        
        # store individual differences for lexicase
        individual.ptscores = np.absolute(y-pred)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        individual.ptscores = np.full(len(y), np.nan)
        fitness = INVALID_FITNESS
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise
    
    if fitness == float("inf"):
        individual.ptscores = np.full(len(y), np.nan)
        return INVALID_FITNESS,
    
    return fitness,
    

def fitness_balacc(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate balanced accuracy as fitness for this individual using points passed

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calcualting fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        balanced accuracy fitness
    """

    x = points[0]
    y = points[1]
    

    if individual.invalid == True:
        return INVALID_FITNESS,

    try:
        pred = eval(individual.phenotype)
        
#         pred2 = eval(compress_weights(individual.phenotype))
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
        
#         fitness_compressed = balanced_accuracy_score(y[~nan_mask],pred2)
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
    

def fitness_balacc_lexicase(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate balanced accuracy fitness for this individual and store differences in
        predicted vs observed outcomes for use in lexicase selection

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calcualting fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        balanced accuracy fitness
    """

    x = points[0]
    y = points[1]
    
    if individual.invalid == True:
        individual.ptscores = np.full(len(y), np.nan)
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
        
        # save individual point scores for use in lexicase selection
        full = np.copy(pred)
        full[~nan_mask] = np.where(pred[~nan_mask] < 0.5, 0, 1)
        individual.ptscores = np.absolute(y-full)
        
        
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = INVALID_FITNESS
        individual.ptscores = np.full(len(y), np.nan)
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise
        
    if fitness == float("inf"):
        individual.ptscores = np.full(len(y), np.nan)
        return INVALID_FITNESS,
    
    return fitness,
    
