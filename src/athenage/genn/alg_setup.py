""" Setup of DEAP structures and additional functions used in GENN algorithm """
# import grape
from deap import creator, base, tools
import athenage.grape.grape as grape
from athenage.genn.functions import activate, PA, PM, PS, PD, pdiv, PAND, PNAND, PXOR, POR, PNOR
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, average_precision_score

INVALID_FITNESS = -1000

def configure_toolbox(fitness: str, selection: str, crosstype:str ='match',
                      init:str ='sensible') -> base.Toolbox:
    """Configure the DEAP toolbox for controlling GE algorithm

    Args:
        fitness: SNP values filename
        selection: type of selection operator
        crosstype: type of crossover operator
        init: scale outcome values from 0 to 1.0

    Returns:
        DEAP base.Toolbox configured for a GE run
    """

    toolbox = base.Toolbox()
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create('Individual', grape.Individual, fitness=creator.FitnessMax)
    if init == 'sensible':
        toolbox.register("populationCreator", grape.sensible_initialization, creator.Individual) 
    else:
        toolbox.register("populationCreator", grape.random_initialization, creator.Individual) 

    if crosstype == 'onepoint':
        toolbox.register("mate", grape.crossover_onepoint)
    elif crosstype == 'match':
        toolbox.register("mate", grape.crossover_match)
    elif crosstype == 'block':
        toolbox.register("mate", grape.crossover_block)

    toolbox.register("mutate", grape.mutation_int_flip_per_codon)
    
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
    elif fitness == 'auc':
        if selection=='lexicase':
            toolbox.register("evaluate", fitness_auc_lexicase)
            toolbox.register("select", grape.selBalAccLexicase)
        else:
            toolbox.register("evaluate", fitness_auc)
            toolbox.register("select", tools.selTournament, tournsize=2)
    elif fitness=='f1_score':
        if selection=='lexicase':
             toolbox.register("evaluate", fitness_f1_lexicase)
             toolbox.register("select", grape.selBalAccLexicase)
        else:
             toolbox.register("evaluate", fitness_f1)
             toolbox.register("select",tools.selTournament, tournsize=2 )
    elif fitness=='auprc':
        if selection=='lexicase':
             toolbox.register("evaluate", fitness_auprc_lexicase)
             toolbox.register("select", grape.selBalAccLexicase)
        else:
             toolbox.register("evaluate", fitness_auprc)
             toolbox.register("select",tools.selTournament, tournsize=2 )    
    else:
        raise ValueError("fitness must be fitness_rsquared or fitness_balacc")
    
    return toolbox


def set_crossover(toolbox: 'deap.base.toolbox', crosstype: str) -> None:
    """Sets crossover type for toolbox

    Args:
        toolbox: DEAP toolbox
        crosstype: specifies type to use

    Returns:
        None
    """
    if crosstype == 'onepoint':
        toolbox.register("mate", grape.crossover_onepoint)
    elif crosstype == 'match':
        toolbox.register("mate", grape.crossover_match)
    elif crosstype == 'block':
        toolbox.register("mate", grape.crossover_block)


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
        points: 2-D list containing inputs and outcome for calculating fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        r-squared fitness
    """
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
        points: 2-D list containing inputs and outcome for calculating fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        r-squared fitness
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
        points: 2-D list containing inputs and outcome for calculating fitness
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
        points: 2-D list containing inputs and outcome for calculating fitness
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

def fitness_f1(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate F1 score (also known as balanced F-score or F-measure) as fitness for this individual using points passed

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calculating fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        f1 score fitness
    """

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
        fitness = f1_score(y[~nan_mask],pred_nonan)
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

def fitness_f1_lexicase(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculatethe F1 score (also known as balanced F-score or F-measure) for this individual and store differences in
        predicted vs observed outcomes for use in lexicase selection

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calculating fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        f1 score fitness
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
        fitness = f1_score(y[~nan_mask],pred_nonan)
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

def fitness_auprc(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate area under Precision-Recall (PR) curve as fitness for this individual using points passed

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calculating fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        f1 score fitness
    """

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
        fitness = average_precision_score(y[~nan_mask],pred_nonan)
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

def fitness_auprc_lexicase(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate the AUPRC (area under Precision-Recall curve) for this individual and store differences in
        predicted vs observed outcomes for use in lexicase selection

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calculating fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        auprc score fitness
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
        fitness = average_precision_score(y[~nan_mask],pred_nonan)
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


def fitness_auc(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate area under the curve (AUC)
    as fitness for this individual using points passed

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calculating fitness
            points[0] contains 2-D np.ndarray of all inputs

    Returns:
        AUC fitness
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
        fitness = roc_auc_score(y[~nan_mask],pred_nonan)
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
    
def fitness_auc_lexicase(individual: 'deap.creator.Individual', points: list) -> float:
    """Calculate area under the curve (AUC) for this individual and store differences in
        predicted vs observed outcomes for use in lexicase selection

    Args:
        individual: solution being evaluated for fitness
        points: 2-D list containing inputs and outcome for calculating fitness
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
        fitness = roc_auc_score(y[~nan_mask],pred_nonan)
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
