"""Algorithm parameters"""
import argparse
from operator import attrgetter
import os


params = {}

def less_than(parameters: dict,smaller: float,bigger: float) -> bool:
    """Check that smaller <= bigger.

    Args:
        parameters: number of splits (cross-validations)
        smaller: key for smaller parameter value
        bigger: key for bigger parameter value


    Returns: 
        True if smaller <= bigger, False otherwise
    """

    if parameters[smaller] > parameters[bigger]:
        print(f"{smaller} must be <= {bigger}")
        return False
    else:
        return True


def valid_parameters(parameters) -> bool:
    """
    Check that all parameters passed are valid for ATHENA run

        Args:
            parameters: Dictionary with key/value pairs for all parameters

        Returns: 
            True if all parameters are valid, False otherwise
    """

    all_valid = True
    # check that mutation rate and crossover are 0-1.0
    for rate in ['P_CROSSOVER', 'P_MUT']:  
        if parameters[rate] < 0.0 or parameters[rate] > 1.0:
            print(f"{rate} must be in range 0-1.0")
            all_valid=False

    # check that missing drop fraction is greater than 0.0 and <= 1.0
    for rate in ['DROP_FRACT']:  
        if parameters[rate] <= 0.0 or parameters[rate] > 1.0:
            print(f"{rate} must be in > 0 and <= 1.0")
            all_valid=False
    
    if  not less_than(parameters, 'ELITE_SIZE', 'HOF_SIZE'):
        all_valid = False
    
    if parameters['INIT'] == 'random' and not \
        less_than(parameters, 'MIN_INIT_GENOME_LENGTH', 'MAX_INIT_GENOME_LENGTH'):
        all_valid = False
    
    if parameters['INIT'] == 'sensible' and not \
        less_than(parameters, 'MIN_INIT_TREE_DEPTH', 'MAX_INIT_TREE_DEPTH'):
        all_valid = False
        
    # OUTCOME_FILE and GRAMMAR_FILES are required 
    if not os.path.isfile(parameters['GRAMMAR_FILE']):
        print(f"GRAMMAR_FILE {parameters['GRAMMAR_FILE']} not found")
        all_valid = False
    
    if parameters['OUTCOME_FILE'] == None or not os.path.isfile(parameters['OUTCOME_FILE']):
        print(f"OUTCOME_FILE {parameters['OUTCOME_FILE']} not found")
        all_valid = False
    else:
        if parameters['CONTIN_FILE']:
            if not os.path.isfile(parameters['CONTIN_FILE']):
                print(f"CONTIN_FILE {parameters['CONTIN_FILE']} not found")
                all_valid = False
        if parameters['GENO_FILE']:
            if not os.path.isfile(parameters['GENO_FILE']):
                print(f"GENO_FILE {parameters['GENO_FILE']} not found")
                all_valid = False
        if not(parameters['CONTIN_FILE'] or parameters['GENO_FILE']):
            print("At least one of CONTIN_FILE and GENO_FILE must be set")
            all_valid = False

    if parameters['FITNESS'] not in ['r-squared', 'balanced_acc', 'auc']:
        print("FITNESS must be either r-squared, balanced_acc or auc")
        all_valid = False
    
    if parameters['INIT'] not in ['random', 'sensible']:
        print("INIT must be either sensible or random")
        all_valid = False
    
    if parameters['CODON_CONSUMPTION'] not in ['eager', 'lazy']:
         print("CODON_CONSUMPTION must be either eager or lazy")
         all_valid = False

    if parameters['CROSSOVER'] not in ['onepoint', 'match', 'block']:
         print("CROSSOVER must be either onepoint, match or block")
         all_valid = False

    if parameters['GENO_ENCODE'] and parameters['GENO_ENCODE'] not in \
        ['add_quad', 'additive']:
        print("GENO_ENCODE must be either 'add_quad or additive")
    
    if parameters['SELECTION'] and parameters['SELECTION'] not in \
        ['tournament', 'lexicase', 'epsilon_lexicase']:
        print("SELECTION must be one of tournament, lexicase, epsilon_lexicase")
        
        
    if parameters['TEST_OUTCOME_FILE']:
        if not os.path.isfile(parameters['TEST_OUTCOME_FILE']):
            print(f"TEST_OUTCOME_FILE {parameters['TEST_OUTCOME_FILE']} not found")
            all_valid = False
        if parameters['TEST_CONTIN_FILE'] and not os.path.isfile(parameters['TEST_CONTIN_FILE']):
                print(f"TEST_CONTIN_FILE {parameters['TEST_CONTIN_FILE']} not found")
                all_valid = False
        if parameters['TEST_GENO_FILE'] and not os.path.isfile(parameters['TEST_GENO_FILE']):
                print(f"TEST_GENO_FILE {parameters['TEST_GENO_FILE']} not found")
                all_valid = False
        if not(parameters['TEST_CONTIN_FILE'] or parameters['TEST_GENO_FILE']):
            print("At least one of TEST_CONTIN_FILE and TEST_GENO_FILE must be set when TEST_OUTCOME_FILE is provided ")
            all_valid = False
        if parameters['CV'] != 1:
            print("CV must be set to 1 when using user provided testing set")
    
    if parameters['COLOR_MAP_FILE']:
        if not os.path.isfile(parameters['COLOR_MAP_FILE']):
            print(f"COLOR_MAP_FILE {parameters['COLOR_MAP_FILE']} not found")
            all_valid = false
            
    if parameters["GENOME_TYPE"] not in ['standard', 'leap', 'mcge']:
        print("GENOME_TYPE must be one of standard, leap, mcge")
    
    if parameters["GEN_CROSS_SWITCH"]:
        if not parameters["CROSSOVER2"]:
            print("GEN_CROSS_SWITCH specified but no second crossover type set (CROSSOVER2)")
            all_valid = False
    if parameters["CROSSOVER2"]:
        if not parameters["GEN_CROSS_SWITCH"]:
            print("CROSSOVER2 specified but not generation at which to perform switch (GEN_CROSS_SWITCH)")
            all_valid = False
    if parameters['CROSSOVER2'] not in ['onepoint', 'match', 'block', None]:
         print("CROSSOVER2 must be either onepoint, match or block")
         all_valid = False

    if parameters['CROSSOVER'] in ['match', 'block'] or parameters['CROSSOVER2'] in ['match', 'block']:
        if parameters['GENOME_TYPE'] != 'standard':
            print("GENOME_TYPE must be set to standard to use match or block for CROSSOVER/CROSSOVER2")
            all_valild = False

    return all_valid    
    

def load_param_file(file_name: str) -> dict:
    """
    Load in a params text file and set the params dictionary directly. The text file must
        be in the format key: value.
    
    Args;
        file_name: The name/location of a parameters file.

    Returns:
        params: dict with paramter string as key and parameter value
    """

    try:
        open(file_name, "r")
    except FileNotFoundError:
        s = "load_param_file\n" \
            "Error: Parameters file not found.\n"
        raise Exception(s)

    with open(file_name, 'r') as parameters:
        # Read the whole parameters file.
        content = parameters.readlines()

        for line in [l for l in content if not l.startswith("#")]:

            split = line.find(":")
            key, value = line[:split], line[split + 1:].strip()

            # Evaluate parameters.
            try:
                value = eval(value)

            except:
                # We can't evaluate, leave value as a string.
                pass

            if key == 'INCLUDEDVARS':
                value = value.split()

            # Set parameter
            params[key] = value
    

class SortedDefaultHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    Sorts the arguments in alphabetical order and inherits from 
    argparse.ArgumentDefaultsHelpFormatter so that default values
    are included in help strings.
    """
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortedDefaultHelpFormatter, self).add_arguments(actions)
        
        
def parse_cmd_args(arguments: list, has_mpi: bool=False) -> dict:
    """Parse command line arguments using argparse. 

    Args:
        arguments: Command-line arguments passed by user
        has_mpi: MPI related arguments included when has_mpi=True

    Returns:
        cmd_args: dict of command line options
    """

    parser = argparse.ArgumentParser(
        formatter_class=SortedDefaultHelpFormatter,
        prog='ATHENA',
        usage=argparse.SUPPRESS,
        description="""Welcome to ATHENA - Help.
        The following are the available command line arguments. Please see
        manual for more detailed information on the parameters.""",
        epilog="""Sample inputs are provided in the example directory of the distributed package.""")

    parser._optionals.title = 'ATHENA command-line usage'
    parser.add_argument('--param-file',
                        dest='PARAM_FILE',
                        type=str,
                        help='Specifies the parameters file to be used. Must '
                             'include the full file extension. Parameters defined '
                             'in the file are overriden by any command line arguments')
    parser.add_argument('--pop-size',
                        dest='POP_SIZE',
                        type=int,
                        default=250,
                        help='Sets population size for GE algorithm')
    parser.add_argument('--gens',
                        dest='GENS',
                        type=int,
                        default=50,
                        help='Sets number of generations in evolution')
    parser.add_argument('--p-crossover',
                        dest='P_CROSSOVER',
                        type=float,
                        default=0.8,
                        help='Sets probability of a crossover during selection')
    parser.add_argument('--crossover',
                        dest='CROSSOVER',
                        type=str,
                        default='onepoint',
                        choices=['onepoint', 'match', 'block'],
                        help='Options are onepoint, match and block. Specifies type of'
                        'crossover to use')
    parser.add_argument('--crossover2',
                        dest='CROSSOVER2',
                        type=str,
                        choices=['onepoint', 'match', 'block'],
                        help='Options are onepoint, match and block. Specifies type of'
                        'crossover to switch GE to utilize during run')
    parser.add_argument('--gen-cross-switch',
                        dest='GEN_CROSS_SWITCH',
                        type=int,
                        help='Switch crossover type at this generation')
    parser.add_argument('--p-mut',
                        dest='P_MUT',
                        type=float,
                        default=0.01,
                        help='Sets probability per codon of mutation')
    parser.add_argument('--nelite',
                        dest='ELITE_SIZE',
                        type=int,
                        default=1,
                        help='Sets number of best networks carried over to next'
                         'generation')
    parser.add_argument('--hof-size',
                        dest='HOF_SIZE',
                        type=int,
                        default=1,
                        help='Sets number of the best networks to save for reporting. '
                        'Must be >= to elite size')
    parser.add_argument('--codon-size',
                        dest='CODON_SIZE',
                        type=int,
                        default=65536,
                        help='Maximum value of a codon in the genome of an individual'
                        'in the evolutionary population. At a minimum it should be'
                        '>= the largest number of choices for a rule in the grammar')
    parser.add_argument('--codon-consumption',
                        dest='CODON_CONSUMPTION',
                        type=str,
                        default='eager',
                        choices=['eager', 'lazy'],
                        help='Options are eager and lazy. Specifies whether grammar '
                        'will consume codons when only one choice for a rule in the '
                        'grammar.')
    parser.add_argument('--cv',
                        dest='CV',
                        type=int,
                        default=5,
                        help='Sets number of cross-validations to split the data into.')
    parser.add_argument('--out',
                        dest='OUT',
                        type=str,
                        default='athena_results',
                        help='Sets basename and location for output files and can include relative'
                        ' or full path.')
    parser.add_argument('--genome-type',
                        dest='GENOME_TYPE',
                        choices=['standard', 'leap', 'mcge'],
                        default='standard',
                        type=str,
                        help='Sets GE genome type to use (standard, leap or mcge) '
                         'generation')
    parser.add_argument('--outcome-file',
                        dest='OUTCOME_FILE',
                        type=str,
                        help='Sets name of file containing outcomes (phenotypes) in'
                        ' input data')
    parser.add_argument('--outcome',
                        dest='OUTCOME',
                        type=str,
                        help='Column header to use in outcome (default is to use the first after the ID)')
    parser.add_argument('--geno-file',
                        dest='GENO_FILE',
                        type=str,
                        help='Sets name of file containing genotypes (0,1,2) in'
                        ' input data')
    parser.add_argument('--contin-file',
                        dest='CONTIN_FILE',
                        type=str,
                        help='Sets name of file containing continuous variables in'
                        ' input data')
    parser.add_argument('--drop-fract',
                        dest='DROP_FRACT',
                        type=float,
                        default=1.0,
                        help='Input variables are dropped if they equal/exceed this fraction of missing across data')
    parser.add_argument('--grammar-file',
                        dest='GRAMMAR_FILE',
                        type=str,
                        help='Sets name of grammar')
    parser.add_argument('--fitness',
                        dest='FITNESS',
                        choices=['r-squared', 'balanced_acc', 'auc'],
                        default='balanced_acc',
                        type=str,
                        help='Sets metric for fitness (balanced_acc, auc or r-squared)')
    parser.add_argument('--random-seed',
                        dest='RANDOM_SEED',
                        type=int,
                        default=12345,
                        help='Sets random seed for the run')
    parser.add_argument('--init',
                        dest='INIT',
                        type=str,
                        choices=['random', 'sensible'],
                        default='sensible',
                        help='Use sensible to use sensible initialization and random '
                        'to use random initialization')
    parser.add_argument('--max-init-tree-depth',
                        dest='MAX_INIT_TREE_DEPTH',
                        type=int,
                        default='11',
                        help='Sets maximum depth for tree created by sensible '
                        'initialization')
    parser.add_argument('--min-init-tree-depth',
                        dest='MIN_INIT_TREE_DEPTH',
                        type=int,
                        default=7,
                        help='Sets minimum depth for tree created by sensible '
                        'initialization')
    parser.add_argument('--min-init-genome-length',
                        dest='MIN_INIT_GENOME_LENGTH',
                        type=int,
                        default=50,
                        help='Sets minimum genome length for random initialization')
    parser.add_argument('--max-init-genome-length',
                        dest='MAX_INIT_GENOME_LENGTH',
                        type=int,
                        default=250,
                        help='Sets maximum genome length for random initializtion')
    parser.add_argument('--geno-encode',
                        dest='GENO_ENCODE',
                        choices=['add_quad', 'additive'],
                        type=str,
                        help='Sets genotype encoding. Must be either add_quad or additive')
    parser.add_argument('--scale-outcome',
                        dest='SCALE_OUTCOME',
                        action="store_true",
                        help="Sets flag for scaling outcome variable from 0 to 1.0")
    parser.add_argument('--scale-contin',
                        dest='SCALE_CONTIN',
                        action="store_true",
                        help="Sets flag for scaling continuous variables from 0 to 1.0")
    parser.add_argument('--missing',
                        dest='MISSING',
                        type=str,
                        help="Sets identifier for missing values input files")
    parser.add_argument('--selection',
                        dest='SELECTION',
                        choices=['tournament', 'lexicase', 'epsilon_lexicase'],
                        default='tournament',
                        type=str,
                        help="Sets selection type (tournament, lexicase, epsilon_lexicase)")
    parser.add_argument('--test-outcome-file',
                        dest='TEST_OUTCOME_FILE',
                        type=str,
                        default=None,
                        help="Set name of outcome file for designated test dataset")
    parser.add_argument('--test-geno-file',
                        dest='TEST_GENO_FILE',
                        type=str,
                        help="Set name of genoype data file for designated test dataset")
    parser.add_argument('--test-contin-file',
                        dest='TEST_CONTIN_FILE',
                        type=str,
                        help="Set name of continuous data file for designated test dataset")
    parser.add_argument('--max-depth',
                        dest='MAX_DEPTH',
                        type=int,
                        default=100,
                        help='Sets max depth for mapping with individuals exceeding this being invalid')
    parser.add_argument('--color-map-file',
                        dest='COLOR_MAP_FILE',
                        type=str,
                        help="File specifying colors to use for specific inputs")
    parser.add_argument('--includedvars',
                        dest='INCLUDEDVARS',
                        nargs='+',
                        type=str,
                        help="Variable names to be included in run (accepts multiple values")
                    
    if has_mpi:
        parser.add_argument('--gens-migrate',
                            dest='GENS_MIGRATE',
                            type=int,
                            default=25,
                            help='Sets generational interval for migrating best individuals for multi-process run')

    args = parser.parse_args(arguments)

    # convert to dict
    cmd_args = {key: value for key, value in vars(args).items()}
    
    # Set "None" values correctly.
    for key in sorted(cmd_args.keys()):
        # Check all specified arguments.

        if type(cmd_args[key]) == str and cmd_args[key].lower() == "none":
            # Allow for people not using correct capitalisation.

            cmd_args[key] = None

    return cmd_args

                         
def set_params(command_line_args: list, has_mpi: bool=False) -> dict:
    """
    Sets all parameters for the run. It parses command line arguments first and 
    then a paramters file if provided. The values in the parameters file will
    supercede any provided on the command line.

    Args:
        command_line_args: Command line arguments specified by the user.
        has_mpi: True when mpi detected so that parameters can be distributred to worker processes

    Returns: 
        params: dict with paramter string as key and parameter value
    """

    cmd_args = parse_cmd_args(command_line_args, has_mpi)

    params.update(cmd_args)
    # LOAD PARAMETERS FILE
    # These parameters in the file overwrites all previously set parameters.
    # if 'PARAM_FILE' in cmd_args:
    if cmd_args['PARAM_FILE'] is not None:
        load_param_file(cmd_args['PARAM_FILE'])

    return params
    
    
