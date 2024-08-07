import argparse
from operator import attrgetter
import os


"""Algorithm parameters"""
params = {

    'POPULATION_SIZE': 250,
    'GENERATIONS': 50,
    
    'P_CROSSOVER':0.8,
    'P_MUTATION':0.01,
    
    'ELITE_SIZE':1,
    # hall of fame must be >= ELITE_SIZE
    'HALLOFFAME_SIZE':1,
    
    'CODON_SIZE':250,
    'CODON_CONSUMPTION':'eager',
    
    # number of cross-validations 
    'CV':5,

    # Prefix for output files, can include path, i.e results/athena
    'OUT': 'athena_results',

    # genome representation (also accepts mcge and leap)
    'GENOME_TYPE': 'standard',

    # Set input files, requires outcome (pheno) file and at least one of geno_file and contin_file
    'GENO_FILE' : None,
    'OUTCOME_FILE' : None,
    'CONTIN_FILE' : None,
    # Set grammar file
    'GRAMMAR_FILE': None,
    # Set parameters file
    'PARAM_FILE': None,
    
    # user-specified testing set
    'TEST_OUTCOME_FILE': None,
    'TEST_GENO_FILE': None,
    'TEST_CONTIN_FILE': None,
    
    #plotting files
    'COLOR_MAP_FILE': None,
    
    # can be r-squared or balanced accuracy for case/control inputs
    'FITNESS': 'r-squared',
    
    'RANDOM_SEED': 12345,
    
    'GENO_ENCODE': None,
    
    # Initialization parameters
    'MIN_INIT_GENOME_LENGTH' : 30,
    'MAX_INIT_GENOME_LENGTH' : 200,
    # Uses sensible initalization
    'INIT' : 'sensible',
    
    'MAX_DEPTH': 50,

    'MAX_INIT_TREE_DEPTH' : 11,
    'MIN_INIT_TREE_DEPTH' : 7,
    
    'GENO_ENCODE' : None,
    'SCALE_OUTCOME': False,
    'SCALE_CONTIN': False,
    
    'MISSING':None,
    'SELECTION': 'tournament',
    
    'GENS_MIGRATE': 25,
    'OUTCOME' : None,
    
    'INCLUDEDVARS' : None

}



def less_than(parameters,smaller,bigger):
    """
    Check that first value <= bigger

    Parameters:
        parameters: Dictionary with key/value pairs for all parameters
        smaller: number
        bigger: numberr
        
    Returns: True if smaller <= bigger
    """
    if parameters[smaller] > parameters[bigger]:
        print(f"{smaller} must be <= {bigger}")
        return False
    else:
        return True


def valid_parameters(parameters):
    """
    Check that all parameters passed are valid for ATHENA run

        Parameters:
            parameters: Dictionary with key/value pairs for all parameters

        Returns: 
            True if all parameters are valid, False otherwise
    """
    
    all_valid = True
    # check that mutation rate and crossover are 0-1.0
    for rate in ['P_CROSSOVER', 'P_MUTATION']:  
        if parameters[rate] < 0.0 or parameters[rate] > 1.0:
            print(f"{rate} must be in range 0-1.0")
            all_valid=False
            
    
    if  not less_than(parameters, 'ELITE_SIZE', 'HALLOFFAME_SIZE'):
        all_valid = False
    
    if parameters['INIT'] == 'random' and not \
        less_than(parameters, 'MIN_INIT_GENOME_LENGTH', 'MAX_INIT_GENOME_LENGTH'):
        all_valid = False
    
    if parameters['INIT'] == 'sensible' and not \
        less_than(parameters, 'MIN_INIT_TREE_DEPTH', 'MAX_INIT_TREE_DEPTH'):
        all_valid = False
        
    # OUTCOME_FILE and GRAMMAR_FILES are required 
    if not os.path.isfile(parameters['GRAMMAR_FILE']):
        print(f"GRAMMAR_FILE {parameters[GRAMMAR_FILE]} not found")
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

    if parameters['FITNESS'] not in ['r-squared', 'balanced_acc']:
        print("FITNESS must be either r-squared or balanced_acc")
        all_valid = False
    
    if parameters['INIT'] not in ['random', 'sensible']:
        print("INIT must be either sensible or random")
        all_valid = False
    
    if parameters['CODON_CONSUMPTION'] not in ['eager', 'lazy']:
         print("CODON_CONSUMPTION must be either eager or lazy")
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
    
    return all_valid    
    

def load_param_file(file_name):
    """
    Load in a params text file and set the params dictionary directly.

    :param file_name: The name/location of a parameters file.
    :return: Nothing.
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

            # Parameters files are parsed by finding the first instance of a
            # colon.
            split = line.find(":")

            # Everything to the left of the colon is the parameter key,
            # everything to the right is the parameter value.
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
        
        
def parse_cmd_args(arguments, has_mpi=False):
    """
    Parser for command line arguments specified by the user. Specified command
    line arguments over-write parameter file arguments, which themselves
    over-write original values in the algorithm.parameters.params dictionary.

    The argument parser structure is set up such that each argument has the
    following information:

        dest: a valid key from the algorithm.parameters.params dictionary
        type: an expected type for the specified option (i.e. str, int, float)
        help: a string detailing correct usage of the parameter in question.

    Optional info:

        default: The default setting for this parameter.
        action : The action to be undertaken when this argument is called.

    NOTE: You cannot add a new parser argument and have it evaluate "None" for
    its value. All parser arguments are set to "None" by default. We filter
    out arguments specified at the command line by removing any "None"
    arguments. Therefore, if you specify an argument as "None" from the
    command line and you evaluate the "None" string to a None instance, then it
    will not be included in the eventual parameters.params dictionary. A
    workaround for this would be to leave "None" command line arguments as
    strings and to eval them at a later stage.

    :param arguments: Command line arguments specified by the user.
    :return: A dictionary of parsed command line arguments, along with a
    dictionary of newly specified command line arguments which do not exist
    in the params dictionary.
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
    parser.add_argument('--param_file',
                        dest='PARAM_FILE',
                        type=str,
                        help='Specifies the parameters file to be used. Must '
                             'include the full file extension. Parameters defined '
                             'in the file are overriden by any command line arguments')
    parser.add_argument('--popsize',
                        dest='POPULATION_SIZE',
                        type=int,
                        default=250,
                        help='Sets population size for GE algorithm')
    parser.add_argument('--gens',
                        dest='GENERATIONS',
                        type=int,
                        default=50,
                        help='Sets number of generations in evolution')
    parser.add_argument('--pcrossover',
                        dest='P_CROSSOVER',
                        type=float,
                        default=0.8,
                        help='Sets probability of a crossover during selection')
    parser.add_argument('--pmut',
                        dest='P_MUTATION',
                        type=float,
                        default=0.01,
                        help='Sets probability per codon of mutation')
    parser.add_argument('--nelite',
                        dest='ELITE_SIZE',
                        type=int,
                        default=1,
                        help='Sets number of best networks carried over to next'
                         'generation')
    parser.add_argument('--hof_size',
                        dest='HALLOFFAME_SIZE',
                        type=int,
                        default=1,
                        help='Sets number of the best networks to save for reporting. '
                        'Must be >= to elite size')
    parser.add_argument('--codon_size',
                        dest='CODON_SIZE',
                        type=int,
                        default=250,
                        help='Maximum value of a codon in the genome of an individual'
                        'in the evolutionary population. At a minimum it should be'
                        '>= the largest number of choices for a rule in the grammar')
    parser.add_argument('--codon_consumption',
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
    parser.add_argument('--genome_type',
                        dest='GENOME_TYPE',
                        choices=['standard', 'leap', 'mcge'],
                        default='standard',
                        type=str,
                        help='Sets GE genome type to use (standard, leap or mcge) '
                         'generation')
    parser.add_argument('--outcome_file',
                        dest='OUTCOME_FILE',
                        type=str,
                        help='Sets name of file containing outcomes (phenotypes) in'
                        ' input data')
    parser.add_argument('--outcome',
                        dest='OUTCOME',
                        type=str,
                        help='Column header to use in outcome (default is to use the first after the ID)')
    parser.add_argument('--geno_file',
                        dest='GENO_FILE',
                        type=str,
                        help='Sets name of file containing genotypes (0,1,2) in'
                        ' input data')
    parser.add_argument('--contin_file',
                        dest='CONTIN_FILE',
                        type=str,
                        help='Sets name of file containing continuous variables in'
                        ' input data')
    parser.add_argument('--grammar',
                        dest='GRAMMAR_FILE',
                        type=str,
                        help='Sets name of grammar')
    parser.add_argument('--fitness',
                        dest='FITNESS',
                        choices=['r-squared', 'balanced_acc'],
                        default='r-squared',
                        type=str,
                        help='Sets metric for fitness (balanced_acc or r-squared)')
    parser.add_argument('--rand',
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
    parser.add_argument('--max_init_tree',
                        dest='MAX_INIT_TREE_DEPTH',
                        type=int,
                        default='11',
                        help='Sets maximum depth for tree created by sensible '
                        'initialization')
    parser.add_argument('--min_init_tree',
                        dest='MIN_INIT_TREE_DEPTH',
                        type=int,
                        default=7,
                        help='Sets minimum depth for tree created by sensible '
                        'initialization')
    parser.add_argument('--min_rand_genome',
                        dest='MIN_INIT_GENOME_LENGTH',
                        type=int,
                        default=30,
                        help='Sets minimum genome length for random intialization')
    parser.add_argument('--max_rand_genome',
                        dest='MAX_INIT_GENOME_LENGTH',
                        type=int,
                        default=200,
                        help='Sets maximum genome length for random initaliztion')
    parser.add_argument('--geno_encode',
                        dest='GENO_ENCODE',
                        choices=['add_quad', 'additive'],
                        type=str,
                        help='Sets genotype encoding. Must be either add_quad or additive')
    parser.add_argument('--scale_outcome',
                        dest='SCALE_OUTCOME',
                        action="store_true",
                        help="Sets flag for scaling outcome variable from 0 to 1.0")
    parser.add_argument('--scale_contin',
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
    parser.add_argument('--test_outcome',
                        dest='TEST_OUTCOME_FILE',
                        type=str,
                        help="Set name of outcome file for designated test dataset")
    parser.add_argument('--test_geno',
                        dest='TEST_GENO_FILE',
                        type=str,
                        help="Set name of genoype data file for designated test dataset")
    parser.add_argument('--test_contin',
                        dest='TEST_CONTIN_FILE',
                        type=str,
                        help="Set name of continuous data file for designated test dataset")
    parser.add_argument('--max_depth',
                        dest='MAX_DEPTH',
                        type=int,
                        default=50,
                        help='Sets max depth for mapping with individuals exceeding this being invalid')
    parser.add_argument('--color_map',
                        dest='COLOR_MAP_FILE',
                        type=str,
                        help="File specifying colors to use for specific inputs")
    parser.add_argument('--included-vars',
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

    # Parse command line arguments using all above information.
    args, unknown = parser.parse_known_args(arguments)

    # All default args in the parser are set to "None". Only take arguments
    # which are not "None", i.e. arguments which have been passed in from
    # the command line.
    cmd_args = {key: value for key, value in vars(args).items() if value is
                not None}
    # Set "None" values correctly.
    for key in sorted(cmd_args.keys()):
        # Check all specified arguments.

        if type(cmd_args[key]) == str and cmd_args[key].lower() == "none":
            # Allow for people not using correct capitalisation.

            cmd_args[key] = None

    return cmd_args, unknown

                         
def set_params(command_line_args, create_files=True, has_mpi=False):
    """
    This function parses all command line arguments specified by the user.
    If certain parameters are not set then defaults are used (e.g. random
    seeds, elite size). Sets the correct imports given command line
    arguments. Sets correct grammar file and fitness function.

    :param command_line_args: Command line arguments specified by the user.
    :return: Nothing.
    """

    cmd_args, unknown = parse_cmd_args(command_line_args, has_mpi)

    if unknown:
        # We currently do not parse unknown parameters. Raise error.
        s = "algorithm.parameters.set_params\nError: " \
            "unknown parameters: %s\nYou may wish to check the spelling, " \
            "add code to recognise this parameter, or use " \
            "--extra_parameters" % str(unknown)
        raise Exception(s)

    # LOAD PARAMETERS FILE
    # NOTE that the parameters file overwrites all previously set parameters.
    if 'PARAM_FILE' in cmd_args:
        load_param_file(cmd_args['PARAM_FILE'])

    # Join original params dictionary with command line specified arguments.
    # NOTE that command line arguments overwrite all previously set parameters.
    params.update(cmd_args)

    return params
    
    