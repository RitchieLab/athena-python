## Using ATHENA
ATHENA can be run from the command-line by executing "athena.py" (or "python athena.py") and specifying the desired inputs, outputs and other optional settings. Options can be either passed on the command-line or can be included in a paramters file that can be passed to ATHENA on the command-line. 

Every option (except for the parameter file option) is available on both the command-line and in the parameter file. The available options are the same no matter where they appear, but are formatted differently. Options on the command line are lower-case, start with two dashes and may contain single dashes to separate words (such as outcome-file), while in a configuration file the same option would be in upper-case, contain no dashes and instead use underscores to separate words (i.e. OUTCOME_FILE).

Parameters in the parameter file take precedence over any passed on the command-line. Many parameters have default values. 

All options are listed here in both their command line and parameter file forms. If an option allows or requires any further arguments, they are also noted along with their default values, if any. Arguments which are required are enclosed in <angle brackets\>, while arguments which are optional are enclosed in [square brackets].


# File Options
| **Command-line** | **Parameter file** | **Arguments** | **Information** |
|---|---|---|---|
| `--color-map-file ` | COLOR_MAP_FILE |[STR] | Default: *NONE*. File for specifying colors in output plots|
| `--contin-file` | CONTIN_FILE | [STR] |Default: *NONE*. File containing continuous input data |
| `--geno-file` | GENO_FILE | [STR] |Default: *NONE*. File containing genotypic input data |
| `--grammar-file` | GRAMMAR_FILE | [STR] |Default: *NONE*. File containing BNF grammar for GE |
| `--outcome-file` | OUTCOME_FILE | [STR] |Default: *NONE*. File containing outcome (phenotype) variables |
| `--param-file` |  | [STR] |Default: *NONE*. File containing parameters |
| `--test-contin-file` | TEST_CONTIN_FILE | [STR] |Default: *NONE*. File containing test continuous input data |
| `--test-geno-file` | TEST_GENO_FILE | [STR] |Default: *NONE*. File containing test genotypic input data |
| `--test-outcome-file` | TEST_OUTCOME_FILE | [STR] |Default: *NONE*. File containing test outcome (phenotype) variables |

# Data Options
| **Command-line** | **Parameter file** | **Arguments** | **Information** |
|---|---|---|---|
| `--cv` | CV | [INT] |Default: *5*. Number of cross-validations to split data |
| `--geno-encode` | GENO_ENCODE | [STR] |Default: *NONE*.  Sets genotype encoding. Must be either add_quad or additive |
| `--includedvars` | INCLUDEDVARS | [STR STR ...] |Default: *NONE*. Variable names to be included in run (accepts multiple arguments) |
| `--missing` | MISSING | [STR] |Default: *NONE*. Value denoting missing data in input files |
| `--outcome` | OUTCOME | [STR] |Default: *NONE*. Column header designating which variable to use. Uses first when not designated. |
| `--scale-contin` | SCALE_CONTIN | |Default: *False*. Scale continuous input variables from 0-1.0 |
| `--scale-outcome` | SCALE_OUTCOME | |Default: *False*. Scale outcome variables form 0-1.0 |

# Reporting Options
| **Command-line** | **Parameter file** | **Arguments** | **Information** |
|---|---|---|---|
| `--hof-size` | HOF_SIZE | [INT] |Default: *1*. Sets number of the best networks to save for reporting. Must be >= to elite size|
| `--out` | OUT | [STR] |Default: *athena_results*. Specifies name prefix for output file|


# Algorithm Options
| **Command-line** | **Parameter file** | **Arguments** | **Information** |
|---|---|---|---|
| `--codon-consumption` | CODON_CONSUMPTION | [STR] |Default: *eager*. Whether grammar will consume codons when only one choice for a rule in the grammar. (eager, lazy) |
| `--codon-size` | CODON_SIZE | [INT] |Default: *NONE*. Maximum value of a codon in the genome of an individual in the evolutionary population. At a minimum it should be>= the largest number of choices for a rule in the grammar|
| `--fitness` | FITNESS | [STR] |Default: *balanced_acc*. Metric for fitness (balanced_acc or r-squared)|
| `--genome-type` | GENOME_TYPE | [STR] |Default: *standard*. GE genome type to use (standard, leap or mcge) generation |
| `--gens` | GENS | [INT] |Default: *50*. Number of generations in evolution |
| `--init` | INIT | [STR] |Default: *sensible*. Initialization procedure (sensible or random)|
| `--max-depth` | MAX_DEPTH | [INT] |Default: *100*. Max depth for mapping with individuals exceeding this being invalid |
| `--max-init-tree-depth` | MAX_INIT_TREE_DEPTH | [INT] |Default: *11* Maximum depth for tree created by sensible initialization |
| `--max-init-genome-length` | MAX_INIT_GENOME_LENGTH | [INT] |Default: *250*. Maximum genome length for random initializtion |                   
| `--min-init-tree-depth` | MIN_INIT_TREE_DEPTH | [INT] |Default: *7*. Minimum depth for tree created by sensible initialization|
| `--min-init-genome-length` | MIN_INIT_GENOME_LENGTH | [INT] |Default: *50*. Minimum genome length for random initializtion |     
| `--nelite` | NELITE | [INT] |Default: *1*. number of best networks carried over to next generation |
| `--p-crossover` | P_CROSSOVER | [FLOAT] |Default: *0.8*. Probability of a crossover during selection |
| `--crossover` | CROSSOVER | [STR] |Default: *onepoint*. Crossover operator type. (onepoint, match, block)|
| `--crossover2` | CROSSOVER2| [STR] |Default: *NONE*. Crossover operator type for algorithm to switch to. (onepoint, match, block)|
| `--gen-cross-switch` | GEN_CROSS_SWITCH | [INT] |Default: *NONE*. Generation at which to switch crossover operator|
| `--p-mut` | P_MUT | [FLOAT] |Default: *0.01*. Probability of a crossover during selection|
| `--pop-size` | POP_SIZE | [INT] |Default: *250*. Population size for GE algorithm|
| `--random-seed` | RANDOM_SEED | [INT] |Default: *12345*. Random seed |
| `--selection` | SELECTION | [STR] |Default: *tournament*. Selection type (tournament, lexicase, epsilon_lexicase) |

# Parellization Options
| **Command-line** | **Parameter file** | **Arguments** | **Information** |
|---|---|---|---|
| `--gens-migrate` | GENS_MIGRATE | [INT] |Default: *25*. Generational interval for migrating best individuals for multi-process run |




