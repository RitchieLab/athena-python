**ATHENA** utilizes machine learning techniques to analyze high-throughput categorical (i.e. single nucleotide polymorphisms) and quantitative (i.e. gene expression levels) predictor variables to generate multivariable models that predict either a categorical (i.e. disease status) or quantitative (i.e. cholesterol levels) outcomes.

## Data 

### Filtering

The default behavior for **ATHENA** is to include all input variables included in the data set and use the first outcome variable in that file. The parameter [INCLUDEDVARS](manual/parameters.md/#data-options) specifies the names of variables to include in a run. The parameter [OUTCOME](manual/parameters.md/#data-options) specifies the name of the outcome to use.

### Missing data

**ATHENA** handles missing data in the input variables by skipping any samples for a network when any single variable in the network is missing. The percentage of missing data for a network is reported in the summary file at the end of the run. The algorithm functions best with no missing data, so it the input data set should be as complete as possible.

### Encoding / normalization

Continuous input and outcome variables can be normalized using the parameters [SCALE_CONTIN](manual/parameters.md/#data-options) and [SCALE_OUTCOME](manual/parameters.md/#data-options). The variables will be scaled from 0 to 1 using min-max normalization.

For genotypes, the encoding can be set as additive (-1,0,1) or the genotype representation created by Jurg Ott can be used. This method creates 2 new "dummy" variables for each SNP genotype: one to encode for linear, or allelic effect and to encode to a non-linear or quadratic effect. The additive encoding in **ATHENA** corresponds to the Linear column below:

| **Genotype** | **Original** | **Linear** | **Quadratic** |
|---|---|---|---|
| AA | 0 | -1 | -1 |
| Aa | 1 | 0 | 2 |
| aa | 2 | 1 | -1 |

### Cross validation

**ATHENA** utilizes cross validatino to evaluate the evolved models on unseen data. The data are divided into equal parts and one part is left out of each cross validation to act as the test set for the evolved models. The parameter [CV](manual/parameters.md/#data-options) controls the number of cross validations used. If CV is set to 1, then all the data will be used for training and there will be not test score reported.

As an alternative, ATHENA will accept data files that are designated as test files (TEST_OUTCOME_FILE,TEST_GENO_FILE,TEST_CONTIN_FILE). In this case, these files will be used as the test set and a single run is performed.


## Grammatical Evolution

### Grammar

**ATHENA** uses a context-free grammar to map the genomes in the population to final networks. The grammar can be modified to change the final networks. 

For example, the sample grammar file begins its mapping:

    <p> ::= <pn>(<pinput>)
    <pn> ::= PA
        | PS
        | PM
        | PD
    <pinput> ::= [<winput>,<winput>]
        | [<winput>,<winput>,<winput>]
        | [<winput>,<winput>,<winput>,<winput>]
        | [<winput>,<winput>,<winput>,<winput>,<winput>]
    <winput> ::= (<cop> * <v>)
            | (<cop> * <p>)

It can be altered to insure that another layer always appears in the final networks by inserting another non-terminal:

    <p> ::= <pn>(<pinput1>)
    <pn> ::= PA
        | PS
        | PM
        | PD
    <pinput> ::= [(<cop> * <p>),(<cop> * <p>)]
        | [(<cop> * <p>), (<cop> * <p>), (<cop> * <p>)]
    <pinput> ::= [<winput>,<winput>]
        | [<winput>,<winput>,<winput>]
    <winput> ::= (<cop> * <v>)
        | (<cop> * <p>)

Alternatively, the grammar can be modified to include only specific node types:

    <p> ::= <pn>(<pinput>)
    <pn> ::= PA
    <pinput> ::= [<winput>,<winput>]
        | [<winput>,<winput>,<winput>]
        | [<winput>,<winput>,<winput>,<winput>]
        | [<winput>,<winput>,<winput>,<winput>,<winput>]
    <winput> ::= (<cop> * <v>)
            | (<cop> * <p>)


### Genomes

**ATHENA** has three genome types available for use. 

1. Standard linear - each production rule consumes a codon sequentially.
2. Multi-chromosomal genome grammatical evolution [(link)](https://core.ac.uk/download/pdf/12529831.pdf) - each non-terminal has its own linear list of codons and codons are consumed sequentially only for that non-terminal.
3. LEAP [(link)](https://dl.acm.org/doi/10.1145/3583133.3590680) - a linear genome divided into frames. A frame consists of a complete set of codons that match the number of non-terminals in the grammar.

The genomes primarily affect the behavior of the [crossover operator](#Crossover). The MCGE and LEAP genomes both maintain the context of the codons along the genome. In the standard linear genome, a codon may be used for different non-terminals depending on the codons that precede it.

### Fitness

**ATHENA** offers five fitness metrics, balanced accuracy, AUC, F1 score, AUPRC and r-squared. [Balanced accuracy](https://en.wikipedia.org/wiki/Precision_and_recall#Imbalanced_data), [F1 score](https://en.wikipedia.org/wiki/F-score), [area under the curve (AUC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) and [area under Precision-Recall (PR) curve (AUPRC)](https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248) should be used with binary outcomes (i.e. disease status). [R squared](https://en.wikipedia.org/wiki/Coefficient_of_determination) should be used with continuous outcome variables (i.e. cholesterol levels).

### Initialization

Two options exist for initializing the population used in grammatical evolution, sensible and random.

#### random

For the standard genome type, genomes are randomly generated codon by codon. The length is set by the parameters MIN_INIT_GENOME_LENGTH and MAX_INIT_GENOME_LENGTH and each codon is less than or equal to the CODON_SIZE parameter.

For the MCGE genome type, each chromosome's size is set by the parameters MIN_INIT_GENOME_LENGTH and MAX_INIT_GENOME_LENGTH.

For the LEAP genome type, the values MIN_INIT_GENOME_LENGTH and MIN_INIT_GENOME_LENGTH are divided by the frame size to determine the minimum and maximum number of frames in a genome. For example, if the number of non-terminals is 8, MIN_INIT_GENOME_LENGTH is 80 and the MAX_INIT_GENOME_LENGTH is 400, then the number of complete frames generated will be between 10 and 50 (length / number of non-terminals).

#### sensible

Sensible initialization operates similarly for all types of genomes. The process generates complete trees using the grammar passed to the software to generate trees. For each non-terminal used, a codon is created and saved to the genome. The parameters MIN_INIT_TREE_DEPTH and MAX_INIT_TREE_DEPTH set the limits on the size of the trees. Sensible initialization in **ATHENA** generates trees grown to the maximum depth specified for 50% of the intiial population. For the other half, the trees are guaranteed to be at least as deep as the MIN_INIT_TREE_DEPTH while being no more than the depth of MAX_INIT_TREE_DEPTH.


### Selection 

The selection of individuals for the next generation during evolution can be set to either tournament, lexicase or epsilon-lexiscase. The tournament is a standard pairwise selection with the better of the two being selected. If lexicase type selection is desired, then [lexicase](https://discourse.pushlanguage.org/t/lexicase-selection/90) is appropriate for use with binary outcomes (case/control) while [epsilon-lexicase](https://discourse.pushlanguage.org/t/epsilon-lexicase-selection/571) should be used with continuous outcomes.


### Crossover

**ATHENA** utilizes one point effective crossover. For the standard linear chromosome the codons past the point of crossover in each genome are crossed. For LEAP genomes, crossover occurs between the frames (a set of contiguous codons with each position corresponding to a non-terminal) to maintain the context of each codon. For MCGE, crossover occurs between a single matching chromosome in each genome.

### Mutation

Mutation is a straightforward process as each codon in all types of genomes are checked against the mutation rate (P_MUT) for possible mutation. If a codon is mutated, the codon value is changed to a random integer up the maximum value specified (CODON_SIZE).


