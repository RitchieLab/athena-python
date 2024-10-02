
# Input Files



## Parameters file

An optional parameters file can be used in an ATHENA run. The format is *keyword:*  *value* and parameters specified in the file take precedence over ones on the command line. The parameters file is a useful way to replicate previous runs of ATHENA. See the *[Parameters](../parameters)* list for all available options.

    GRAMMAR_FILE:   example/genn.bnf
    OUTCOME_FILE:   example/outcome_contin.txt
    CONTIN_FILE:    example/contin.txt
    SCALE_CONTIN:   True
    SCALE_OUTCOME:  True
    GENO_FILE:      example/genos.txt
    GENO_ENCODE:    add_quad
    OUT:            results/athena
    FITNESS:        r-squared
    POP_SIZE:       250
    GENERATIONS:    5

## Data files

ATHENA data files are whitespace delimited and the first row must contain the column headers. A data file must have an *ID* column followed by the labels for the data (genotypic, continuous, or outcome) included in the file. The ID column is used to match the data presented in rows across the various input files.

An outcome file and at least one of the genotype and continuous data files are required for an ATHENA run.

### Genotype file
A genotype input file must contain the ID column. The genotypes must be 0,1,2 or the value designated by the *MISSING* parameter

    ID SNP1 SNP2 SNP3 SNP4
    id1 1 2 0 0
    id2 0 2 0 0
    id3 1 1 1 0
    id4 0 0 1 1
    id5 0 1 1 0
    id6 0 1 1 1
    id7 0 0 1 1

### Continuous data file
A continuous data file must contain the ID column. Data can be any numeric values or the value designated by the *MISSING* parameter

    ID C1 C2 C3 C4
    id1 -0.19824033325 -0.25884851921 1.29742330517 0.04194009721
    id2 -0.32127593922 0.04965043519 1.52750412357 0.04550146096
    id3 -0.89618956836 -0.2082694151 0.83338920376 -0.12484128176
    id4 -0.00847565279 0.15470277072 0.26226553399 -0.00364302721
    id5 -0.52438207799 0.19822901217 1.19583036534 -0.04283847354
    id6 0.23387312884 -0.11317690779 0.64984994477 -0.07093586693
    id7 -0.23458885217 0.27120910755 1.18926694272 -0.10301740165

### Outcome file
The outcome file is required and must contain an ID column

    ID outcome1 outcome2
    id1 0.4056314684 0.7949572102
    id2 -0.7938889637 -0.7737498065
    id3 1.291416007 -0.0263485549
    id4 -0.241405327 1.064138677
    id5 -0.1028663478 0.0355249568
    id6 0.1577496992 -0.0535040572
    id7 0.5924636053 -0.8850819371

### Testing files

The testing files are designated by parameters TEST_OUTCOME_FILE, TEST_GENO_FILE and TEST_CONTIN_FILE file. The formats match the equivalent input files described above. The testing files are used in a run where training is done on the data files provided by OUTCOME_FILE, GENO_FILE and CONTIN_FILE and the testing is done on these files. In other words, it allows the user to split the data before a run.

## Grammar file

The grammar file defines the collection of terminal symbols, non-terminal symbols, production rules and the start symbol for creating networks. **ATHENA** automatically adjusts the production rule for the <v> non-terminal to include all the input variables loaded.

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
    <cop> ::= float(<num>)
            | <e>
    <e>   ::= (<cop> + <cop>)
            | (<cop> - <cop>)
            | (<cop> * <cop>)
            | pdiv(<cop>,<cop>)
    <num> ::= <dig>.<dig>
            | .<dig>
            | .<dig><dig>
            | <dig>.<dig><dig>
            | <dig><dig>.<dig><dig>
    <dig> ::= 0
            | 1
            | 2
            | 3
            | 4
            | 5
            | 6
            | 7
            | 8
            | 9
    <v> ::= x[0] | x[1] | x[2] | x[3] | x[4] | x[5] | x[6] | x[7] | x[8] | x[9]


## Color mapping file

The color mapping file specifies colors for specific variables in the input files. If the variables appear in the networks plotted at the end of the run, those network nodes will appear in the designated colors.

The file has 3 columns that are tab-delimited: *Category*, *Color*, *Inputs*. The Color column accepts [X11/CSS colors](https://en.wikipedia.org/wiki/Web_colors#CSS_colors) or hex colors (#ff5733) 

    Category	Color	Inputs
    APOE	orange	SNP3 SNP92 SNP22
    ABCA7	lightblue	SNP12 SNP27
    Environmental	lightgreen	C30 C2
    Response to cytokine	royalblue	SNP79

