## Running ATHENA

To run ATHENA, you can pass the main file to Python (version 3.10 or later):

```python athena.py```

or, if you have made the file executable:

```athena.py```

When passing no arguments, ATHENA will output the command-line options available to it:

    >> python athena.py
    Welcome to ATHENA - Help. The following are the available command line arguments. Please see manual
    for more detailed information on the parameters.

    ATHENA command-line usage:
    --codon-consumption {eager,lazy}
                            Options are eager and lazy. Specifies whether grammar will consume codons
                            when only one choice for a rule in the grammar. (default: eager)
    --codon-size CODON_SIZE
                            Maximum value of a codon in the genome of an individualin the evolutionary
                            population. At a minimum it should be>= the largest number of choices for a
                            rule in the grammar (default: 250)
    --color-map-file COLOR_MAP_FILE ```

## Example files

The files used in this tutorial are distributed in the example directory. The tutorial assumes you are running in the parent directory of the examples directory.

## Minimal options

At a minimum, ATHENA requires a grammar file, an outcome file and either a genotype data or continuous data input file. This example shows a minimal run where all other options us the default values.

    athena.py --grammar-file example/genn.bnf --contin-file example/contin.txt --outcome-file example/outcome_casecon.txt

    CV:  1

    gen = 0 , Best fitness = (0.565,)
    gen = 1 , Best fitness = (0.565,) , Number of invalids = 133
    gen = 2 , Best fitness = (0.565,) , Number of invalids = 93
    gen = 3 , Best fitness = (0.565,) , Number of invalids = 100
    gen = 4 , Best fitness = (0.565,) , Number of invalids = 77

ATHENA will print its progress and list the cross validation number, the current generation, the best fitness value for a network in its population and the number of invalid networks in the population.

The output files will be written to the current working directory and have the basename of 'athena_results'. ATHENA writes a plot of the best network and log of each cross-validation interval. It also creates a summary file listing the networks and fitnesses of the best networks across all cross-validations.

    athena_results.cv1.log		athena_results.cv3.log		athena_results.cv5.log
    athena_results.cv1.png		athena_results.cv3.png		athena_results.cv5.png
    athena_results.cv2.log		athena_results.cv4.log		athena_results_summary.txt
    athena_results.cv2.png		athena_results.cv4.png

## Specifying outputs name and location

The *--out/OUT* parameter can contain a path (either relative or full) along with the basename portion of the output files. For example, to have the files written to a results directory under your current directory and named experiment1:

    athena.py --grammar-file example/genn.bnf --contin-file example/contin.txt --outcome-file example/outcome_casecon.txt --out results/example1

The directory must exist prior to the run. ATHENA will overwrite files with the same names.

## Using genotype data

ATHENA will accept genotype data using the *--geno-file/GENO_FILE* parameter. The values in the genotype file should all be 0,1,2 or the value for missing (set by *--missing/MISSING* ). The values can be encoded as either [additive or add_quad](methodology.md/#encoding-normalization).


    athena.py --grammar-file example/genn.bnf --contin-file example/contin.txt --outcome-file example/outcome_casecon.txt --out results/example1 --geno-file example/genos.txt --geno-encode add_quad


## Using a parameter file

All options (except for *--param-file* itself) can be passed to ATHENA in a [parameter file](manual/input_files.md#parameters-file). The parameters file can serve as a record for the parameters used in a run. The options can be mixed between the command-line and the file with the value in the parameter file taking precendence when an option appears in both.

    python athena.py --param-file example/rsquared.config


## Limiting variables in the algorithm

The option *--includedvars/INCLUDEDVARS* filters the input variables to only the variables included in the list.

    python athena.py --param-file example/included.config

## Color coding plots

A [color mapping file](manual/input_files.md#color-mapping-file) maps specific variables name to colors and can be included to color those nodes in the network plots.

    python athena.py --param-file example/included.config --color-map-file example/color_map.txt


## Parallel run

If MPI and the Python package mpi4py are installed, ATHENA can run in parallel mode. 

    mpirun -np 4 athena.py --grammar-file example/genn.bnf --contin-file example/contin.txt --outcome-file example/outcome_casecon.txt --out results/parallel

