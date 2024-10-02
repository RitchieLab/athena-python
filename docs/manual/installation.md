
# Installation 

## Prerequisites
The following prerequisites are required to run [ATHENA](https://github.com/RitchieLab/athena-python):

* Python3 (version 3.10+)
* Python modules 
    * DEAP (evolutionary algorithm framework)
    * scikit-learn (machine learning)
    * netgraph (graph visualizations)
    * networkx (network creation/manipulation)
    * pandas

## Parallelization
ATHENA supports parallelization of the algorithm and depends on the mpi4py python module. That feature is optional and the software will run without it.

## Installing ATHENA
The ATHENA code can be accessed on [GitHub](https://github.com/RitchieLab/athena-python). The current method to use it is to clone the repository 

```
git clone https://github.com/RitchieLab/athena-python.git
```

```
python setup.py install
```

This will place the Biofilter and LOKI files in your system’s usual place for Python-based software, which is typically alongside Python itself. The installation can also be done in a different location by using the “`--prefix`” or “`--exec-prefix`” options.

## Compiling Prior Knowledge
The LOKI prior knowledge database **must** be generated before Biofilter can be used. This is done with the “`loki-build.py`” script which was installed along with Biofilter. There are several options for this utility which are detailed below, but to get started, you just need “`--knowledge`” and “`--update`”:

```
loki-build.py --knowledge loki.db --update
```

This will download and process the bulk data files from all supported knowledge sources, storing the result in the file “`loki.db`” (which we recommend naming after the current date, such as “loki-20240521.db”). The update process may take as few as 4 hours or as many as 24 depending on the speed of your internet connection, processor and filesystem, and requires up to 30 GB of free disk space: 10-20 GB of temporary storage (“C:\TEMP” on Windows, “/tmp” on Linux, etc) plus another 5-10 GB for the final knowledge database file.

By default, the LOKI build script will delete all sources’ bulk data downloads after they have been processed. If the knowledge database will be updated frequently, it is recommended to keep these bulk files available so that any unchanged files will not need to be downloaded again. This can be accomplished with the “`--archive`” option.


