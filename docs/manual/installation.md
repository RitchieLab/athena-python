
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

Then from the main directory:
```
pip install .
```
This will place the ATHENA and files in your system’s usual place for Python-based software, which is typically alongside Python itself. It will also add the commmand 'athena.py' to your path. The installation can also be done in a different location by using the “`--prefix`” or “`--exec-prefix`” options. 

