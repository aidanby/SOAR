# api-representation-learning
learning an API representation for library migration with program synthesis


## Dev Environment Setup

- Prerequisite:
  - python 3.8.5 (anaconda)
  
- Python packages dependencies:
  - click
  - z3-solver
  - tensorflow
  - pytorch
  - rpy2
  - transformers
  - fuzzywuzzy
  - matplotlib
  - numpy 
  - nltk
  - lark-parser
  - mkl
  - tqdm

## Run benchmarks:

The DL benchmarks can be found at ```api-learning-representation/autotesting/benchmarks```, and the 
data wrangling benchmarks at ```api-learning-representation/synthesis/synthesizer/dplyr_to_pd/dplyr```.

- Add the repository to the PYTHONPATH environment variable:
```
    $ export PYTHONPATH=/path/to/api-learning-representation
```
- To run a specific benchmark (DL) do the following:
```
    $ cd synthesis/synthesizer/tf_to_torch
    $ python3 torch_synthesizer.py -b name
```
- Conversly, to run a data wrangling task execute the following code:
```
    $ cd synthesis/synthesizer/dplyr_to_pandas
    $ python3 pd_synthesizer.py -b name
```

  
