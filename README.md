just edited it from my phone so it is untested for now
and unfinished

Simple tool That implement KNN Imputation for GPUs

## Dependencies

- Cupy
- Pandas
- argparse

## How it works

```
python KnnImputer.py -h
usage: KnnImputer.py [-h] -file FILE [-n_neigh N_NEIGH]

KNN Imputer for GPU

optional arguments:
  -h, --help        show this help message and exit
  -file FILE        Name of the file to be imputed
  -n_neigh N_NEIGH  Number of neighbors in Knn

```

## Example running 

``` 
python KnnImputer.py -file "dataframe.csv" -n_neigh 5
```
